import glob
import json
import pathlib
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from agents import SACAgent
from tqdm import trange
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent

import ogbench.locomaze  # noqa

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'pointmaze-random-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'navigate', 'Dataset type.')
flags.DEFINE_string('restore_path', 'experts/ant', 'Expert agent restore path.')
flags.DEFINE_integer('restore_epoch', 400000, 'Expert agent restore epoch.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.2, 'Gaussian action noise level.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Maximum number of steps in an episode.')


def main(_):
    assert FLAGS.dataset_type in ['path', 'navigate', 'stitch', 'explore']
    # 'path': Reach a single goal and stay there.
    # 'navigate': Repeatedly reach randomly sampled goals in a single episode.
    # 'stitch': Reach a nearby goal that is 4 cells away and stay there.
    # 'explore': Repeatedly follow random directions sampled every 10 steps.

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        ob_type='pixels',
        terminate_at_goal=False,
        max_episode_steps=FLAGS.max_episode_steps,
    )

    # Initialize oracle agent.
    if 'point' in FLAGS.env_name:
        def actor_fn(ob, temperature):
            return ob[-2:]
    else:
        # Load agent config.
        restore_path = FLAGS.restore_path
        candidates = glob.glob(restore_path)
        assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

        with open(candidates[0] + '/flags.json', 'r') as f:
            agent_config = json.load(f)['agent']

        # Load agent.
        agent = SACAgent.create(
            FLAGS.seed,
            np.zeros(env.observation_space.shape[0]),
            env.action_space.sample(),
            agent_config,
        )
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)

    # Collect data.
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        maze_id = np.random.randint(len(env.unwrapped._maze_maps))
        maze_map = env.unwrapped._maze_maps[maze_id]

        # Store all empty cells and vertex cells.
        empty_cells = []
        vertex_cells = []
        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] == 0:
                    empty_cells.append((i, j))

                    # Exclude hallway cells.
                    if (
                        maze_map[i - 1, j] == 0
                        and maze_map[i + 1, j] == 0
                        and maze_map[i, j - 1] == 1
                        and maze_map[i, j + 1] == 1
                    ):
                        continue
                    if (
                        maze_map[i, j - 1] == 0
                        and maze_map[i, j + 1] == 0
                        and maze_map[i - 1, j] == 1
                        and maze_map[i + 1, j] == 1
                    ):
                        continue
                    vertex_cells.append((i, j))
    
        # Sample an initial state from all cells.
        init_ij = empty_cells[np.random.randint(len(empty_cells))]
        # Sample a goal state from vertex cells.
        valid_goals = [c for c in vertex_cells if c != init_ij]
        goal_ij = valid_goals[np.random.randint(len(valid_goals))]
    
        # Reset the environment with the sampled maze, initial state, and goal.
        ob, _ = env.reset(options=dict(task_info=dict(maze_id=maze_id, init_ij=init_ij, goal_ij=goal_ij)))

        done = False
        step = 0

        while not done:
            # Get the oracle subgoal and compute the direction.
            subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(env.unwrapped.get_xy(), env.unwrapped.cur_goal_xy)
            subgoal_dir = subgoal_xy - env.unwrapped.get_xy()
            subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

            # Query the oracle agent for the action.
            agent_ob = env.unwrapped.get_ob(ob_type='states')
            # Exclude the agent's position and add the subgoal direction.
            agent_ob = np.concatenate([agent_ob[2:], subgoal_dir])
            action = actor_fn(agent_ob, temperature=0)
            
            # Add Gaussian noise to the action.
            action = action + np.random.normal(0, FLAGS.noise, action.shape)
            action = np.clip(action, -1, 1)

            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = info['success']

            # Sample a new goal state when the current goal is reached.
            if success and FLAGS.dataset_type == 'navigate':
                goal_ij = vertex_cells[np.random.randint(len(vertex_cells))]
                env.unwrapped.set_goal(goal_ij)

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminations'].append(terminated)
            dataset["truncations"].append(truncated)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])

            ob = next_ob
            step += 1

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminations' or k == 'truncations':
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)
