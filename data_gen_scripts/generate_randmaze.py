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
flags.DEFINE_string('restore_path', 'experts/ant', 'Expert agent restore path.')
flags.DEFINE_integer('restore_epoch', 400000, 'Expert agent restore epoch.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.2, 'Gaussian action noise level.')
flags.DEFINE_integer('num_steps', 1000000, 'Number of steps.')


def main(_):
    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        ob_type='pixels',
        max_episode_steps=100000, # Ensure never hit the max episode steps limit.
        width=84,
        height=84,
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
    pbar = trange(FLAGS.num_steps, total=FLAGS.num_steps, desc='Collecting data')
    step = 0
    while step < FLAGS.num_steps:
        # Sample a random maze
        maze_id = np.random.randint(len(env.unwrapped._maze_maps))
        maze_map = env.unwrapped._maze_maps[maze_id]

        # Sample random initial and goal states.
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
        # Sample initial state from all cells.
        init_ij = empty_cells[np.random.randint(len(empty_cells))]
        # Sample goal state from vertex cells.
        valid_goals = [c for c in vertex_cells if c != init_ij]
        goal_ij = valid_goals[np.random.randint(len(valid_goals))]
    
        # Reset the environment with the sampled maze, initial state, and goal.
        ob, _ = env.reset(options=dict(task_info=dict(maze_id=maze_id, init_ij=init_ij, goal_ij=goal_ij)))

        done = False
        while not done:
            # Get the oracle subgoal and compute the direction.
            subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(env.unwrapped.get_xy(), env.unwrapped.cur_goal_xy)
            subgoal_dir = subgoal_xy - env.unwrapped.get_xy()
            subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

            # Query the oracle agent for the action.
            agent_ob = env.unwrapped.get_ob(ob_type='states')
            agent_ob = np.concatenate([agent_ob[2:], subgoal_dir]) # exclude position and add subgoal direction.
            action = actor_fn(agent_ob, temperature=0)
            
            # Add Gaussian noise to the action.
            action = action + np.random.normal(0, FLAGS.noise, action.shape)
            action = np.clip(action, -1, 1)

            next_ob, reward, terminated, truncated, info = env.step(action)
            assert not truncated, 'Data collection should not trigger truncation.'
            done = terminated or truncated
            
            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['dones'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])

            ob = next_ob
            step += 1
            pbar.update(1)
    
    print("Total number of steps: ", step)

    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminations' or k == 'truncations':
            dtype = bool
        else:
            dtype = np.float32
        dataset[k] = np.array(v, dtype=dtype)
    pathlib.Path(FLAGS.save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FLAGS.save_path, **dataset)


if __name__ == '__main__':
    app.run(main)
