import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from gymnasium.spaces import Box

from ogbench.locomaze.ant import AntEnv
from ogbench.locomaze.humanoid import HumanoidEnv
from ogbench.locomaze.point import PointEnv


def make_rand_maze_env(loco_env_type, *args, **kwargs):
    """Factory function for creating a random maze environment.

    Args:
        loco_env_type: Locomotion environment type. One of 'point', 'ant', or 'humanoid'.
        *args: Additional arguments to pass to the target class.
        **kwargs: Additional keyword arguments to pass to the target class.
    """
    if loco_env_type == 'point':
        loco_env_class = PointEnv
    elif loco_env_type == 'ant':
        loco_env_class = AntEnv
    elif loco_env_type == 'humanoid':
        loco_env_class = HumanoidEnv
    else:
        raise ValueError(f'Unknown locomotion environment type: {loco_env_type}')

    class RandMazeEnv(loco_env_class):
        """Maze environment.

        It inherits from the locomotion environment and adds a maze to it.
        """

        def __init__(
            self,
            maze_map_dir,
            maze_unit=4.0,
            maze_height=0.5,
            terminate_at_goal=True,
            ob_type='states',
            add_noise_to_goal=True,
            reward_task_id=None,
            use_oracle_rep=False,
            mark_goal=True,
            recolor_pixel_ob=False,
            *args,
            **kwargs,
        ):
            """Initialize the maze environment.

            Args:
                maze_map_dir: Directory containing the maze map files.
                maze_unit: Size of a maze unit block.
                maze_height: Height of the maze walls.
                terminate_at_goal: Whether to terminate the episode when the goal is reached.
                ob_type: Observation type. Either 'states' or 'pixels'.
                add_noise_to_goal: Whether to add noise to the goal position.
                reward_task_id: Task ID for single-task RL. If this is not None, the environment operates in a
                    single-task mode with the specified task ID. The task ID must be either a valid task ID or 0, where
                    0 means using the default task.
                use_oracle_rep: Whether to use oracle goal representations.
                mark_goal: Whether to mark the goal in the environment.
                *args: Additional arguments to pass to the parent locomotion environment.
                **kwargs: Additional keyword arguments to pass to the parent locomotion environment.
            """
            self._maze_map_dir = maze_map_dir
            self._maze_unit = maze_unit
            self._maze_height = maze_height
            self._terminate_at_goal = terminate_at_goal
            self._ob_type = ob_type
            self._add_noise_to_goal = add_noise_to_goal
            self._reward_task_id = reward_task_id
            self._use_oracle_rep = use_oracle_rep
            self._mark_goal = mark_goal
            self._recolor_pixel_ob = recolor_pixel_ob
            assert ob_type in ['states', 'pixels']

            # Define constants.
            self._offset_x = 4
            self._offset_y = 4
            self._noise = 1
            self._goal_tol = 1.0 if loco_env_type == 'point' else 0.5

            # Define maze map.
            self._maze_maps = []
            for filename in sorted(os.listdir(maze_map_dir)):
                maze_map = np.loadtxt(os.path.join(maze_map_dir, filename), delimiter=",", dtype=int)
                self._maze_maps.append(maze_map)
            maze_sizes = set([m.shape for m in self._maze_maps])
            assert len(maze_sizes) == 1, "All maze maps must have the same size."
            self._maze_size = maze_sizes.pop()

            # Update XML file.
            tree = ET.parse(self.xml_file)
            self._initialize_tree(tree)
            _, maze_xml_file = tempfile.mkstemp(text=True, suffix='.xml')
            tree.write(maze_xml_file)
            super().__init__(xml_file=maze_xml_file, *args, **kwargs)

            # map (i,j) → geom-index in sim.model
            self._block_geom_ids = {}
            for gid in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if name and name.startswith('block_'):
                    _, i, j = name.split('_')
                    self._block_geom_ids[gid] = (int(i), int(j))

            # Make custom camera.
            if self.camera_id is None and self.camera_name is None:
                # Use a custom default view.
                camera = mujoco.MjvCamera()
                camera.lookat[0] = 2 * (self._maze_size[1] - 3)
                camera.lookat[1] = 2 * (self._maze_size[0] - 3)
                camera.distance = 5 * (self._maze_size[1] - 2)
                camera.elevation = -90
                self.custom_camera = camera
            else:
                self.custom_camera = self.camera_id or self.camera_name

            # Set task goals.
            self.task_infos = self._initialize_tasks()
            self.num_tasks = len(self.task_infos)
            self.cur_task_id = None
            self.cur_task_info = None
            self.cur_goal_xy = np.zeros(2)

            self.custom_renderer = None
            if self._ob_type == 'pixels':
                self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

                # Manually color the floor to enable the agent to infer its position from the observation.
                if self._recolor_pixel_ob:
                    tex_grid = self.model.tex('grid')
                    tex_height = tex_grid.height[0]
                    tex_width = tex_grid.width[0]
                    # MuJoCo 3.2.1 changed the attribute name from 'tex_rgb' to 'tex_data'.
                    attr_name = 'tex_rgb' if hasattr(self.model, 'tex_rgb') else 'tex_data'
                    tex_rgb = getattr(self.model, attr_name)[tex_grid.adr[0] : tex_grid.adr[0] + 3 * tex_height * tex_width]
                    tex_rgb = tex_rgb.reshape(tex_height, tex_width, 3)
                    for x in range(tex_height):
                        for y in range(tex_width):
                            min_value = 0
                            max_value = 192
                            r = int(x / tex_height * (max_value - min_value) + min_value)
                            g = int(y / tex_width * (max_value - min_value) + min_value)
                            tex_rgb[x, y, :] = [r, g, 128]
                self._initialize_renderer()
            else:
                ex_ob = self.get_ob()
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

        def _initialize_tree(self, tree):
            """Initialize the XML tree for an empty maze"""
            worldbody = tree.find('.//worldbody')

            # Put wall at every position in the maze.
            for i in range(self._maze_size[0]):
                for j in range(self._maze_size[1]):
                    ET.SubElement(
                        worldbody,
                        'geom',
                        name=f'block_{i}_{j}',
                        pos=f'{j * self._maze_unit - self._offset_x} {i * self._maze_unit - self._offset_y} {self._maze_height / 2 * self._maze_unit}',
                        size=f'{self._maze_unit / 2} {self._maze_unit / 2} {self._maze_height / 2 * self._maze_unit}',
                        type='box',
                        contype='1',
                        conaffinity='1',
                        material='wall',
                    )

            # Adjust floor size (assume all mazes have the same size).
            center_x, center_y = 2 * (self._maze_size[1] - 3), 2 * (self._maze_size[0] - 3)
            size_x, size_y = 2 * self._maze_size[1], 2 * self._maze_size[0]
            floor = tree.find('.//geom[@name="floor"]')
            floor.set('pos', f'{center_x} {center_y} 0')
            floor.set('size', f'{size_x} {size_y} 0.2')

            if self._ob_type == 'pixels' and self._recolor_pixel_ob:
                # Color wall.
                wall = tree.find('.//material[@name="wall"]')
                wall.set('rgba', '.6 .6 .6 1')
                # Remove ambient light.
                light = tree.find('.//light[@name="global"]')
                light.attrib.pop('ambient')
                # Remove torso light.
                torso_light = tree.find('.//light[@name="torso_light"]')
                torso_light_parent = tree.find('.//light[@name="torso_light"]/..')
                torso_light_parent.remove(torso_light)
                # Remove texture repeat.
                grid = tree.find('.//material[@name="grid"]')
                grid.set('texuniform', 'false')
                if loco_env_type == 'ant':
                    # Color one leg white to break symmetry.
                    tree.find('.//geom[@name="aux_1_geom"]').set('material', 'self_white')
                    tree.find('.//geom[@name="left_leg_geom"]').set('material', 'self_white')
                    tree.find('.//geom[@name="left_ankle_geom"]').set('material', 'self_white')

            # Mark the target position.
            if self._mark_goal:
                ET.SubElement(
                    worldbody,
                    'geom',
                    name='target',
                    type='cylinder',
                    size='.5 .05',
                    pos='0 0 .05',
                    material='target',
                    contype='0',
                    conaffinity='0',
                )

        def _initialize_tasks(self, num_tasks=10000):
            # `tasks` is a list of tasks, where each task is a list of three items: (map_ind, init_ij, goal_ij).
            rng = np.random.default_rng(42)

            tasks = []
            for i in range(num_tasks):
                maze_id = rng.integers(len(self._maze_maps))
                maze_map = self._maze_maps[maze_id]
                
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

                # Sample initial cell.
                if len(empty_cells) < 2:
                    raise ValueError(f'Not enough empty cells in maze {maze_id} for task {i}.')
                init_cell = empty_cells[rng.integers(len(empty_cells))]

                # Sample goal cell.
                valid_goals = [c for c in vertex_cells if c != init_cell]
                if len(valid_goals) < 1:
                    raise ValueError(f"Can't find valid goal cells in maze {maze_id} for task {i}. ")
                goal_cell = valid_goals[rng.integers(len(valid_goals))]

                tasks.append([maze_id, init_cell, goal_cell])

            task_infos = []
            for i, task in enumerate(tasks):
                task_infos.append(
                    dict(
                        task_name=f'task{i + 1}',
                        maze_id=task[0],
                        init_ij=task[1],
                        init_xy=self.ij_to_xy(task[1]),
                        goal_ij=task[2],
                        goal_xy=self.ij_to_xy(task[2]),
                    )
                )
            return task_infos

        def _initialize_renderer(self):
            # Make custom renderer.
            self.custom_renderer = mujoco.Renderer(
                self.model,
                width=self.width,
                height=self.height,
            )
            self.render()

        def _reset_maze_map(self, maze_map):
            for gid, (i, j) in self._block_geom_ids.items():
                if maze_map[i, j] == 1:
                    # turn wall "on"
                    self.model.geom_size[gid] = np.array([self._maze_unit/2, self._maze_unit/2, (self._maze_height/2)*self._maze_unit])
                    self.model.geom_contype[gid] = 1
                    self.model.geom_conaffinity[gid] = 1
                else:
                    # turn wall "off"
                    self.model.geom_size[gid] = np.zeros(3)
                    self.model.geom_contype[gid] = 0
                    self.model.geom_conaffinity[gid] = 0
        
        @property
        def maze_map(self):
            if self.cur_task_info is None:
                raise ValueError('Attempting to access maze_map before resetting the environment.')
            return self._maze_maps[self.cur_task_info['maze_id']]

        def reset(self, options=None, *args, **kwargs):
            if options is None:
                options = {}
            # Set the task goal.
            if self._reward_task_id is not None:
                # Use the pre-defined task.
                assert 1 <= self._reward_task_id <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = self._reward_task_id
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_id' in options:
                # Use the pre-defined task.
                assert 1 <= options['task_id'] <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = options['task_id']
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_info' in options:
                # Use the provided task information.
                self.cur_task_id = None
                self.cur_task_info = options['task_info']
            else:
                # Randomly sample a task.
                self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]

            # Whether to provide a rendering of the goal.
            render_goal = False
            if 'render_goal' in options:
                render_goal = options['render_goal']

            # Get initial and goal positions with noise.
            init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['init_ij']))
            goal_xy = self.ij_to_xy(self.cur_task_info['goal_ij'])
            if self._add_noise_to_goal:
                goal_xy = self.add_noise(goal_xy)

            # Reset the maze map
            maze_map = self._maze_maps[self.cur_task_info["maze_id"]]
            self._reset_maze_map(maze_map)

            # First, force set the position to the goal position to obtain the goal observation.
            super().reset(*args, **kwargs)

            # Do a few random steps to stabilize the environment.
            num_random_actions = 40 if loco_env_type == 'humanoid' else 5
            for _ in range(num_random_actions):
                super().step(self.action_space.sample())

            # Save the goal observation.
            self.set_goal(goal_xy=goal_xy)
            self.set_xy(goal_xy)
            goal_ob = self.get_oracle_rep() if self._use_oracle_rep else self.get_ob()
            if render_goal:
                goal_rendered = self.render()

            # Now, do the actual reset.
            ob, info = super().reset(*args, **kwargs)
            self.set_goal(goal_xy=goal_xy)
            self.set_xy(init_xy)
            ob = self.get_ob()
            info['goal'] = goal_ob
            if render_goal:
                info['goal_rendered'] = goal_rendered
            info['task_id'] = self.cur_task_id
            info['task_info'] = self.cur_task_info

            return ob, info

        def step(self, action):
            ob, reward, terminated, truncated, info = super().step(action)

            # Check if the agent has reached the goal.
            if np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= self._goal_tol:
                if self._terminate_at_goal:
                    terminated = True
                info['success'] = 1.0
                reward = 1.0
            else:
                info['success'] = 0.0
                reward = 0.0

            # If the environment is in the single-task mode, modify the reward.
            if self._reward_task_id is not None:
                reward = reward - 1.0  # -1 (failure) or 0 (success).

            return ob, reward, terminated, truncated, info

        def render(self):
            if self.custom_renderer is None:
                self._initialize_renderer()
            self.custom_renderer.update_scene(self.data, camera=self.custom_camera)
            return self.custom_renderer.render()

        def get_ob(self, ob_type=None):
            ob_type = self._ob_type if ob_type is None else ob_type
            if ob_type == 'states':
                return super().get_ob()
            else:
                return self.render()

        def get_oracle_rep(self):
            """Return the oracle goal representation (i.e., the goal position)."""
            return np.array(self.cur_goal_xy)

        def set_goal(self, goal_ij=None, goal_xy=None):
            """Set the goal position and update the target object."""
            if goal_xy is None:
                self.cur_goal_xy = self.ij_to_xy(goal_ij)
                if self._add_noise_to_goal:
                    self.cur_goal_xy = self.add_noise(self.cur_goal_xy)
            else:
                self.cur_goal_xy = goal_xy
            if self._mark_goal:
                self.model.geom('target').pos[:2] = goal_xy

        def get_oracle_subgoal(self, start_xy, goal_xy):
            """Get the oracle subgoal for the agent.

            If the goal is unreachable, it returns the current position as the subgoal.

            Args:
                start_xy: Starting position of the agent.
                goal_xy: Goal position of the agent.
            Returns:
                A tuple of the oracle subgoal and the BFS map.
            """
            maze_map = self._maze_maps[self.cur_task_info["maze_id"]]
            start_ij = self.xy_to_ij(start_xy)
            goal_ij = self.xy_to_ij(goal_xy)

            # Run BFS to find the next subgoal.
            bfs_map = maze_map.copy()
            for i in range(maze_map.shape[0]):
                for j in range(maze_map.shape[1]):
                    bfs_map[i][j] = -1

            bfs_map[goal_ij[0], goal_ij[1]] = 0
            queue = [goal_ij]
            while len(queue) > 0:
                i, j = queue.pop(0)
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < maze_map.shape[0]
                        and 0 <= nj < maze_map.shape[1]
                        and maze_map[ni, nj] == 0
                        and bfs_map[ni, nj] == -1
                    ):
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))

            # Find the subgoal that attains the minimum BFS value.
            subgoal_ij = start_ij
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = start_ij[0] + di, start_ij[1] + dj
                if (
                    0 <= ni < maze_map.shape[0]
                    and 0 <= nj < maze_map.shape[1]
                    and maze_map[ni, nj] == 0
                    and bfs_map[ni, nj] < bfs_map[subgoal_ij[0], subgoal_ij[1]]
                ):
                    subgoal_ij = (ni, nj)

            if subgoal_ij == goal_ij:
                # If the subgoal cell contains the goal, return the exact goal position.
                subgoal_xy = np.array(goal_xy)
            else:
                # Otherwise return the center of the subgoal cell.
                subgoal_xy = np.array(self.ij_to_xy(subgoal_ij))
            return subgoal_xy, bfs_map

        def xy_to_ij(self, xy):
            maze_unit = self._maze_unit
            i = int((xy[1] + self._offset_y + 0.5 * maze_unit) / maze_unit)
            j = int((xy[0] + self._offset_x + 0.5 * maze_unit) / maze_unit)
            return i, j

        def ij_to_xy(self, ij):
            i, j = ij
            x = j * self._maze_unit - self._offset_x
            y = i * self._maze_unit - self._offset_y
            return x, y

        def add_noise(self, xy):
            random_x = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            random_y = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            return xy[0] + random_x, xy[1] + random_y

    return RandMazeEnv(*args, **kwargs)
