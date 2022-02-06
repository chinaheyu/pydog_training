import random
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces, core
from data_preprocess import load_processed_data


# core.Env是gym的环境基类,自定义的环境就是根据自己的需要重写其中的方法；
# 必须要重写的方法有:
# __init__()：构造函数
# reset()：初始化环境
# step()：环境动作,即环境对agent的反馈
# render()：如果要进行可视化则实现

# ↓x  →y
# 0:up 1:down 2:left 3:right

class OdorEnvA(core.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, np.Inf, [5, 11, 11])

        # 读取数据
        self.processed_data = load_processed_data()

        self.map_prefix = None
        self.map_grid = None
        self.concentration_grid = None
        self.velocity_x_grid = None
        self.velocity_y_grid = None
        self.source_position = None

        # 机器人观测过的数据
        self.trajectory_matrix = None
        self.concentration_matrix = None
        self.airflow_x_matrix = None
        self.airflow_y_matrix = None

        # 机器人位置
        self.agent_position = None

        # 计数
        self.step_count = 0

    def _choose_spawn_point(self):
        while True:
            point = tuple(np.random.randint(0, 30, 2))
            if self.map_grid[point] < 1 and (abs(point[0] - self.source_position[0]) > 5 or abs(point[1] - self.source_position[1]) > 5):
                return point

    def _normalize_data(self, data):
        if np.max(data) - np.min(data) > 0.0:
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def _update_observation_matrix(self):
        self.trajectory_matrix[self.agent_position] += 1
        self.concentration_matrix[self.agent_position] = self.concentration_grid[self.agent_position]
        self.airflow_x_matrix[self.agent_position] = self.velocity_x_grid[self.agent_position]
        self.airflow_y_matrix[self.agent_position] = self.velocity_y_grid[self.agent_position]

    def _get_observation(self):
        # 填充边界
        trajectory_matrix_pad = np.pad(self.trajectory_matrix, (5, 5), 'constant', constant_values=0)
        map_matrix_pad = np.pad(self.map_grid, (5, 5), 'constant', constant_values=1)
        # concentration_matrix_pad = np.pad(self._normalize_data(self.concentration_matrix), (5, 5), 'constant', constant_values=0)
        # airflow_x_matrix_pad = np.pad(self._normalize_data(self.airflow_x_matrix), (5, 5), 'constant', constant_values=0)
        # airflow_y_matrix_pad = np.pad(self._normalize_data(self.airflow_y_matrix), (5, 5), 'constant', constant_values=0)
        concentration_matrix_pad = np.pad(self.concentration_matrix, (5, 5), 'constant', constant_values=0)
        airflow_x_matrix_pad = np.pad(self.airflow_x_matrix, (5, 5), 'constant', constant_values=0)
        airflow_y_matrix_pad = np.pad(self.airflow_y_matrix, (5, 5), 'constant', constant_values=0)

        # 组合
        observation_matrix = np.stack((trajectory_matrix_pad,
                                       map_matrix_pad,
                                       concentration_matrix_pad,
                                       airflow_x_matrix_pad,
                                       airflow_y_matrix_pad), axis=0)

        # 剪裁观测区域
        observation_matrix = observation_matrix[:, self.agent_position[0]:self.agent_position[0] + 11, self.agent_position[1]:self.agent_position[1] + 11]
        return observation_matrix

    def _will_collision(self, action):
        next_x = self.agent_position[0]
        next_y = self.agent_position[1]
        if action == 0:
            next_x -= 1
        if action == 1:
            next_x += 1
        if action == 2:
            next_y -= 1
        if action == 3:
            next_y += 1

        # 边界
        if next_x < 0 or next_x > 29 or next_y < 0 or next_y > 29:
            return True

        # 障碍物
        if self.map_grid[next_x, next_y] > 0.5:
            return True

        return False

    def _move_agent(self, action):
        # 移动agent
        if action == 0:
            self.agent_position = (self.agent_position[0] - 1, self.agent_position[1])
        if action == 1:
            self.agent_position = (self.agent_position[0] + 1, self.agent_position[1])
        if action == 2:
            self.agent_position = (self.agent_position[0], self.agent_position[1] - 1)
        if action == 3:
            self.agent_position = (self.agent_position[0], self.agent_position[1] + 1)

    def _is_reach_source(self):
        return abs(self.agent_position[0] - self.source_position[0]) < 2 and abs(self.agent_position[1] - self.source_position[1]) < 2

    def _repeat_punishment(self):
        return -(self.trajectory_matrix[self.agent_position] - 1.0) * 0.1

    def _concentration_punishment(self):
        return self._normalize_data(self.concentration_matrix)[self.agent_position] - 1.0

    def _distance_punishment(self):
        return -(abs(self.agent_position[0] - self.source_position[0]) + abs(self.agent_position[1] - self.source_position[1])) / 30.0

    def step(self, action):
        self.step_count += 1
        info = {'map_grid': self.map_grid,
                'agent_position': self.agent_position,
                'source_position': self.source_position,
                'state': 'unfinished'}

        # 碰撞惩罚
        if self._will_collision(action):
            info['state'] = 'failure'
            return self._get_observation(), -10.0, True, info
        self._move_agent(action)
        self._update_observation_matrix()

        info['agent_position'] = self.agent_position

        # 成功奖励
        if self._is_reach_source():
            info['state'] = 'success'
            return self._get_observation(), 20.0, True, info

        if self.step_count > 450:
            info['state'] = 'failure'
            return self._get_observation(), self._repeat_punishment() - 0.1, True, info
        else:
            return self._get_observation(), self._repeat_punishment() - 0.1, False, info

    def _transpose_environment(self):
        self.map_grid = self.map_grid.T
        self.concentration_grid = self.concentration_grid.T
        self.velocity_x_grid, self.velocity_y_grid = self.velocity_y_grid.T, self.velocity_x_grid.T
        self.source_position = (self.source_position[1], self.source_position[0])

    def _flip_environment_y(self):
        self.map_grid = self.map_grid[:, ::-1]
        self.concentration_grid = self.concentration_grid[:, ::-1]
        self.velocity_x_grid = self.velocity_x_grid[:, ::-1]
        self.velocity_y_grid = -self.velocity_y_grid[:, ::-1]
        self.source_position = (self.source_position[0], 29 - self.source_position[1])

    def _flip_environment_x(self):
        self.map_grid = self.map_grid[::-1, :]
        self.concentration_grid = self.concentration_grid[::-1, :]
        self.velocity_x_grid = -self.velocity_x_grid[::-1, :]
        self.velocity_y_grid = self.velocity_y_grid[::-1, :]
        self.source_position = (29 - self.source_position[0], self.source_position[1])

    def _rotate_environment(self):
        # TODO: 旋转90°
        pass

    def _random_change_environment(self):
        # 随机转置或翻转环境
        rng = random.random()
        if rng < 0.2:
            self._transpose_environment()
        elif rng < 0.4:
            self._flip_environment_x()
        elif rng < 0.6:
            self._flip_environment_y()
        elif rng < 0.8:
            self._rotate_environment()
        else:
            pass

    def reset(self):
        # 随机选择一个地图
        map_prefix = random.choice(['a', 'b', 'c', 'd'])
        # map_prefix = random.choice(['e'])

        # 加载数据
        self.map_grid = self.processed_data[map_prefix]['map_grid']
        self.concentration_grid = self.processed_data[map_prefix]['concentration_grid']
        self.velocity_x_grid = self.processed_data[map_prefix]['velocity_x_grid']
        self.velocity_y_grid = self.processed_data[map_prefix]['velocity_y_grid']
        self.source_position = self.processed_data[map_prefix]['source_position']

        # 浓度标准化
        self.concentration_grid = self._normalize_data(self.concentration_grid)

        # 训练环境随机转置或翻转
        # self._random_change_environment()

        # 重置机器人观测过的数据
        self.trajectory_matrix = np.zeros((30, 30))
        self.concentration_matrix = np.zeros((30, 30))
        self.airflow_x_matrix = np.zeros((30, 30))
        self.airflow_y_matrix = np.zeros((30, 30))

        # 机器人位置
        self.agent_position = self._choose_spawn_point()

        self._update_observation_matrix()

        self.step_count = 0

        return self._get_observation()

    def debug_reset(self, map_prefix, initial_position):
        # 加载数据
        self.map_grid = self.processed_data[map_prefix]['map_grid']
        self.concentration_grid = self.processed_data[map_prefix]['concentration_grid']
        self.velocity_x_grid = self.processed_data[map_prefix]['velocity_x_grid']
        self.velocity_y_grid = self.processed_data[map_prefix]['velocity_y_grid']
        self.source_position = self.processed_data[map_prefix]['source_position']

        # 浓度标准化
        self.concentration_grid = self._normalize_data(self.concentration_grid)

        # 训练环境随机转置或翻转
        # self._random_change_environment()

        # 重置机器人观测过的数据
        self.trajectory_matrix = np.zeros((30, 30))
        self.concentration_matrix = np.zeros((30, 30))
        self.airflow_x_matrix = np.zeros((30, 30))
        self.airflow_y_matrix = np.zeros((30, 30))

        # 机器人位置
        self.agent_position = initial_position

        self._update_observation_matrix()

        self.step_count = 0

        return self._get_observation()


    def render(self, mode='human'):
        if mode == 'ascii':
            for i in range(30):
                for j in range(30):
                    if self.map_grid[i, j] > 0.5:
                        print('X', end='')
                    elif self.source_position[0] == i and self.source_position[1] == j:
                        print('?', end='')
                    elif self.agent_position[0] == i and self.agent_position[1] == j:
                        print('.', end='')
                    else:
                        print(' ', end='')
                print('')
        elif mode == 'human':
            ob = self._get_observation()
            plt.subplot(221)
            plt.imshow(ob[0, :, :])
            plt.colorbar()
            plt.subplot(222)
            plt.imshow(ob[1, :, :])
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(ob[2, :, :])
            plt.colorbar()
            ax = plt.subplot(224)
            x, y = np.mgrid[0:15:11j, 0:15:11j]
            plt.quiver(y, x, ob[4, :, :], ob[3, :, :], color='C0', angles='xy')
            ax.invert_yaxis()
            plt.colorbar()
            plt.show()

    def close(self):
        pass


class OdorEnvB(core.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, np.Inf, [5, 11, 11])

        # 读取数据
        self.processed_data = load_processed_data()

        self.map_prefix = None
        self.map_grid = None
        self.concentration_grid = None
        self.velocity_x_grid = None
        self.velocity_y_grid = None
        self.source_position = None

        # 机器人观测过的数据
        self.trajectory_matrix = None
        self.concentration_matrix = None
        self.airflow_x_matrix = None
        self.airflow_y_matrix = None

        # 机器人位置
        self.agent_position = None

        # 计数
        self.step_count = 0

    def _choose_spawn_point(self):
        while True:
            point = tuple(np.random.randint(0, 30, 2))
            if self.map_grid[point] < 1 and (abs(point[0] - self.source_position[0]) > 5 or abs(point[1] - self.source_position[1]) > 5):
                return point

    def _normalize_data(self, data):
        if np.max(data) - np.min(data) > 0.0:
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def _update_observation_matrix(self):
        self.trajectory_matrix[self.agent_position] += 1
        self.concentration_matrix[self.agent_position] = self.concentration_grid[self.agent_position]
        self.airflow_x_matrix[self.agent_position] = self.velocity_x_grid[self.agent_position]
        self.airflow_y_matrix[self.agent_position] = self.velocity_y_grid[self.agent_position]

    def _get_observation(self):
        # 填充边界
        trajectory_matrix_pad = np.pad(self.trajectory_matrix, (5, 5), 'constant', constant_values=0)
        map_matrix_pad = np.pad(self.map_grid, (5, 5), 'constant', constant_values=1)
        # concentration_matrix_pad = np.pad(self._normalize_data(self.concentration_matrix), (5, 5), 'constant', constant_values=0)
        # airflow_x_matrix_pad = np.pad(self._normalize_data(self.airflow_x_matrix), (5, 5), 'constant', constant_values=0)
        # airflow_y_matrix_pad = np.pad(self._normalize_data(self.airflow_y_matrix), (5, 5), 'constant', constant_values=0)
        concentration_matrix_pad = np.pad(self.concentration_matrix, (5, 5), 'constant', constant_values=0)
        airflow_x_matrix_pad = np.pad(self.airflow_x_matrix, (5, 5), 'constant', constant_values=0)
        airflow_y_matrix_pad = np.pad(self.airflow_y_matrix, (5, 5), 'constant', constant_values=0)

        # 组合
        observation_matrix = np.stack((trajectory_matrix_pad,
                                       map_matrix_pad,
                                       concentration_matrix_pad,
                                       airflow_x_matrix_pad,
                                       airflow_y_matrix_pad), axis=0)

        # 剪裁观测区域
        observation_matrix = observation_matrix[:, self.agent_position[0]:self.agent_position[0] + 11, self.agent_position[1]:self.agent_position[1] + 11]
        return observation_matrix

    def _will_collision(self, action):
        next_x = self.agent_position[0]
        next_y = self.agent_position[1]
        if action == 0:
            next_x -= 1
        if action == 1:
            next_x += 1
        if action == 2:
            next_y -= 1
        if action == 3:
            next_y += 1

        # 边界
        if next_x < 0 or next_x > 29 or next_y < 0 or next_y > 29:
            return True

        # 障碍物
        if self.map_grid[next_x, next_y] > 0.5:
            return True

        return False

    def _move_agent(self, action):
        # 移动agent
        if action == 0:
            self.agent_position = (self.agent_position[0] - 1, self.agent_position[1])
        if action == 1:
            self.agent_position = (self.agent_position[0] + 1, self.agent_position[1])
        if action == 2:
            self.agent_position = (self.agent_position[0], self.agent_position[1] - 1)
        if action == 3:
            self.agent_position = (self.agent_position[0], self.agent_position[1] + 1)

    def _is_reach_source(self):
        return abs(self.agent_position[0] - self.source_position[0]) < 2 and abs(self.agent_position[1] - self.source_position[1]) < 2

    def _repeat_punishment(self):
        return -(self.trajectory_matrix[self.agent_position] - 1.0) * 0.1

    def _concentration_punishment(self):
        return self._normalize_data(self.concentration_matrix)[self.agent_position] - 1.0

    def _distance_punishment(self):
        return -(abs(self.agent_position[0] - self.source_position[0]) + abs(self.agent_position[1] - self.source_position[1])) / 30.0

    def step(self, action):
        self.step_count += 1
        info = {'map_grid': self.map_grid,
                'agent_position': self.agent_position,
                'source_position': self.source_position}

        # 碰撞惩罚
        if self._will_collision(action):
            return self._get_observation(), -10.0, True, info
        self._move_agent(action)
        self._update_observation_matrix()

        info['agent_position'] = self.agent_position

        # 成功奖励
        if self._is_reach_source():
            return self._get_observation(), 20.0, True, info

        if self.step_count > 450:
            return self._get_observation(), self._repeat_punishment() - 0.1, True, info
        else:
            return self._get_observation(), self._repeat_punishment() - 0.1, False, info

    def _transpose_environment(self):
        self.map_grid = self.map_grid.T
        self.concentration_grid = self.concentration_grid.T
        self.velocity_x_grid, self.velocity_y_grid = self.velocity_y_grid.T, self.velocity_x_grid.T
        self.source_position = (self.source_position[1], self.source_position[0])

    def _flip_environment_y(self):
        self.map_grid = self.map_grid[:, ::-1]
        self.concentration_grid = self.concentration_grid[:, ::-1]
        self.velocity_x_grid = self.velocity_x_grid[:, ::-1]
        self.velocity_y_grid = -self.velocity_y_grid[:, ::-1]
        self.source_position = (self.source_position[0], 29 - self.source_position[1])

    def _flip_environment_x(self):
        self.map_grid = self.map_grid[::-1, :]
        self.concentration_grid = self.concentration_grid[::-1, :]
        self.velocity_x_grid = -self.velocity_x_grid[::-1, :]
        self.velocity_y_grid = self.velocity_y_grid[::-1, :]
        self.source_position = (29 - self.source_position[0], self.source_position[1])

    def _random_change_environment(self):
        # 随机转置或翻转环境
        rng = random.random()
        if rng < 0.25:
            self._transpose_environment()
        elif rng < 0.5:
            self._flip_environment_x()
        elif rng < 0.75:
            self._flip_environment_y()
        else:
            pass

    def reset(self):
        # 随机选择一个地图
        # map_prefix = random.choice(['a', 'b', 'c', 'd'])
        map_prefix = random.choice(['e'])

        # 加载数据
        self.map_grid = self.processed_data[map_prefix]['map_grid']
        self.concentration_grid = self.processed_data[map_prefix]['concentration_grid']
        self.velocity_x_grid = self.processed_data[map_prefix]['velocity_x_grid']
        self.velocity_y_grid = self.processed_data[map_prefix]['velocity_y_grid']
        self.source_position = self.processed_data[map_prefix]['source_position']

        # 浓度标准化
        self.concentration_grid = self._normalize_data(self.concentration_grid)

        # 训练环境随机转置或翻转
        # self._random_change_environment()

        # 重置机器人观测过的数据
        self.trajectory_matrix = np.zeros((30, 30))
        self.concentration_matrix = np.zeros((30, 30))
        self.airflow_x_matrix = np.zeros((30, 30))
        self.airflow_y_matrix = np.zeros((30, 30))

        # 机器人位置
        self.agent_position = self._choose_spawn_point()

        self._update_observation_matrix()

        self.step_count = 0

        return self._get_observation()

    def render(self, mode='human'):
        if mode == 'ascii':
            for i in range(30):
                for j in range(30):
                    if self.map_grid[i, j] > 0.5:
                        print('X', end='')
                    elif self.source_position[0] == i and self.source_position[1] == j:
                        print('?', end='')
                    elif self.agent_position[0] == i and self.agent_position[1] == j:
                        print('.', end='')
                    else:
                        print(' ', end='')
                print('')
        elif mode == 'human':
            ob = self._get_observation()
            plt.subplot(221)
            plt.imshow(ob[0, :, :])
            plt.colorbar()
            plt.subplot(222)
            plt.imshow(ob[1, :, :])
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(ob[2, :, :])
            plt.colorbar()
            ax = plt.subplot(224)
            x, y = np.mgrid[0:15:11j, 0:15:11j]
            plt.quiver(y, x, ob[4, :, :], ob[3, :, :], color='C0', angles='xy')
            ax.invert_yaxis()
            plt.colorbar()
            plt.show()

    def close(self):
        pass