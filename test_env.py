import pathlib

import gym
import yaml

import odor_env
from dqn_model import OdorDQN
import torch
import tianshou as ts
import numpy as np
import matplotlib.pyplot as plt
# import pygame
import matplotlib.gridspec as gridspec


env = gym.make('OdorEnvA-v0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = OdorDQN(device).to(device)
policy.load_state_dict(torch.load('output.pth'))
policy.eval()

plt.ion()

play_times = 5
score = 0.0

obs = env.reset()
obs, reward, done, info = env.step(0)
obs = env.reset()

# pygame.init()
# pygame.joystick.init()
# controller = pygame.joystick.Joystick(0)
# controller.init()

robot_path = []

while True:
    # 自动
    action = policy(np.expand_dims(obs, 0))[0].max(1)[1].view(1, 1).item()
    obs, reward, done, info = env.step(action)
    score += reward

    # 记录坐标
    robot_path.append({'x': int(info['agent_position'][0]), 'y': int(info['agent_position'][1])})

    # 手动，手柄操作
    # hat = controller.get_hat(0)
    # if hat[0] == -1:
    #     obs, reward, done, info = env.step(2)
    #     score += reward
    #     print(f'score: {score}')
    # if hat[0] == 1:
    #     obs, reward, done, info = env.step(3)
    #     score += reward
    #     print(f'score: {score}')
    # if hat[1] == 1:
    #     obs, reward, done, info = env.step(0)
    #     score += reward
    #     print(f'score: {score}')
    # if hat[1] == -1:
    #     obs, reward, done, info = env.step(1)
    #     score += reward
    #     print(f'score: {score}')

    plt.clf()
    gs = gridspec.GridSpec(2, 4)
    plt.subplot(gs[0:2, 0:2])
    plt.imshow(info['map_grid'], cmap='gray')
    plt.gca().add_patch(plt.Rectangle((info['source_position'][1] - 0.5, info['source_position'][0] - 0.5), 1.0, 1.0, facecolor="yellow"))
    plt.gca().add_patch(plt.Rectangle((info['agent_position'][1] - 0.5, info['agent_position'][0] - 0.5), 1.0, 1.0, facecolor="red"))
    plt.subplot(gs[0, 2])
    plt.imshow(obs[0, :, :])
    plt.colorbar()
    plt.subplot(gs[0, 3])
    plt.imshow(obs[1, :, :])
    plt.colorbar()
    plt.subplot(gs[1, 2])
    plt.imshow(obs[2, :, :])
    plt.colorbar()
    ax = plt.subplot(gs[1, 3])
    x, y = np.mgrid[0:15:11j, 0:15:11j]
    plt.quiver(y, x, obs[4, :, :], obs[3, :, :], color='C0', angles='xy')
    ax.invert_yaxis()
    plt.pause(0.01)

    if done:
        print(f'score: {score}')
        obs = env.reset()
        play_times -= 1

        # 保存
        # log_path = pathlib.Path(f'result/path_{play_times}.yaml')
        # log_path.write_text(yaml.dump({'result_score': float(score), 'result_path': robot_path}, default_flow_style=False))

        robot_path = []
        score = 0.0

        if play_times < 1:
            plt.ioff()
            plt.show()
            break
