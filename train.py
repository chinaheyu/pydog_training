import gym, torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from dqn_model import OdorDQN
import odor_env


# Define some hyper-parameters:
train_task = 'OdorEnvA-v0'
test_task = 'OdorEnvA-v0'
lr, epoch, batch_size = 1e-4, 100, 64
train_num, test_num = 50, 100
gamma, n_step, target_freq = 0.95, 3, 320
buffer_size = 20000
eps_train, eps_test = 1.0, 0.0
eps_train_final = 0.1
step_per_epoch, step_per_collect = 10000, 10
logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make environments:
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(train_task) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(test_task) for _ in range(test_num)])

# Define the network:
env = gym.make(train_task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = OdorDQN(device).to(device)

optim = torch.optim.Adam(net.parameters(), lr=lr)

# Setup policy and collectors:
policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

# 继续训练
# policy.load_state_dict(torch.load('dqn.pth'))


# Let's train it:
def train_fn(epoch, env_step):
    # nature DQN setting, linear decay in the first 1M steps
    if env_step <= 1e4:
        eps = eps_train - env_step / 1e4 * \
              (eps_train - eps_train_final)
    else:
        eps = eps_train_final
    policy.set_eps(eps)


def save_checkpoint_fn(epoch, env_step, gradient_step):
    # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    ckpt_path = 'checkpoint.pth'
    torch.save({'model': policy.state_dict()}, ckpt_path)


result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
    test_num, batch_size, update_per_step=1 / step_per_collect,
    train_fn=train_fn,
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= 19,
    logger=logger,
    save_checkpoint_fn=save_checkpoint_fn
)
print(f'Finished training! Use {result["duration"]}')

# Save / load the trained policy
torch.save(policy.state_dict(), 'dqn.pth')

torch.save(net.state_dict(), 'output.pth')
