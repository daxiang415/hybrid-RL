import os
import click
import time
import numpy as np
import gym
import gym_platform
from agents.paddpg import PADDPGAgent
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from simple_rl import Takasago_ENV
import random
import torch


def pad_action(act, act_param):
    params = [np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
    params[act][:] = act_param
    return act, params
def seed_everything(seed):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)

def evaluate(env, agent, episodes=1):
    returns = []

    for _ in range(episodes):
        state = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, _, _ = agent.act(state, Train=False)
            action = pad_action(act, act_param)
            state, _, reward, terminal, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return np.array(returns)








@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=10000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=50, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=32, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=128, help='Number of transitions required to start learning.',
              type=int)
@click.option('--use-ornstein-noise', default=True,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=100000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau-critic', default=0.01, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor', default=0.01, help='Soft target network update averaging factor.', type=float)
@click.option('--learning-rate-critic', default=1e-3, help="Critic network learning rate.", type=float)
@click.option('--learning-rate-actor', default=1e-4, help="Actor network learning rate.", type=float)
@click.option('--initialise-params', default=False, help='Initialise action parameters.', type=bool)
@click.option('--clip-grad', default=0.5, help="Parameter gradient clipping limit.", type=float)
@click.option('--layers', default='[256,128]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
@click.option('--title', default="PADDPG", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, save_dir,
        epsilon_steps, epsilon_final, tau_actor, tau_critic, use_ornstein_noise,
        learning_rate_actor, learning_rate_critic, clip_grad, layers, initialise_params, title):
    env = Takasago_ENV()



    agent = PADDPGAgent(observation_space=env.observation_space,
                        action_space=env.action_space,
                        batch_size=batch_size,
                        learning_rate_actor=learning_rate_actor,
                        learning_rate_critic=learning_rate_critic,
                        epsilon_steps=epsilon_steps,
                        epsilon_final=epsilon_final,
                        gamma=gamma,
                        clip_grad=clip_grad,
                        tau_actor=tau_actor,
                        tau_critic=tau_critic,
                        initial_memory_threshold=initial_memory_threshold,
                        use_ornstein_noise=use_ornstein_noise,
                        replay_memory_size=replay_memory_size,
                        inverting_gradients=inverting_gradients,
                        adam_betas=(0.9, 0.999),
                        critic_kwargs={'hidden_layers': layers, 'init_type': "kaiming"},
                        actor_kwargs={'hidden_layers': layers, 'init_type': "kaiming", 'init_std': 0.0001,
                                      'squashing_function': False},
                        seed=seed)
    agent.load_model(actor_path='paddpg_actor.pth', critic_path='paddpg_critic.pth')

    eval_returns = evaluate(env, agent, 1)
    print(sum(eval_returns) / len(eval_returns))






if __name__ == '__main__':
    #seed_everything(0)
    run()





