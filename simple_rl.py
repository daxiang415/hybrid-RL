import time                 # to measure the computation time
import gym
from gym import spaces, core
import numpy as np
import random
import pandas as pd
import math
import os


import matplotlib.pyplot as plt
import seaborn as sns



# 如果把168小时的工作日和休息日全部求和，再平均值，作为一个状态


class Takasago_ENV(gym.Env):
  """A building energy system operational optimization for OpenAI gym and takasago"""

  def __init__(self):

    super(Takasago_ENV, self).__init__()
    self.data = pd.read_csv('train.csv')
    self.data = self.data.rename(columns=lambda x: x.strip())

    self.action_space = spaces.Tuple((
        spaces.Discrete(3),
        spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    ))
    self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    self.Max_temp = self.data.max()['temperature']
    self.Max_pv = self.data.max()['PV_output']
    self.Max_solar = self.data.max()['Solar']
    self.Max_hour = 23
    self.Max_power = self.data.max()['Whole']
    self.history_length = 24
    self.optimize_length = 168
    self.battery_change = 200

    # 数据标记
    self.time = 0
    self.t = 0

    # 电机动作差
    self.bio_action = 1

  def _next_observation(self):

    history_frame = np.array([
      (self.data.loc[self.current_step, 'holiday'] * 2) - 1,
      (self.data.loc[self.current_step, 'hour'] / self.Max_hour * 2) - 1,
      (self.data.loc[self.current_step, 'Whole'] / self.Max_power * 2) - 1,
      (self.data.loc[self.current_step, 'PV_output'] / self.Max_pv * 2) - 1,
      (self.data.loc[self.current_step, 'Solar'] / self.Max_solar * 2) - 1,
    ])

    step_in_epo = ((self.current_step - self.original_step) / self.optimize_length * 2) - 1

    obs = np.append(history_frame, (self.battery_state * 2) - 1)
    obs = np.append(obs, step_in_epo)

    return obs.astype(np.float32)

  def reset(self):
    # Reset the state of the environment to an initial state
    self.battery_state = np.random.randint(20, 80) * 0.01
    #self.battery_state = 0.5
    # Set the current step to a random point within the data frame
    self.current_step = random.randint(0, self.data.shape[0] - self.optimize_length - 1)
    #self.current_step = 4596

    self.original_step = self.current_step

    return self._next_observation()

  def reward(self, error, act, battery, battery_action):
    u = -10  # 均值μ
    # u01 = -4
    sig = math.sqrt(4)  # 标准差δ
    deta = 5 * 1.6
    reward_1 = (deta * np.exp(-(error - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)) - 0.6


    if battery <= 0.2 and battery_action < 0:
      # reward_3 = 0.25 * battery - 0.1
      reward_3 = -1.2

    elif battery > 0.8 and battery_action > 0:
      # reward_3 = (-battery + 0.6) * 0.25
      reward_3 = -1.2
    else:
      reward_3 = 0

    self.bio_action = act

    rewards = reward_1
    return rewards + reward_3

  def step(self, action):

      # 顺序是chp，pv，battery



    # 读取计算结果，计算reward
    if self.current_step - self.original_step > self.optimize_length - 2:
      done = True
    else:
      done = False

    current_battery = self.battery_state
    chp_action = action[0]
    pv_action = action[1][chp_action][0]

    pv_action = (pv_action + 1) / 2


    battery_action = action[1][chp_action][1]

    # 电池动作
    if (battery_action * self.battery_change) / 4595 + self.battery_state < 0:
      battery_action = - self.battery_state  # 改变动作，只能放这么多电
      reward_battery = -0.1  # 动作错误，一个负奖励
      self.battery_state = 0

    elif (battery_action * self.battery_change) / 4595 + self.battery_state > 1:
      battery_action = 1 - self.battery_state
      reward_battery = -0.1
      self.battery_state = 1

    else:
      reward_battery = 0
      self.battery_state = self.battery_state + (battery_action * self.battery_change) / 4595
      battery_action = battery_action




    # 动作归一化


    pv_gen = pv_action * self.data.iloc[self.current_step]['PV_output']
    bio_gen = chp_action * 40

    error = bio_gen + pv_gen - self.data.iloc[self.current_step]['Whole'] - battery_action * self.battery_change


    reward = self.reward(error, chp_action, self.battery_state, battery_action)

    reward = reward
    #
    self.current_step += 1
    state = self._next_observation()

    return state, self.current_step, reward, done, {'step': self.current_step, 'error': error, 'pv_action': pv_action,
                                 'bio_action': chp_action, 'battery_action': battery_action, 'battery_state': current_battery}





# #
# if __name__ == "__main__":
#     env = Takasago_ENV()
#
#
#
#     PV_gen = []
#     Power_demand = []
#     batteray_usage = []
#     bio_gen = []
#     error = []
#     battery_state = []
#     pv_action = []
#     reward_all = []
#
#     obs = env.reset()
#     done = False
#     data = pd.read_csv('train.csv').rename(columns=lambda x: x.strip())
#     df = pd.DataFrame(columns=['bio_gen', 'battery_usage', 'pv_gen', 'power', 'error', 'battery_state', 'pv_action', 'reward'])
#     print('开始最终测试')
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#
#         Power_demand.append(data.loc[info['step'] - 1, 'Whole'])
#         PV_gen.append(info['pv_action'] * data.loc[info['step'] - 1, 'PV_output'])
#         bio_gen.append(info['bio_action'] * 80)
#         batteray_usage.append(- info['battery_action'] * env.battery_change)
#         error.append(info['error'])
#         battery_state.append(info['battery_state'])
#         pv_action.append(info['pv_action'])
#         reward_all.append(reward)
#         if -20 < info['error'] < 0:
#             #print('good')
#             #print(info['error'])
#             continue
#         else:
#             print('not good')
#             print(info['error'])
#
#     df['battery_usage'] = batteray_usage
#     df['bio_gen'] = bio_gen
#
#     df['pv_gen'] = PV_gen
#     df['power'] = Power_demand
#     df['error'] = error
#     df['battery_state'] = battery_state
#     df['pv_action'] = pv_action
#     df['reward']  = reward_all
#
#     df.to_csv('rl.csv')























  
  

