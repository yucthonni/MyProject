import gym
from gym import register
from gym import spaces
import numpy as np
import torch.nn as nn

epsilon=0.005

class RewardEnv(gym.Env):
    def __init__(self):
        super(RewardEnv,self).__init__()
        
        #Init State Space
        self.observation_space=spaces.Box(low=-1e6,high=1e6,shape=(3,10))
        
        #Init Action Space
        self.action_space=spaces.Box(low=-1e3,high=1e3,shape=(1,))
        
    def reset(self):
        return self.observation_space.sample()
    
    def step(self, action):
        next_state=self.observation_space.sample()
        reward=np.random.randn(1)*action
        done=False
        if np.random.randn(1)<epsilon:
            done=True
        info=None
        return next_state,reward,done,info
    
    def render(self):
        pass
    
register(
    id='RewardEnv-v1',
    entry_point='reward_env:RewardEnv'
)
    
