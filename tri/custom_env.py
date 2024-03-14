import gym
import numpy as np
from gym import spaces
import gym
import torch.nn as nn
import torch

class CustomEnv(gym.Env):
    def __init__(self):
        self.observation_space=spaces.Box(-1e3,1e3,(10,5))
        self.action_space=spaces.Box(-1,1,(1,))
        self.discriminator=Discriminator(10*5+1,64,1)
        self.state=np.zeros((10,5),dtype=np.float32)
    
    def step(self,action):
        reward=self.calc_reward(self.state,action)
        next_state=np.random.randn(10,5)
        done=False
        if np.random.random(1)<0.05:
            done=True
        self.state=next_state
        return next_state,reward,done,{}
    def reset(self):
        self.state=np.zeros((10,5),dtype=np.float32)
        return self.state
    
    def calc_reward(self,state,action):
        s=torch.FloatTensor(state).view(50)
        a=torch.FloatTensor(action)
        x=torch.cat([s,a],dim=0)
        return self.discriminator.discriminator(x)
    
    def render(self):
        pass
    
class Discriminator():
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Discriminator,self).__init__()
        self.discriminator=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
    def forward(self,x):
        return self.discriminator(x)
    
gym.register(
    id='CustomEnv-v1',
    entry_point='custom_env:CustomEnv'
)
        