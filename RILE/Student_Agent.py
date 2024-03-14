from SAC import SAC
import gym
import numpy as np
import torch.nn as nn

class StudentAgent():
    def __init__(self,envid,hidden_dim,max_action):
        self.env=gym.make(envid)
        self.state_dim=self.env.observation_space.shape[0]
        self.action_dim=self.env.action_space.shape[0]
        self.SAC=SAC(self.state_dim,self.action_dim,hidden_dim,max_action)
    
    def Train(self,total_step):
        for i in range(total_step):
            state=self.env.reset()
            done=False
            while not done:
                action=self.SAC.select_action(state)
                next_state,reward,done,_=self.env.step(action)
                self.SAC.store(state,action,reward,next_state,done)
                self.SAC.update()
                state=next_state
            if i%10==0:
                print(i/10)
        
    def Evaluate(self,total_step):
        rewards=[]
        for i in range(total_step):
            state=self.env.reset()
            done=False
            ep_reward=0
            while not done:
                action=self.SAC.select_action(state)
                next_state,reward,done,_=self.env.step(action)
                ep_reward+=reward
            rewards.append(ep_reward)
        return np.mean(rewards)
    
class RewardFucntion(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,output_scale):
        super(RewardFucntion,self).__init__()
        
        # self.input_dim=input_dim
        # self.output_dim=output_dim
        self.output_scale=output_scale
        self.reward=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
            nn.Tanh()
        )
        
    def forward(self,x):
        return self.reward(x)*self.output_scale