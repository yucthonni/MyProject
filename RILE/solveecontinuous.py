import torch
import numpy as np
from tqdm import tqdm

from PPOee import PPO
import gym
from CustomEnv import CustomEnv
from Student_Agent import StudentAgent
from TeacherAgent import TeacherAgent
from ExpertTrajectory import ExpertTrajectory


import network_sim
from datetime import datetime

import os
import sys
import argparse
from glob import glob


device='cuda'
BUFFER_SIZE=28800


class Student(StudentAgent):
    def __init__(self, state_dim, action_dim, env):
        super().__init__(state_dim, action_dim, env)
        
    def generate_trajectory(self, step: int):
        pb=tqdm(range(step))
        for _ in pb:
            s=self.env.reset()
            d=False
            while not d:
                a,action,l,v=self.model.select_action(s)
                s_,r,d,_=self.env.step(a)
                self.model.buffer.store(s[-3:],a,l,s_,r,v,d)
                s=s_
            pb.update()
            
            
class Teacher(TeacherAgent):
    def __init__(self, state_dim, action_dim, env:CustomEnv, expert_trajectory):
        super().__init__(state_dim, action_dim, env, expert_trajectory)
        self.model = PPO(10, action_dim, 256, self.replay_buffer, False)
        
    def ComputeReward(self):
        pb=tqdm(range(min(self.trajectory_buffer.index,self.trajectory_buffer.buffer_size)))
        for i in pb:
            length=len(self.trajectory_buffer.state[0])
            s=torch.FloatTensor(self.trajectory_buffer.get_state_padding(self.trajectory_buffer.state[int(i/length)],i%length,10,True)[1]).view(-1)
            r=torch.FloatTensor(self.trajectory_buffer.get_reward_padding(self.trajectory_buffer.reward,i,10,length,True)[1]).view(-1)
            reward,tensor_reward,l,v=self.model.select_action(r)
            sa_pair=torch.cat((s.to(device),tensor_reward.view(-1)),-1)
            self.replay_buffer.store(
                r,
                reward,
                l,
                self.trajectory_buffer.next_state[i],
                self.discriminator.model(sa_pair).detach().cpu().numpy(),
                v,
                self.trajectory_buffer.done[i]
            )
            self.trajectory_buffer.reward[i]=reward
            pb.update()


env=CustomEnv('PccNs-v0',BUFFER_SIZE)

sa=Student(np.prod(env.get_state_dim()),env.get_action_dim()[0],env)
ta=Teacher(np.prod(env.get_state_dim()),env.get_action_dim()[0],env,None)

parser=argparse.ArgumentParser()
parser.add_argument('--smodel',type=str,required=True,help='Path to Student Model')
parser.add_argument('--tmodel',type=str,required=True,help='Path to Teacher Model')
parser.add_argument('--iter',type=int,required=True)
args=parser.parse_args()
smodel_path=args.smodel
tmodel_path=args.tmodel
iter=args.iter

stupath='model/student/'+datetime.now().strftime('%d%H%M')
os.makedirs(stupath)
# smodel_path=glob(smodel_dir+'/*.zip')[-1]
# tmodel_path=glob(tmodel_dir+'/*.zip')[-1]


with open(smodel_path,'rb') as fp:
    sm=torch.load(fp)
    sa.model.policy_old.load_state_dict(sm.state_dict())
    sa.model.policy_old.eval()
    sa.model.policy.load_state_dict(sm.state_dict())
    sa.model.policy.eval()
    
    
with open(tmodel_path,'rb') as fp:
    tm=torch.load(fp)
    ta.model.policy_old.load_state_dict(tm)
    ta.model.policy_old.eval()
    ta.model.policy.load_state_dict(tm)
    ta.model.policy.eval()


for i in range(iter):
    # with contextlib.redirect_stderr(None):
    print('第',i+1,'次更新')
    print('Cleaning Buffer ......')
    sa.replay_buffer.clean()
    print('<Student> Generating Trajectorys ......')
    sa.generate_trajectory(72)
    print('<Teacher> is Computing Rewards of Trajectorys ......')
    ta.ComputeReward()
    print('<Student> Training by Given Rewards ......')
    sa.train(1000,2048)
    if i%10==0:
        with open(stupath+'/studentmodel'+datetime.now().strftime('%H%M')+'.pt','wb') as f:    
            torch.save(sa.model.policy_old.state_dict(),f,_use_new_zipfile_serialization=False)


