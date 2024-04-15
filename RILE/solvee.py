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


device='cuda'
BUFFER_SIZE=14400


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
        self.model = PPO(10, action_dim,
                         256, self.replay_buffer)
        
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
        self.discriminator.collect_expert()


env=CustomEnv('PccNs-v0',BUFFER_SIZE)


et=ExpertTrajectory(4000)
sa=Student(np.prod(env.get_state_dim()),env.get_action_dim()[0],env)
ta=Teacher(np.prod(env.get_state_dim()),env.get_action_dim()[0],env,et)



et.load_expert_model('verygood.zip')
et.generate_trajectory(env)

stupath='model/student/'+datetime.now().strftime('%d%H%M')
teapath='model/teacher/'+datetime.now().strftime('%d%H%M')
os.makedirs(stupath)
os.makedirs(teapath)

for i in range(int(sys.argv[1])):
    # with contextlib.redirect_stderr(None):
    print('第',i+1,'次更新')
    print('Cleaning Buffer ......')
    sa.replay_buffer.clean()
    print('<Student> Generating Trajectorys ......')
    sa.generate_trajectory(36)
    print('<Teacher> is Computing Rewards of Trajectorys ......')
    ta.ComputeReward()
    print('<Student> Training by Given Rewards ......')
    sa.train(1000,512)
    if i%5==0:
        with open(stupath+'/studentmodel'+datetime.now().strftime('%H%M')+'.zip','wb') as f:    
            torch.save(sa.model,f)
    if i%10==0:
        print('<Teacher> Updating Discriminator ......')
        ta.discriminator.update(512,True)
        print('<Teacher> Training by Discriminator ......')
        ta.trainPPO(2000,1024,True)
        with open(teapath+'/teachermodel'+datetime.now().strftime('%H%M')+'.zip','wb') as f:    
            torch.save(ta.model,f)


