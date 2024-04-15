from CustomEnv import CustomEnv
from Student_Agent import StudentAgent
from TeacherAgent import TeacherAgent
import network_sim
from PPOee import PPO

import torch
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import argparse
import numpy as np
from glob import glob

parser=argparse.ArgumentParser()
parser.add_argument('--smodel',type=str,required=True,help='Path to Student Model')
parser.add_argument('--tmodel',type=str,required=True,help='Path to Teacher Model')

batch_size=1024

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
        
args=parser.parse_args()
smodel_dir=args.smodel
tmodel_dir=args.tmodel

env=CustomEnv('PccNs-v0',8000)
sa=Student(np.prod(env.get_state_dim()),env.get_action_dim()[0],env)
ta=Teacher(np.prod(env.get_state_dim()),env.get_action_dim()[0],env,None)
writer=SummaryWriter('./runs/compare_reward/')

smodel_path=glob(smodel_dir+'/*.zip')[-1]
tmodel_path=glob(tmodel_dir+'/*.zip')[-1]

temp_model=PPO(np.prod(env.get_state_dim()),env.get_action_dim()[0],256,None)

with open(smodel_path,'rb') as sf:
    temp_model=torch.load(sf)
    sa.model.policy_old=temp_model.policy_old
    sa.model.policy=temp_model.policy
    
with open(tmodel_path,'rb') as tf:
    temp_model=torch.load(tf)
    ta.model.policy_old=temp_model.policy_old
    ta.model.policy=temp_model.policy
    
sa.generate_trajectory(20)
index=sa.model.buffer.sample(batch_size,True)
pb=tqdm(index)

for i in pb:
    studnet_reward=sa.model.buffer.reward[i]
    length=len(ta.trajectory_buffer.state[0])
    s=torch.FloatTensor(np.array(ta.trajectory_buffer.get_state_padding(ta.trajectory_buffer.state[int(i/length)],i%length,10,True)[1])).view(-1)
    sa_pair=torch.cat((s,ta.trajectory_buffer.action[i]),-1)
    teacher_reward,_,_,_=ta.model.select_action(sa_pair)
    writer.add_scalar('student reward',studnet_reward,i)
    writer.add_scalar('teacher reward',teacher_reward,i)
    pb.update()
    
writer.close()
