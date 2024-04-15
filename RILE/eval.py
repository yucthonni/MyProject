from os import write
import torch
from Student_Agent import StudentAgent
from CustomEnv import CustomEnv
import numpy as np
import network_sim
import sys
from glob import glob
import json
from torch.utils.tensorboard.writer import SummaryWriter


env=CustomEnv('PccNs-v0',8192)
sa=StudentAgent(np.prod(env.get_state_dim()),env.get_action_dim()[0],env)


path=sys.argv[2]
# filelist=glob('model/student/'+path+'/studentmodel*.zip')
filelist=glob('model/student/'+path+'/studentmodel*.pt')
filelist.sort()
# evalfile=open('student_model_evaluation','a')
json_records=[]
writer=SummaryWriter('./runs/'+path+'/')

if len(sys.argv)==4:
    while not filelist[0][33:37]==sys.argv[3]:
        filelist.pop(0)

for i in range(len(filelist)):
    with open(filelist[i],'rb') as f:
        # sa.model=torch.load(f)
        state_dict=torch.load(f)
        sa.model.policy.load_state_dict=state_dict
        sa.model.policy_old.load_state_dict=state_dict
        
    
    rewards=[]
    
    print('evaluating Model',filelist[i][33:37],'......')
    
    for _ in range(int(sys.argv[1])):
        s=env.reset()
        d=False
        reward=0
        while not d:
            a,_,_,_=sa.model.select_action(s)
            s,r,d,_=env.step(a)
            reward+=r
        rewards.append(reward)
        
    mean_reward=np.mean(rewards)
    std_reward=np.std(rewards)
    
    writer.add_scalar('mean_reward',mean_reward,i)
    writer.add_scalar('reward_std',std_reward,i)
    
    
    # string='Student Model <{}> Reward: {} Variance: {} \n'.format(file[33:37],mean_reward,std_reward)
    # evalfile.write(string)
    json_record={
        'model_name':filelist[i][33:37],
        'reward_mean':mean_reward,
        'reward_var':std_reward
    }
    json_records.append(json_record)

json_data=json.dumps(json_records,indent=5)


with open('eval/'+path+'.json','w') as evalfile:
    evalfile.write(json_data)
        