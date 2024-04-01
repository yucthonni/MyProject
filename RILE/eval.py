import torch
from Student_Agent import StudentAgent
from CustomEnv import CustomEnv
import gym
import network_sim
import networkx
import numpy as np
import sys
from glob import glob
import json

env=CustomEnv('PccNs-v0',8192)
sa=StudentAgent(np.prod(env.get_state_dim()),env.get_action_dim()[0],env)

path=sys.argv[2]
filelist=glob('model/student/'+path+'/studentmodel*.zip')
filelist.sort()
# evalfile=open('student_model_evaluation','a')
json_records=[]

if sys.argv[3]:
    while not filelist[0][33:37]==sys.argv[3]:
        filelist.pop(0)

for file in filelist:
    with open(file,'rb') as f:
        sa.model=torch.load(f)
    
    rewards=[]
    
    print('evaluating Model',file[33:37],'......')
    
    for i in range(int(sys.argv[1])):
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
    
    
    # string='Student Model <{}> Reward: {} Variance: {} \n'.format(file[33:37],mean_reward,std_reward)
    # evalfile.write(string)
    json_record={
        'model_name':file[33:37],
        'reward_mean':mean_reward,
        'reward_var':std_reward
    }
    json_records.append(json_record)

json_data=json.dumps(json_records,indent=5)


with open('eval/'+path+'.json','w') as evalfile:
    evalfile.write(json_data)
        