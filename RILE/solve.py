import numpy as np
from sympy import false
import network_sim
import pickle
import torch
from datetime import datetime

import os
import sys


from ExpertTrajectory import ExpertTrajectory
from CustomEnv import CustomEnv
from Student_Agent import StudentAgent
from TeacherAgent import TeacherAgent



device='cuda'
BUFFER_SIZE=14400

env=CustomEnv('PccNs-v0',BUFFER_SIZE)

# et=ExpertTrajectory(4000)
sa=StudentAgent(np.prod(env.get_state_dim()),env.get_action_dim()[0],env)
# ta=TeacherAgent(np.prod(env.get_state_dim()),env.get_action_dim()[0],env,et)



# et.load_expert_model('verygood.zip')
# et.generate_trajectory(env)

stupath='model/student/'+datetime.now().strftime('%d%H%M')
teapath='model/teacher/'+datetime.now().strftime('%d%H%M')
os.makedirs(stupath)
os.makedirs(teapath)

for i in range(100):
    print('第',i+1,'次更新')
    sa.replay_buffer.clean()
    sa.generate_trajectory(36)
    # ta.ComputeReward()
    sa.train(1000,1024)
    # ta.discriminator.update(100,True)
    # ta.trainPPO(100,1024)
    
    if i%50==0:
        with open(stupath+'/studentmodel'+datetime.now().strftime('%H%M')+'.pt','wb') as f:    
            torch.save(sa.model.policy.state_dict(),f,_use_new_zipfile_serialization=False)
        # with open(teapath+'/teachermodel'+datetime.now().strftime('%H%M')+'.zip','wb') as f:    
        #     torch.save(ta.model,f)


