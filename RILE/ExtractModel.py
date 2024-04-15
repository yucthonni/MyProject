from Student_Agent import StudentAgent
import sys
from PPOee import PPO
import torch
from glob import glob
import os
from tqdm import tqdm

# model=PPO(30,1,256,None)
path=sys.argv[1]
filelist=glob('model/student/'+path+'/studentmodel*.zip')

for file in filelist:
    with open(file,'rb') as f:
        model=torch.load(f)
        filename,_=os.path.splitext(os.path.basename(file))
        torch.save(model.policy_old.state_dict,'model/student/'+path+'/'+filename+'.pt')
    os.remove(file)