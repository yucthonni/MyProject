
from Student_Agent import StudentAgent
import sys
from PPOee import ActorCritic
import torch
from glob import glob
import os
from tqdm import tqdm

# model=PPO(30,1,256,None)
isStu=True
path=sys.argv[1]
if isStu:
    filelist=glob('model/student/'+path+'/studentmodel*.pt')
else:
    filelist=glob('model/teacher/'+path+'/teachermodel*.pt')

model=ActorCritic(30,1,256,True,0.6)

for file in filelist:
    with open(file,'rb') as f:
        model=torch.load(f)
        model.eval()
        filename,_=os.path.splitext(os.path.basename(file))
        torch.save(model.state_dict(),'model/student/'+path+'/'+filename+'.pt',_use_new_zipfile_serialization=False)