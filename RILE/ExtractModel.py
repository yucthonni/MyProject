from Student_Agent import StudentAgent
import sys
from PPOee import PPO
import torch
from glob import glob
import os
from tqdm import tqdm

# model=PPO(30,1,256,None)
isStu=False
path=sys.argv[1]
if isStu:
    filelist=glob('model/student/'+path+'/studentmodel*.zip')
else:
    filelist=glob('model/teacher/'+path+'/teachermodel*.zip')

for file in filelist:
    with open(file,'rb') as f:
        model=torch.load(f)
        filename,_=os.path.splitext(os.path.basename(file))
        if isStu:
            torch.save(model.policy_old.state_dict(),'model/student/'+path+'/'+filename+'.pt',_use_new_zipfile_serialization=False)
        else:
            torch.save(model.policy_old.state_dict(),'model/teacher/'+path+'/'+filename+'.pt',_use_new_zipfile_serialization=False)
    os.remove(file)