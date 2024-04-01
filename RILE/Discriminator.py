import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

device='cuda'

class Discriminator_network(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Discriminator_network,self).__init__()
        self.discriminator=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,x):
        x=self.discriminator(x)
        return nn.functional.sigmoid(x)
    
class Discriminator():
    def __init__(self,input_dim,hidden_dim,batch_size,sbuffer,ebuffer):
        self.model=Discriminator_network(input_dim,hidden_dim)
        self.sbuffer=sbuffer
        self.ebuffer=ebuffer
        self.batch_size=batch_size
        self.policy_dist=None
        self.expert_dist=None
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        self.loss_criterion=nn.BCELoss()
        
    def collect_expert(self):
        self.expert_dist=torch.cat((torch.stack(self.ebuffer.state,dim=0),torch.stack(self.ebuffer.action,dim=0)),1)
        
    def collect_dist(self,batch_size):
        index=self.sbuffer.sample(batch_size,True)
        state=self.sbuffer.from_after_state(index)
        action=torch.stack([self.sbuffer.action[i] for i in index],dim=0)
        self.policy_dist=torch.cat((state,action),1)
        
    def update(self,total_timestep:int,progress=True):
        if progress:
            pb=tqdm(range(total_timestep))
            for i in pb:
                stu_traj=self.collect_dist(self.batch_size)
                eo=self.model(self.expert_dist)
                so=self.model(self.policy_dist)
                loss=self.loss_criterion(eo,torch.ones(len(self.expert_dist),1))+\
                    self.loss_criterion(so,torch.zeros(self.batch_size,1))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pb.update()
        else:
            for i in range(total_timestep):
                stu_traj=self.collect_dist(self.batch_size)
                eo=self.model(self.expert_dist)
                so=self.model(self.policy_dist)
                loss=self.loss_criterion(eo,torch.ones(len(self.expert_dist),1))+\
                    self.loss_criterion(so,torch.zeros(self.batch_size,1))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()