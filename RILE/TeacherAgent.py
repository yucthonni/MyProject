from Discriminator import Discriminator
import torch
from tqdm import tqdm
from PPOee import PPO

hidden_dim=64
device='cuda'

class TeacherAgent():
    def __init__(self,state_dim,action_dim,env,expert_trajectory):
        self.trajectory_buffer=env.student_buffer
        self.replay_buffer=env.teacher_buffer
        self.model=PPO(state_dim+action_dim,1,self.replay_buffer)
        self.discriminator=Discriminator(state_dim+action_dim,hidden_dim,64,self.trajectory_buffer,expert_trajectory)
        
    def ComputeReward(self):
        pb=tqdm(range(min(self.trajectory_buffer.index,self.trajectory_buffer.buffer_size)))
        for i in pb:
            sa_pair=torch.cat((self.trajectory_buffer.state[i],self.trajectory_buffer.action[i]),-1)
            reward,_,l,v=self.model.select_action(sa_pair)
            self.replay_buffer.store(sa_pair,
                                     reward,
                                     l,
                                     self.trajectory_buffer.next_state[i],
                                     self.discriminator.model(sa_pair).detach().cpu().numpy(),
                                     v,
                                     self.trajectory_buffer.done[i],
                                     )
            self.trajectory_buffer.reward[i]=reward
            pb.update()
        self.discriminator.collect_expert()
            
    def trainPPO(self,total_timestep:int,batch_size:int):
        for i in range(total_timestep):
            self.model.update(batch_size)
            
    def trainDiscriminator(self,total_timestep:int):
        self.discriminator.update(total_timestep,False)
        
    def train(self,total_timestep:int,PPO_timestep:int,D_timestep:int):
        pb=tqdm(range(total_timestep))
        for i in pb:
            print('Computing reward...')
            self.ComputeReward()
            print('Training Discriminator...')
            self.trainDiscriminator(D_timestep)
            print('Training PPO...')
            self.trainPPO(PPO_timestep)
            pb.update()
        
    def save(self):
        pass