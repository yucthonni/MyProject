import torch
import numpy as np

class ReplayBuffer:
    def __init__(self,buffer_size):
        self.buffer_size=buffer_size
        self.index=0
        self.state=[]
        self.action=[]
        self.logprobs=[]
        self.next_state=[]
        self.reward=[]
        self.state_value=[]
        self.done=[]
        self.teacher_reward=torch.zeros(buffer_size)
        self.islatest=False
        
        
    def store(self,state,action,log_prob=None,next_state=None,reward=None,state_value=None,done=None):
        state=torch.FloatTensor(state).view(-1)
        action=torch.FloatTensor(action)
        # log_prob=torch.FloatTensor(log_prob)
        next_state=torch.FloatTensor(next_state).view(-1)
        reward=torch.FloatTensor(np.array(reward))
        # state_value=torch.FloatTensor(state_value)
        # done=torch.BoolTensor(done)
        if self.index>=self.buffer_size:
            index=self.index%self.buffer_size
            self.state[index]=state
            self.action[index]=action
            self.logprobs[index]=log_prob
            self.next_state[index]=next_state
            self.reward[index]=reward
            self.state_value[index]=state_value
            self.done[index]=done
        else:
            self.state.append(state)
            self.action.append(action)
            self.logprobs.append(log_prob)
            self.next_state.append(next_state)
            self.reward.append(reward)
            self.state_value.append(state_value)
            self.done.append(done)
        self.index+=1
        
        
    def sample(self,batch_size,return_index=False):
        sample_index=np.random.choice(min(self.buffer_size,self.index),batch_size,replace=True)
        if return_index:
            return sample_index
        state=[self.state[i] for i in sample_index]
        action=[self.action[i] for i in sample_index]
        log_prob=[self.logprobs[i] for i in sample_index]
        next_state=[self.next_state[i] for i in sample_index]
        reward=[self.reward[i] for i in sample_index]
        state_value=[self.state_value[i] for i in sample_index]
        done=[self.done[i] for i in sample_index]
        teacher_reward=[self.teacher_reward[i] for i in sample_index]
        return state,action,log_prob,next_state,reward,state_value,done,teacher_reward
    
    def clean(self):#这里暂时有个bug，不能解决append那里，不过暂时用不上这个函数，就先不管了
        self.index=0
        self.state=[]
        self.action=[]
        self.logprobs=[]
        self.next_state=[]
        self.reward=[]
        self.state_value=[]
        self.done=[]
        self.teacher_reward=[]
        
        
# # class ReplayBuffer:
# #     def __init__(self,buffetqdm(range(1000))tqdm(range(1000))r_size,state_dim,action_dim):
# #         self.buffer_size=buffer_size
# #         self.index=0
# #         self.state=np.zeros((bu8192ffer_size,state_dim))
# #         self.action=np.zeros((buffer_size,action_dim))
# #         self.next_state=np.zeros((buffer_size,state_dim))
# #         self.reward=np.zeros(buffer_size,dtype=float)
# #         self.done=np.zeros(buffer_size,dtype=bool)
        
# #     def store(self,state,action,next_state,done):
# #         index=self.index%self.buffer_size
# #         self.state[index]=state8192
# #         self.action[index]=action
# #         self.next_state[index]=next_state
# #         self.reward[index]=0.0
# #         self.done[index]=done
# #         self.index+=1
# #
# #     def sample(self,batch_size,return_index=False):
# #         sample_index=np.random.choice(batch_size,self.buffer_size,replace=True)
# #         if return_index:
# #             return sample_index
# #         state=self.state[sample_index]
# #         action=self.action[sample_index]
# #         next_state=self.next_state[sample_index]
# #         reward=self.reward[sample_index]
# #         done=self.done[sample_index]
# #         return state,action,next_state,reward,done
    
# #     def clean(self):
# #         self.index=0
# class ReplayBuffer:
#     def __init__(self,buffer_size):
#         self.buffer_size=buffer_size
#         self.index=0
#         self.state=[]
#         self.action=[]
#         self.logprobs=[]
#         self.next_state=[]
#         self.reward=[]
#         self.state_value=[]
#         self.done=[]
        
        
#     def store(self,state,action,log_prob,next_state,reward,state_value,done):
#         #state=torch.FloatTensor(state).to(device)
#         if self.index>=self.buffer_size:
#             index=self.index%self.buffer_size
#             self.state[index]=state
#             self.action[index]=action
#             self.logprobs[index]=log_prob
#             self.next_state[index]=next_state
#             self.reward[index]=reward
#             self.state_value[index]=state_value
#             self.done[index]=done
#         else:
#             self.state.append(state)
#             self.action.append(action)
#             self.logprobs.append(log_prob)
#             self.next_state.append(next_state)
#             self.reward.append(reward)
#             self.state_value.append(state_value)
#             self.done.append(done)
        
#         self.index+=1
        
        
#     def sample(self,batch_size,return_index=False):
#         #这里之前一直搞反了
#         sample_index=np.random.choice(self.buffer_size,batch_size,replace=True)
#         if return_index:
#             return sample_index
#         state=[self.state[i] for i in sample_index]
#         action=[self.action[i] for i in sample_index]
#         log_prob=[self.logprobs[i] for i in sample_index]
#         next_state=[self.next_state[i] for i in sample_index]
#         reward=[self.reward[i] for i in sample_index]
#         state_value=[self.state_value[i] for i in sample_index]
#         done=[self.done[i] for i in sample_index]
#         return state,action,log_prob,next_state,reward,state_value,done
    
#     def clean(self):#这里暂时有个bug，不能解决append那里，不过暂时用不上这个函数，就先不管了
#         self.index=0
#         self.state=[]
#         self.action=[]
#         self.logprobs=[]
#         self.next_state=[]
#         self.reward=[]
#         self.state_value=[]
#         self.done=[]