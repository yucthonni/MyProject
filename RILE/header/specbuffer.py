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
        self.state_buffer=[]
        
        
    def store(self,state,action,log_prob=None,next_state=None,reward=None,state_value=None,done=None):
        #buffer_size必须是400的倍数
        self.state_buffer.append(state)
        
        next_state=torch.FloatTensor(next_state).view(-1)
        action=torch.FloatTensor(action)
        reward=torch.FloatTensor(np.array(reward))
        if self.index>=self.buffer_size:
            index=self.index%self.buffer_size
            if done:
                self.state[int(index/400)]=self.state_buffer
                self.state_buffer=[]
            self.action[index]=action
            self.logprobs[index]=log_prob
            self.next_state[index]=next_state
            self.reward[index]=reward
            self.state_value[index]=state_value
            self.done[index]=done
        else:
            if done:
                self.state.append(self.state_buffer)
                self.state_buffer=[]
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
        length=len(self.state[0])
        state=[torch.FloatTensor(self.get_state_padding(self.state[int(i/length)],i%length,10,True)[0]).view(-1) for i in sample_index]
        action=[self.action[i] for i in sample_index]
        log_prob=[self.logprobs[i] for i in sample_index]
        next_state=[self.next_state[i] for i in sample_index]
        reward=[self.reward[i] for i in sample_index]
        state_value=[self.state_value[i] for i in sample_index]
        done=[self.done[i] for i in sample_index]
        teacher_reward=[self.teacher_reward[i] for i in sample_index]
        return state,action,log_prob,next_state,reward,state_value,done,teacher_reward
    
    def get_state_padding(self,a:list,index:int,offset:int,zero_padding:bool):
        if not zero_padding:
            # return a[max(0,index-offset):min(index+offset,len(a))]
            return a[max(0,index-offset+1):index+1],a[index+1:min(index+offset+1,len(a)+1)]
        b=a.copy()
        zero_padding=np.copy(b[0])
        for _ in range(offset):
            b.insert(0,zero_padding)
            b.append(zero_padding)
        # return b[max(0,index):min(index+2*offset,len(a)+2*offset)]
        return b[max(0,index+1):index+offset+1],b[index+offset+1:min(index+2*offset+1,len(a)+2*offset+1)]
    
    def state_from_index(self,index:list):
        length=len(self.state[0])
        state=[torch.FloatTensor(self.get_state_padding(self.state[int(i/length)],i%length,10,True)[0]).view(-1) for i in index]
        return torch.stack(state,dim=0).detach().cuda()
        
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