import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ReplayBuffer():
    def __init__(self,memsize,state_dim,action_dim):
        self.memsize=memsize
        self.cntr=0
        self.s=np.zeros((self.memsize,state_dim))
        self.a=np.zeros((self.memsize,action_dim))
        self.r=np.zeros(self.memsize)
        self.s_=np.zeros((self.memsize,state_dim))
        self.d=np.zeros(self.memsize,dtype=np.bool8)
        
    def add(self,state,action,reward,next_state,done):
        index=self.cntr%self.memsize
        self.s[index]=state
        self.a[index]=action
        self.r[index]=reward
        self.s_[index]=next_state
        self.d[index]=done
        self.cntr+=1
        
    def sample(self,batch_size):
        maxmem=min(self.memsize,self.cntr)
        index=np.random.choice(maxmem,batch_size,replace=True)
        states=self.s[index]
        actions=self.a[index]
        rewards=self.r[index]
        next_states=self.s_[index]
        dones=self.d[index]
        
        return states,actions,rewards,next_states,dones
    
class Actor(nn.Module):
    def __init__(self, state_dim,hidden_dim,max_action,max_log_std=2):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.mu_head=nn.Linear(hidden_dim,1)
        self.log_std_head=nn.Linear(hidden_dim,1)
        self.max_action=max_action
        self.reparam_noise=1e-6
        
        self.max_log_std=max_log_std
        
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        mu=self.mu_head(x)
        log_std_head=self.log_std_head(x)
        log_std_head=torch.clamp(log_std_head,self.reparam_noise,self.max_log_std)
        return mu,log_std_head
    
    def select_action(self,state):
        mu,log_std_head=self.forward(state)
        probs=torch.distributions.Normal(mu,log_std_head)
        action_=probs.sample()
        action=torch.tanh(action_)*torch.tensor(self.max_action)
        log_probs=probs.log_prob(action_)
        log_probs-=torch.log(1-action.pow(2)+self.reparam_noise)
        #print(log_probs)
        log_probs=log_probs.sum(1,keepdim=True)
        
        return action,log_probs
    
class Q(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super(Q,self).__init__()
        self.q=nn.Sequential(
            nn.Linear(state_dim+action_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
        
    def forward(self,state,action):
        x=torch.cat([state,action],dim=-1)
        return self.q(x)
    
class Critic(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Critic,self).__init__()
        self.critic=nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
        
    def forward(self,state):
        return self.critic(state)
    
class SAC:
    def __init__(self,state_dim,action_dim,hidden_dim,
                 max_action,actor_lr=1e-3,value_lr=1e-3,
                 q_lr=1e-3,gamma=0.99,tau=0.005,memory_size=8196,batch_size=64):
        #读取参数
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.hidden_dim=hidden_dim
        self.max_action=max_action
        self.actor_lr=actor_lr
        self.q_lr=q_lr
        self.value_lr=value_lr
        self.gamma=gamma
        self.tau=tau
        self.memory_size=memory_size
        self.batch_size=batch_size
        
        #初始化各部件
        ##回放缓存
        self.replay_buffer=ReplayBuffer(self.memory_size,self.state_dim,self.action_dim)
        ##策略网络
        self.actor=Actor(self.state_dim,self.action_dim,self.hidden_dim,self.max_action)
        #self.target_actor=Actor(self.state_dim,self.action_dim,self.hidden_dim,self.max_action)
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=actor_lr)
        ##价值网络
        self.value=Critic(self.state_dim,self.hidden_dim)
        self.target_value=Critic(self.state_dim,self.hidden_dim)
        self.value_optimizer=optim.Adam(self.value.parameters(),lr=value_lr)
        ##动作价值网络
        self.qvalue=Q(self.state_dim,self.action_dim,self.hidden_dim)
        #self.target_qvalue=Q(self.state_dim,self.action_dim,self.hidden_dim)
        self.qvalue_optimizer=optim.Adam(self.qvalue.parameters(),lr=q_lr)
        
    def store(self,state,action,reward,next_state,done):
        self.replay_buffer.add(state,action,reward,next_state,done)
    
    def sample(self):
        state,action,reward,next_state,done=self.replay_buffer.sample(self.batch_size)
        state=torch.FloatTensor(state)
        action=torch.FloatTensor(action)
        reward=torch.FloatTensor(reward)
        next_state=torch.FloatTensor(next_state)
        done=torch.BoolTensor(done)
        return state,action,reward,next_state,done
    
    def select_action(self,state):
        state=torch.FloatTensor(state).unsqueeze(0)
        action,_=self.actor.select_action(state)
        return action.detach().numpy()[0]
    
    def update(self):
        #采样
        state,action,reward,next_state,done=self.sample()
        
        #计算alue值
        value=self.value(state).view(-1)
        next_value=self.target_value(next_state).view(-1)
        
        #选择动作
        action_sampled,log_prob=self.actor.select_action(state)
        log_prob=log_prob.view(-1)
        
        #计算q值
        q_value=self.qvalue(state,action_sampled).view(-1)
        
        #更新value网络
        self.value_optimizer.zero_grad()
        target_value=q_value-log_prob
        value_loss=0.5*F.mse_loss(value,target_value)
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()
        
        #更新Actor网络
        self.actor_optimizer.zero_grad()
        actor_loss=log_prob-q_value
        actor_loss=torch.mean(actor_loss)
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        #更新Q网络
        self.qvalue_optimizer.zero_grad()
        q_hat=reward+self.gamma*next_value
        q_value=self.qvalue(state,action)
        q_loss=F.mse_loss(q_value.view(-1),q_hat)
        q_loss.backward()
        self.qvalue_optimizer.step()
        
        self.update_parameters()
        
    def update_parameters(self):
        value_state_dict=dict(self.value.named_parameters())
        target_value_state_dict=dict(self.target_value.named_parameters())
        
        for name in value_state_dict:
            value_state_dict[name]=self.tau*value_state_dict[name].clone()+\
                (1-self.tau)*target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)
