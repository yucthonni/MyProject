import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

################################## set device ##################################
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

################################## PPO Policy ##################################
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
        
        
    def store(self,state,action,log_prob,next_state,reward,state_value,done):
        state=torch.FloatTensor(state).to(device)
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
        sample_index=np.random.choice(batch_size,self.buffer_size,replace=True)
        if return_index:
            return sample_index
        state=self.state
        action=self.action
        log_prob=self.logprobs
        next_state=self.next_state
        reward=self.reward
        state_value=self.state_value
        done=self.done
        return state,action,log_prob,next_state,reward,state_value,done
    
    def clean(self):#这里暂时有个bug，不能解决append那里，不过暂时用不上这个函数，就先不管了
        self.index=0
        self.state=[]
        self.action=[]
        self.logprobs=[]
        self.next_state=[]
        self.reward=[]
        self.state_value=[]
        self.done=[]
# class RolloutBuffer:
#     def __init__(self):
#         self.actions = []
#         self.states = []
#         self.logprobs = []
#         self.rewards = []
#         self.state_values = []
#         self.is_terminals = []
    
#     def clear(self):
#         del self.actions[:]
#         del self.states[:]
#         del self.logprobs[:]
#         del self.rewards[:]
#         del self.state_values[:]
#         del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        # if self.has_continuous_action_space:
        #     action_mean = self.actor(state)
            
        #     action_var = self.action_var.expand_as(action_mean)
        #     cov_mat = torch.diag_embed(action_var).to(device)
        #     dist = MultivariateNormal(action_mean, cov_mat)
            
        #     # For Single Action Environments.
        #     if self.action_dim == 1:
        #         action = action.reshape(-1, self.action_dim)
        # else:
        #     action_probs = self.actor(state)
        #     dist = Categorical(action_probs)
        action_mean=self.actor(state)
        action_var=self.action_var.expand_as(action_mean)
        cov_mat=torch.diag_embed(action_var).to(device)
        dist=MultivariateNormal(action_mean,cov_mat)
        if self.action_dim==1:
            action=action.reshape(-1,self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, buffer_size, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, K_epochs=80, eps_clip=0.2, has_continuous_action_space=True, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = ReplayBuffer(buffer_size)

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        # self.buffer.states.append(state)
        # self.buffer.actions.append(action)
        # self.buffer.logprobs.append(action_logprob)
        # self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten(),action,action_logprob,state_val

    def update(self,batch_size):
        #改成sample一些buffer
        index=self.buffer.sample(batch_size,True)
        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed([self.buffer.reward[i] for i in index]), reversed([self.buffer.done[i] for i in index])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack([self.buffer.state[i] for i in index]), dim=0).detach().to(device)
        old_actions = torch.squeeze(torch.stack([self.buffer.action[i] for i in index]), dim=0).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack([self.buffer.logprobs[i] for i in index], dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack([self.buffer.state_value[i] for i in index], dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        #self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       