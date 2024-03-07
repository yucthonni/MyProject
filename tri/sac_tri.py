import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

class ReplayBuffer():
    def __init__(self):
        self.s=[]
        self.a=[]
        self.r=[]
        self.s_=[]
        self.d=[]
    def add(self,state,action,reward,next_state,done):
        self.s.append(state)
        self.a.append(action)
        self.r.append(reward)
        self.s_.append(next_state)
        self.d.append(done)
        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SAC(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return action

    def update(self, replay_buffer, batch_size=100):
        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Compute the target Q-value
        with torch.no_grad():
            next_action = self.actor(next_state)
            target_Q1, target_Q2 = self.critic(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * (target_Q + self.alpha * next_action.pow(2).sum(dim=1))

        # Get current Q-value estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean() + self.alpha * (self.actor(state).pow(2).sum(dim=1) + self.target_entropy).mean()

        # Optimize the actor and critic
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update alpha
        alpha_loss = -(self.log_alpha * (actor_loss + self.target_entropy).detach()).mean()
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()

        self.alpha = self.log_alpha.exp()

# Hyperparameters
state_dim = 3
action_dim = 1
gamma = 0.99
tau = 0.005

# Initialize SAC agent
sac = SAC(state_dim, action_dim)

# Initialize replay buffer
replay_buffer = ReplayBuffer()

env=gym.Env('CartPole-v1')

# Train SAC agent
for episode in range(1000):
    # Collect experience
    state = env.reset()
    for t in range(1000):
        action = sac.select_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        # Update SAC agent
        sac.update(replay_buffer)
