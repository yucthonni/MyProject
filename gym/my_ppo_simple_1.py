'''一个简单的PPO实现，其中的actor和critic使用相同网络'''
#try custom PPO on PccNs-v0
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import network_sim

# 定义策略网络
class ActorCriticNetwork(nn.Module):
    def __init__(self,n_states,n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, n_actions)  # 输出动作概率

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=-1)

# 创建 PPO 代理
class PPOAgent:
    def __init__(self,n_states,n_actions,device):
        self.device=device
        self.actor_critic = ActorCriticNetwork(n_states,n_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=1e-3)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor_critic(state)
        action = probs.multinomial(1)[0].tolist()
        return action

    def train(self, states, actions, rewards, values):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        # 计算优势函数
        advantages = rewards - values

        # 计算策略梯度
        log_probs = torch.log(self.actor_critic(states)[range(len(states)), actions]).to(self.device)
        actor_loss = -torch.mean(log_probs * advantages).to(self.device)

        # 计算价值函数损失
        critic_loss = F.mse_loss(self.actor_critic(states).squeeze(-1), rewards).to(self.device)

        # 计算总损失
        loss = actor_loss + 0.5 * critic_loss

        # 更新网络参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

env=gym.make('PccNs-v0')

n_states=env.observation_space.shape
n_actions=env.action_space.shape

agent=PPOAgent(n_states[0],n_actions[0],'cuda')
s=env.reset()

num_eval_episodes=100
eval_rewards=[]
for episode in range(num_eval_episodes):
    state=env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state

    eval_rewards.append(episode_reward)
        
print(f"Average evaluation reward: {np.mean(eval_rewards)}")

num_iteration=10
num_episodes=100
for i in range(num_iteration):
    for episode in range(num_episodes):
        state=env.reset()
        done=False
        episode_reward=0
        
        while not done:
            action=agent.act(state)
            next_state,reward,done,_=env.step(action)
            episode_reward+=reward
            agent.train([state],[action],[reward],[0])
            state=next_state
            
    print(f"Epoch {i+1}/{num_iteration}：Reward {episode_reward}")

num_eval_episodes=100
eval_rewards=[]
for episode in range(num_eval_episodes):
    state=env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state

    eval_rewards.append(episode_reward)
        
print(f"Average evaluation reward: {np.mean(eval_rewards)}")


