#try custom PPO on CartPole
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 创建 CartPole 环境
env = gym.make('CartPole-v1')

# 定义策略网络
class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)  # 输出动作概率

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

# 创建 PPO 代理
class PPOAgent:
    def __init__(self):
        self.actor_critic = ActorCriticNetwork()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=1e-3)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor_critic(state)
        action = probs.multinomial(1).item()
        return action

    def train(self, states, actions, rewards, values):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)

        # 计算优势函数
        advantages = rewards - values

        # 计算策略梯度
        log_probs = torch.log(self.actor_critic(states)[range(len(states)), actions])
        actor_loss = -torch.mean(log_probs * advantages)

        # 计算价值函数损失
        critic_loss = F.mse_loss(self.actor_critic(states).squeeze(-1), rewards)

        # 计算总损失
        loss = actor_loss + 0.5 * critic_loss

        # 更新网络参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
agent=PPOAgent()
num_episodes=1000
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
        
    print(f"Episode {episode}：Reward {episode_reward}")
    
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