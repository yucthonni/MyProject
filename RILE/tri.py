import network_sim
from my_ppo_2 import PPO
import gym

env=gym.make("PccNs-v0")
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.shape[0]

model=PPO(state_dim,action_dim,8192)

for i in range(1000):
    s=env.reset()
    done=False
    while not done:
        a,action,l,v=model.select_action(s)
        s_,r,done,_=env.step(a)
        model.buffer.store(s,action,l,s_,r,v,done)

    if i%10==0 & i==0:
        model.update(100)
        print(i/10)