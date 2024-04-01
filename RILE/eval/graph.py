import matplotlib.pyplot as plt
import json
import sys

with open(sys.argv[1]+'.json','r') as f:
    datas=json.load(f)
    
rewards=[]
deviation=[]

for data in datas:
    rewards.append(data.get('reward_mean'))
    deviation.append(data.get('reward_var'))
    
plt.plot(rewards)
plt.show()