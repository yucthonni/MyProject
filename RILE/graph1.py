# %%
import re

# %%
pattern = r"Reward: (?P<reward>[-]*[0-9.]+) Variance: (?P<variance>[0-9.]+)"

# %%
os.getcwd()

# %%
reward=[]
std=[]

# %%
with open('student_model_evaluation','r') as f:
    datas=f.readlines()
    for data in datas:
        match=re.search(pattern,data)
        reward.append(match.group('reward'))
        std.append(match.group('variance'))

# %%
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# %%
for i in range(len(reward)):
    reward[i]=float(reward[i])
    std[i]=float(std[i])

# %%
import numpy as np
mean=np.mean(reward)
var=np.std(reward)

# %%
for i in range(len(reward)):
    reward[i]=(reward[i]-mean)/var

# %%
reward

# %%
plt.legend()
# plt.boxplot(list(zip(reward,std)))
plt.plot(reward)


