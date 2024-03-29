# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gymnasium as gym
import network_sim_exp

#from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines3 import PPO
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common.simple_arg_parse import arg_or_default
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
import numpy as np

print("--------------Make Env.--------------")
env = gym.make('custom/PccNs-v1')

venv=make_vec_env(
    'custom/PccNs-v1',
    rng=np.random.default_rng(),
    n_envs=4,
    post_wrappers=[
        lambda env,_: RolloutInfoWrapper(env)
    ],
)

#env = gym.make('CartPole-v0')

# model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)
print("--------------Build Model-------------")
model=PPO("MlpPolicy",env, verbose=1, batch_size=1024, gamma=0.99,tensorboard_log='./outputs/')
print("--------------Ready to Learn--------------")
model.learn(total_timesteps=32000*410)
model.save("model_12.zip")
