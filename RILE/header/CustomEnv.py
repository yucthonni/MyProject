import gym
from buffer import ReplayBuffer
import contextlib

class CustomEnv:
    def __init__(self,env_id,buffer_size):
        with contextlib.redirect_stderr(None):
            self.env=gym.make(env_id)
        self.student_buffer=ReplayBuffer(buffer_size)
        self.teacher_buffer=ReplayBuffer(buffer_size)
        
    def reset(self):
        with contextlib.redirect_stderr(None),contextlib.redirect_stdout(None):
            return self.env.reset()
    
    def step(self,action):
        with contextlib.redirect_stderr(None),contextlib.redirect_stdout(None):
            return self.env.step(action)
    
    def get_state_dim(self):
        return self.env.observation_space.shape
    
    def get_action_dim(self):
        return self.env.action_space.shape