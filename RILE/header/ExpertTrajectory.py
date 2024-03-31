from buffer import ReplayBuffer
from stable_baselines3 import PPO as sb3ppo
from tqdm import tqdm

class ExpertTrajectory(ReplayBuffer):
    def __init__(self, buffer_size):
        super().__init__(buffer_size)
        self.model=None
        self.max_step=buffer_size*5
        
    def load_expert_model(self,path:str):
        self.model=sb3ppo.load(path)
    
    def generate_trajectory(self,env,max_step=0):
        if max_step:
            self.max_step=max_step
        with tqdm(range(self.max_step)) as pb:
            while True:
                s=env.reset()
                d=False
                while not d:
                    a,_=self.model.predict(s)
                    s_,r,d,_=env.step(a)
                    self.store(s,a,None,s_,r,None,d)
                    s=s_
                    pb.update()
                if pb.n>=self.max_step:
                    break