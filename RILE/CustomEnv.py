class CustomEnv:
    def __init__(self,env_id):
        self.env=gym.make(env_id)
        self.student_buffer=ReplayBuffer(8192)
        self.teacher_buffer=ReplayBuffer(8192)
        
    def reset(self):
        return self.env.reset()
    
    def step(self,action):
        return self.env.step(action)
    
    def get_state_dim(self):
        return self.env.observation_space.shape
    
    def get_action_dim(self):
        return self.env.action_space.shape