from tqdm import tqdm
from PPOee import PPO

hidden_dim = 256


class StudentAgent:
    def __init__(self, state_dim, action_dim, env):
        self.env = env
        # self.replay_buffer=self.env.student_buffer
        self.replay_buffer = env.student_buffer
        self.model = PPO(state_dim, action_dim, hidden_dim, self.replay_buffer)

    def generate_trajectory(self, step: int):
        pb = tqdm(range(step))
        num = 0
        for i in pb:
            s = self.env.reset()
            d = False
            while not d:
                a, _, l, v = self.model.select_action(s)
                s_, r, d, _ = self.env.step(a)
                self.model.buffer.store(s, a, l, s_, r, v, d)
                s = s_
                num += 1
            # pb.update()
        print('生成', num, '条轨迹')

    def train(self, total_timestep, batch_size):
        pb = tqdm(range(total_timestep))
        for i in pb:
            self.model.update(batch_size)
            pb.update()
