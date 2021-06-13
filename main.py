from agent import Agent
from env import TicTacToe
from setting import *
from tqdm import trange


class LeraningSimulation(object):
    def __init__(self) -> None:
        self.env = TicTacToe()
        self.agents = {"Cross": Agent(self.env.action_num), "Circle": Agent(self.env.action_num)}
        self.state = None
        self.result = []
        self.sim_result = []

    def start(self):
        for sim in trange(SIMULATION_NUM, desc='simulation loop'):
            self.agents = {"Cross": Agent(self.env.action_num), "Circle": Agent(self.env.action_num)}
            self.state = self.env.reset()
            self.sim_result = []
            self.one_simulation()
            self.result.append(self.sim_result)

    def one_simulation(self):
        for epi in trange(EPISODE_NUM, desc='episode loop'):
            is_step_done = False
            while not is_step_done:
                action = self.agents["Cross"].select(self.state)
                next_state, reward, done, _, is_step_done = self.env.step(action)
                self.agents["Cross"].save_memory(self.state, action, reward, next_state, done)
                if not is_step_done:
                    self.agents["Cross"].learning()

            if done:
                # 引き分けは1
                self.sim_result.append(1)
                if reward == 1.0:
                    # バツが勝つと0
                    self.sim_result.append(0)
                    self.agents["Circle"].change_last_reward(-1.0)
                self.agents["Cross"].learning()
                self.agents["Circle"].learning()
                continue

            if epi == 0:
                # 初回はなにも行動していないので学習もしない
                self.agents["Circle"].learning()

            self.state = next_state

            is_step_done = False
            while not is_step_done:
                action = self.agents["Circle"].select(self.state)
                next_state, reward, done, _, is_step_done = self.env.step(action)
                self.agents["Circle"].save_memory(self.state, action, reward, next_state, done)
                if not is_step_done:
                    self.agents["Circle"].learning()

            if done:
                # 引き分けは1
                self.sim_result.append(1)
                if reward == 1.0:
                    # マルが勝つと2
                    self.sim_result.append(2)
                    self.agents["Cross"].change_last_reward(-1.0)
                self.agents["Cross"].learning()
                self.agents["Circle"].learning()
                continue

            self.agents["Cross"].learning()

            self.state = next_state

if __name__ == '__main__':
    LeraningSimulation().start()
