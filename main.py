from agent import Agent
from env import TicTacToe
from setting import *
from tqdm import trange


class LeraningSimulation(object):
    def __init__(self) -> None:
        self.env = TicTacToe()
        self.agents = {"Cross": Agent(), "Circle": Agent()}
        self.state = None
        self.result = []
        self.sim_result = []

    def start(self):
        for sim in trange(SIMULATION_NUM, desc='simulation loop'):
            self.agents = {"Cross": Agent(), "Circle": Agent()}
            self.state = self.env.reset()
            self.sim_result = []
            self.one_simulation()
            self.result.append(self.sim_result)

    def one_simulation(self):
        for epi in trange(EPISODE_NUM, desc='episode loop', leave=False):
            # 先行はバツ
            action = self.agents["Cross"].select(self.state, self.env.get_available_select_action())
            next_state, reward, done, _ = self.env.step(action)
            self.agents["Cross"].save_memory(self.state, action, reward, next_state, done)

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

            # 後攻はマル
            action = self.agents["Circle"].select(self.state, self.env.get_available_select_action())
            next_state, reward, done, _ = self.env.step(action)
            self.agents["Circle"].save_memory(self.state, action, reward, next_state, done)

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
