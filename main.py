from agent import Agent, RandomAgent
from env import TicTacToe
import pandas as pd
from setting import *
from tqdm import trange


class LeraningSimulation(object):
    def __init__(self) -> None:
        self.env = TicTacToe()
        self.agents = {"Cross": Agent(), "Circle": Agent()}
        self.state = None
        self.sim_result = []

    def start(self):
        result = []
        for sim in trange(SIMULATION_NUM, desc='simulation loop'):
            self.agents = {"Cross": Agent(), "Circle": Agent()}
            self.sim_result = []
            self._one_simulation()
            result.append(self.sim_result)
        self._make_csv(result, "result", "result.csv")
        self.agents["Cross"].save("cross")
        self.agents["Circle"].save("circle")

    def _one_simulation(self):
        for epi in trange(EPISODE_NUM, desc='episode loop', leave=False):
            self.state = self.env.reset()
            self.sim_result.append(self._one_episode(epi))

    def _one_episode(self, epi):
        done = False
        while not done:
            # 引き分けなら1
            result = 1

            # 先行はバツ
            action = self.agents["Cross"].select(self.state, self.env.get_available_select_action())
            next_state, reward, done, _ = self.env.step(action)
            self.agents["Cross"].save_memory(self.state, action, reward, next_state, done)

            if done:
                if reward == 2.0:
                    # バツが勝つと0
                    result = 0
                    self.agents["Circle"].change_last_reward(-1.0)
                self.agents["Cross"].learning()
                self.agents["Circle"].learning()
                return result

            if epi == 0:
                # 初回はなにも行動していないので学習もしない
                self.agents["Circle"].learning()

            self.state = next_state

            # 後攻はマル
            action = self.agents["Circle"].select(self.state, self.env.get_available_select_action())
            next_state, reward, done, _ = self.env.step(action)
            self.agents["Circle"].save_memory(self.state, action, reward, next_state, done)

            if done:
                if reward == 2.0:
                    # マルが勝つと2
                    result = 2
                    self.agents["Cross"].change_last_reward(-1.0)
                self.agents["Cross"].learning()
                self.agents["Circle"].learning()
                return result

            self.agents["Cross"].learning()

            self.state = next_state

    def _make_csv(self, lst, kind, file_name):
        csv_lst = []
        cols = ['episode']
        simulation_num = len(lst)
        episode_num = len(lst[0])

        for sim in range(simulation_num):
            cols.append("{}sim_{}".format(sim+1, kind))

        for epi in range(episode_num):
            one_line = [epi+1]
            for sim in range(simulation_num):
                one_line.append(lst[sim][epi])
            csv_lst.append(one_line)

        df = pd.DataFrame(csv_lst, columns=cols)
        df.to_csv('output/' + file_name, index=False)


if __name__ == '__main__':
    LeraningSimulation().start()
