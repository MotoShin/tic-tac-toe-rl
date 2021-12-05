from enum import Enum
import sys
from agent import Agent, RandomAgent
from env import TicTacToe, Agents
import pandas as pd
from setting import *
from tqdm import trange


class LeraningSimulation(object):
    def __init__(self) -> None:
        self.env = TicTacToe()
        self.agents = {Agents.CROSS: Agent(), Agents.CIRCLE: Agent()}
        self.state = None
        self.sim_result = []

    def start(self):
        result = []
        for sim in trange(SIMULATION_NUM, desc='simulation loop'):
            self.env = TicTacToe()
            self.agents = {Agents.CROSS: Agent(), Agents.CIRCLE: Agent()}
            self.sim_result = []
            self._one_simulation()
            result.append(self.sim_result)
        self._make_csv(result, "result", "result.csv")
        self.agents[Agents.CROSS].save(Agents.CROSS.value)
        self.agents[Agents.CIRCLE].save(Agents.CIRCLE.value)

    def _one_simulation(self):
        for _ in trange(EPISODE_NUM, desc='episode loop', leave=False):
            self.state = self.env.reset()
            self.sim_result.append(self._one_episode())

    def _one_episode(self):
        episode_done = False

        while not episode_done:
            # 先行はバツ
            next_player = Agents.CROSS
            while True:
                next_player, turn_player, done, draw_flg = self._one_turn(next_player)
                self.agents[turn_player].learning()
                
                if done:
                    if not draw_flg:
                        # 勝負がついた場合、負けたplayerの最終報酬を-1.0に更新
                        self.agents[next_player].change_last_reward(-1.0)
                    episode_done = True
                    break

        if draw_flg:
            # 引き分けなら1
            return 1
        else:
            if turn_player is Agents.CROSS:
                # バツ勝利は0
                return 0
            elif turn_player is Agents.CIRCLE:
                # マル勝利は2
                return 2

    def _one_turn(self, turn_agent):
        action = self.agents[turn_agent].select(self.state)

        next_state, reward, done, turn_player, next_turn_player = self.env.step(action)

        if turn_agent is not turn_player:
            # simulationのturn_agentとenvのturn playerが一致していない場合はエラー
            print("Error: different turn player.")
            sys.exit(1)

        self.agents[turn_agent].save_memory(self.state, action, reward, next_state, done)

        self.state = next_state

        draw_flg = True
        if reward == 1.0:
            draw_flg = False

        return next_turn_player, turn_player, done, draw_flg

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
