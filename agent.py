import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from network import DqnNetwork, NetworkUtil, Variable
from setting import *

SAMPLE = (list, list, list, list)

class Agent(object):
    def __init__(self) -> None:
        self.value_net = NetworkUtil.initialize(DqnNetwork())
        self.target_net = NetworkUtil.copy_param(self.value_net, DqnNetwork())
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=NW_LEARNING_RATE, eps=NW_EPS)
        self.memory = ReplayBuffer(NUM_REPLAY_BUFFER)

        self.behavior_policy = Egreedy(eps_time_steps=EPS_TIMESTEPS, eps_start=EPS_START, eps_end=EPS_END)
        self.target_policy = Greedy
        self.learning_count = 0

    def learning(self) -> None:
        if not self.memory.can_sample(BATCH_SIZE):
            return

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.memory.sample(BATCH_SIZE)

        obs_batch = Variable(torch.from_numpy(obs_batch))
        act_batch = Variable(torch.from_numpy(act_batch))
        rew_batch = Variable(torch.from_numpy(rew_batch))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch))
        not_done_mask = Variable(torch.from_numpy(1 - done_mask))

        if torch.cuda.is_available():
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()
            not_done_mask = not_done_mask.cuda()

        # Q values
        self.value_net.eval()
        current_Q_values = self.value_net(NetworkUtil.to_binary(obs_batch)).gather(1, act_batch.unsqueeze(1)).squeeze(1)
        # target Q values
        self.target_net.eval()
        next_max_Q = self.target_policy.select(self.target_net(NetworkUtil.to_binary(next_obs_batch)))
        next_Q_values = not_done_mask * next_max_Q
        target_Q_values = rew_batch + (GAMMA * next_Q_values)
        # loss
        self.value_net.train()
        loss = F.smooth_l1_loss(current_Q_values.float(), target_Q_values.float())

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learning_count % 100 == 0:
            self.update_target_network()
        self.learning_count += 1

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.value_net.state_dict())

    def save_memory(self, state, action, reward, next_state, done) -> None:
        self.memory.append(state, action, reward, next_state, done)

    def change_last_reward(self, reward) -> None:
        self.memory.reward[-1] = reward

    def select(self, state, available_select_action) -> int:
        with torch.no_grad():
            state = Variable(NetworkUtil.to_binary(torch.from_numpy(state)))
            output = self.value_net(state)
        available = output[0][available_select_action]
        selected = self.behavior_policy.select(torch.stack([available], dim=0))
        return available_select_action[selected]

    def save(self, kind) -> None:
        torch.save(self.value_net.to('cpu').state_dict(), "output/{}.pth".format(kind))


class RandomAgent(object):
    def learning(self):
        pass

    def select(self, state, available_select_action) -> int:
        return random.choice(available_select_action)

    def change_last_reward(self, reward) -> None:
        pass

    def save(self, kind) -> None:
        pass


class TestAgent(object):
    def __init__(self, path) -> None:
        self.value_net = NetworkUtil.initialize(DqnNetwork())
        self.value_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.policy = Greedy

    def select(self, state, available_select_action) -> int:
        with torch.no_grad():
            state = Variable(torch.from_numpy(state))
            output = self.value_net(NetworkUtil.to_binary(state))
        print(output)
        available = output[0][available_select_action]
        selected = self.policy.select(torch.stack([available], dim=0))
        return available_select_action[selected]


class ReplayBuffer(object):
    def __init__(self, size) -> None:
        self.size = size

        self.obs = deque([])
        self.action = deque([])
        self.reward = deque([])
        self.next_obs = deque([])
        self.done = deque([])

    def append(self, obs, action, reward, next_obs, done) -> None:
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)

        if (len(self.obs) > self.size):
            self.obs.popleft()
            self.action.popleft()
            self.reward.popleft()
            self.next_obs.popleft()
            self.done.popleft()

    def can_sample(self, batch_size) -> bool:
        return len(self.obs) >= batch_size

    def sample(self, batch_size) -> SAMPLE:
        assert self.can_sample(batch_size)
        indexes = random.sample([i for i in range(len(self.obs))], batch_size)

        np_obs = np.array(self.obs)
        np_action = np.array(self.action)
        np_reward = np.array(self.reward)
        np_next_obs = np.array(self.next_obs)
        np_done = np.array(self.done)

        return np_obs[indexes], np_action[indexes], np_reward[indexes], np_next_obs[indexes], np_done[indexes]


class Greedy(object):
    def select(lst: torch.Tensor) -> int:
        if len(lst) == 1:
            # simulation
            return lst.max(1)[1].view(1, 1).item()
        # leraning
        return lst.detach().max(1)[0]

class Egreedy(object):
    def __init__(self, eps_time_steps, eps_start, eps_end) -> None:
        self.select_count = 0
        
        self.eps_time_steps = eps_time_steps
        self.eps_start = eps_start
        self.eps_end = eps_end

    def select(self, lst: torch.Tensor) -> int:
        sample = random.random()
        value = self._shcedule()

        if sample > value:
            selected = Greedy.select(lst)
        else:
            selected = random.randrange(len(lst[0]))

        self.select_count += 1
        return selected

    def _shcedule(self) -> float:
        fraction = min(float(self.select_count) / self.eps_time_steps, 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)
