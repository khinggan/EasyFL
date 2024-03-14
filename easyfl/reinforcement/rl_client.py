import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from itertools import count
logger = logging.getLogger(__name__)
from easyfl.client import BaseClient
from easyfl.reinforcement.replaybuffer import ReplayMemory, Transition




class RL_Client(BaseClient):
    def __init__(self, cid, conf, train_data=None, test_data=None, device=None, env=None, **kwargs):
        super(RL_Client, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.env = env
        self.state, _ = self.env.reset()
        self.n_actions = self.env.action_space.n
        self.n_observation = len(self.state)


class DQN_Client(RL_Client):
    def __init__(self, cid, conf, train_data=None, test_data=None, device=None, env=None, **kwargs):
        super().__init__(cid, conf, train_data, test_data, device, env, **kwargs)

        self.env = env

        # self.state, _ = self.env.reset()
        # self.n_actions = self.env.action_space.n
        # self.n_observation = len(self.state)

        # self.policy_net = self.model(self.n_observation, self.n_actions).to(device)
        # self.target_net = self.model(self.n_observation, self.n_actions).to(device)
        # self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(10000)
        self.episode_durations = []
        self.rewards = []
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000.0
        self.gamma = 0.99
        self.tau = 0.005,
        
        self.step_done = 0
        self.batch_size = 128

        
    def pre_train(self):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()
        self.state, _ = self.env.reset()
        self.n_actions = self.env.action_space.n
        self.n_observation = len(self.state)

        self.policy_net = self.model
        self.target_net = self.model
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, conf, device="cpu"):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        self.local_ep = conf.local_epoch
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        start_time = time.time()
        self.train_loss = []
        for i in range(conf.local_epoch):    # local training episode number
            self.state, info = self.env.reset()
            self.state = torch.tensor(self.state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0
            for t in count():
                action = self.select_action()
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(self.state, action, next_state, reward)

                episode_reward += reward.item()

                # Move to the next state
                self.state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    # self.plot_durations()
                    self.rewards.append(episode_reward)
                    break
            logger.debug("Client {}, local epoch: {}, reward: {}".format(self.cid, i, episode_reward))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def select_action(self):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.step_done / self.eps_decay)
        self.step_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(self.state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
