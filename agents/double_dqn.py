from copy import copy, deepcopy
import torch
import numpy as np
from numpy.random import choice
from agent import Agent
from transitions import Event_Handler
import torch.nn as nn
from torch import Tensor
from network import make_seq

hidde_size_out = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CriticNet(nn.Module):
    def __init__(self, state_size, out_size, discr_state_info, hidde_size):
        super(CriticNet, self).__init__()
        self.__in_size = state_size
        self.__out_size = out_size
        self.__is_discr_state, self.__discr_size = discr_state_info

        if self.__is_discr_state:
            assert self.__discr_size is not None , 'discr_size must be not None if is_discr_state is True'
            self.layers = make_seq(state_size-1, hidde_size_out, hidde_size)
            self.embed_layer = torch.nn.Embedding(self.__discr_size, hidde_size_out)
            self.output_layer = torch.nn.Linear(hidde_size_out, out_size)
        else:
            self.layers = make_seq(state_size-1, out_size, hidde_size)

    def forward(self, x) -> Tensor:
        if self.__is_discr_state:
            conti_x = x[:, 0:2]
            discr_x = x[:, 2].long()

            hidde_conti_x = self.layers(conti_x)
            hidde_discr_x = self.embed_layer(discr_x)
            hidde_x = torch.add(hidde_conti_x, hidde_discr_x)
            output = self.output_layer(hidde_x)
            return output
        else:
            conti_x = x[:, 0:2]
            output = self.layers(conti_x)
            return output


class DoubleDQN(Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(DoubleDQN, self).__init__(config, agent_config, is_eval)

        self.__epsilon = agent_config["epsilon"]
        self.__t = agent_config["t"]
        self.__update_counter = agent_config["update_counter"]
        self.__target_update_freq = agent_config["target_update_freq"]

        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=False,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold,
                                           w=self._w, is_off_policy=True)
        self.__is_discr_state = agent_config["is_discr_state"]

        if self.__is_discr_state:
            discr_state_info = (
                self.__is_discr_state, self._config.stop_num)
        else:
            discr_state_info = (self.__is_discr_state, None)

        self.__critic_net = CriticNet(self._state_size, self._action_size, discr_state_info, self._hidde_size).to(device)
        self.__critic_target_net = deepcopy(self.__critic_net).to(device)

        if not self._is_eval:
            for param in self.__critic_target_net.parameters():
                param.requires_grad = False
            self.__critic_optim = torch.optim.Adam(
                self.__critic_net.parameters(), lr=self._lr)
            self.__check_poins = [150]
        else:
            model = self.load_model(self._agent_config)
            self.__critic_net.load_state_dict(model)

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        # write to wandb
        headw_varia = super().reset(episode, is_record_wandb)

        if not self._is_eval:
            self.__event_handl.clear_events()
            if (episode+1) in self.__check_poins:
                self.save_model(self.__critic_net.state_dict(),
                                self._agent_config)

        return headw_varia

    def __str__(self) -> str:
        return 'DoubleDQN'

    def cal_hold_time(self, snapshot):
        state = copy(snapshot.local_state)
        state.append(snapshot.curr_stop_id)
        action = self.act(state)
        hold_time = (action/(self._action_size-1)) * self._max_hold
        # record departure event and log reward
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, action, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)
        if not self._is_eval:
            self.schedule_hyperparameters(self.__t)
            # if accumulated event num is enough, push it to buffer
            if self.__event_handl.get_trans_num_by_bus() >= self._batch_size:
                self.__event_handl.push_transition_to_buffer()
            # update policy network parameters
            self.learn()
            self.__t += 1

        return hold_time

    def act(self, state):
        if not self._is_eval:
            u = np.random.uniform(low=0.0, high=1.0, size=1)
            if u < self.__epsilon:
                a = np.random.choice([i for i in range(self._action_size)])
                return a

        state = torch.tensor(state, dtype=torch.float32).reshape(1, self._state_size).to(device)
        a = torch.argmax(self.__critic_net(state))
        return a.item()

    def schedule_hyperparameters(self, step):
        self.__epsilon = np.exp(-0.001 * step)

    def learn(self):
        if self.__event_handl.get_buffer_size() < self._batch_size:
            return

        states, actions, rewards, next_states= self.__event_handl.sample_transition(
            self._batch_size)
        s = torch.tensor(
            states, dtype=torch.float32).reshape(-1, self._state_size).to(device)
        a = torch.tensor(actions, dtype=torch.long).to(device).view(-1, 1)
        r = torch.tensor(rewards, dtype=torch.float32).to(device).view(-1, 1)
        n_s = torch.tensor(
            next_states, dtype=torch.float32).reshape(-1, self._state_size).to(device)

        # select the maximum q-value from the network for the next state
        with torch.no_grad():
            max_next_q = torch.max(self.__critic_target_net(n_s), 1).values.unsqueeze(1)
            y_hat = r + self._gamma * max_next_q
        # select the action value for the value network in the current state
        target = self.__critic_net(s).gather(1, a)

        # update critic parameters
        L = torch.nn.MSELoss()(y_hat, target)
        self.__critic_optim.zero_grad()
        L.backward()
        self.__critic_optim.step()

        # Finally, update target networks by polyak averaging.
        self.__update_counter += 1
        if self.__update_counter % self.__target_update_freq == 0:
            for p, p_targ in zip(self.__critic_net.parameters(), self.__critic_target_net.parameters()):
                p_targ.data.copy_(self._polya * p_targ.data + (1 - self._polya) * p.data)