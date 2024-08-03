from copy import deepcopy, copy
import numpy as np
from transitions import Event_Handler
import torch
from agent import Agent
import torch.nn as nn
from torch import Tensor
from network import make_seq

hidde_size_out = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNet(nn.Module):
    def __init__(self, state_size, out_size, hidde_size, discr_state_info, output_activation="None"):
        super(ActorNet, self).__init__()
        self.__in_size = state_size
        self.__out_size = out_size
        self.__is_discr_state, self.__discr_size = discr_state_info
        self.__output_activation = output_activation

        if self.__is_discr_state:
            assert self.__discr_size is not None, 'discr_size must be not None if is_embed_discr_sate is True'
            self.layers = make_seq(state_size-1, hidde_size_out, hidde_size)
            self.embed_layer = torch.nn.Embedding(self.__discr_size, hidde_size_out)
            self.output_layer = torch.nn.Linear(hidde_size_out, out_size)
        else:
            self.layers = make_seq(state_size-1, out_size, hidde_size, self.__output_activation)

    def forward(self, x) -> Tensor:
        if self.__is_discr_state:
            conti_x = x[:, 0:2]
            discr_x = x[:, 2].long()

            hidde_conti_x = self.layers(conti_x)
            hidde_discr_x = self.embed_layer(discr_x)
            hidde_x = torch.add(hidde_conti_x, hidde_discr_x)
            output = torch.sigmoid(self.output_layer(hidde_x))
            return output
        else:
            conti_x = x[:, 0:2]
            output = self.layers(conti_x)
            return output


class CriticNet(nn.Module):
    def __init__(self, state_size, out_size, discr_state_info, hidde_size):
        super(CriticNet, self).__init__()
        self.__in_size = state_size
        self.__out_size = out_size
        self.__is_discr_state, self.__discr_size = discr_state_info

        if self.__is_discr_state:
            assert self.__discr_size is not None , 'discr_size must be not None if is_discr_state is True'
            self.layers = make_seq(state_size-1+1, hidde_size_out, hidde_size)
            self.embed_layer = torch.nn.Embedding(self.__discr_size, hidde_size_out)
            self.output_layer = torch.nn.Linear(hidde_size_out, out_size)
        else:
            self.layers = make_seq(state_size-1+1, out_size, hidde_size)

    def forward(self, x) -> Tensor:
        if self.__is_discr_state:        
            conti_x = x[:, 0:2]
            discr_x = x[:, 2].long()
            a = x[:, 3].unsqueeze(1)
            conti_x_a = torch.cat((conti_x, a), dim=1)
            hidde_conti_x_a = self.layers(conti_x_a)
            hidde_discr_x = self.embed_layer(discr_x)
            hidde_x = torch.add(hidde_conti_x_a, hidde_discr_x)
            output = self.output_layer(hidde_x)
            return output
        else:
            conti_x = x[:, 0:2]
            a = x[:, 3].unsqueeze(1)
            conti_x_a = torch.cat((conti_x, a), dim=1)
            output = self.layers(conti_x_a)
            return output


class DDPG(Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(DDPG, self).__init__(config, agent_config, is_eval)

        self.__init_noise_level = agent_config['init_noise_level']
        self.__decay_rate = agent_config['decay_rate']
        self.__layer_init_type = agent_config['layer_init_type']
        self.__is_discr_state = agent_config["is_discr_state"]
        self.__is_discr_act = agent_config['is_discr_act']
        if self.__is_discr_state:
            discr_state_info = (
                self.__is_discr_state, self._config.stop_num)
        else:
            discr_state_info = (self.__is_discr_state, None)

        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=False,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold,
                                           w=self._w, is_off_policy=True)

        self.__actor_net = ActorNet(self._state_size, out_size=1,  hidde_size=self._hidde_size,
                                    discr_state_info=discr_state_info, output_activation="Sigmoid").to(device)

        if not self._is_eval:
            self.__criti_net = CriticNet(self._state_size, 1, discr_state_info, self._hidde_size).to(device)

            # a list of episode num that need to save model
            self.__check_poins = [150]
            self.__targe_actor_net = deepcopy(self.__actor_net).to(device)
            self.__targe_criti_net = deepcopy(self.__criti_net).to(device)

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for param in self.__targe_actor_net.parameters():
                param.requires_grad = False
            for param in self.__targe_criti_net.parameters():
                param.requires_grad = False

            self.__actor_optim = torch.optim.Adam(
                self.__actor_net.parameters(), lr=self._lr)
            self.__critic_optim = torch.optim.Adam(
                self.__criti_net.parameters(), lr=self._lr)
            # maintain update counting
            self.__add_event_count = 0
            # update every self.__update_cycle counts
            self.__updat_cycle = 1
            # maintain noise maker
            self.__noise_level = self.__init_noise_level
        else:
            model = self.load_model(self._agent_config)
            self.__actor_net.load_state_dict(model)

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        # write to wandb
        headw_varia = super().reset(episode, is_record_wandb)

        if not self._is_eval:
            self.__event_handl.clear_events()
            self.__noise_level = self.__decay_rate ** episode * self.__init_noise_level
            # print(self.__noise_level)
            if (episode+1) in self.__check_poins:
                self.save_model(self.__actor_net.state_dict(),
                                self._agent_config)

        return headw_varia

    def __str__(self) -> str:
        return 'DDPG'

    def cal_hold_time(self, snapshot):
        state = copy(snapshot.local_state)
        state.append(snapshot.curr_stop_id)
        action = self.act(state)
        hold_time = action * self._max_hold
        # record departure event and log reward
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, action, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)
        if not self._is_eval:
            self.__add_event_count += 1
            # if accumulated event num is enough, push it to buffer
            if self.__event_handl.get_trans_num_by_bus() >= self._batch_size:
                self.__event_handl.push_transition_to_buffer()
            # update policy network parameters
            self.learn()

        return hold_time

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).reshape(1, self._state_size).to(device)
        with torch.no_grad():
            a = self.__actor_net(state)
            # when training, add noise
            if not self._is_eval:
                noise = torch.from_numpy(np.array(np.random.normal(0, self.__noise_level))).to(device)
                a = (a + noise).clip(0, 1)
        return a.item()

    def learn(self):
        if self.__add_event_count % self.__updat_cycle != 0 or self.__event_handl.get_buffer_size() < self._batch_size:
            return

        states, actions, rewards, next_states= self.__event_handl.sample_transition(
            self._batch_size, self.__is_discr_act)
        s = torch.tensor(
            states, dtype=torch.float32).reshape(-1, self._state_size).to(device)
        # LongTensor for idx selection
        a = torch.tensor(actions, dtype=torch.float32).to(device).view(-1, 1)
        r = torch.tensor(rewards, dtype=torch.float32).to(device).view(-1, 1)
        n_s = torch.tensor(
            next_states, dtype=torch.float32).reshape(-1, self._state_size).to(device)

        # update critic network
        self.__critic_optim.zero_grad()
        # current estimate
        s_a = torch.concat((s, a), dim=1)
        for param in self.__criti_net.parameters():
            param.requires_grad = True
        Q = self.__criti_net(s_a)

        # Bellman backup for Q function
        n_targe_a = self.__targe_actor_net(n_s)  # (batch_size, 1)
        n_targe_s_a = torch.concat((n_s, n_targe_a), dim=1)
        with torch.no_grad():
            n_targe_q = self.__targe_criti_net(n_targe_s_a)
            # r is (batch_size, ), need to align with output from NN
            y_hat = r + self._gamma * n_targe_q
        # MSE loss against Bellman backup
        # Unfreeze Q-network so as to optimize it
        q_loss = torch.nn.MSELoss()(y_hat, Q)
        # update critic parameters
        q_loss.backward()
        self.__critic_optim.step()

        # update actor network
        self.__actor_optim.zero_grad()
        next_a = self.__actor_net(s)
        next_s_a = torch.concat((s, next_a), dim=1)
        # Freeze Q-network to save computational efforts
        for param in self.__criti_net.parameters():
            param.requires_grad = False
        actor_loss = torch.mean(-self.__criti_net(next_s_a))
        actor_loss.backward()
        self.__actor_optim.step()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.__actor_net.parameters(), self.__targe_actor_net.parameters()):
                p_targ.data.copy_(self._polya * p_targ.data + (1 - self._polya) * p.data)
            for p, p_targ in zip(self.__criti_net.parameters(), self.__targe_criti_net.parameters()):
                p_targ.data.copy_(self._polya * p_targ.data + (1 - self._polya) * p.data)
