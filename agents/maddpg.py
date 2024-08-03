from copy import deepcopy, copy
import numpy as np
from transitions import Event_Handler
import torch
from network import make_seq, Event_Critic_Net
from torch_geometric.data import Data, Batch
from agent import Agent
import torch.nn as nn
from torch import Tensor

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
            output = self.output_layer(hidde_x)
            return output
        else:
            conti_x = x[:, 0:2]
            output = self.layers(conti_x)
            return output


class CriticNet(nn.Module):
    def __init__(self, state_size, out_size, discr_state_info, hidde_size=(64, )):
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


class MADDPG(Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(MADDPG, self).__init__(config, agent_config, is_eval)

        self.__init_noise_level = agent_config['init_noise_level']
        self.__decay_rate = agent_config['decay_rate']
        self.__layer_init_type = agent_config['layer_init_type']
        self.__gat_state_size = 4        #state:2, action:1, augme_info:1
        self.__gat_hidde_size = agent_config['gat_hidde_size']
        self.__is_discr_state = agent_config['is_discr_state']
        self.__is_discr_act = agent_config['is_discr_act']
        if self.__is_discr_state:
            discr_state_info = (
                self.__is_discr_state, self._config.stop_num)
        else:
            discr_state_info = (self.__is_discr_state, None)

        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=False,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold,
                                           w=self._w, is_off_policy=True, is_graph=True)

        self.__actor_net = ActorNet(self._state_size, out_size=1, discr_state_info=discr_state_info, hidde_size=self._hidde_size,
                                    output_activation="Sigmoid").to(device)

        if not self._is_eval:
            self.__ego_criti_net = CriticNet(state_size=self._state_size, out_size=1, discr_state_info=discr_state_info,
                                         hidde_size=self._hidde_size).to(device)
            self.__globa_criti_net = Event_Critic_Net(self.__gat_state_size, self.__gat_hidde_size, out_size=1).to(device)

            # a list of episode num that need to save model
            self.__check_poins = [150]
            self.__targe_actor_net = deepcopy(self.__actor_net).to(device)
            self.__targe_ego_criti_net = deepcopy(self.__ego_criti_net).to(device)
            self.__targe_globa_criti_net = deepcopy(self.__globa_criti_net).to(device)

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for param in self.__targe_actor_net.parameters():
                param.requires_grad = False
            for param in self.__targe_ego_criti_net.parameters():
                param.requires_grad = False
            for param in self.__targe_globa_criti_net.parameters():
                param.requires_grad = False

            self.__actor_optim = torch.optim.Adam(
                self.__actor_net.parameters(), lr=self._lr)
            self.__ego_critic_optim = torch.optim.Adam(
                self.__ego_criti_net.parameters(), lr=self._lr)
            self.__globa_criti_optim = torch.optim.Adam(
                self.__globa_criti_net.parameters(), lr=self._lr)

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

            if (episode+1) in self.__check_poins:
                self.save_model(self.__actor_net.state_dict(),
                                self._agent_config)

        return headw_varia

    def __str__(self) -> str:
        return 'MADDPG'

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
                self.__event_handl.push_transition_graph_to_buffer()
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

        states, actions, rewards, next_states, graphs, next_graphs = self.__event_handl.sample_transition_graph(
            self._batch_size, self.__is_discr_act)
        up_data, down_data = self.construct_graph(graphs)
        batch_up_data = Batch.from_data_list(up_data).to(device)
        batch_down_data = Batch.from_data_list(down_data).to(device)

        next_up_data, next_down_data = self.construct_graph(next_graphs)
        next_batch_up_data = Batch.from_data_list(next_up_data).to(device)
        next_batch_down_data = Batch.from_data_list(next_down_data).to(device)

        s = torch.tensor(
            states, dtype=torch.float32).reshape(-1, self._state_size).to(device)
        # LongTensor for idx selection
        a = torch.tensor(actions, dtype=torch.float32).to(device).view(-1, 1)
        r = torch.tensor(rewards, dtype=torch.float32).to(device).view(-1, 1)
        n_s = torch.tensor(
            next_states, dtype=torch.float32).reshape(-1, self._state_size).to(device)

        # update critic network
        self.__ego_critic_optim.zero_grad()
        self.__globa_criti_optim.zero_grad()
        # current estimate
        s_a = torch.concat((s, a), dim=1)
        for param in self.__ego_criti_net.parameters():
            param.requires_grad = True
        for param in self.__globa_criti_net.parameters():
            param.requires_grad = True
        ego_Q = self.__ego_criti_net(s_a)
        globa_Q = self.__globa_criti_net(batch_up_data, batch_down_data)
        Q = ego_Q + globa_Q

        # Bellman backup for Q function
        n_targe_a = self.__targe_actor_net(n_s)
        n_targe_s_a = torch.concat((n_s, n_targe_a), dim=1)
        with torch.no_grad():
            ego_q_polic_targe = self.__targe_ego_criti_net(n_targe_s_a)
            globa_q_polic_targe = self.__targe_globa_criti_net(next_batch_up_data, next_batch_down_data)
            n_targe_q = ego_q_polic_targe + globa_q_polic_targe
            # r is (batch_size, ), need to align with output from NN
            y_hat = r + self._gamma * n_targe_q
        # MSE loss against Bellman backup
        # Unfreeze Q-network so as to optimize it
        q_loss = torch.nn.MSELoss()(y_hat, Q)
        # update critic parameters
        q_loss.backward()
        self.__ego_critic_optim.step()
        self.__globa_criti_optim.step()

        # update actor network
        self.__actor_optim.zero_grad()
        next_a = self.__actor_net(s)
        next_s_a = torch.concat((s, next_a), dim=1)
        # Freeze Q-network to save computational efforts
        for param in self.__ego_criti_net.parameters():
            param.requires_grad = False
        for param in self.__globa_criti_net.parameters():
            param.requires_grad = False
        actor_loss = torch.mean(-self.__ego_criti_net(next_s_a))
        actor_loss.backward()
        self.__actor_optim.step()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.__actor_net.parameters(), self.__targe_actor_net.parameters()):
                p_targ.data.copy_(self._polya * p_targ.data + (1 - self._polya) * p.data)
            for p, p_targ in zip(self.__ego_criti_net.parameters(), self.__targe_ego_criti_net.parameters()):
                p_targ.data.copy_(self._polya * p_targ.data + (1 - self._polya) * p.data)
            for p, p_targ in zip(self.__globa_criti_net.parameters(), self.__targe_globa_criti_net.parameters()):
                p_targ.data.copy_(self._polya * p_targ.data + (1 - self._polya) * p.data)

    def construct_graph(self, graps):
        # there are 'batch_size' graphs, construct them
        up_graph_list = []
        down_graph_list = []
        # for each graph, construct edge_index and x
        # each graph is a list containing events (nodes in namaedtuple)
        for graph in graps:
            up_edges, down_edges = [], []
            up_edge_count, down_edge_count = 0, 0
            up_hs, down_hs = [], []
            for event in graph:
                # TODO check self augme is 0
                if event.up_or_down == 'up':
                    up_edge_count += 1
                    # up_edges.append([0, up_edge_count])
                    up_edges.append([up_edge_count, 0])
                    up_h = []
                    up_h.extend(event.state)
                    up_h.append(event.action)
                    up_h.append(event.augme_info)   #augme_info  distance
                    up_hs.append(up_h)
                elif event.up_or_down == 'down':
                    down_edge_count += 1
                    # down_edges.append([0, down_edge_count])
                    down_edges.append([down_edge_count, 0])
                    down_h = []
                    down_h.extend(event.state)
                    down_h.append(event.action)
                    down_h.append(event.augme_info)
                    down_hs.append(down_h)
                elif event.up_or_down == 'self':
                    up_edges.append([0, 0])
                    up_h = []
                    up_h.extend(event.state)
                    up_h.append(event.action)
                    up_h.append(event.augme_info)
                    up_hs.append(up_h)

                    down_edges.append([0, 0])
                    down_h = []
                    down_h.extend(event.state)
                    down_h.append(event.action)
                    down_h.append(event.augme_info)
                    down_hs.append(down_h)

            up_edge_index = torch.tensor(up_edges)
            up_x = torch.tensor(up_hs)

            down_edge_index = torch.tensor(down_edges)
            down_x = torch.tensor(down_hs)

            # must be transposed so that the first row is the source node and the second row is the target node
            up_data = Data(x=up_x, edge_index=up_edge_index.t().contiguous())
            down_data = Data(x=down_x, edge_index=down_edge_index.t().contiguous())

            up_graph_list.append(up_data)
            down_graph_list.append(down_data)

        return up_graph_list, down_graph_list
