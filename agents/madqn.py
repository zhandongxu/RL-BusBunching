from copy import copy
import torch
import numpy as np
from agent import Agent
from transitions import Event_Handler
import torch.nn as nn
from torch import Tensor
from network import make_seq, Event_Critic_Net
from torch_geometric.data import Data, Batch


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


class MADQN(Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(MADQN, self).__init__(config, agent_config, is_eval)

        self.__epsilon = agent_config["epsilon"]
        self.__t = agent_config["t"]
        self.__gat_state_size = 4
        self.__gat_hidde_size = agent_config['gat_hidde_size']
        self.__is_discr_state = agent_config["is_discr_state"]
        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=True,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold,
                                           w=self._w, is_off_policy=True, is_graph=True)

        if self.__is_discr_state:
            discr_state_info = (
                self.__is_discr_state, self._config.stop_num)
        else:
            discr_state_info = (self.__is_discr_state, None)

        self.__critic_net = CriticNet(self._state_size, self._action_size, discr_state_info, self._hidde_size).to(device)
        self.__globa_criti_net = Event_Critic_Net(self.__gat_state_size, self.__gat_hidde_size, out_size=self._action_size).to(device)
        if not self._is_eval:
            self.__check_poins = [150]
            self.__critic_optim = torch.optim.Adam(
                self.__critic_net.parameters(), lr=self._lr)
            self.__globa_criti_optim = torch.optim.Adam(
                self.__globa_criti_net.parameters(), lr=self._lr)
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
        return 'MADQN'

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
                self.__event_handl.push_transition_graph_to_buffer()
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

        states, actions, rewards, next_states, graphs, next_graphs= self.__event_handl.sample_transition_graph(self._batch_size)
        up_data, down_data = self.construct_graph(graphs)
        batch_up_data = Batch.from_data_list(up_data).to(device)
        batch_down_data = Batch.from_data_list(down_data).to(device)

        next_up_data, next_down_data = self.construct_graph(next_graphs)
        next_batch_up_data = Batch.from_data_list(next_up_data).to(device)
        next_batch_down_data = Batch.from_data_list(next_down_data).to(device)

        s = torch.tensor(states, dtype=torch.float32).reshape(-1, self._state_size).to(device)
        a = torch.tensor(actions, dtype=torch.long).to(device).view(-1, 1)
        r = torch.tensor(rewards, dtype=torch.float32).to(device).view(-1, 1)
        n_s = torch.tensor(next_states, dtype=torch.float32).reshape(-1, self._state_size).to(device)
        # select the maximum q-value from the network for the next state
        ego_Q = self.__critic_net(s)
        global_Q = self.__globa_criti_net(batch_up_data, batch_down_data)
        # global_Q = torch.zeros_like(ego_Q)
        Q = torch.add(ego_Q, global_Q)

        with torch.no_grad():
            next_ego_Q = self.__critic_net(n_s)
            next_global_Q = self.__globa_criti_net(next_batch_up_data, next_batch_down_data)
            # next_global_Q = torch.zeros_like(next_ego_Q)
            nexe_Q = torch.add(next_ego_Q, next_global_Q)
            max_next_q = torch.max(nexe_Q, 1).values.unsqueeze(1)
            y_hat = r + self._gamma * max_next_q
        # select the action value for the value network in the current state
        target = Q.gather(1, a)

        self.__critic_optim.zero_grad()
        self.__globa_criti_optim.zero_grad()
        q_loss = torch.nn.MSELoss()(y_hat, target)
        q_loss.backward()
        self.__critic_optim.step()
        self.__globa_criti_optim.step()

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
