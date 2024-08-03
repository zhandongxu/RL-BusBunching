from copy import copy
import torch
from agent import Agent
from transitions import Event_Handler
from torch.distributions import Categorical
import torch.nn as nn
from torch import Tensor
from network import make_seq, Event_Critic_Net
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

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
            output = torch.softmax(self.output_layer(hidde_x), dim=1)
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


class MAPPO(Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(MAPPO, self).__init__(config, agent_config, is_eval)
        self.__K_epoch = agent_config["K_epoch"]
        self.__clip = agent_config["clip"]
        self.__lmbda = agent_config["lmbda"]
        self.__is_discr_state = agent_config["is_discr_state"]
        self.max_grad_norm = 0.5
        self.__gat_state_size = 4  # state:2, action:1, augme_info:1
        self.__gat_hidde_size = agent_config['gat_hidde_size']
        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=False,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold,
                                           w=self._w, is_off_policy=False, is_graph=True)
        if self.__is_discr_state:
            discr_state_info = (
                self.__is_discr_state, self._config.stop_num)
        else:
            discr_state_info = (self.__is_discr_state, None)

        self.__actor_net = ActorNet(self._state_size, out_size=self._action_size, hidde_size=self._hidde_size,
                                    discr_state_info=discr_state_info, output_activation="Softmax").to(device)

        if not self._is_eval:
            self.__criti_net = CriticNet(self._state_size, 1, discr_state_info, self._hidde_size).to(device)
            self.__globa_criti_net = Event_Critic_Net(self.__gat_state_size, self.__gat_hidde_size, out_size=1).to(device)

            # a list of episode num that need to save model
            self.__check_poins = [150]

            self.__actor_optim = torch.optim.Adam(
                self.__actor_net.parameters(), lr=self._lr)
            self.__critic_optim = torch.optim.Adam(
                self.__criti_net.parameters(), lr=self._lr)
            self.__globa_criti_optim = torch.optim.Adam(
                self.__globa_criti_net.parameters(), lr=self._lr)

        else:
            model = self.load_model(self._agent_config)
            self.__actor_net.load_state_dict(model)

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        # write to wandb
        headw_varia = super().reset(episode, is_record_wandb)

        if not self._is_eval:
            self.learn()
            self.__event_handl.clear_memor(is_off_policy=False)
            self.__event_handl.clear_events()
            if (episode + 1) in self.__check_poins:
                self.save_model(self.__actor_net.state_dict(),
                                self._agent_config)

        return headw_varia

    def __str__(self) -> str:
        return 'MAPPO'

    def cal_hold_time(self, snapshot):
        state = copy(snapshot.local_state)
        state.append(snapshot.curr_bus_id)
        action = self.act(state)
        hold_time = (action / (self._action_size - 1)) * self._max_hold
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, action, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)

        return hold_time

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).reshape(1, self._state_size).to(device)
        probs = self.__actor_net(state)
        dist = Categorical(probs)
        a = dist.sample()

        return a.item()

    def make_batch(self, transitions):
        s_list, a_list, r_list, next_s_list, graphs_list, next_graphs_list = [], [], [], [], [], []
        for transition in transitions:
            s, a, r, next_s, graph, next_graph = transition
            s_list.append(s)
            a_list.append(a)
            r = r + self._w * math.exp(-a / 3)
            r_list.append(r)
            next_s_list.append(next_s)
            graphs_list.append(graph)
            next_graphs_list.append(next_graph)

        return s_list, a_list, r_list, next_s_list, graphs_list, next_graphs_list

    def learn(self):
        transitions = self.__event_handl.push_transition_graph_to_buffer()

        s_list, a_list, r_list, next_s_list, graphs_list, next_graphs_list = self.make_batch(transitions)
        up_data, down_data = self.construct_graph(graphs_list)
        batch_up_data = Batch.from_data_list(up_data).to(device)
        batch_down_data = Batch.from_data_list(down_data).to(device)

        next_up_data, next_down_data = self.construct_graph(next_graphs_list)
        next_batch_up_data = Batch.from_data_list(next_up_data).to(device)
        next_batch_down_data = Batch.from_data_list(next_down_data).to(device)

        s = torch.tensor(
            s_list, dtype=torch.float32).reshape(-1, self._state_size).to(device)
        a = torch.tensor(a_list, dtype=torch.long).to(device).view(-1, 1)
        r = torch.tensor(r_list, dtype=torch.float32).to(device).view(-1, 1)
        next_s = torch.tensor(
            next_s_list, dtype=torch.float32).reshape(-1, self._state_size).to(device)

        next_ego_q = self.__criti_net(next_s)
        next_global_q = self.__globa_criti_net(next_batch_up_data, next_batch_down_data)
        next_q_target = next_ego_q + next_global_q
        td_target = r + self._gamma * next_q_target
        ego_q = self.__criti_net(s)
        global_q = self.__globa_criti_net(batch_up_data, batch_down_data)
        q_value = ego_q + global_q
        td_error = td_target - q_value

        # GAE
        td_error = td_error.cpu().detach().numpy()
        advantage = 0  
        advantages_list = []
        for error in td_error[::-1]: 
            advantage = self._gamma * self.__lmbda * advantage + error
            advantages_list.append(advantage)

        advantages_list.reverse()
        advantages = torch.tensor(advantages_list, dtype=torch.float32).to(device)

        old_log_probs = torch.log(self.__actor_net(s).gather(1, a)).detach()

        for _ in range(self.__K_epoch):
            new_log_probs = torch.log(self.__actor_net(s).gather(1, a))
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.__clip, 1 + self.__clip) * advantages

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            ego_q = self.__criti_net(s)
            global_q = self.__globa_criti_net(batch_up_data, batch_down_data)
            q_value = ego_q + global_q
            critic_loss = torch.mean(F.mse_loss(q_value, td_target.detach()))

            self.__actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.__actor_net.parameters(), self.max_grad_norm)
            self.__actor_optim.step()

            self.__critic_optim.zero_grad()
            self.__globa_criti_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.__criti_net.parameters(), self.max_grad_norm)
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
