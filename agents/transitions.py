from collections import namedtuple, defaultdict
import numpy as np
from copy import copy
from collections import deque
import random
import csv
import math


class Event_Handler:
    def __init__(self, config, is_rewar_globa, is_rewar_track_globa, is_state_globa, max_hold, w, is_off_policy=False,
                 is_graph=False) -> None:
        self.__config = config
        # bus_id -> [event]
        self.__bus_events = defaultdict(list)
        # time -> [event]
        self.__time_events = defaultdict(list)
        # departure event namedtuple: state is local
        self.__depar_event = namedtuple('depar_event', ['episode', 'bus_id', 'stop_id',
                                                        'time', 'state', 'bus_load', 'pax_demand','globa_state',
                                                        'pos_idx', 'globa_relat_state', 'equal_rewar', 'action', 'action_log'])
        # reward for training is global or local
        self._is_rewar_globa = is_rewar_globa
        # tracking reward is global or local
        self.__is_rewar_track_globa = is_rewar_track_globa
        # state is global or local
        self._is_state_globa = is_state_globa
        self._max_hold = max_hold
        self._w = w
        self.__is_graph = is_graph
        if is_off_policy:
            self.__memor = deque(maxlen=10000)
        else:
            self.__memor = []
        if is_graph:
            # augme_info is the distance to the current bus
            self.__node_feat = namedtuple(
                'node_feature', ['bus_id', 'stop_id', 'time', 'up_or_down', 'augme_info', 'state', 'action'])
            self.__is_graph_state_globa = False

    def clear_events(self):
        self.__time_events = defaultdict(list)
        self.__bus_events = defaultdict(list)

    def add_event(self, snapshot, action, hold_time, w, action_log=0):
        equal_rewar = self.calcu_equal_rewar(snapshot, self._is_rewar_globa)
        event = self.__depar_event(snapshot.episode, snapshot.curr_bus_id, snapshot.curr_stop_id, snapshot.ct,
                                   snapshot.local_state, snapshot.bus_load, snapshot.pax_demand,
                                   snapshot.globa_state, snapshot.pos_idx, snapshot.globa_relat_state, equal_rewar,
                                   action, action_log)
        self.__bus_events[snapshot.curr_bus_id].append(event)
        self.__time_events[snapshot.ct].append(event)

        track_equal_rewar, track_inten_rewar, track_rewar = self.track_rewar(
            snapshot, self.__is_rewar_track_globa, hold_time, w)
        return track_equal_rewar, track_inten_rewar, track_rewar

    def write_transition_to_file(self, type_dict):
        file = 'data/'
        file = file + type_dict['behav_polic'] + '_sg_'
        file = file + str(type_dict['is_state_globa']) + '_rg_'
        file = file + str(type_dict['is_rewar_globa'])
        if 'NONLINEAR' in type_dict['behav_polic']:
            file += '_p_' + str(type_dict['pertu_range'])

        if type_dict['is_graph'] == False:
            file += '_trans.csv'
            trans = self.__get_transitions(is_full=True)
            with open(file, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(trans)
        # else:
        #     file += '_eg_trans.pickle'
        #     trans = self.__get_event_graph_trans(is_full=True)
        #     for tran in trans:
        #         with open(file, 'ab') as f:
        #             dill.dump(tran, f)

        # clear event records after writing to csv
        self.clear_events()

    def __get_transitions(self, is_full=False):
        # get each bus's each transition and share
        # if "is_full", also record bus_id, stop_id, time
        trans = []
        for bus_id, events in self.__bus_events.items():
            for prev_event, curr_event in zip(events[0:-1], events[1:]):
                if self._is_state_globa == False:
                    s = copy(prev_event.state)
                    # s.append(prev_event.bus_load)
                    # s.append(prev_event.pax_demand)
                    s.append(prev_event.stop_id)
                    n_s = copy(curr_event.state)
                    # n_s.append(curr_event.bus_load)
                    # n_s.append(curr_event.pax_demand)
                    n_s.append(curr_event.stop_id)
                else:
                    s = copy(prev_event.globa_relat_state)
                    n_s = copy(curr_event.globa_relat_state)

                a = copy(prev_event.action)
                a_log = copy(prev_event.action_log)
                # equalized reward of current departure
                equal_r = copy(curr_event.equal_rewar)
                # balanced reward between prev action and current departure reward
                if is_full:
                    bus_id = copy(prev_event.bus_id)
                    stop_id = copy(prev_event.stop_id)
                    ct = copy(prev_event.time)
                    ep = copy(prev_event.episode)
                    trans.append([ep, bus_id, stop_id, ct, s, a, equal_r, n_s])
                else:
                    trans.append([s, a, a_log, equal_r, n_s])
        return trans

    def push_transition_to_buffer(self):
        trans = self.__get_transitions(is_full=False)
        for tran in trans:
            self.__memor.append(tran)
        # clear event records after pushing it to memory
        self.clear_events()
        return self.__memor

    def clear_memor(self, is_off_policy=False):
        if is_off_policy:
            self.__memor = deque(maxlen=10000)
        else:
            self.__memor = []

    def sample_transition(self, batch_size, is_discr_act=True):
        # for off policy training
        states = []
        actions = []
        rewards = []
        next_states = []
        samples = random.sample(self.__memor, batch_size)
        for sample in samples:
            state, action, _, reward, next_state= sample
            states.append(state)
            actions.append(action)
            if is_discr_act:
                reward = reward + self._w * math.exp(-action / 3)
            else:
                reward = reward + self._w * math.exp(-action)
            rewards.append(reward)
            next_states.append(next_state)
        return states, actions, rewards, next_states

    def get_buffer_size(self):
        return len(self.__memor)

    def get_trans_num_by_bus(self):
        # last transition hasn't received reward, hence omitted
        avail_trans_size = sum((len(steps[0: -1])
                                for _, steps in self.__bus_events.items()))
        return avail_trans_size

    def calcu_equal_rewar(self, snapshot, is_globa):
        if is_globa:
            globa_state = snapshot.globa_state
            spacs = np.diff(globa_state)
            spacs = np.append(spacs, 1 - globa_state[-1] + globa_state[0])

            # 1.
            # equal_rewar = -np.std(spacs)

            # 2.
            # print(spacs)
            devia = np.abs(np.diff(spacs))
            devia = np.append(devia, np.abs(spacs[0] - spacs[-1]))
            equal_rewar = -sum(devia) / len(devia)

            # 3.
            # even_spac = 1/len(spacs)
            # devia = np.abs(spacs - even_spac)
            # equal_rewar = -np.mean(devia)
        else:
            state = snapshot.local_state
            # 1.
            # equal_rewar = -abs(state[0] - state[1])

            # 2.
            equal_rewar = math.exp(-abs(state[0] - state[1]))

            # 3.
            # equal_rewar = -abs(state[0] * snapshot.corri_lengt / (30/3.6) - snapshot.equ_headway)
        return equal_rewar

        # def balance_rewar(self, equal_rewar, actio):
        #     return equal_rewar - self._w*actio

    def track_rewar(self, snapshot, is_globa, hold_time, w):
        if is_globa:
            globa_state = snapshot.globa_state
            spacs = np.diff(globa_state)
            spacs = np.append(spacs, 1 - globa_state[-1] + globa_state[0])
            devia = np.abs(np.diff(spacs))
            devia = np.append(devia, np.abs(spacs[0] - spacs[-1]))
            equal_rewar = -sum(devia) / len(devia)
            inten_rewar = -hold_time / self._max_hold
            rewar = equal_rewar + w * inten_rewar
        else:
            state = snapshot.local_state
            #1
            # equal_rewar = -abs(state[0] - state[1])
            # inten_rewar = -hold_time / self._max_hold
            #2
            equal_rewar = math.exp(-abs(state[0] - state[1]))
            inten_rewar = math.exp(-hold_time / self._max_hold)
            #3
            # equal_rewar = -abs(state[0] * snapshot.corri_lengt / (30/3.6) - snapshot.equ_headway)
            # inten_rewar = 0
            rewar = equal_rewar + w * inten_rewar
        return equal_rewar, inten_rewar, rewar


    # event graphs about multi-agent

    def push_transition_graph_to_buffer(self):
        trans = self.__get_event_graph_trans(is_full=False)
        for tran in trans:
            self.__memor.append(tran)
        # clear event records after pushing it to memory
        self.clear_events()
        return self.__memor

    def __get_event_graph_trans(self, is_full=False):
        trans = []
        for bus_id, events in self.__bus_events.items():
            for prev_event, curr_event, next_event in zip(events[0: -2], events[1:-1], events[2:]):
                if self.__is_graph_state_globa:
                    prev_state = copy(prev_event.globa_relat_state)
                    curr_state = copy(curr_event.globa_relat_state)
                else:
                    prev_state = copy(prev_event.state)
                    curr_state = copy(curr_event.state)

                evens_betwe_node_feats = self.__get_events_between(
                    prev_event.time, curr_event.time, prev_event.stop_id)

                self_node_feat = self.__node_feat(
                    prev_event.bus_id, prev_event.stop_id, prev_event.time, 'self', 0, prev_state, prev_event.action)
                evens_betwe_node_feats.append(self_node_feat)

                next_evens_betwe_node_feats = self.__get_events_between(
                    curr_event.time, next_event.time, curr_event.stop_id)
                next_self_node_feat = self.__node_feat(
                    curr_event.bus_id, curr_event.stop_id, curr_event.time, 'self', 0, curr_state, curr_event.action)
                next_evens_betwe_node_feats.append(next_self_node_feat)

                s = copy(prev_event.globa_relat_state) if self._is_state_globa else copy(
                    prev_event.state)
                s.append(prev_event.stop_id)

                n_s = copy(curr_event.globa_relat_state) if self._is_state_globa else copy(
                    curr_event.state)
                n_s.append(curr_event.stop_id)

                a = copy(prev_event.action)
                # equalized reward of current depature
                equal_r = copy(curr_event.equal_rewar)
                # balanced reward between prev action and current depature reward
                if is_full:
                    # for offline learning
                    bus_id = copy(prev_event.bus_id)
                    stop_id = copy(prev_event.stop_id)
                    ct = copy(prev_event.time)
                    ep = copy(prev_event.episode)
                    trans.append([ep, bus_id, stop_id, ct, s, a, equal_r, n_s,
                                 evens_betwe_node_feats, next_evens_betwe_node_feats])
                else:
                    # for online learning
                    trans.append(
                        [s, a, equal_r, n_s, evens_betwe_node_feats, next_evens_betwe_node_feats])
        return trans

    def __get_events_between(self, prev_time, curr_time, prev_stop_id):
        # get events between prev_event and curr_event
        time_evens_betwe_dict = {time: events for time, events in self.__time_events.items(
        ) if time > prev_time and time < curr_time}
        stop_id_evens_betwe = {}
        for time, events in time_evens_betwe_dict.items():
            for event in events:
                stop_id_evens_betwe[event.stop_id] = event
        sort_stop_ids = sorted(stop_id_evens_betwe.keys())

        stop_num = self.__config.stop_num
        upstr_stop_ids, downs_stop_ids = self.get_up_and_downstream_stop_ids(
            prev_stop_id, stop_num)

        node_feats = []
        for idx, stop_id in enumerate(sort_stop_ids):
            bus_id = stop_id_evens_betwe[stop_id].bus_id
            ct = stop_id_evens_betwe[stop_id].time

            if self.__is_graph_state_globa:
                state = stop_id_evens_betwe[stop_id].globa_relat_state
            else:
                state = stop_id_evens_betwe[stop_id].state

            action = stop_id_evens_betwe[stop_id].action
            dist, up_or_down = None, None
            if stop_id in upstr_stop_ids:
                if prev_stop_id - stop_id >= 0:
                    dist = prev_stop_id - stop_id
                else:
                    dist = prev_stop_id - stop_id + stop_num
                up_or_down = 'up'
            elif stop_id in downs_stop_ids:
                if stop_id - prev_stop_id >= 0:
                    dist = stop_id - prev_stop_id
                else:
                    dist = stop_id - prev_stop_id + stop_num
                up_or_down = 'down'
            else:
                # event is at the same stop as prev_event
                dist = 0
                up_or_down = 'down'
            dist = dist / stop_num
            node_feat = self.__node_feat(
                bus_id, stop_id, ct, up_or_down, dist, state, action)
            node_feats.append(node_feat)

        return node_feats

    def sample_transition_graph(self, batch_size, is_discr_act=True):
        states = []
        actions = []
        rewards = []
        next_states = []
        graphs = []
        next_graphs = []
        samples = random.sample(self.__memor, batch_size)
        for sample in samples:
            state, action, reward, next_state, graph, next_graph = sample
            states.append(state)
            actions.append(action)
            if is_discr_act:
                reward = reward + self._w * math.exp(-action / 3)
            else:
                reward = reward + self._w * math.exp(-action)
            rewards.append(reward)
            next_states.append(next_state)
            graphs.append(graph)
            next_graphs.append(next_graph)
        return states, actions, rewards, next_states, graphs, next_graphs

    def get_up_and_downstream_stop_ids(self, stop_id, stop_num):
        if stop_num % 2 == 0:
            upstr_stop_num = int(stop_num / 2)
            downs_stop_num = int(stop_num / 2) - 1
        else:
            upstr_stop_num = int((stop_num - 1) / 2)
            downs_stop_num = int((stop_num - 1) / 2)
        upstr_stop_ids = []
        for i in range(1, upstr_stop_num + 1):
            if stop_id - i >= 0:
                upstr_stop_ids.append(stop_id - i)
            else:
                upstr_stop_ids.append(stop_num + stop_id - i)
        downs_stop_ids = []
        for i in range(1, downs_stop_num + 1):
            if stop_id + i <= stop_num - 1:
                downs_stop_ids.append(stop_id + i)
            else:
                downs_stop_ids.append(stop_id + i - stop_num)
        return upstr_stop_ids, downs_stop_ids


