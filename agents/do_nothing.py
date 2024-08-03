import numpy as np
from collections import defaultdict, namedtuple, deque
import torch
from torch.utils.data import DataLoader
import wandb

from transitions import Event_Handler
from agent import Agent


class Do_Nothing(Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(Do_Nothing, self).__init__(config, agent_config, is_eval)
        self.__is_graph = agent_config['is_graph']
        self.__type_dict = {'behav_polic': 'DO_NOTHING', 'is_state_globa': self._is_state_globa,
                            'is_rewar_globa': self._is_rewar_globa, 'is_graph': self.__is_graph}

        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=True,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold, w=self._w, is_graph=self.__is_graph)

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        # at the final, write it to csv
        if is_record_transition:
            self.__event_handl.write_transition_to_file(self.__type_dict)
        headw_varia = super().reset(episode, is_record_wandb)

        return headw_varia

    def __str__(self) -> str:
        return 'DO_NOTHING'

    def cal_hold_time(self, snapshot):
        action = 0.0
        hold_time = 0.0
        # record departure event and log reward
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, action, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)

        return hold_time
