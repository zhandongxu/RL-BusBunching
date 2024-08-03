import numpy as np
from agent import Agent
import wandb
from transitions import Event_Handler
from itertools import product

class Daganzo(Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(Daganzo, self).__init__(config, agent_config, is_eval)
        self.__H = config.equ_headw
        self.__behav_polic = agent_config['behav_polic']
        self.__is_graph = agent_config['is_graph']
        self.__pertu_range = agent_config['pertu_range']

        self.__type_dict = {'behav_polic': self.__behav_polic, 'is_state_globa': self._is_state_globa,
                            'is_rewar_globa': self._is_rewar_globa, 'is_graph': self.__is_graph,
                            'pertu_range': self.__pertu_range}

        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=False,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold, w=self._w,
                                           is_graph=self.__is_graph)

        if self.__behav_polic == 'NONLINEAR_RANDOM':
            self.__alpha, self.__H = self.sample_alpha_and_H()
        elif self.__behav_polic == 'NONLINEAR_FIX':
            self.__alpha = 0.2
        elif self.__behav_polic == 'NONLINEAR_ENUME':
            self.__alphs = np.arange(0.01, 1.01, 0.01).tolist()
            self.__Hs = [x * 60 for x in [6]]
            self.__paras_enume = list(product(self.__Hs, self.__alphs))
            self.__alpha = self.__alphs[0]
            self.__H = self.__Hs[0]

    def __str__(self) -> str:
        return 'Daganzo'

    def sample_alpha_and_H(self):
        # return np.random.uniform(0.01, 0.99), np.random.uniform(5*60, 7*60)
        return np.random.uniform(0.01, 1.01), self.__H

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        headw_varia = super().reset(episode, is_record_wandb)
        # at the end of each episode, write it to csv
        if is_record_transition:
            self.__event_handl.write_transition_to_file(self.__type_dict)

        # update interacting policy parameter
        if self.__behav_polic == 'NONLINEAR_RANDOM':
            self.__alpha, self.__H = self.sample_alpha_and_H()
        elif self.__behav_polic == 'NONLINEAR_ENUME':
            self.__H, self.__alpha = self.__paras_enume[episode]
        elif self.__behav_polic == 'NONLINEAR_FIX':
            pass
        print(self.__alpha, self.__H, '---------------')
        return headw_varia

    def cal_hold_time(self, snapshot):
        ct = snapshot.ct
        assert ct == snapshot.last_depar_times[-1]
        last_depar_time = snapshot.last_depar_times[-2]
        if last_depar_time == 0:
            hold_time = 0
        else:
            devia = self.__H - (ct - last_depar_time)
            pertu = np.random.uniform(-self.__pertu_range,
                                      self.__pertu_range) if self.__pertu_range > 0 else 0
            devia += pertu
            hold_time = max(0, self.__alpha * devia)
            hold_time = min(self._max_hold, hold_time)

        # record departure event and log reward
        action = hold_time / self._max_hold
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, action, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)

        return hold_time