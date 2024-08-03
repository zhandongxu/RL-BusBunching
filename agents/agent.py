import numpy as np
from collections import deque
import torch
import wandb
import os
import json
import uuid


class Agent:
    def __init__(self, config, agent_config, is_eval) -> None:
        # hyperparameters
        self._config = config
        self._agent_config = agent_config
        self._is_eval = is_eval
        self._is_state_globa = agent_config['is_state_globa']
        self._max_hold = agent_config['max_hold']
        self._w = agent_config['w']
        self._is_rewar_globa = agent_config['is_rewar_globa']
        # store model path with respect to agent config
        self.__json_file_path = f'../model/config_model_map.json'

        if agent_config['agent_name'] in ['Daganzo', 'DO_NOTHING']:
            # for analytical agents
            pass
        else:
            # RL agents
            self._state_size = 3 if not self._is_state_globa else config.bus_num
            self._action_size = 4
            self._gamma = agent_config['gamma']
            self._lr = agent_config['lr']
            self._polya = agent_config['polya']
            self._batch_size = agent_config['batch_size']
            self._hidde_size = agent_config['hidde_size']
            # self._is_embed_discr_state = agent_config['is_embed_discr_state']

        # metric tracking
        self._track_rewas = []
        self._track_equal_rewas = []
        self._track_inten_rewas = []
        self._track_heads = []
        self._track_hold_times = []
        self._stop_wait_time_dict = {}
        self._stop_arriv_dict = {}
        self._bus_ride_time_dict = {}
        self._bus_ride_pax_dict = {}

    def reset(self, episode, is_record_wandb=True):
        sum_rewar = float(sum(self._track_rewas) / len(self._track_rewas))
        equal_rewar = float(sum(self._track_equal_rewas) / \
            len(self._track_equal_rewas))
        inten_rewar = float(sum(self._track_inten_rewas) / \
            len(self._track_inten_rewas))
        hold_time = float(sum(self._track_hold_times) / len(self._track_hold_times))
        headw_varia = np.std(self._track_heads)

        # total waiting time in seconds
        total_wait_time = sum(self._stop_wait_time_dict.values())
        # total arrivals
        total_arriv = sum(self._stop_arriv_dict.values())
        # average waiting time in minutes
        avg_wait_time = (total_wait_time / total_arriv) / 60.0

        # total riding time in seconds
        total_ride_time = sum(self._bus_ride_time_dict.values())
        total_ride = sum(self._bus_ride_pax_dict.values())
        # average riding time in minutes
        avg_ride_time = (total_ride_time / total_ride) / 60.0

        if is_record_wandb:
            wandb.log({'sum': sum_rewar, 'equalized reward': equal_rewar, 'intensity reward': inten_rewar, 'headway variation': headw_varia,
                      'hold time': hold_time, 'average waiting time': avg_wait_time, 'average riding time': avg_ride_time})

        self._track_rewas = []
        self._track_equal_rewas = []
        self._track_inten_rewas = []
        self._track_heads = []
        self._track_hold_times = []
        # record cumulative waiting time and arrivals at stops
        self._stop_wait_time_dict = {}
        self._stop_arriv_dict = {}
        # record cumulative ride time and boarding pax at buses
        self._bus_ride_time_dict = {}
        self._bus_ride_pax_dict = {}
        # record average waiting time from headway of each stop
        self._stop_headway_dict = {}
        self._stop_bus_load_dict = {}

        return headw_varia

    def cal_hold_time(self, snapshot):
        raise NotImplementedError

    def get_config_map(self):
        # load the JSON file if it exists
        if os.path.isfile(self.__json_file_path):
            with open(self.__json_file_path, 'r') as f:
                exist_confi = json.load(f)
        else:
            exist_confi = {}
        return exist_confi

    def save_model(self, model_state_dict, agent_config):
        # generate a UUID and use it to create the model file path
        model_uuid = str(uuid.uuid4())
        model_path = f'../model/{model_uuid}.pt'
        exist_confi = self.get_config_map()
        for num, model_info in exist_confi.items():
            # if model config exists, update model path
            if model_info['config'] == agent_config:
                old_path = model_info['model_path']
                model_info['model_path'] = model_path
                exist_confi[num] = model_info
                # delete the old model file
                print(f'Deleting old model file {old_path}')
                os.remove(old_path)
                break
        # else:
            # if no model config exists, add new model config
        confi_num = len(exist_confi)
        json_dict = {confi_num: {
            'model_path': model_path, 'config': agent_config}}
        exist_confi.update(json_dict)

        # save the updated model config to the JSON file
        with open(self.__json_file_path, 'w') as f:
            json.dump(exist_confi, f)
        torch.save(model_state_dict, model_path)

    def load_model(self, agent_config):
        exist_confi = self.get_config_map()

        for num, model_info in exist_confi.items():
            # if model config exists, update model path
            if model_info['config'] == agent_config:
                model_path = model_info['model_path']
                return torch.load(model_path)
        else:
            raise KeyError("no model config exists, please check")

    def track(self, track_rewar, track_equal_rewar, track_inten_rewar, track_hold_time):
        self._track_rewas.append(track_rewar)
        self._track_equal_rewas.append(track_equal_rewar)
        self._track_inten_rewas.append(track_inten_rewar)
        self._track_hold_times.append(track_hold_time)

    def track_headway_deviations(self, snapshot):
        last_depar_time = snapshot.last_depar_times[-2]
        h = snapshot.ct - last_depar_time
        self._track_heads.append(h)

    def record_wait_time(self, stop_wait_time_dict):
        self._stop_wait_time_dict = stop_wait_time_dict

    def record_arrival(self, stop_arrival_dict):
        self._stop_arriv_dict = stop_arrival_dict

    def record_ride_time(self, bus_ride_time_dict):
        self._bus_ride_time_dict = bus_ride_time_dict

    def record_boarding(self, bus_ride_pax_dict):
        self._bus_ride_pax_dict = bus_ride_pax_dict

    def record_occupancy(self, stop_bus_load_dict):
        self._stop_bus_load_dict = stop_bus_load_dict

    def record_headway(self, stop_headway_dict):
        self._stop_headway_dict = stop_headway_dict
