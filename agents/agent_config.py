import itertools


class Agent_Config(object):
    def __init__(self, agent_name):
        # some common parameters
        self._max_hold = [60.0]
        self.__agent_config = {'max_hold': self._max_hold, }

        if agent_name == "DQN":
            self.__set_DQN_combine()
        elif agent_name == "DoubleDQN":
            self.__set_DoubleDQN_combine()
        elif agent_name == "Liu2023":
            self.__set_Liu2023_combine()
        elif agent_name == "DuelingDQN":
            self.__set_DuelingDQN_combine()
        elif agent_name == "DDPG":
            self.__set_DDPG_combine()
        elif agent_name == "PPO_C":
            self.__set_PPOC_combine()
        elif agent_name == "PPO_D":
            self.__set_PPOD_combine()
        elif agent_name == 'DO_NOTHING':
            self.__set_Do_Nothing_combine()
        elif agent_name == "Daganzo":
            self.__set_Daganzo_combine()
        elif agent_name == "MADDPG":
            self.__set_MADDPG_combine()
        elif agent_name == "MAPPO":
            self.__set_MAPPO_combine()
        elif agent_name == "MAPPO_C":
            self.__set_MAPPOC_combine()
        elif agent_name == "MADQN":
            self.__set_MADQN_combine()

        self.__combine = self.__combine_config()

    def __combine_config(self):
        combinations = list(itertools.product(*self.__agent_config.values()))
        combine = []
        for combo in combinations:
            result = {}
            for i, key in enumerate(self.__agent_config.keys()):
                result[key] = combo[i]
            combine.append(result)
        return combine

    def get_combine(self):
        return self.__combine

    def __set_Do_Nothing_combine(self):
        self.__agent_config['w'] = [0]
        self.__agent_config['is_state_globa'] = [False]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_graph'] = [False]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['agent_name'] = ['DO_NOTHING']

    def __set_Daganzo_combine(self):
        self.__agent_config['w'] = [0]
        self.__agent_config['is_state_globa'] = [False]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_graph'] = [False]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['pertu_range'] = [300]
        self.__agent_config['behav_polic'] = ['NONLINEAR_FIX']
        self.__agent_config['agent_name'] = ['Daganzo']

    def __set_DQN_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config["epsilon"] = [1]
        self.__agent_config["t"] = [1]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [True]
        self.__agent_config['agent_name'] = ['DQN']

    def __set_DoubleDQN_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config["epsilon"] = [1]
        self.__agent_config["t"] = [1]
        self.__agent_config["update_counter"] = [0]
        self.__agent_config["target_update_freq"] = [200]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [True]
        self.__agent_config['agent_name'] = ['DoubleDQN']

    def __set_Liu2023_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config["epsilon0"] = [1]
        self.__agent_config["t"] = [1]
        self.__agent_config["sigma"] = [1e8]
        self.__agent_config["update_counter"] = [0]
        self.__agent_config["target_update_freq"] = [200]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['agent_name'] = ['Liu2023']

    def __set_DuelingDQN_combine(self):     
        self.__set_RL_agent_common_config()
        self.__agent_config["epsilon"] = [1]
        self.__agent_config["t"] = [1]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [True]
        self.__agent_config['agent_name'] = ['DuelingDQN']

    def __set_DDPG_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config['init_noise_level'] = [0.15]
        self.__agent_config['decay_rate'] = [0.98]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['is_discr_act'] = [False]
        self.__agent_config['agent_name'] = ['DDPG']

    def __set_PPOC_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config["clip"] = [0.2]
        self.__agent_config["K_epoch"] = [10]
        self.__agent_config["lmbda"] = [0.95]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['is_discr_act'] = [False]
        self.__agent_config['agent_name'] = ['PPO_C']

    def __set_PPOD_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config["clip"] = [0.2]
        self.__agent_config["K_epoch"] = [10]
        self.__agent_config["lmbda"] = [0.95]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['is_discr_act'] = [True]
        self.__agent_config['agent_name'] = ['PPO_D']

    def __set_A2C_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config["epsilon"] = [1]
        self.__agent_config["t"] = [1]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [False, True]
        self.__agent_config['agent_name'] = ['A2C']

    def __set_MADDPG_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config['is_graph'] = [True]
        self.__agent_config['init_noise_level'] = [0.2]
        self.__agent_config['decay_rate'] = [0.98]
        self.__agent_config['gat_hidde_size'] = [8]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['is_discr_act'] = [False]
        self.__agent_config['agent_name'] = ['MADDPG']

    def __set_MAPPO_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config["clip"] = [0.2]
        self.__agent_config["K_epoch"] = [10]
        self.__agent_config["lmbda"] = [0.95]
        self.__agent_config['gat_hidde_size'] = [8]
        self.__agent_config['is_graph'] = [True]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['agent_name'] = ['MAPPO']

    def __set_MAPPOC_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config["clip"] = [0.2]
        self.__agent_config["K_epoch"] = [10]
        self.__agent_config["lmbda"] = [0.95]
        self.__agent_config['gat_hidde_size'] = [8]
        self.__agent_config['is_graph'] = [True]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['is_discr_act'] = [False]
        self.__agent_config['agent_name'] = ['MAPPO_C']

    def __set_MADQN_combine(self):
        self.__set_RL_agent_common_config()
        self.__agent_config["epsilon"] = [1]
        self.__agent_config["t"] = [1]
        self.__agent_config['gat_hidde_size'] = [10]
        self.__agent_config['is_graph'] = [True]
        self.__agent_config['is_rewar_globa'] = [False, True]
        self.__agent_config['is_discr_state'] = [False]
        self.__agent_config['agent_name'] = ['MADQN']

    def __set_RL_agent_common_config(self):
        self.__agent_config['layer_init_type'] = ['default']
        self.__agent_config['w'] = [   0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8
                                    ]
        self.__agent_config['lr'] = [5e-3, 1e-3] # 5e-3
        self.__agent_config['gamma'] = [0.95, 0.9, 0.99]  #0.99
        self.__agent_config['is_state_globa'] = [False]
        self.__agent_config['batch_size'] = [128]
        self.__agent_config['polya'] = [0.99]
        self.__agent_config['hidde_size'] = [[64], [64,64]]
        # self.__agent_config['line_name'] = [27]
