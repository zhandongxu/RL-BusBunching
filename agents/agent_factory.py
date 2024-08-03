from maddpg import MADDPG
from ddpg import DDPG
from dqn import DQN
from double_dqn import DoubleDQN
from dueling_dqn import DuelingDQN
from do_nothing import Do_Nothing
from ppo_conti import PPO_C
from ppo_disc import PPO_D
from Liu2023 import Liu
from madqn import MADQN
from mappo import MAPPO
from mappo_conti import MAPPO_C
from daganzo2009 import Daganzo


class Agent_Factory(object):
    @staticmethod
    def produce_agent(config, agent_config, is_eval):
        if agent_config['agent_name'] == "DQN":
            return DQN(config, agent_config, is_eval)
        elif agent_config['agent_name'] == "DoubleDQN":
            return DoubleDQN(config, agent_config, is_eval)
        elif agent_config['agent_name'] == "DuelingDQN":
            return DuelingDQN(config, agent_config, is_eval)
        elif agent_config['agent_name'] == "DDPG":
            return DDPG(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'PPO_C':
            return PPO_C(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'PPO_D':
            return PPO_D(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'DO_NOTHING':
            return Do_Nothing(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'Daganzo':
            return Daganzo(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'Liu2023':
            return Liu(config, agent_config, is_eval)
        elif agent_config['agent_name'] == "MADQN":
            return MADQN(config, agent_config, is_eval)
        elif agent_config['agent_name'] == "MAPPO":
            return MAPPO(config, agent_config, is_eval)
        elif agent_config['agent_name'] == "MAPPO_C":
            return MAPPO_C(config, agent_config, is_eval)
        elif agent_config['agent_name'] == "MADDPG":
            return MADDPG(config, agent_config, is_eval)