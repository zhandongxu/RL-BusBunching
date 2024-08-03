import wandb
from agents.agent_factory import Agent_Factory
from agents.agent_config import Agent_Config
from visualize import plot_time_space_diagram
from config import Config
from utils import set_seed
from simulate.simulator import Simulator
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'


set_seed(1)
config = Config(line=34)  #input line information

# corresponding algorithm name 
# agent_name = "DO_NOTHING"
# agent_name = "Daganzo"
# agent_name = "DQN"
# agent_name = "DoubleDQN"
# agent_name = "DuelingDQN"
agent_name = "DDPG"
# agent_name = "PPO_C"
# agent_name = "PPO_D"
# agent_name = "MADQN"
# agent_name = "MAPPO"
# agent_name = "MAPPO_C"
# agent_name = "MADDPG"
# agent_name = "Liu2023"

# is_eval = True
is_eval = False
# is_record_link_traje = True
is_record_link_traje = False
is_record_wandb = True
# is_record_wandb = False

if is_eval == False:        # wandb project name
    project_name = "new bus bunching study"
else:
    # project_name = "eva for K2"
    # project_name = "eva for 27"
    project_name = "eva"
    # project_name = "vehicle-robust"
combine_config = Agent_Config(agent_name).get_combine()

for agent_config in combine_config:
    headw_varis = []
    print("total combination:", len(combine_config),
          ", current combination:", combine_config.index(agent_config)+1)
    print('agent_name:', agent_config['agent_name'], 'combination:', agent_config)
    agent = Agent_Factory.produce_agent(config, agent_config, is_eval)
    if is_record_wandb:
        wandb.init(project=project_name, config=agent_config)
    simulator = Simulator(config, agent, is_record_link_traje)
    for episode in range(150):
        if episode % 20 == 0:
            print('------------', episode, '------------')
        simulator.reset(episode=episode)
        simulator.simulate(agent_config)
        headw_varia = agent.reset(
            episode=episode, is_record_wandb=is_record_wandb, is_record_transition=is_record_trans)
        headw_varis.append(headw_varia)
        # if episode > 100 and min(headw_varis[-10:]) > 150:
        #     break

    plot_time_space_diagram(simulator.get_buses_for_plot(), config)
    wandb.finish()



