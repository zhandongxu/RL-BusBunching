# **Reinforcement Learning (RL) to slove bus bunching**  

A framework where RL agent tries to choose the correct holding time at an station to improve the efficiency of bus operation. This repository allows us to compare the improvement of different algorithms in solving the bus bunching problem, and to compare the impact of these RL components on the results by setting different state, action, and reward functions.

I have uploaded this here to help anyone searching for a good starting point for deep reinforcement learning with bus bunching. 

## The code structure
The main file is **main.py**. It performs agent training or evaluation based on input line information (e.g., `config = Config(line=34)`). Depending on the parameters selected for training or evaluation, headway variation, waiting time, holding time and reward are then recorded in the wandb.

Overall the code is divided into classes that handle different parts of the training.
* **main.py**: Training and evaluating after identifying bus routes, reinforcement learning methods (e.g., DQN, Double DQN, Dueling DQN, DDPG, PPO) and parameters recorded by wandb.
  
* **config.py**: Determination of parameters related to the simulated operation of bus routes. Through the real-world data processing of station information and operation data, further processing and obtaining the relevant parameters required to simulate the operation of bus simulation.
  
* **utils.py**：Setting of random seed parameters.
  
* **visualize.py**：Used for plotting space-trajectory diagram.
  
* The **agent** class contains code for different reinforcement learning methods as well as neural network setups.

* The **simulate** calss contains code for the bus operation simulator, compiled from bus, stop, link, line, passenger. Of these, the **simulator.py** is responsible for the main simulation run, and the **snapshot.py** responsible for recording some information (state and other operation information) about the agent before making a decision.
  
* The **data** class contains information about one of the lines used for training.

## The setting explained

Our default settings for both training and evaluation episodes are 150. The settings contained in the file **main.py** and **config.py** are the following:
* **agent_name**: Algorithm corresponding to the agent for training or evaluation.
* **is_eval**: Whether to train or evaluate, 'False' means training, 'True' means evaluation.
* **is_record_link_traje**: Whether to record information on the spatial and temporal trajectories of vehicles during simulation modelling.
* **is_record_wandb**: Whether to log data in wandb.
* **sim_duration**: Duration of the simulation
* **output_file**：Input line information, wherein the fields include 'seq',	'link_length',	'lambda',	'mean_tt',	'cv_tt'. **'seq'**: The station's sequence number on the line. **'link_length'**：Link length from this station to the next station. **'lambda'**：Passenger arrival rate， i.e., number of passengers arriving in a minute. **'mean_tt'/'cv_tt'**: Average travel time and its coefficient of variation corresponding to the link lengths. In the **data** file, I put a sample file for formatting reference.

Parameters related to reinforcement learning are in their corresponding names or 'agent/agent_config.py'.
* **max_hold**: the maximum holding time *T*.
* **epsilon**：coefficient of greedy control.
* **t**：controlling the size of *epsilon*.
* **target_update_freq**: update frequency of the target network.
* **is_discr_state**: whether or not to consider station sequences.
* **w**: hyperparameters of the reward function.
* **lr**: the learning rate defined for the neural network.
* **gamma**: the gamma parameter of the Bellman equation.
* **batch_size**: the number of samples retrieved from the memory for each training iteration.
* **polya**: update factor for the target network
* **hidde_size**：hidden layer construction in neural networks.

## RL component settings

* **Action**: In the problem of bus bunching, the action output is one-dimensional. We can adjust the action space by **max_hold** and **action_size** (in the 'agent/agent.py', i.e. if action_size = 4, then the discrete action space is denoted as [0, 1/3, 2/3, 1]). Then we sample an action in this collection.
* **State**: The state space we collect is the the three-dimensional of front and rear vehicle spacing *h-*, *h+* respectively and station index *d*, [*h-*, *h+*, *d*]. Determine if station index is considered based on the parameter  **is_discr_state**.
* **Reward function**: In our settings, *r = math.exp(-|(h-) - (h+)|) + w · math.exp(-a)*. It can be adjusted in 'agent/transitions.py' to suit your needs.

## Implementation
1.Firstly, the bus routes selected for training and the reinforcement learning method are identified, and the relevant parameters are modified in the code corresponding to the name and in ‘agent/agent_config.py’. 

2.Then, as mentioned above, set the corresponding state, action, and reward in 'agent/transitions.py' and you can see the recorded training trajectory in the wandb. 

3.Finally，setting '**is_eval**' to True will then allow you to evaluate the effectiveness of the model training.

## Contact
I sincerely hope this repository can help you. If you have any questions about it, you can open an issue on the issues page.


