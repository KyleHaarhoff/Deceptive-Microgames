#Imports
import copy
import os
import threading
import time

from the_resistance.the_resistance.env.the_resistance import TheResistance
from RL_Implementations.DQN import qModel, optimize_model, egreedy_policy, memory_buffer
import pandas as pd

import torch
import torch.optim as optim
from tqdm import tqdm
import math
import numpy as np

from itertools import zip_longest
#Seed
seed = 83348902
# Label - for multiple results tracking
file_label = "run_1" 
#Cuda
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

#Hyperparams
epsilon = 0.8
timesteps = 6000
learning_rate = 0.001
experience_max_length = 1000
batch_size = 64
gamma = 0.98
target_update_frequency = 100
epsilon_decay_rate = 0.9999 #Epsilon will day multiplicatively with this number every timestep
epsilon_min = 0.05
checkpoint_rate = 500

#Env Params
num_agents = 5


# 
root_directory = os.path.dirname(os.path.abspath(__file__))
#Create Folders for Models

os.makedirs(root_directory + "/models", exist_ok=True)
os.makedirs(root_directory + "/models/DQN", exist_ok=True)
os.makedirs(root_directory + "/models/DQN/The_Resistance", exist_ok=True)
#Create Folders for Results
os.makedirs(root_directory + "/results", exist_ok=True)
os.makedirs(root_directory + "/results/DQN", exist_ok=True)
os.makedirs(root_directory + "/results/DQN/The_Resistance", exist_ok=True)

if seed == None:
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=99999999)

print(f"Using seed: {seed}")
rng = np.random.default_rng(seed)


env = TheResistance(num_agents=num_agents)

agents = env.possible_agents
target_models = {}
policy_models = {}
memories = {}
optimizers = {}

#
agent_labels = ["spy", "agent"]
agent_to_label_map = {}
for agent in agents:
    if "spy" in agent:
        agent_to_label_map[agent] = "spy"
    else:
        agent_to_label_map[agent] = "agent" 

ave_rewards = {}
saved_total_rewards = {}
saved_game_results = {"results":[],
                        "total_spy_to_spy_votes":[],
                        "total_spy_to_spy_votes_yes":[],
                        "total_infested_missions":[],
                        "total_infested_missions_sabotaged":[],
                        "total_agent_not_in_vote":[],
                        "total_agent_not_in_vote_yes":[]}
game_lengths = {"lengths":[]}

# Need to add a dimension for the DQN
num_inputs = math.prod(env.observation_space("spy1").shape)

num_outputs =   env.action_space("spy1").n.item()
target_models["spy"] = qModel(num_inputs, num_outputs).to(device)
policy_models["spy"] = qModel(num_inputs, num_outputs).to(device)
target_models["spy"].load_state_dict(policy_models["spy"].state_dict())
optimizers["spy"] = optim.Adam(target_models["spy"].parameters(), lr=learning_rate)


num_inputs = math.prod(env.observation_space("agent1").shape)
num_outputs =   env.action_space("agent1").n.item()
target_models["agent"] = qModel(num_inputs, num_outputs).to(device)
policy_models["agent"] = qModel(num_inputs, num_outputs).to(device)
target_models["agent"].load_state_dict(policy_models["agent"].state_dict())
optimizers["agent"] = optim.Adam(target_models["agent"].parameters(), lr=learning_rate)


for agent in agents:
    #tracking
    ave_rewards[agent] = []
    saved_total_rewards[agent] = []
    memories[agent] = memory_buffer(max_length=experience_max_length)



total_steps = 0
for i in tqdm(range(timesteps)):
    #Complete Game
    observations, _ = env.reset(seed=rng.integers(low=0, high=99999999))
    
    done = False
    iteration = 0


    for agent in agents:
        #Expand dimension for DQN
        observations[agent] = observations[agent].flatten()
    #tracking
    total_rewards = {}
    for agent in agents:
        total_rewards[agent] = 0
    zsteps = 0
    
        
    
    while not done:
        total_steps+=1
        zsteps +=1

        #act
        actions = {}
        for agent in agents: 
            agent_label = agent_to_label_map[agent]
            action = egreedy_policy(np.array([observations[agent]]), policy_models[agent_label], epsilon, 
                                    num_actions=env.action_space(agent).n.item())
            actions[agent] = action


        next_state, rewards, dones, truncations, infos = env.step(actions)

        #record in memory
        for agent in agents: 
            #Expand dimension for DQN
            next_state[agent] = next_state[agent].flatten()
            memories[agent].append(observations[agent], actions[agent], 
                                    rewards[agent], next_state[agent], dones[agent])
        
            
        
        # Optimise the models
        for agent in agents:
            total_rewards[agent] += rewards[agent]
            agent_label = agent_to_label_map[agent]
            optimize_model(memories[agent], policy_models[agent_label], target_models[agent_label], 
                           optimizers[agent_label], batch_size = batch_size, gamma=gamma)
            
        if total_steps % target_update_frequency == 0:
            for agent_label in agent_labels:
                target_models[agent_label].load_state_dict(policy_models[agent_label].state_dict())

        #update the state
        observations = next_state
        #
        for agent in agents:
         done = done or dones[agent]

    for agent in agents:
        #tracking
        ave_rewards[agent].append(total_rewards[agent]/zsteps)
        saved_total_rewards[agent].append(total_rewards[agent])
    saved_game_results["results"].append(env.game_state)
    saved_game_results["total_spy_to_spy_votes"].append(env.total_spy_to_spy_votes)
    saved_game_results["total_spy_to_spy_votes_yes"].append(env.total_spy_to_spy_votes_yes)
    saved_game_results["total_infested_missions"].append(env.total_infested_missions)
    saved_game_results["total_infested_missions_sabotaged"].append(env.total_infested_missions_sabotaged)
    saved_game_results["total_agent_not_in_vote"].append(env.total_agent_not_in_vote)
    saved_game_results["total_agent_not_in_vote_yes"].append(env.total_agent_not_in_vote_yes)
    game_lengths["lengths"].append(zsteps)

    #If we have enough experience
    epsilon*= epsilon_decay_rate
    epsilon = max(epsilon, epsilon_min)

    #Checkpoint Model
    for agent_label in agent_labels:
        if i>=checkpoint_rate:
            torch.save(target_models[agent_label].state_dict(), 
                       root_directory +f"/models/DQN/The_Resistance/{agent_label}-Policy-{i}-{file_label}.pth")

df = pd.DataFrame(ave_rewards)
df.to_csv(root_directory +f"/results/DQN/The_Resistance/ave-rewards-{file_label}.csv", index=False)

df = pd.DataFrame(saved_game_results)
df.to_csv(root_directory +f"/results/DQN/The_Resistance/game-results-{file_label}.csv", index=False)

df = pd.DataFrame(saved_total_rewards)
df.to_csv(root_directory +f"/results/DQN/The_Resistance/total-rewards-{file_label}.csv", index=False)


df = pd.DataFrame(game_lengths)
df.to_csv(root_directory +f"/results/DQN/The_Resistance/game_lengths-{file_label}.csv", index=False)