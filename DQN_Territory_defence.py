#Imports
import copy
import os
import threading
import time

from Territory_Defence.territory_defence.env.territory_defence import TerritoryDefence
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
epsilon = 0.9
timesteps = 60000
learning_rate = 0.001
experience_max_length = 1000
batch_size = 64
gamma = 0.95
target_update_frequency = 100
epsilon_decay_rate = 0.9999 #Epsilon will day multiplicatively with this number every timestep
epsilon_min = 0.05
checkpoint_rate = 500

#Env Params
negative_rewards = False
instant_rewards = True 

env_memory_length = 2
max_steps = 25
start_map = [
        [' ',' ','D',' ', ' '],
        ['G','W',' ','W', 'G'],
        [' ','W','G','W', ' '],
        [' ','C',' ','C', ' '],
        ['C',' ',' ',' ', 'C'],
        ['C',' ',' ',' ', 'C'],
        [' ','C','I','C', ' ']
    ]


# 
root_directory = os.path.dirname(os.path.abspath(__file__))
#Create Folders for Models
os.makedirs(root_directory + "/models", exist_ok=True)
os.makedirs(root_directory + "/models/DQN", exist_ok=True)
os.makedirs(root_directory + "/models/DQN/Territory_Defence", exist_ok=True)

#Create Folders for Results
os.makedirs(root_directory + "/results", exist_ok=True)
os.makedirs(root_directory + "/results/DQN", exist_ok=True)
os.makedirs(root_directory + "/results/DQN/Territory_Defence", exist_ok=True)

if seed == None:
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=99999999)

print(f"Using seed: {seed}")
rng = np.random.default_rng(seed)


env = TerritoryDefence(start_map = start_map, memory_length = env_memory_length,
                       negative_rewards=negative_rewards, instant_rewards=instant_rewards, 
                       max_steps = max_steps) 
agents = env.possible_agents
target_models = {}
policy_models = {}
memories = {}
optimizers = {}

#Tracking
ave_rewards = {}
saved_total_rewards = {}
saved_game_results = {"results":[]}
game_lengths = {"lengths":[]}
learning_stats = {"invader_towards_goal_average":[],
                  "defender_towards_invader_average":[]}


for agent in agents:
    num_inputs  =  math.prod(env.observation_space(agent)["observation"].shape)
    num_outputs =   env.action_space(agent).n.item()
    target_models[agent] = qModel(num_inputs, num_outputs).to(device)
    policy_models[agent] = qModel(num_inputs, num_outputs).to(device)
    target_models[agent].load_state_dict(policy_models[agent].state_dict())
    memories[agent] = memory_buffer(max_length=experience_max_length)
    optimizers[agent] = optim.Adam(target_models[agent].parameters(), lr=learning_rate)

    #tracking
    ave_rewards[agent] = []
    saved_total_rewards[agent] = []



total_steps = 0
for i in tqdm(range(timesteps)):
    #Complete Game
    env.reset(seed=rng.integers(low=0, high=99999999))
    
    done = False
    iteration = 0
    prev_memories= {}

    for agent in agents:
        prev_memories[agent] = None

    #tracking
    total_rewards = {}
    for agent in agents:
        total_rewards[agent] = 0
    zsteps = 0
    invader_towards_goal_steps = 0
    defender_towards_invader_steps = 0
    for agent in env.agent_iter():
        if agent == agents[0]:
            total_steps+=1
            zsteps +=1

        #get observation
        observation, reward, termination, truncation, info = env.last()
        
        mask = observation["action_mask"]
        observation = observation["observation"]

        #take action
        if termination or truncation:
            action = None
        else:
            #Get Actions
            action = egreedy_policy(np.array([observation.flatten()]), policy_models[agent], epsilon, mask=mask,
                                    num_actions=env.action_space(agent).n.item())
        env.step(action)
        
        #record in memory
        if prev_memories[agent] is not None:
            memories[agent].append(prev_memories[agent]["state"], prev_memories[agent]["action"], 
                                   reward, observation.flatten(), termination, mask)
        
        prev_memories[agent] = {
            "state":observation,
            "action":action
        }
        
        total_rewards[agent] +=reward
        optimize_model(memories[agent], policy_models[agent], target_models[agent], optimizers[agent], batch_size = batch_size, gamma=gamma)

        if total_steps % target_update_frequency == 0:
            target_models[agent].load_state_dict(policy_models[agent].state_dict())
        #Tracking
        if env.is_invader[agent]:
            invader_towards_goal_steps += env.invader_moved_closer_to_goal
        else:
            defender_towards_invader_steps += env.defender_moved_closer_to_attacker
    for agent in agents:
        #tracking
        ave_rewards[agent].append(total_rewards[agent]/zsteps)
        saved_total_rewards[agent].append(total_rewards[agent])
    learning_stats["defender_towards_invader_average"].append(defender_towards_invader_steps/zsteps)
    learning_stats["invader_towards_goal_average"].append(invader_towards_goal_steps/zsteps)
    saved_game_results["results"].append(env.game_state)
    game_lengths["lengths"].append(zsteps)

    #If we have enough experience
    epsilon*= epsilon_decay_rate
    epsilon = max(epsilon, epsilon_min)

    #Checkpoint Model
    for agent in env.possible_agents:
        if i%checkpoint_rate == 0:
            torch.save(target_models[agent].state_dict(), 
                       root_directory +f"/models/DQN/Territory_Defence/{agent}-Policy-{i}-{file_label}.pth")

df = pd.DataFrame(ave_rewards)
df.to_csv(root_directory +f"/results/DQN/Territory_Defence/ave-rewards-{file_label}.csv", index=False)

df = pd.DataFrame(saved_game_results)
df.to_csv(root_directory +f"/results/DQN/Territory_Defence/game-results-{file_label}.csv", index=False)

df = pd.DataFrame(saved_total_rewards)
df.to_csv(root_directory +f"/results/DQN/Territory_Defence/total-rewards-{file_label}.csv", index=False)


df = pd.DataFrame(game_lengths)
df.to_csv(root_directory +f"/results/DQN/Territory_Defence/game_lengths-{file_label}.csv", index=False)

df = pd.DataFrame(learning_stats)
df.to_csv(root_directory +f"/results/DQN/Territory_Defence/learning-stats-{file_label}.csv", index=False)