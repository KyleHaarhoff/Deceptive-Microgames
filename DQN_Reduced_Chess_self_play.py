#Imports
import copy
import os
import threading
import time

from reduced_chess.reduced_chess.env.reduced_chess_v2 import ReducedChess
from RL_Implementations.DQN import qModel, optimize_model, egreedy_policy, memory_buffer
import pandas as pd

import torch
import torch.optim as optim
from tqdm import tqdm
import math
import numpy as np

from itertools import zip_longest
#Seed
seed = None
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
gamma = 0.98
target_update_frequency = 100
epsilon_decay_rate = 0.9999 #Epsilon will day multiplicatively with this number every timestep
epsilon_min = 0.05
checkpoint_rate = 10000

#Env Params
env_memory_length = 6



# 
root_directory = os.path.dirname(os.path.abspath(__file__))
#Create Folders for Models

os.makedirs(root_directory + "/models", exist_ok=True)
os.makedirs(root_directory + "/models/DQN_self", exist_ok=True)
os.makedirs(root_directory + "/models/DQN_self/Reduced_Chess", exist_ok=True)
#Create Folders for Results
os.makedirs(root_directory + "/results", exist_ok=True)
os.makedirs(root_directory + "/results/DQN_self", exist_ok=True)
os.makedirs(root_directory + "/results/DQN_self/Reduced_Chess", exist_ok=True)

if seed == None:
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=99999999)

print(f"Using seed: {seed}")
rng = np.random.default_rng(seed)


env = ReducedChess(memory_length = env_memory_length)

agents = env.possible_agents
policy_models = {}
memories = {}

#Tracking
ave_rewards = {}
saved_total_rewards = {}
saved_game_results = {"results":[],
                        "king_captured":[],
                        "final_rank":[],
                        "pieces_revealed":[]}

game_lengths = {"lengths":[]}

#Only need one agent
num_inputs  =  env.observation_space("white_pieces")["observation"].shape
num_outputs =   env.action_space("white_pieces").n.item()
target_model = qModel(num_inputs, num_outputs).to(device)
policy_model = qModel(num_inputs, num_outputs).to(device)
target_model.load_state_dict(policy_models.state_dict())
optimizer = optim.Adam(target_model.parameters(), lr=learning_rate)


for agent in agents:
    num_inputs  =  env.observation_space(agent)["observation"].shape
    num_outputs =   env.action_space(agent).n.item()
    policy_models[agent] = qModel(num_inputs, num_outputs).to(device)
    memories[agent] = memory_buffer(max_length=experience_max_length)

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
            action = egreedy_policy(np.array([observation]), policy_models[agent], epsilon, mask=mask, 
                                    num_actions=env.action_space(agent).n.item())
        env.step(action)
        
        #record in memory
        if prev_memories[agent] is not None:
            memories[agent].append(prev_memories[agent]["state"], prev_memories[agent]["action"], reward, observation, termination, mask)
        
        prev_memories[agent] = {
            "state":observation,
            "action":action
        }
        
        total_rewards[agent] +=reward
        optimize_model(memories[agent], policy_models[agent], target_model, optimizer, batch_size = batch_size, gamma=gamma)

        if total_steps % target_update_frequency == 0:
            target_model.load_state_dict(policy_models.state_dict())

    for agent in agents:
        #tracking
        ave_rewards[agent].append(total_rewards[agent]/zsteps)
        saved_total_rewards[agent].append(total_rewards[agent])
    saved_game_results["results"].append(env.game_state)
    saved_game_results["king_captured"].append(env.king_capture)
    saved_game_results["final_rank"].append(env.final_rank)
    saved_game_results["pieces_revealed"].append(env.pieces_revealed)
    game_lengths["lengths"].append(zsteps)

    #If we have enough experience
    epsilon*= epsilon_decay_rate
    epsilon = max(epsilon, epsilon_min)

    #Checkpoint Model
    for agent in env.possible_agents:
        if i%checkpoint_rate == 0:
            torch.save(target_model.state_dict(), 
                       root_directory +f"/models/DQN_self/Reduced_Chess/{agent}-Policy-{i}-{file_label}.pth")

df = pd.DataFrame(ave_rewards)
df.to_csv(root_directory +f"/results/DQN_self/Reduced_Chess/ave-rewards-{file_label}.csv", index=False)

df = pd.DataFrame(saved_game_results)
df.to_csv(root_directory +f"/results/DQN_self/Reduced_Chess/game-results-{file_label}.csv", index=False)

df = pd.DataFrame(saved_total_rewards)
df.to_csv(root_directory +f"/results/DQN_self/Reduced_Chess/total-rewards-{file_label}.csv", index=False)


df = pd.DataFrame(game_lengths)
df.to_csv(root_directory +f"/results/DQN_self/Reduced_Chess/game_lengths-{file_label}.csv", index=False)