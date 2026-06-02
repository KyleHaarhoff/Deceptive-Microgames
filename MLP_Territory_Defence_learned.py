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
file_label = "learned_hlp_run_1" 
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
checkpoint_rate = 500
num_goals = 3 #Used for the number of policies

#Env Params
negative_rewards = False
instant_rewards = True 

#env params
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










root_directory = os.path.dirname(os.path.abspath(__file__))
#Create Folders for Models
os.makedirs(root_directory + "/models", exist_ok=True)
os.makedirs(root_directory + "/models/MLP", exist_ok=True)
os.makedirs(root_directory + "/models/MLP/Territory_Defence", exist_ok=True)

#Create Folders for Results
os.makedirs(root_directory + "/results", exist_ok=True)
os.makedirs(root_directory + "/results/MLP", exist_ok=True)
os.makedirs(root_directory + "/results/MLP/Territory_Defence", exist_ok=True)

if seed == None:
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=99999999)

print(f"Using seed: {seed}")
rng = np.random.default_rng(seed)


env = TerritoryDefence(start_map = start_map, memory_length = env_memory_length,
                       negative_rewards=negative_rewards, instant_rewards=instant_rewards, 
                       max_steps = max_steps) 
agents = env.possible_agents
invader_memories = {}
invader_optimizers = {}

#Tracking
ave_rewards = {}
saved_total_rewards = {}
saved_game_results = {"results":[]}
game_lengths = {"lengths":[]}
learning_stats = {"invader_towards_goal_average":[],
                  "defender_towards_invader_average":[]}


#Setup Invader (MLP)

invader_target_models = {}
invader_policy_models = {}
num_inputs  =  env.observation_space("invader")["observation"].shape
num_outputs =   env.action_space("invader").n.item()
#lower level policies
for policy_num in range(num_goals):
    invader_target_models[policy_num] = qModel(num_inputs, num_outputs).to(device)
    invader_policy_models[policy_num] = qModel(num_inputs, num_outputs).to(device)
    invader_target_models[policy_num].load_state_dict(invader_policy_models[policy_num].state_dict())
    invader_memories[policy_num] = memory_buffer(max_length=experience_max_length)
    invader_optimizers[policy_num] = optim.Adam(invader_target_models[policy_num].parameters(), lr=learning_rate)

#higher level policies
invader_hlp_target_model = qModel(num_inputs, num_goals).to(device)
invader_hlp_policy_model = qModel(num_inputs, num_goals).to(device)
invader_hlp_target_model.load_state_dict(invader_hlp_policy_model.state_dict())
invader_hlp_memories = memory_buffer(max_length=experience_max_length)
invader_hlp_optimizer = optim.Adam(invader_target_models[policy_num].parameters(), lr=learning_rate)


# Function to distinguish between the paths
def choose_policy(observation):
    return egreedy_policy(np.array([observation]), invader_hlp_policy_model, epsilon, mask=None,
                                    num_actions=num_goals)
    



#Setup Defender (Normal DQN)
num_inputs  =  env.observation_space("defender")["observation"].shape
num_outputs =   env.action_space("defender").n.item()
defender_target_model = qModel(num_inputs, num_outputs).to(device)
defender_policy_model = qModel(num_inputs, num_outputs).to(device)
defender_target_model.load_state_dict(defender_policy_model.state_dict())
defender_memories = memory_buffer(max_length=experience_max_length)
defender_optimizer = optim.Adam(defender_target_model.parameters(), lr=learning_rate)

for agent in agents:
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
            if agent == "defender":
                #Normal DQN
                action = egreedy_policy(np.array([observation]), defender_policy_model, epsilon, mask=mask,
                                        num_actions=env.action_space(agent).n.item())
            else:
                policy_num = choose_policy(observation=observation)
                action = egreedy_policy(np.array([observation]), invader_policy_models[policy_num], epsilon, mask=mask,
                                        num_actions=env.action_space(agent).n.item())

        env.step(action)
        
        #record in memory
        if prev_memories[agent] is not None:
            if agent == "defender":
                defender_memories.append(prev_memories[agent]["state"], prev_memories[agent]["action"], 
                                    reward, observation, termination, mask)
            else:
                #append to the higher policy
                invader_hlp_memories.append(prev_memories[agent]["state"], prev_memories[agent]["policy_num"], 
                                    reward, observation, termination)
                #append to the correct lower policy
                prev_policy_num = prev_memories[agent]["policy_num"]
                invader_memories[prev_policy_num].append(prev_memories[agent]["state"], prev_memories[agent]["action"], 
                                    reward, observation, termination, mask)

            #optimisation step
            if agent == "defender":
                optimize_model(defender_memories, defender_policy_model, defender_target_model, defender_optimizer, batch_size = batch_size, gamma=gamma)
            else:
                #HLP
                optimize_model(invader_hlp_memories, invader_hlp_policy_model, invader_hlp_target_model, 
                            invader_hlp_optimizer, batch_size = batch_size, gamma=gamma)
                #LLP
                prev_policy_num = prev_memories[agent]["policy_num"]
                optimize_model(invader_memories[prev_policy_num], invader_policy_models[prev_policy_num], invader_target_models[prev_policy_num], 
                            invader_optimizers[prev_policy_num], batch_size = batch_size, gamma=gamma)


        if agent == "defender":
            prev_memories[agent] = {
            "state":observation,
            "action":action
        }
        else:
            prev_memories[agent] = {
            "state":observation,
            "action":action,
            "policy_num":policy_num
        }

        
        total_rewards[agent] +=reward

        if total_steps % target_update_frequency == 0:
            if agent == "defender":
                defender_target_model.load_state_dict(defender_policy_model.state_dict())

            else:
                invader_hlp_target_model.load_state_dict(invader_hlp_policy_model.state_dict())
                for policy_num in range(num_goals):
                    invader_target_models[policy_num].load_state_dict(invader_policy_models[policy_num].state_dict())
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
            if agent == "defender":
                torch.save(defender_target_model.state_dict(), 
                        root_directory +f"/models/MLP/Territory_Defence/{agent}-Policy-{i}-{file_label}.pth")

            else:
                torch.save(invader_hlp_target_model.state_dict(), 
                        root_directory +f"/models/MLP/Territory_Defence/{agent}-HLP-Policy-{i}-{file_label}.pth")
                for policy_num in range(num_goals):
                    torch.save(invader_target_models[policy_num].state_dict(), 
                            root_directory +f"/models/MLP/Territory_Defence/{agent}-LLP-{policy_num}-Policy-{i}-{file_label}.pth")
                    

df = pd.DataFrame(ave_rewards)
df.to_csv(root_directory +f"/results/MLP/Territory_Defence/ave-rewards-{file_label}.csv", index=False)

df = pd.DataFrame(saved_game_results)
df.to_csv(root_directory +f"/results/MLP/Territory_Defence/game-results-{file_label}.csv", index=False)

df = pd.DataFrame(saved_total_rewards)
df.to_csv(root_directory +f"/results/MLP/Territory_Defence/total-rewards-{file_label}.csv", index=False)


df = pd.DataFrame(game_lengths)
df.to_csv(root_directory +f"/results/MLP/Territory_Defence/game_lengths-{file_label}.csv", index=False)

df = pd.DataFrame(learning_stats)
df.to_csv(root_directory +f"/results/DQN/Territory_Defence/learning-stats-{file_label}.csv", index=False)