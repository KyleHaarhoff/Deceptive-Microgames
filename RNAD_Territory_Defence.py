
import os
#Imports
import copy
import os
import threading
import time

from Territory_Defence.territory_defence.env.territory_defence import TerritoryDefence
from RL_Implementations.PPO import compute_GAE, trajectory_memory, Actor_Critic
from RL_Implementations.RNAD import RNAD
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
file_label = "run_1_set" 
#Cuda
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
#Hyper params:
gamma = 0.98  # Discount factor
lr = 0.0001  
epochs = 5  
mini_batch_size = 4  
max_memory_length = 50
gae_lambda = 0.95
epsilon_clip = 0.05
num_timesteps = 120000
#RNAD params
regularised_reward_coefficient = 0.1
num_reg_policy_updates = 100
num_timesteps_per_reg_update = num_timesteps//num_reg_policy_updates

checkpoint_rate = 10000
#Env Params
negative_rewards = False
instant_rewards = False 

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
os.makedirs(root_directory + "/models/RNAD", exist_ok=True)
os.makedirs(root_directory + "/models/RNAD/Territory_Defence", exist_ok=True)

#Create Folders for Results
os.makedirs(root_directory + "/results", exist_ok=True)
os.makedirs(root_directory + "/results/RNAD", exist_ok=True)
os.makedirs(root_directory + "/results/RNAD/Territory_Defence", exist_ok=True)

if seed == None:
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=99999999)

print(f"Using seed: {seed}")
rng = np.random.default_rng(seed)


env = TerritoryDefence(start_map = start_map, memory_length = env_memory_length,
                       negative_rewards=negative_rewards, instant_rewards=instant_rewards, 
                       max_steps = max_steps) 
agents = env.possible_agents
#stores the current trajectory
memories = {}
#stores the trajectory set for the next training iteration
training_memories = {}

#Tracking
ave_rewards = {}
saved_total_rewards = {}
saved_game_results = {"results":[]}
game_lengths = {"lengths":[]}
learning_stats = {"invader_towards_goal_average":[],
                  "defender_towards_invader_average":[]}

online_models = {}
reg_policies = {}

agent_to_team_mapping= {}
team_count_dict = {}

for agent in agents:
    num_inputs  =  math.prod(env.observation_space(agent)["observation"].shape)
    num_outputs =   env.action_space(agent).n.item()
    online_models[agent] = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)
    reg_policies[agent] = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)
    memories[agent] = trajectory_memory()
    training_memories[agent] = trajectory_memory()
    
    agent_to_team_mapping[agent] = agent
    team_count_dict[agent] = 1


    #tracking
    ave_rewards[agent] = []
    saved_total_rewards[agent] = []

rnad = RNAD(agents, reg_policies, regularised_reward_coefficient, 
            agent_to_team_mapping, team_count_dict, device)

total_steps = 0
for _ in range(num_reg_policy_updates):
    rnad.update_reg_policies(online_models)
    for i in tqdm(range(num_timesteps_per_reg_update)):
        total_steps+=1
        #Complete Game
        env.reset(seed=rng.integers(low=0, high=99999999))
        prev_trajectories = {}
        for agent in agents:
            prev_trajectories[agent] = []

        #tracking
        total_rewards = {}
        for agent in agents:
            total_rewards[agent] = 0
        zsteps = 0
        invader_towards_goal_steps = 0
        defender_towards_invader_steps = 0
        for agent in env.agent_iter():
            if agent == agents[0]:
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
                action, log_prob, value = online_models[agent].act(observation.flatten(), mask)
            env.step(action)
            
            #record in memory
            if prev_trajectories[agent] == []:
                prev_trajectories[agent] = [observation.flatten(), action, log_prob, value, mask]
            else:
                memories[agent].append(prev_trajectories[agent][0], prev_trajectories[agent][1], prev_trajectories[agent][2], reward-1, prev_trajectories[agent][3], termination, mask=prev_trajectories[agent][4])
                prev_trajectories[agent] = [observation.flatten(), action, log_prob, value, mask]

           
            
            total_rewards[agent] +=reward
            #Tracking
            if env.is_invader[agent]:
                invader_towards_goal_steps += env.invader_moved_closer_to_goal
            else:
                defender_towards_invader_steps += env.defender_moved_closer_to_attacker


        #Setup dictionaries for RNAD reward transformation
        observation_dict = {}
        action_dict = {}
        policy_dict = {}
        rewards_dict = {}
        mask_dict = {}
        log_probs_dict= {}
        values_dict = {}
        for agent in agents:
            observation_dict[agent] = memories[agent].states
            action_dict[agent] = memories[agent].actions
            policy_dict[agent] = online_models[agent]
            rewards_dict[agent] = memories[agent].rewards
            mask_dict[agent] = memories[agent].masks
            log_probs_dict[agent] = memories[agent].log_probs
            values_dict[agent] =  memories[agent].log_probs
        transformed_rewards_dict =rnad.get_transformed_rewards(observation_dict, action_dict,
                                        policy_dict,rewards_dict, mask_dict)
        
        #replace rewards
        for agent in agents:
            memories[agent].rewards = transformed_rewards_dict[agent].tolist()
            training_memories[agent].append_all(memories[agent])
            memories[agent].clear()

        #check if we have enough information for training
        if len(training_memories[agent]) >= max_memory_length:
            for agent in agents:
                advantages = compute_GAE(training_memories[agent].rewards,training_memories[agent].values, training_memories[agent].dones, gamma, gae_lambda)
                #Repeat Epoch No Times
                for epoch in range(epochs):
                    # Do learning / optimise actor and critic
                    online_models[agent].optimise( mini_batch_size, training_memories[agent].states, training_memories[agent].actions, training_memories[agent].log_probs, \
                                            training_memories[agent].values, advantages, critic_coefficient = 0.5, entropy_coef = 0.01, masks = training_memories[agent].masks)
            
            #clear memories
            training_memories[agent].clear()

        for agent in agents:
            #tracking
            ave_rewards[agent].append(total_rewards[agent]/zsteps)
            saved_total_rewards[agent].append(total_rewards[agent])

        learning_stats["defender_towards_invader_average"].append(defender_towards_invader_steps/zsteps)
        learning_stats["invader_towards_goal_average"].append(invader_towards_goal_steps/zsteps)
        saved_game_results["results"].append(env.game_state)
        game_lengths["lengths"].append(zsteps)


        #Checkpoint Model
        if total_steps%checkpoint_rate == 0:
            for agent in agents:
                online_models[agent].save(root_directory +f"/models/RNAD/Territory_Defence/{agent}-actor-{total_steps}-{file_label}.pth",
                                root_directory +f"/models/RNAD/Territory_Defence/{agent}-critic-{total_steps}-{file_label}.pth") 

df = pd.DataFrame(ave_rewards)
df.to_csv(root_directory +f"/results/RNAD/Territory_Defence/ave-rewards-{file_label}.csv", index=False)

df = pd.DataFrame(saved_game_results)
df.to_csv(root_directory +f"/results/RNAD/Territory_Defence/game-results-{file_label}.csv", index=False)

df = pd.DataFrame(saved_total_rewards)
df.to_csv(root_directory +f"/results/RNAD/Territory_Defence/total-rewards-{file_label}.csv", index=False)


df = pd.DataFrame(game_lengths)
df.to_csv(root_directory +f"/results/RNAD/Territory_Defence/game_lengths-{file_label}.csv", index=False)

df = pd.DataFrame(learning_stats)
df.to_csv(root_directory +f"/results/RNAD/Territory_Defence/learning-stats-{file_label}.csv", index=False)