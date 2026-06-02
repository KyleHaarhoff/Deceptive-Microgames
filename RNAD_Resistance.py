#Imports
import copy
import os
import threading
import time

from the_resistance.the_resistance.env.the_resistance import TheResistance
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
file_label = "run_1" 
#Cuda
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

#Hyper params:
gamma = 0.95  # Discount factor
lr = 0.0001  
epochs = 5  
mini_batch_size = 8  
max_memory_length = 50
gae_lambda = 0.95
epsilon_clip = 0.01
checkpoint_rate = 10000

num_agents = 5
num_timesteps = 100000
#RNAD params
regularised_reward_coefficient = 0.2
num_reg_policy_updates = 100
num_timesteps_per_reg_update = num_timesteps//num_reg_policy_updates


# 
root_directory = os.path.dirname(os.path.abspath(__file__))
#Create Folders for Models

os.makedirs(root_directory + "/models", exist_ok=True)
os.makedirs(root_directory + "/models/RNAD", exist_ok=True)
os.makedirs(root_directory + "/models/RNAD/The_Resistance", exist_ok=True)
#Create Folders for Results
os.makedirs(root_directory + "/results", exist_ok=True)
os.makedirs(root_directory + "/results/RNAD", exist_ok=True)
os.makedirs(root_directory + "/results/RNAD/The_Resistance", exist_ok=True)

if seed == None:
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=99999999)

print(f"Using seed: {seed}")
rng = np.random.default_rng(seed)


env = TheResistance(num_agents=num_agents)

agents = env.possible_agents
reg_policies = {}
policy_models = {}
agent_to_model_dict = {}
#stores the current trajectory
memories = {}
#stores the trajectory set for the next training iteration
training_memories = {}

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

#Create Models
num_inputs  =  math.prod(env.observation_space("spy1").shape)
num_outputs =   env.action_space("spy1").n.item()

policy_models["spy"] = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)

#Agent
num_inputs  =  math.prod(env.observation_space("agent1").shape)
num_outputs =   env.action_space("agent1").n.item()

policy_models["agent"] = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)


agent_to_team_mapping= {}
team_count_dict = {}
for agent in agents:
    #tracking
    ave_rewards[agent] = []
    saved_total_rewards[agent] = []
    memories[agent] = trajectory_memory()
    training_memories[agent] = trajectory_memory()

    #RNAD
    num_inputs  =  math.prod(env.observation_space(agent).shape)
    num_outputs =   env.action_space(agent).n.item()
    reg_policies[agent] = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)

    agent_to_team_mapping[agent] = agent_to_label_map[agent]
    team_count_dict[agent] = 2 if "spy" in agent else 3

rnad = RNAD(agents, reg_policies, regularised_reward_coefficient, 
            agent_to_team_mapping, team_count_dict, device)

total_steps = 0
for i in (range(num_reg_policy_updates)):
    #Update Reg Policy
    for agent in agents:
        agent_to_model_dict[agent] = policy_models[agent_to_label_map[agent]]
    rnad.update_reg_policies(agent_to_model_dict)

    for j in tqdm(range(num_timesteps_per_reg_update)):
        #COLLECT TRAJECTORIES
        observations, _ = env.reset(seed=rng.integers(low=0, high=99999999)) 

        episode_ended_flag = False
        #tracking
        total_rewards = {}
        for agent in agents:
            total_rewards[agent] = 0
        zsteps = 0
        while not episode_ended_flag:
            zsteps+=1
            #Get Actions
            actions = {}
            log_probs = {}
            values = {}
            for agent in agents: 
                agent_label = agent_to_label_map[agent]
                action, log_prob, value = policy_models[agent_label].act(observations[agent].flatten())
                actions[agent] = action
                log_probs[agent] = log_prob
                values[agent] = value


            next_state, rewards, dones, truncations, infos = env.step(actions)

            #tracking
            for agent in agents:
                total_rewards[agent] += rewards[agent]
            
            #store tansitions
            for agent in agents: 
                memories[agent].append(observations[agent].flatten(), actions[agent], log_probs[agent], rewards[agent], values[agent], dones[agent])
            #update state
            observations = next_state
            #check termination - no truncations occur in this env
            for agent in agents:
                episode_ended_flag = episode_ended_flag or dones[agent]


        #Tranform Rewards

        #Optimise Models
        #Setup dictionaries for RNAD reward transformation
        observation_dict = {}
        action_dict = {}
        policy_dict = {}
        rewards_dict = {}
        log_probs_dict= {}
        values_dict = {}
        for agent in agents:
            agent_label = agent_to_label_map[agent]
            observation_dict[agent] = memories[agent].states
            action_dict[agent] = memories[agent].actions
            policy_dict[agent] = policy_models[agent_label]
            rewards_dict[agent] = memories[agent].rewards
            log_probs_dict[agent] = memories[agent].log_probs
            values_dict[agent] =  memories[agent].log_probs
        transformed_rewards_dict =rnad.get_transformed_rewards(observation_dict, action_dict,
                                        policy_dict,rewards_dict, None)

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
                    agent_label = agent_to_label_map[agent]
                    # Do learning / optimise actor and critic
                    policy_models[agent_label].optimise( mini_batch_size, training_memories[agent].states, training_memories[agent].actions, training_memories[agent].log_probs, \
                                            training_memories[agent].values, advantages, critic_coefficient = 0.5, entropy_coef = 0.01)
                
                #clear memories
                training_memories[agent].clear()

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


        #Checkpoint Model
        for agent_label in agent_labels:
            if total_steps%checkpoint_rate == 0:
                policy_models[agent_label].save(root_directory +f"/models/RNAD/The_Resistance/{agent_label}-actor-{i}-{file_label}.pth",
                                    root_directory +f"/models/RNAD/The_Resistance/{agent_label}-critic-{i}-{file_label}.pth") 

df = pd.DataFrame(ave_rewards)
df.to_csv(root_directory +f"/results/RNAD/The_Resistance/ave-rewards-{file_label}.csv", index=False)

df = pd.DataFrame(saved_game_results)
df.to_csv(root_directory +f"/results/RNAD/The_Resistance/game-results-{file_label}.csv", index=False)

df = pd.DataFrame(saved_total_rewards)
df.to_csv(root_directory +f"/results/RNAD/The_Resistance/total-rewards-{file_label}.csv", index=False)


df = pd.DataFrame(game_lengths)
df.to_csv(root_directory +f"/results/RNAD/The_Resistance/game_lengths-{file_label}.csv", index=False)