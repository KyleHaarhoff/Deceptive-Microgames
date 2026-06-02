#Imports
import copy
import os
import threading
import time

from the_resistance.the_resistance.env.the_resistance import TheResistance
from RL_Implementations.PPO import compute_GAE, trajectory_memory, Actor_Critic
import pandas as pd

import torch
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

#Hyper params:
gamma = 0.95  # Discount factor
lr = 0.0001  
epochs = 5  
mini_batch_size = 8  
max_memory_length = 50
gae_lambda = 0.95
epsilon_clip = 0.01
checkpoint_rate = 50

num_agents = 5
num_timesteps = 100000
num_threads = 16 #Will divide timesteps over each thread

root_directory = os.path.dirname(os.path.abspath(__file__))
#Create Folders for Models

os.makedirs(root_directory + "/models", exist_ok=True)
os.makedirs(root_directory + "/models/PPO", exist_ok=True)
os.makedirs(root_directory + "/models/PPO/The_Resistance", exist_ok=True)
#Create Folders for Results
os.makedirs(root_directory + "/results", exist_ok=True)
os.makedirs(root_directory + "/results/PPO", exist_ok=True)
os.makedirs(root_directory + "/results/PPO/The_Resistance", exist_ok=True)
#Create RNG seeds for each 

if seed == None:
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=99999999)

print(f"Using seed: {seed}")
rng = np.random.default_rng(seed)

RNGs = []
for i in range(num_threads):
    RNGs.append(np.random.default_rng(rng.integers(low=0, high=99999999)))


#Create Models
env = TheResistance( num_agents=num_agents)
env.reset()

agent_labels = ["spy", "agent"]
online_models = {}
full_memories = {}

# We will only use a single model for spies and a single model for normal 'agents'

num_inputs  =  math.prod(env.observation_space("spy1").shape)
num_outputs =   env.action_space("spy1").n.item()
online_models["spy"] = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)
full_memories["spy"] = trajectory_memory()


num_inputs  =  math.prod(env.observation_space("agent1").shape)
num_outputs =   env.action_space("agent1").n.item()
online_models["agent"] = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)
full_memories["agent"] = trajectory_memory()

#Create Threads
threads = []

shared_counter = 0
done_threads = 0
total_iter = 0
zCheckpoints = 1
counter_lock = threading.RLock() # Also known as a mutex

def synchronised_training(iteration):
    global shared_counter
    global done_threads
    global total_iter
    global zCheckpoints
    with counter_lock:
        #Critical Section
        shared_counter += 1
        total_iter+=iteration
        #If all threads are stopped then train
        if shared_counter >= num_threads-done_threads:
            for agent_label in agent_labels:
                #Gather all the training data
                for thread in threads:
                    memory = thread.memories[agent_label]
                    #advantages have already been calculated
                    #gather and clear the threads memory for the agent
                    full_memories[agent_label].append_all(memory)
                    memory.clear()

                memory = full_memories[agent_label]
                #Repeat Epoch No Times
                for epoch in range(epochs):
                    # Do learning / optimise actor and critic
                    online_models[agent_label].optimise( mini_batch_size, memory.states, memory.actions, memory.log_probs, \
                                            memory.values, memory.advantages, critic_coefficient = 0.5, entropy_coef = 0.01)

                #reset memory
                memory.clear()

                #Checkpoint this model if needed
                if total_iter>=checkpoint_rate*zCheckpoints:
                    online_models[agent_label].save(root_directory +f"/models/PPO/The_Resistance/{agent_label}-actor-{total_iter}-{file_label}.pth",
                                    root_directory +f"/models/PPO/The_Resistance/{agent_label}-critic-{total_iter}-{file_label}.pth") 
            if total_iter>=checkpoint_rate*zCheckpoints: 
                zCheckpoints+=1 # so that we checkpoint roughly at the rate
            #unpause all the threads and update all models
            for thread in threads:
                for agent_label in agent_labels:
                    thread.models[agent_label].copy_online_net(online_models[agent_label])
                thread.resume()
            
            #reset counter
            shared_counter = 0

def finish_stepping():
    global done_threads
    global shared_counter
    with counter_lock:
        done_threads+=1
        #Need to account for the case that threads finish while others are waiting 
        if shared_counter >= num_threads-done_threads:
            synchronised_training(0)


class RL_Agent_Thread(threading.Thread):
    def __init__(self, rng):
        super().__init__()
        self._pause_event = threading.Event()
        self._pause_event.set() 
        #set rng
        self.rng = rng

        #set up environment and models
        
        self.env = TheResistance(num_agents = num_agents)
        self.env.reset()

        self.agent_labels = ["spy","agent"]
        self.models = {}
        self.memories = {}

        #set up models
        for agent_label in self.agent_labels:
            num_inputs  =  math.prod(env.observation_space(agent_label+"1").shape)
            num_outputs =   env.action_space(agent_label+"1").n.item()
            self.models[agent_label] = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)
            self.models[agent_label].copy_online_net(online_models[agent_label])
            self.memories[agent_label] = trajectory_memory()
        
        #set up memories
        self.agents = self.env.agents
        for agent in self.agents:
            self.memories[agent] = trajectory_memory()

        #mapping agent to correct model
        self.agent_to_label_map = {}
        for agent in self.agents:
            if "spy" in agent:
                self.agent_to_label_map[agent] = "spy"
            else:
                self.agent_to_label_map[agent] = "agent" 

        #For checkpointing step progress
        self.unreported_timesteps = 0
        #for_plotting 
        self.ave_rewards = {}
        self.saved_total_rewards = {}
        self.saved_game_results = {"results":[],
                                   "total_spy_to_spy_votes":[],
                                   "total_spy_to_spy_votes_yes":[],
                                   "total_infested_missions":[],
                                   "total_infested_missions_sabotaged":[],
                                   "total_agent_not_in_vote":[],
                                   "total_agent_not_in_vote_yes":[]}
        for agent in self.agents:
            self.ave_rewards[agent] = []
            self.saved_total_rewards[agent] = []

    '''
    Function to run the model 
    Will run as normal and once the memory is full, it will wait until all threads are also ready for training
    The thread will then copy over the updated Network Weights and continue gathering experience
    '''

    def run(self):
        memory_length = 0
        for i in tqdm(range(num_timesteps//num_threads)): #COLLECT TRAJECTORIES
            self.unreported_timesteps += 1
            '''
            '''
            #COLLECT TRAJECTORIES
            observations, _ = self.env.reset(seed=self.rng.integers(low=0, high=99999999)) 

            episode_ended_flag = False
            #tracking
            total_rewards = {}
            for agent in self.agents:
                total_rewards[agent] = 0
            zsteps = 0
            while not episode_ended_flag:
                zsteps+=1
                #Get Actions
                actions = {}
                log_probs = {}
                values = {}
                for agent in self.agents: 
                    agent_label = self.agent_to_label_map[agent]
                    action, log_prob, value = self.models[agent_label].act(observations[agent].flatten())
                    actions[agent] = action
                    log_probs[agent] = log_prob
                    values[agent] = value
                
                #Take Action
                next_state, rewards, dones, truncations, infos = self.env.step(actions)

                #tracking
                for agent in self.agents:
                    total_rewards[agent] += rewards[agent]
                
                #store tansitions
                memory_length+=1
                for agent in self.agents: 
                    self.memories[agent].append(observations[agent].flatten(), actions[agent], log_probs[agent], rewards[agent], values[agent], dones[agent])
                #update state
                observations = next_state
                #check termination - no truncations occur in this env
                for agent in self.agents:
                    episode_ended_flag = episode_ended_flag or dones[agent]

                #check if it is time to learn
                if memory_length >= max_memory_length:
                    #first organise each of the individual memories for the specific models
                    for agent in self.agents: 
                        memory = self.memories[agent]
                        #CALC ADVANTAGES
                        advantages = compute_GAE(memory.rewards,memory.values, memory.dones, gamma, gae_lambda)
                        agent_label = self.agent_to_label_map[agent]
                        memory.advantages = advantages
                        self.memories[agent_label].append_all(memory)
                        #reset memory
                        memory.clear()
                    memory_length = 0

                    
                    #pause event
                    self.pause()
                    #add self to training count
                    synchronised_training(self.unreported_timesteps)
                    #wait until training is complete and we unpause
                    self._pause_event.wait()
                    self.unreported_timesteps = 0
            #reset for new round
            for agent in self.agents:
                #tracking
                self.ave_rewards[agent].append(total_rewards[agent]/zsteps)
                self.saved_total_rewards[agent].append(total_rewards[agent])
            self.saved_game_results["results"].append(self.env.game_state)
            self.saved_game_results["total_spy_to_spy_votes"].append(self.env.total_spy_to_spy_votes)
            self.saved_game_results["total_spy_to_spy_votes_yes"].append(self.env.total_spy_to_spy_votes_yes)
            self.saved_game_results["total_infested_missions"].append(self.env.total_infested_missions)
            self.saved_game_results["total_infested_missions_sabotaged"].append(self.env.total_infested_missions_sabotaged)
            self.saved_game_results["total_agent_not_in_vote"].append(self.env.total_agent_not_in_vote)
            self.saved_game_results["total_agent_not_in_vote_yes"].append(self.env.total_agent_not_in_vote_yes)
        finish_stepping()
    def pause(self):
        self._pause_event.clear()  # Clear the flag to block the thread
        
    def resume(self):
        self._pause_event.set()    # Set the flag to unblock the thread

for i in range(num_threads):
    my_thread = RL_Agent_Thread(RNGs[i])
    my_thread.start()
    threads.append(my_thread)



#wait for all threads to finish
for thread in threads:
    thread.join()   

#Save result information for viewing
ave_rewards = {}

for agent in env.possible_agents:
    #code to interleave results for each round correctly
    lists = []
    for thread in threads:
        lists.append(thread.ave_rewards[agent])

    result = [
        x
        for group in zip_longest(*lists)
        for x in group
        if x is not None
    ]
    ave_rewards[agent] = result

saved_total_rewards = {}
for agent in env.possible_agents:
    #code to interleave results for each round correctly
    lists = []
    for thread in threads:
        lists.append(thread.saved_total_rewards[agent])

    result = [
        x
        for group in zip_longest(*lists)
        for x in group
        if x is not None
    ]
    saved_total_rewards[agent] = result

def tracking_conversion(saved_game_results, key, threads):
    pass
    lists = []
    for thread in threads:
        lists.append(thread.saved_game_results[key])
    result = [
            x
            for group in zip_longest(*lists)
            for x in group
            if x is not None
        ]
    saved_game_results[key] = result 

saved_game_results = {}

tracking_conversion(saved_game_results, "results", threads)
tracking_conversion(saved_game_results, "total_spy_to_spy_votes", threads)
tracking_conversion(saved_game_results, "total_spy_to_spy_votes_yes", threads)
tracking_conversion(saved_game_results, "total_infested_missions", threads)
tracking_conversion(saved_game_results, "total_infested_missions_sabotaged", threads)
tracking_conversion(saved_game_results, "total_agent_not_in_vote", threads)
tracking_conversion(saved_game_results, "total_agent_not_in_vote_yes", threads)


df = pd.DataFrame(ave_rewards)
df.to_csv(root_directory +f"/results/PPO/The_Resistance/ave-results-{file_label}.csv", index=False)

df = pd.DataFrame(saved_game_results)
df.to_csv(root_directory +f"/results/PPO/The_Resistance/game-res-{file_label}.csv", index=False)

df = pd.DataFrame(saved_total_rewards)
df.to_csv(root_directory +f"/results/PPO/The_Resistance/results-{file_label}.csv", index=False)

