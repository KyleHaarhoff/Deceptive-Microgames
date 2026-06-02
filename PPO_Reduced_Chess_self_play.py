#Imports
import copy
import os
import threading
import time

from reduced_chess.reduced_chess.env.reduced_chess import ReducedChess
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
mini_batch_size = 4  
max_memory_length = 50
gae_lambda = 0.95
epsilon_clip = 0.01
checkpoint_rate = 20000

#env params
env_memory_length = 2
num_timesteps = 500000
num_threads = 16 #Will divide timesteps over each thread

root_directory = os.path.dirname(os.path.abspath(__file__))
#Create Folders for Models

os.makedirs(root_directory + "/models", exist_ok=True)
os.makedirs(root_directory + "/models/PPO_self", exist_ok=True)
os.makedirs(root_directory + "/models/PPO_self/Reduced_Chess", exist_ok=True)
#Create Folders for Results
os.makedirs(root_directory + "/results", exist_ok=True)
os.makedirs(root_directory + "/results/PPO_self", exist_ok=True)
os.makedirs(root_directory + "/results/PPO_self/Reduced_Chess", exist_ok=True)
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
env = ReducedChess( memory_length = env_memory_length)
env.reset()

agents = env.agents
full_memories = {}


for agent in agents:
    num_inputs  =  math.prod(env.observation_space(agent)["observation"].shape)
    num_outputs =   env.action_space(agent).n.item()
    full_memories[agent] = trajectory_memory()

#Simply use on the single model now
online_model = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)

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
            for agent in agents:
                #Gather all the training data
                full_advantages = None
                first_memory = True
                for thread in threads:
                    memory = thread.memories[agent]
                    advantages = compute_GAE(memory.rewards,memory.values, memory.dones, gamma, gae_lambda)
                    if first_memory:
                        full_advantages = advantages
                        first_memory = False
                    else:
                        full_advantages = np.append(full_advantages, advantages)

                    #gather and clear the threads memory for the agent
                    full_memories[agent].append_all(memory)
                    memory.clear()

                memory = full_memories[agent]
                #Repeat Epoch No Times
                for epoch in range(epochs):
                    # Do learning / optimise actor and critic
                    online_model.optimise( mini_batch_size, memory.states, memory.actions, memory.log_probs, \
                                            memory.values, full_advantages, critic_coefficient = 0.5, entropy_coef = 0.01, masks = memory.masks)

                #reset memory
                memory.clear()

                #Checkpoint this model if needed
                if total_iter>=checkpoint_rate*zCheckpoints:
                    online_model.save(root_directory +f"/models/PPO_self/Reduced_Chess/{agent}-actor-{total_iter}-{file_label}.pth",
                                    root_directory +f"/models/PPO_self/Reduced_Chess/{agent}-critic-{total_iter}-{file_label}.pth") 
            if total_iter>=checkpoint_rate*zCheckpoints: 
                zCheckpoints+=1 # so that we checkpoint roughly at the rate
            #unpause all the threads and update all models
            for thread in threads:
                for agent in agents:
                    thread.models[agent].copy_online_net(online_model)
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
        
        self.env = ReducedChess(memory_length = env_memory_length)
        self.env.reset()

        agents = self.env.agents
        self.models = {}
        self.memories = {}

        for agent in agents:
            num_inputs  =  math.prod(env.observation_space(agent)["observation"].shape)
            num_outputs =   env.action_space(agent).n.item()
            self.models[agent] = Actor_Critic(num_inputs, num_outputs, lr, device, epsilon_clip)
            self.models[agent].copy_online_net(online_model)
            self.memories[agent] = trajectory_memory()
        #For checkpointing step progress
        self.unreported_timesteps = 0
        #for_plotting 
        self.ave_rewards = {}
        self.saved_total_rewards = {}
        self.saved_game_results = {"results":[],
                                   "king_captured":[],
                                   "final_rank":[],
                                   "pieces_revealed":[]}
        for agent in agents:
            self.ave_rewards[agent] = []
            self.saved_total_rewards[agent] = []

    '''
    Function to run the model 
    Will run as normal and once the memory is full, it will wait until all threads are also ready for training
    The thread will then copy over the updated Network Weights and continue gathering experience
    '''

    def run(self):
        for i in tqdm(range(num_timesteps//num_threads)):#COLLECT TRAJECTORIES
            self.unreported_timesteps += 1
            self.env.reset(seed=self.rng.integers(low=0, high=99999999)) 
            agents = copy.deepcopy(self.env.agents)
            
            prev_trajectories = {}
            for agent in agents:
                prev_trajectories[agent] = []

            #tracking
            total_rewards = {}
            for agent in agents:
                total_rewards[agent] = 0
            zsteps = 0

            for agent in self.env.agent_iter():
                zsteps+= (1/len(agents))
                observation, reward, termination, truncation, info = self.env.last()
                mask = observation["action_mask"]
                observation = observation["observation"]
                if termination or truncation:
                    action = None
                else:
                    #Get Actions
                    action, log_prob, value = self.models[agent].act(observation.flatten(), mask)
                    
                
                self.env.step(action)

                #tracking
                total_rewards[agent] += reward

                #
                if prev_trajectories[agent] == []:
                    prev_trajectories[agent] = [observation.flatten(), action, log_prob, value, mask]
                else:
                    self.memories[agent].append(prev_trajectories[agent][0], prev_trajectories[agent][1], prev_trajectories[agent][2], reward-1, prev_trajectories[agent][3], termination, mask=prev_trajectories[agent][4])
                    prev_trajectories[agent] = [observation.flatten(), action, log_prob, value, mask]

            
                if len(self.memories[agent]) >= max_memory_length:
                    #pause event
                    self.pause()
                    #add self to training count
                    synchronised_training(self.unreported_timesteps)
                    self.unreported_timesteps = 0
                    #wait until training is complete and we unpause
                    self._pause_event.wait()

            #reset for new round
            for agent in agents:
                prev_trajectories[agent] = []
                #tracking
                self.ave_rewards[agent].append(total_rewards[agent]/zsteps)
                self.saved_total_rewards[agent].append(total_rewards[agent])
            self.saved_game_results["results"].append(self.env.game_state)
            self.saved_game_results["king_captured"].append(self.env.king_capture)
            self.saved_game_results["final_rank"].append(self.env.final_rank)
            self.saved_game_results["pieces_revealed"].append(self.env.pieces_revealed)
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
for agent in agents:
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
for agent in agents:
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


results = []
captures = []
final_ranks = []
pieces_revealed = []

for thread in threads:
    results.append(thread.saved_game_results["results"])
    captures.append(thread.saved_game_results["king_captured"])
    final_ranks.append(thread.saved_game_results["final_rank"])
    pieces_revealed.append(thread.saved_game_results["pieces_revealed"])
    
results = [
        x
        for group in zip_longest(*results)
        for x in group
        if x is not None
    ]
captures = [
        x
        for group in zip_longest(*captures)
        for x in group
        if x is not None]
final_ranks = [
        x
        for group in zip_longest(*final_ranks)
        for x in group
        if x is not None]
pieces_revealed = [
        x
        for group in zip_longest(*pieces_revealed)
        for x in group
        if x is not None]


saved_game_results = {"results":results,
                      "king_captured":captures,
                      "final_rank":final_ranks,
                      "pieces_revealed":pieces_revealed,
                      } 


df = pd.DataFrame(ave_rewards)
df.to_csv(root_directory +f"/results/PPO_self/Reduced_Chess/ave-results-{file_label}.csv", index=False)

df = pd.DataFrame(saved_game_results)
df.to_csv(root_directory +f"/results/PPO_self/Reduced_Chess/game-res-{file_label}.csv", index=False)

df = pd.DataFrame(saved_total_rewards)
df.to_csv(root_directory +f"/results/PPO_self/Reduced_Chess/results-{file_label}.csv", index=False)

