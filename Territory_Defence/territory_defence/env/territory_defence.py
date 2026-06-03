import functools
import random
import copy
import os
import pygame

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, Box, Dict
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector





class TerritoryDefence(AECEnv):
    metadata = {"render_modes": ["human"], "name": "territory_defence"}

    def __init__(self, render_mode = None, start_map = None, 
                 num_invaders:int = 1, num_defenders:int =1, memory_length = 0, 
                 instant_rewards = True, negative_rewards = False, max_steps = 100):
        '''
        Initialise the environment with a default map
        Allow for the map to be user defined

        W = wall
        G = goal
        ' ' = floor
        I = invader start
        D = defender start
        C = Ice

        Actions are
        0 = left
        1 = up
        2 = right
        3 = down

        '''
        self.max_steps = max_steps
        self.render_mode = render_mode
        #Start variables
        if start_map == None:
            self.start_map = [
            [' ',' ',' ','D',' ',' ', ' '],
            ['G','W',' ','G',' ','W', 'G'],
            [' ','W',' ',' ',' ','W', ' '],
            [' ','W',' ',' ',' ','W', ' '],
            [' ',' ',' ',' ',' ',' ', ' '],
            [' ',' ',' ',' ',' ',' ', ' '],
            [' ',' ',' ','I',' ',' ', ' ']
        ]
        else:
            self.start_map= copy.deepcopy(start_map)

        self.goal_positions = [(i, j) for i, row in enumerate(self.start_map) for j, val in enumerate(row) if val == 'G']
        self.num_goals = len(self.goal_positions)
        self.selected_goal = -1 # Used in MLP
        self.invader_positions = [(i, j) for i, row in enumerate(self.start_map) for j, val in enumerate(row) if val == 'I']
        self.defender_positions = [(i, j) for i, row in enumerate(self.start_map) for j, val in enumerate(row) if val == 'D']
        self.wall_positions = [(i, j) for i, row in enumerate(self.start_map) for j, val in enumerate(row) if val == 'W']
        self.ice_positions = [(i, j) for i, row in enumerate(self.start_map) for j, val in enumerate(row) if val == 'C']
        
        #Run variables
        self.num_invaders = num_invaders 
        self.num_defenders = num_defenders  
        self.instant_rewards = instant_rewards # Gives reward every step based on proximity to goal
        self.negative_rewards = negative_rewards # Gives -1 reward every step

        #Checks
        if not (num_invaders>=1):
            raise ValueError(f"num_invaders must be greater than, got {num_invaders}")
        if not (num_defenders>=1):
            raise ValueError(f"num_defenders must be between 0 and 8, got {num_defenders}")
            
        if not (len(self.goal_positions)>=1):
            raise ValueError(f"there must be at least one goal, found {len(self.goal_positions)}")
        if not (len(self.invader_positions)>=num_invaders):
            raise ValueError(f"the number of invader positions must be >= the num_invaders, found {len(self.invader_positions)} positions for {num_invaders} invaders")
        if not (len(self.defender_positions)>=num_defenders):
            raise ValueError(f"the number of defender positions must be >= the num_defenders, found {len(self.defender_positions)} positions for {num_defenders} defenders")
            
        self.possible_agents = []
        self.is_invader = {}
        for i in range(0, num_invaders):
            self.possible_agents.append(f"invader_{i}")
            self.is_invader[f"invader_{i}"] = True
        for i in range(0, num_defenders):
            self.possible_agents.append(f"defender_{i}")
            self.is_invader[f"defender_{i}"] = False
        

        #start positions
        self.agent_positions = {}
        for player in self.possible_agents:
            self.agent_positions[player] = (None, None)

        
        if self.render_mode == "human":
            #used to keep track of which images need updating 
            self.prev_coords = []

        self.current_map = None
        #Tracking
        self.game_state = 0
        self.invader_moved_closer_to_goal = 0
        self.defender_moved_closer_to_attacker = 0
        self.timestep = None

        #memory
        self.memory_length = memory_length
        if memory_length > 0:
            self.memory = np.zeros((len(self.start_map), len(self.start_map[0]), memory_length*2), dtype=int) #memory length x2 since we need two bits for attackers and defenders


    def push_to_memory(self, item):
        self.memory = np.roll(self.memory , shift=-1, axis=2)  #shift left
        self.memory [:, :, -1] = item                #insert at the end

    # Observation space
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # 6 needed to return current state of the board, memory length provides previous information
        return Dict({
            "observation": Box(
                low=0,
                high=1,
                shape=(len(self.start_map), len(self.start_map[0]), 6+self.memory_length*2),
                dtype=np.int8
            ),
            "action_mask": MultiBinary(4),
        })
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)


    def reset(self, seed=None, options=None):

        # seeding
        if seed is not None:
            random.seed(int(seed))
            self.np_random, self.np_random_seed = seeding.np_random(int(seed))

        
        #Reset the map
        self.current_map = copy.deepcopy(self.start_map)
        #choose the goal        
        self.goal_x, self.goal_y =  random.choice(self.goal_positions)
        self.selected_goal = self.goal_positions.index((self.goal_x, self.goal_y))
        self.agents = copy.copy(self.possible_agents)
        self.timestep = 0

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        #Tracking
        self.game_state = 0
        self.invader_moved_closer_to_goal = 0
        self.defender_moved_closer_to_attacker = 0


        

        
        #choose agents positions
        #order is invaders then defenders
        invader_random_positions = random.sample(self.invader_positions,  self.num_invaders)
        defender_random_positions = random.sample(self.defender_positions,  self.num_defenders)
        
        for player in self.agents:
            if self.is_invader[player]:
                coord = invader_random_positions.pop()
                self.agent_positions[player] = (coord[0], coord[1])
            else:
                coord = defender_random_positions.pop()
                self.agent_positions[player] = (coord[0], coord[1])
        
        
        
        #Load images at correct size
        if self.render_mode == "human":
            self.first_render = True
            # Initialize Pygame
            pygame.init()

            # Get the directory of the current Python file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Screen size
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 500, 500
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Territory Defence")
            
            rows, cols = len(self.current_map), len(self.current_map[0])
            
            self.wall_image = pygame.image.load(current_dir+"/wall.png")
            self.ice_image = pygame.image.load(current_dir+"/ice.png")
            self.invader_image = pygame.image.load(current_dir+"/invader.png")
            self.defender_image = pygame.image.load(current_dir+"/defender.png")
            self.floor_image = pygame.image.load(current_dir+"/floor.png")
            self.goal_image = pygame.image.load(current_dir+"/goal.png")
            self.grey_goal_image = pygame.image.load(current_dir+"/grey_goal.png")
            self.tracker = pygame.image.load(current_dir+"/tracker.png")
            self.tracker_faded = pygame.image.load(current_dir+"/tracker_faded.png")
            
            self.wall_image = pygame.transform.scale(self.wall_image, (self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows))  
            self.ice_image = pygame.transform.scale(self.ice_image, (self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows))  
            self.invader_image = pygame.transform.scale(self.invader_image, (self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows))  
            self.defender_image = pygame.transform.scale(self.defender_image, (self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows))  
            self.floor_image = pygame.transform.scale( self.floor_image, (self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows))  
            self.goal_image = pygame.transform.scale(self.goal_image, (self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows))  
            self.grey_goal_image = pygame.transform.scale(self.grey_goal_image, (self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows))
            self.tracker = pygame.transform.scale(self.tracker, (self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows))  
            self.tracker_faded = pygame.transform.scale(self.tracker_faded, (self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows))
        
            self.faded_locations = {}
            for agent in self.possible_agents:
                self.faded_locations[agent] = []
        #cycle to next agent
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()


    def observe(self, agent):
        '''
        provide obs as np array of grid positions, 
        each position is a binary array what observation exists at that tile
        [mypos, invaderpos, enemypos, wall, ice, goal]
        '''
        #create array
        obs  = np.zeros((len(self.start_map), len(self.start_map[0]), 6), dtype=np.int8)
        #my position
        x = self.agent_positions[agent][0]
        y = self.agent_positions[agent][1]
        obs[x][y][0] = 1
        #other agent positions
        for other_player in self.agents:
            if not(other_player == agent):
                x = self.agent_positions[other_player][0]
                y = self.agent_positions[other_player][1]
                #invader vs defender position
                if self.is_invader[other_player]:
                    obs[x][y][1] = 1
                else:
                    obs[x][y][2] = 1

        #walls
        for x, y in self.wall_positions:
            obs[x][y][3] = 1
        #ice
        for x, y in self.ice_positions:
            obs[x][y][4] = 1

        #attackers see selected goal
        #defenders see all goals

        if not self.is_invader[agent]:
            for x,y in self.goal_positions:
                obs[x][y][5] = 1
        else:
            #selected goal
            obs[self.goal_x][self.goal_y][5] = 1
        

        if self.memory_length >0:
            observation = np.zeros((len(self.start_map), len(self.start_map[0]), 6+self.memory_length*2), dtype=np.int8)
            observation[:,:,0:6] = obs
            observation[:,:,6:] = self.memory.copy()
            obs = observation 
        

        action_mask = self.get_action_mask(self.agent_selection)

        return {
            "observation": obs,
            "action_mask": action_mask
        }

   

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        
        #Check if this is the first agent -- I.E. record gamestate in memory
        
        if self._agent_selector.is_first():
            if self.memory_length > 0:

                attackers_locations  = np.zeros((len(self.start_map), len(self.start_map[0])))
                defenders_locations  = np.zeros((len(self.start_map), len(self.start_map[0])))
                #other agent positions
                for player in self.agents:
                    x,y = self.agent_positions[player]
                    if self.is_invader[player]:
                        attackers_locations[x][y] = 1
                    else:
                        defenders_locations[x][y] = 1
                #get and append attackers
                self.push_to_memory(attackers_locations)
                #get and append defenders
                self.push_to_memory(defenders_locations)
        '''
        Actions are
        0 = left
        1 = up
        2 = right
        3 = down

        '''
        self.prev_coords = []


        #Movement logic
        #my position
        x = self.agent_positions[agent][0]
        y = self.agent_positions[agent][1]

        self.prev_coords.append((x,y))
            
        while True:
            #calculate new position
            if action == 0:
                y-=1
            if action == 1:
                x-=1
            if action == 2:
                y+=1
            if action == 3:
                x+=1

            
            flag = False
            #check that it is in bounds
            if not(y>=0 and y<len(self.start_map[0]) and x>=0 and x<len(self.start_map)):
                flag = True
                
            #Check that its not a wall
            if (x,y) in self.wall_positions: 
                flag = True
            #Check  that another agent is not already there
            if (x,y) in self.agent_positions.values(): 
                flag = True

            #Make the move
            if not flag:
                self.agent_positions[agent] = (x, y)
            if flag or (x,y) not in self.ice_positions: #if we are not on ice, or if we hit the wall while slipping on ice then stop
                break

        # check rewards and terminations
        # agent should start again at 0
        defender_rewards = 0

        #reward
         #Invader reaches goal
        if self.is_invader[agent]:
            x = self.agent_positions[agent][0]
            y = self.agent_positions[agent][1]

            if x== self.goal_x and y == self.goal_y:
                self.rewards[agent] += 10
                defender_rewards-=10
                self.terminations[agent] = True
                self.game_state = 1
            elif self.instant_rewards:
                self.rewards[agent] = 1/(abs(x - self.goal_x) + abs(y - self.goal_y)+1)
            
            #Tracking
            prev_x, prev_y = self.prev_coords[0]
            if abs(self.goal_x-prev_x) > abs(self.goal_x - x) or abs(self.goal_y - prev_y) > abs(self.goal_y - y):
                self.invader_moved_closer_to_goal = 1
            else:
                self.invader_moved_closer_to_goal = 0

            
        
        if self.terminations[agent] != True:
            #Is defender adjacent to invader
            x = self.agent_positions[agent][0]
            y = self.agent_positions[agent][1]

            #agent specific incentive to move closer to invaders
            min_dist = len(self.start_map) + len(self.start_map)
            #Tracking
            min_other_x = 0
            min_other_y = 0
            for other_player in self.agents:
                flag = (self.is_invader[agent] and not self.is_invader[other_player]) or \
                    (not self.is_invader[agent] and self.is_invader[other_player])

                if flag and not(self.terminations[other_player]):
                    other_x = self.agent_positions[other_player][0]
                    other_y = self.agent_positions[other_player][1]
                    #check adjacency 
                    if abs(x - other_x) + abs(y - other_y) == 1:
                        if(self.is_invader[agent]):
                            self.rewards[agent] = -10
                            self.terminations[agent] = True
                        else:
                            self.rewards[other_player] = -10
                            self.terminations[other_player] = True
                        defender_rewards+=10
                        min_dist = 0
                        self.game_state = -1 

                    else:
                        if min_dist > abs(x - other_x) + abs(y - other_y):
                            min_other_x = other_x
                            min_other_y = other_y

                        min_dist = min(min_dist, abs(x - other_x) + abs(y - other_y))
            if(not self.is_invader[agent]):
                if min_dist != 0 and self.instant_rewards:
                    self.rewards[agent] = 1/(min_dist+1)

                #Tracking
                prev_x, prev_y = self.prev_coords[0]
                if abs(min_other_x-prev_x) > abs(min_other_x - x) or abs(min_other_y - prev_y) > abs(min_other_y - y):
                    self.defender_moved_closer_to_attacker = 1
                else:
                    self.defender_moved_closer_to_attacker = 0

        #Check if negative rewards are applied
        if self.negative_rewards:
                self.rewards[agent] -= 1


        #assign defender rewards
        for player in self.agents:
            #Invader reaches goal
            if not self.is_invader[player]:
                self.rewards[player] += defender_rewards  


        


        # check terms and truncs
        # Check if all invaders are out of play
        all_done = True
        for player in self.agents:
            if self.is_invader[player] and not(self.terminations[player]):
                all_done = False

        if all_done:
            self.terminations = {a: True for a in self.agents}

        if self._agent_selector.is_last():
            self.timestep += 1
        # The truncations dictionary must be updated for all players.
        # Check truncation conditions
        if self.timestep > self.max_steps:
            self.game_state = -1 
            self.terminations = {a: True for a in self.agents}
            self.truncations = {a: True for a in self.agents}
            #give defenders the win in timeout
            for player in self.agents:
                if self.is_invader[player]:
                    self.rewards[player] += -10 
                else:
                    self.rewards[player] += 10


        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


        if self.render_mode == "human":
            self.faded_locations[agent].insert(0, self.prev_coords[0])
            self.render()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()



    
    def render(self, frame_delay = 200, return_image_arr = False):
        rows, cols = len(self.start_map), len(self.start_map[0])
        if self.render_mode == "human":
            cell_width, cell_height = self.SCREEN_WIDTH/cols, self.SCREEN_HEIGHT/rows
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass
            if self.first_render:
                self.first_render = False
                self.screen.fill((255, 255, 255))
    
                
                # Draw the grid of images
                for row in range(rows):
                    for col in range(cols):
                        x = col * cell_width
                        y = row * cell_height
                        if self.start_map[row][col] == 'W':
                            self.screen.blit(self.wall_image, (x, y))
                        elif self.start_map[row][col] == 'C':
                            self.screen.blit(self.ice_image, (x, y))
                        else:
                            self.screen.blit(self.floor_image, (x, y))
                #add goals
                for (x, y) in self.goal_positions:
                    self.screen.blit(self.grey_goal_image, (y* cell_width, x* cell_height))
    
                self.screen.blit(self.goal_image, (self.goal_y* cell_width, self.goal_x* cell_height))

                # agents
                for player in self.agents:
                    x = self.agent_positions[player][0]
                    y = self.agent_positions[player][1]
                    if self.is_invader[player]:
                        #invaders
                        self.screen.blit(self.invader_image, (y* cell_width, x* cell_height))
                    else:
                        self.screen.blit(self.defender_image, (y* cell_width, x* cell_height))
                        # Update display
                pygame.display.flip()
            else:
                
                # agents
                for player in self.agents:

                    #Check fade squares
                    locations = self.faded_locations[player]
                    for i in range(len(locations)):
                    # paint over previous coords
                        row, col = locations[i]
                        y = col * cell_width
                        x = row * cell_height
                        if i == 0:
                            self.screen.blit(self.tracker, (y,x))
                        elif i ==1:
                            self.screen.blit(self.tracker_faded, (y,x))
                        else:
                            if self.start_map[row][col] == ' ' or self.start_map[row][col] == 'D' or self.start_map[row][col] == 'I':
                                self.screen.blit(self.floor_image, (y,x))
                            elif self.start_map[row][col] == 'C':
                                self.screen.blit(self.ice_image, (y,x))
                            else:
                                #the only other traversible terrain are goals
                                if self.goal_x == row and self.goal_y == col:
                                    self.screen.blit(self.goal_image, (y,x))
                                else:
                                    self.screen.blit(self.grey_goal_image, (y,x))
                            locations.pop(2)

                    #Actual agents
                    x = self.agent_positions[player][0]
                    y = self.agent_positions[player][1]
                    
                    if self.is_invader[player]:
                        #invaders
                        self.screen.blit(self.invader_image, (y* cell_width, x* cell_height))
                    else:
                        self.screen.blit(self.defender_image, (y* cell_width, x* cell_height))      
                pygame.display.flip()
            pygame.time.delay(frame_delay)

            if return_image_arr:
                array = pygame.surfarray.array3d(self.screen) 
                return np.transpose(array, (1, 0, 2))
    def close(self):
        pygame.quit()
        super().close()

    
    def get_action_mask(self, agent:str) -> list[int]:
        action_mask = []
        '''
        Need to check that we are not moving into a wall or out of bounds

        W = wall
        G = goal
        ' ' = floor
        I = invader start
        D = defender start
        C = Ice

        Actions are
        0 = left
        1 = up
        2 = right
        3 = down
        '''
        
        for action in range (0,4):
            x = self.agent_positions[agent][0]
            y = self.agent_positions[agent][1]
            #calculate new position
            if action == 0:
                y-=1
            if action == 1:
                x-=1
            if action == 2:
                y+=1
            if action == 3:
                x+=1
            
            flag = False
            #check that it is in bounds
            if not(y>=0 and y<len(self.start_map[0]) and x>=0 and x<len(self.start_map)):
                flag = True
                
            #Check that its not a wall
            if (x,y) in self.wall_positions: 
                flag = True
            #Check  that another agent is not already there
            if (x,y) in self.agent_positions.values(): 
                flag = True
                
            action_mask.append(0 if flag else 1)
        
        return np.array(action_mask, dtype=np.int8)