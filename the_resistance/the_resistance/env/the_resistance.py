import functools
import random
import copy
import os

import math


import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from pettingzoo import ParallelEnv
import pygame
from gymnasium.utils import seeding

class TheResistance(ParallelEnv):
    metadata = {
        "name": "the_resistance_v0",
    }

    def __init__(self, num_agents = 5, render_mode = None):
        '''

        '''




        self.render_mode = render_mode

            

            
        #Start variables
        #teams
        #Run variables
        self.is_vote = True # Either a vote or a mission


        #game is over when one team reaches 3
        self.spy_points = 0
        self.agent_points = 0

        self.votes = []
        self.player_order = []
        self.mission_team = []
        self.round_num = 0
        self.step_count = 0
        #Tracking
        self.game_state = 0

        self.total_spy_to_spy_votes = 0
        self.total_spy_to_spy_votes_yes = 0

        self.total_infested_missions = 0
        self.total_infested_missions_sabotaged = 0

        self.total_agent_not_in_vote = 0
        self.total_agent_not_in_vote_yes = 0


        #set up round structure and players
        
        if not (5 <= num_agents <= 8):
            raise ValueError(f"num_agents must be between 0 and 8, got {num_agents}")


        self.round_structure = [] #sizes of the mission teams each round

        #players
        if num_agents == 5:
            self.round_structure = [2,3,2,3,3] 
            self.possible_agents = ["spy1", "spy2", "agent1", "agent2", "agent3"]
        if num_agents == 6:
            self.possible_agents = ["spy1", "spy2", "agent1", "agent2", "agent3", "agent4"]
        if num_agents == 7:
            self.possible_agents = ["spy1", "spy2", "spy3", "agent1", "agent2", "agent3", "agent4"]
        if num_agents == 8:
            self.possible_agents = ["spy1", "spy2", "spy3", "agent1", "agent2", "agent3", "agent4", "agent5"]

        self.agent_roles = {}
        for player in self.possible_agents:
            self.agent_roles[player] = "spy" if "spy" in player else "agent"

        #Set up memory
        #Memory will need to accomodate each agents votes in each round and end mission
        # add 2 more slots to accomodate round type and success/whether or not the mission failed
         
        self.memory = np.zeros((num_agents+2, 5*(num_agents+1)), dtype=int)

    def reset(self, seed=None, options=None):
        self.agents = copy.copy(self.possible_agents)
        # seeding
        if seed is not None:
            random.seed(int(seed))
            self.np_random, self.np_random_seed = seeding.np_random(int(seed))
        #Reset the game
        self.is_vote = True
        self.spy_points = 0
        self.agent_points = 0
        self.player_orientation = []
        self.votes = []
        self.mission_team = []
        self.round_num = 0
        self.step_count = 0
        #tracking
        self.game_state = 0
        
        self.total_spy_to_spy_votes = 0
        self.total_spy_to_spy_votes_yes = 0

        self.total_infested_missions = 0
        self.total_infested_missions_sabotaged = 0

        self.total_agent_not_in_vote = 0
        self.total_agent_not_in_vote_yes = 0
        #Randomize player order - just matters for orientation not turn order
        
        self.player_order = list(self.agents)
        random.shuffle(self.player_order)

        #Human render
        #Load images at correct size
        if self.render_mode == "human":
            # Get the directory of the current Python file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Initialize Pygame
            pygame.init()

            # Screen size
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 500, 500
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("The resistance")
            self.first_display = True
            

            #Images
            self.agent_image = pygame.image.load(current_dir+"/agent.png").convert()
            self.agent_selected_image = pygame.image.load(current_dir+"/agent_selected.png").convert()
            self.spy_image = pygame.image.load(current_dir+"/spy.png").convert()
            self.spy_selected_image = pygame.image.load(current_dir+"/spy_selected.png").convert()
            self.agent_yes_image = pygame.image.load(current_dir+"/agent_yes.png").convert()
            self.agent_no_image = pygame.image.load(current_dir+"/agent_no.png").convert()
            self.spy_yes_image = pygame.image.load(current_dir+"/spy_yes.png").convert()
            self.spy_no_image = pygame.image.load(current_dir+"/spy_no.png").convert()
            self.background_image = pygame.image.load(current_dir+"/office.png")
            self.yes_image = pygame.image.load(current_dir+"/yes.png").convert()
            self.no_image = pygame.image.load(current_dir+"/no.png").convert()
            

            self.player_width, self.player_height = (self.SCREEN_WIDTH/(len(self.agents)), self.SCREEN_HEIGHT/(len(self.agents)))
            self.background_image = pygame.transform.scale(self.background_image, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)) 


            self.agent_image = pygame.transform.scale(self.agent_image, (self.player_width, self.player_height))  
            self.flipped_agent_image = pygame.transform.flip(self.agent_image, True, False)

            self.agent_selected_image = pygame.transform.scale(self.agent_selected_image, (self.player_width, self.player_height))  
            self.flipped_agent_selected_image = pygame.transform.flip(self.agent_selected_image, True, False)

            self.spy_image = pygame.transform.scale(self.spy_image, (self.player_width, self.player_height))  
            self.flipped_spy_image = pygame.transform.flip(self.spy_image, True, False)

            self.spy_selected_image = pygame.transform.scale( self.spy_selected_image, (self.player_width, self.player_height))  
            self.flipped_spy_selected_image = pygame.transform.flip(self.spy_selected_image, True, False)

            
            self.spy_yes_image = pygame.transform.scale( self.spy_yes_image, (self.player_width, self.player_height))  
            self.flipped_spy_yes_image = pygame.transform.flip(self.spy_yes_image, True, False)
            
            self.agent_yes_image = pygame.transform.scale( self.agent_yes_image, (self.player_width, self.player_height))  
            self.flipped_agent_yes_image = pygame.transform.flip(self.agent_yes_image, True, False)


            self.spy_no_image = pygame.transform.scale( self.spy_no_image, (self.player_width, self.player_height))  
            self.flipped_spy_no_image = pygame.transform.flip(self.spy_no_image, True, False)

            self.agent_no_image = pygame.transform.scale( self.agent_no_image, (self.player_width, self.player_height))  
            self.flipped_agent_no_image = pygame.transform.flip(self.agent_no_image, True, False)
            
            self.yes_image = pygame.transform.scale(self.yes_image, (20,20))  
            self.no_image = pygame.transform.scale(self.no_image, (20,20))

            #Make white transparent for sprites
            
            ''''''
            self.agent_image.set_colorkey((0, 255, 0))
            self.flipped_agent_image.set_colorkey((0, 255, 0))

            self.agent_selected_image.set_colorkey((0, 255, 0))
            self.flipped_agent_selected_image.set_colorkey((0, 255, 0))

            self.spy_image.set_colorkey((0, 255, 0))
            self.flipped_spy_image.set_colorkey((0, 255, 0))
            
            self.spy_selected_image.set_colorkey((0, 255, 0))
            self.flipped_spy_selected_image.set_colorkey((0, 255, 0))

            self.yes_image.set_colorkey((0, 255, 0))
            self.no_image.set_colorkey((0, 255, 0))

            self.title_font = pygame.font.SysFont("Arial", 24)
            self.score_font = pygame.font.SysFont("Arial", 20)
            self.Title_text = self.title_font.render("Voting Round!", True, (0, 0, 0))
            self.SpyPointText = self.score_font.render("Spy Team (red) Score: 0", True, (0, 0, 0))
            self.AgentPointText = self.score_font.render("Agent Team (blue) Score: 0", True, (0, 0, 0))


        """
        Return initial observations
        Spies know other spies 
        Agents have no starting info
        """

        self.memory = np.zeros((len(self.possible_agents) +2, 5*(len(self.possible_agents)+1)), dtype=int)

        observations = {}
        self.player_orientation = [1 if "spy" in k else 0 for k in self.player_order] # Player info
        self.player_orientation.append(0) #Padding for 'Box' observation
        self.player_orientation.append(0)

        for player in self.player_order:
            self_location = [1 if player == k else 0 for k in self.player_order]
            self_location.append(0) #Padding for 'Box' observation
            self_location.append(0)
            if self.agent_roles[player] == "spy":
                observation = copy.deepcopy(self.player_orientation)
                initial_vote = copy.deepcopy(self.memory)
                observation = np.column_stack((observation, initial_vote))
                observation = np.column_stack((self_location, observation))
                observations[player] =   observation
            else:
                initial_vote = copy.deepcopy(self.memory)
                observation = np.column_stack((self_location, initial_vote))
                observations[player] = observation


        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """
        Takes in an action for the current agent (specified by agent_selection).
        """

        

        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        observations = {}
        # Get dummy infos 
        infos = {a: {} for a in self.agents}

        # Execute actions
        '''
        Actions are
        0 = No
        1 = Yes

        If it is a vote, count all votes and return the vote as an observation
        If it is the last round of voting record the agents that will be in the mission
        '''
        
        '''
        If it is a mission, look only at spies in the mission
        If there is a spy in the mission that selects "yes" score a point for the spies
        Otherwise score for the agents
        '''
        
        vote_count = 0
        num_infiltrations = 0
        mission_failed = False
        ballot  = []
        #Tracking
        current_candidate = self.round_num % (len(self.player_order)+1) #Every out of bounds number will not be a vote round

        # Save info for rendering 
        if self.render_mode == "human":
            self.step_history = {
                "is_vote" : self.is_vote,
                "actions" : copy.deepcopy(actions),
                "mission_team" : copy.deepcopy(self.mission_team),
                "player_idx" : len(self.votes)
            }
        for player in self.player_order:
            if self.is_vote:
                #Count vote
                if int(actions[player]) == 1:
                    vote_count+=1
                    
                #Tracking
                current_candidate_name = self.player_order[current_candidate]
                if self.agent_roles[player] == "spy" and self.agent_roles[current_candidate_name] == "spy":
                    self.total_spy_to_spy_votes+=1
                    if int(actions[player]) == 1:
                        self.total_spy_to_spy_votes_yes+=1

                if self.agent_roles[player] == "agent" and player != current_candidate:
                    self.total_agent_not_in_vote+=1
                    if int(actions[player]) == 1:
                        self.total_agent_not_in_vote_yes+=1


                ballot.append(actions[player])
            else:
                #Tracking
                infested_flag = False
                if player in self.mission_team and self.agent_roles[player] == "spy":  
                    #Tracking
                    infested_flag = True
                    if self.round_num == 4 and len(self.player_order) >= 7: #specific rule in higher player counts
                        if int(actions[player]) == 1:
                            num_infiltrations += 1
                            if num_infiltrations >=2:
                                mission_failed = True
                    else:
                        if int(actions[player]) == 1:
                            mission_failed = True

                #Tracking
                if infested_flag:
                    self.total_infested_missions+=1
                    if mission_failed:
                        self.total_infested_missions_sabotaged+=1
        
        if self.is_vote:
            self.votes.append(vote_count)
            #check if it is the last vote
            if len(self.votes) == len(self.player_order):
                self.is_vote = False
                #select the mission team
                team_size = self.round_structure[self.round_num]
                
                top_x_idx = np.argsort(self.votes)[-team_size:][::-1] #argsort will bias towards list order on ties
                self.mission_team = [self.player_order[k]  for k in top_x_idx]

            

            
            ballot.append(0) # used to provide signal for round type
            ballot.append(0) # used to provide signal for win loss on mission
            #Append the ballot to memory
            self.push_to_memory(np.array(ballot))
            #construct observations
            for player in self.player_order:
                self_location = [1 if player == k else 0 for k in self.player_order]
                self_location.append(0) #Padding for 'Box' observation
                self_location.append(0)
                if self.agent_roles[player] == "spy":
                    observation = copy.deepcopy(self.player_orientation)

                    votes = copy.deepcopy(self.memory)
                    observation = np.column_stack((observation, votes))
                    observation = np.column_stack((self_location, observation))
                    observations[player] =   observation
                else:
                    observation = copy.deepcopy(self.memory)
                    observation = np.column_stack((self_location, observation))
                    observations[player] = observation
        else:
            #prep next round and construct rewards
            self.is_vote = True
            self.round_num+=1
            self.votes = []

            #observations
            team_comp = []
            for player in self.player_order:
                if player in self.mission_team:
                    team_comp.append(1)
                else:
                    team_comp.append(0)
            team_comp.append(1) # round type
            #reset mission team
            self.mission_team = []

            #rewards
            if self.render_mode == "human":
                self.step_history["mission_failed"] = mission_failed
                
            if mission_failed:
                team_comp.append(0) # signal for mission state
                self.spy_points += 1
                for player in self.player_order:
                    if self.agent_roles[player] == "spy":
                        rewards[player] = 1
                    else:
                        rewards[player] = -1
            else:
                team_comp.append(1) # signal for mission state
                self.agent_points +=1
                for player in self.player_order:
                    if self.agent_roles[player] == "spy":
                        rewards[player] = -1
                    else:
                        rewards[player] = 1
            #Add to memory
            self.push_to_memory(np.array(team_comp))
            #construct observations
            for player in self.player_order:
                self_location = [1 if player == k else 0 for k in self.player_order]
                self_location.append(0) #Padding for 'Box' observation
                self_location.append(0)
                if self.agent_roles[player] == "spy":
                    observation = copy.deepcopy(self.player_orientation)

                    votes = copy.deepcopy(self.memory)
                    observation = np.column_stack((observation, votes))
                    observation = np.column_stack((self_location, observation))
                    observations[player] =   observation
                else:
                    observation = copy.deepcopy(self.memory)
                    observation = np.column_stack((self_location, observation))
                    observations[player] = observation
        #check terminal conditions
        if (self.agent_points == 3 or self.spy_points == 3):
            for player in self.player_order:
                rewards[player]*=10
            for player in self.player_order:
                terminations[player] = True
            
            self.game_state = 1 if self.agent_points == 3 else -1
                
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def push_to_memory(self, item):
        self.memory [:, self.step_count] = item #Add item to leftmost unused memory location
        self.step_count+=1

    def render(self, frame_delay = 200, return_image_arr = False):
        if self.render_mode == "human":
            mid_x, mid_y = self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2
            if self.first_display == True:
                #set up the screen

                self.screen.fill((255, 255, 255))
                self.screen.blit(self.background_image, (0, 0))
                self.surfaces = {}
                theta = math.radians(360) / len(self.agents)
                radius = self.SCREEN_HEIGHT/4


                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        #Todo check what I should do here
                        placebo = False

                #set up all players
                i = 0
                for player in self.player_order:
                    x = mid_x + radius * math.cos(theta*i)
                    y = mid_y + radius * math.sin(theta*i)


                    x,y = x-self.player_width/2, y-self.player_height/2

                    #get background surface for easier render updates
                    patch_rect = pygame.Rect(x, y, self.player_width, self.player_height)
                    surface = self.screen.subsurface(patch_rect).copy()

                    self.surfaces[player] = {"surface":surface, "coords":(x,y)}

                    surface_copy = surface.copy()

                    if self.agent_roles[player] == "spy":
                        #add player to patch copy
                        surface_copy.blit(self.spy_image if x+self.player_width/2 < mid_x else self.flipped_spy_image , (0, 0))
                        #add patch to screen
                        self.screen.blit(surface_copy, (x, y))

                    else:
                        surface_copy.blit(self.agent_image if x+self.player_width < mid_x else self.flipped_agent_image, (0, 0))
                        self.screen.blit(surface_copy, (x, y))
                    i+=1

                #Set up title and score text

                pygame.draw.rect(self.screen, (255, 255, 255), (175, 0, 180, 50))
                self.screen.blit(self.Title_text, (mid_x-30, 10))

                
                pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, 220, 80))
                self.screen.blit(self.AgentPointText, (10, 10))
                self.screen.blit(self.SpyPointText, (10, 50))

                self.first_display = False
            else:
                if self.step_history["is_vote"]:
                    # If it is a voting round, select the player being voted on
                    # Display all votes at the top right of the players
                    for player in self.player_order:
                        surface  = self.surfaces[player]["surface"]
                        x, y = self.surfaces[player]["coords"]
                        surface_copy = surface.copy()
                        #select this player
                        if player == self.player_order[self.step_history["player_idx"]]:
                            if self.agent_roles[player] == "spy":
                                #add player to patch copy
                                surface_copy.blit(self.spy_selected_image if x+self.player_width/2 < mid_x else self.flipped_spy_selected_image , (0, 0))

                            else:
                                surface_copy.blit(self.agent_selected_image if x+self.player_width < mid_x else self.flipped_agent_selected_image, (0, 0))
                        else:
                            if self.agent_roles[player] == "spy":
                                #add player to patch copy
                                surface_copy.blit(self.spy_image if x+self.player_width/2 < mid_x else self.flipped_spy_image , (0, 0))

                            else:
                                surface_copy.blit(self.agent_image if x+self.player_width < mid_x else self.flipped_agent_image, (0, 0))
                        #if they votes yes
                        if int(self.step_history["actions"][player]) == 1:
                            surface_copy.blit(self.yes_image , (self.player_width-20, 0))
                            self.screen.blit(surface_copy, (x, y))
                        else:
                            surface_copy.blit(self.no_image , (self.player_width-20, 0))
                            self.screen.blit(surface_copy, (x, y))


                    # Ensure correct title text
                    self.Title_text = self.title_font.render("Voting Round!", True, (0, 0, 0))
                    pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, 500, 50))
                    self.screen.blit(self.Title_text, (mid_x-30, 10))
                else:
                    # If it is a non voting round select all players in the mission
                    for player in self.player_order:
                        surface  = self.surfaces[player]["surface"]
                        x, y = self.surfaces[player]["coords"]
                        surface_copy = surface.copy()

                        #Choose image based on in team and if they are a spy
                        if player in self.step_history["mission_team"]:
                            if self.agent_roles[player] == "spy":
                                #add player to patch copy
                                surface_copy.blit(self.spy_selected_image if x+self.player_width/2 < mid_x else self.flipped_spy_selected_image , (0, 0))
                                #add patch to screen
                                self.screen.blit(surface_copy, (x, y))

                            else:
                                surface_copy.blit(self.agent_selected_image if x+self.player_width < mid_x else self.flipped_agent_selected_image, (0, 0))
                                self.screen.blit(surface_copy, (x, y))
                        else:
                            if self.agent_roles[player] == "spy":
                                #add player to patch copy
                                surface_copy.blit(self.spy_image if x+self.player_width/2 < mid_x else self.flipped_spy_image , (0, 0))
                                #add patch to screen
                                self.screen.blit(surface_copy, (x, y))

                            else:
                                surface_copy.blit(self.agent_image if x+self.player_width < mid_x else self.flipped_agent_image, (0, 0))
                                self.screen.blit(surface_copy, (x, y))

                    #Display the mission Result and update the scores
                    title_text = ("Spies Score" if self.step_history["mission_failed"] else "Resistance Scores")
                    self.Title_text = self.title_font.render(title_text, True, (0, 0, 0))
                    pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, 500, 50))
                    self.screen.blit(self.Title_text, (mid_x-30, 10))


                self.SpyPointText = self.score_font.render("Spy Team (red) Score: "+str(self.spy_points), True, (0, 0, 0))
                self.AgentPointText = self.score_font.render("Agent Team (blue) Score: "+str(self.agent_points), True, (0, 0, 0))
                pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, 220, 80))
                self.screen.blit(self.AgentPointText, (10, 10))
                self.screen.blit(self.SpyPointText, (10, 50))

            pygame.display.flip()
    
            pygame.time.delay(frame_delay)
            if return_image_arr:
                array = pygame.surfarray.array3d(self.screen) 
                return np.transpose(array, (1, 0, 2))
    def close(self):
        if self.render_mode == "human":
            pygame.quit()

        super().close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        
        if self.agent_roles[agent] == "spy":
            return Box(low=0, high=1, shape=(len(self.possible_agents) +2, # 2 additional bits for round type and scoring signals
                                             5*(len(self.possible_agents)+1)+1+1), # 5 rounds * (voting for each player + mission) + player orientation + self location
                                             dtype=np.int8)
        else:
            return Box(low=0, high=1, shape=(len(self.possible_agents) +2, 
                                             5*(len(self.possible_agents)+1)+1),  # 5 rounds * (voting for each player + mission) + player orientation + self location
                                             dtype=np.int8)
    

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)