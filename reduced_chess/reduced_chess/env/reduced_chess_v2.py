#Chess piece SVGs
#Black King By Cburnett - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=1499803
#White King By Cburnett - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=1499806
#Black Bishop By Cburnett - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=1499800
#white Bishop By Cburnett - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=1499801
#black Knight By Cburnett - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=1499807
#white Knight By Cburnett - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=1499808

import functools
import random
import copy
import os

import math
from gymnasium.utils import seeding


import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, Box, Dict
from pettingzoo.utils import agent_selector

from pettingzoo import AECEnv
import pygame

#Helper class to store piece information
class piece():
    def __init__(self, owner, type):
        self.pos = (0,0)
        self.prev_pos = (0,0)
        self.is_alive = True
        self.owner = owner
        self.type = type
        self.revealed = False
        # Information to keep track of revealed info
        # If a move impossible for a specific piece is played, the relevant signal will be given
    
        self.image = None
        self.hidden_image = None

    def reset(self):
        self.pos = (0,0)
        self.prev_pos = (0,0)
        self.is_alive = True
        self.revealed = False
        self.image = None
        self.hidden_image = None


class ReducedChess(AECEnv):
    #colors for render
    dark_square_rgb = (99, 71, 48)
    light_square_rgb = (237, 218, 185)

    action_to_delta_position_mapping = [
        (-1,1), #King
        (0,1),
        (1,1),
        (-1,0),
        (1,0),
        (-1,-1),
        (0,-1),
        (1,-1),

        (-1, 2), #Knight
        (1, 2),
        (-2,1),
        #(0,1),
        (2,1),
        #(-1,0),
        #(1,0),
        (-2,-1),
        #(0,-1),
        (2,-1),
        (-1,-2),
        (1,-2),

        (-2,2),#Bishop
        (2,2),
        (-1,1), 
        #(0,1),
        (1,1),
        #(-1,0),
        #(1,0),
        (-1,-1),
        #(0,-1),
        (1,-1),
        (-2,-2),
        (2,-2)
    ]
    # maps which actions reveal that a piece CANNOT be a certain type
    # (king, knight, bishop)
    action_to_piece_exclusion_mapping = [
        (0,1,0), #King - only diagonal movements exclude the knight
        (0,1,1), #Horizontal/Vertical exludes knight and bishop
        (0,1,0), #Diagonal
        (0,1,1),
        (0,1,1),
        (0,1,0), #Diagonal
        (0,1,1),
        (0,1,0), #Diagonal

        (1,0,1), #Knight - 'L' movement excludes king and bishop
        (1,0,1), # 'L' movement
        (1,0,1), # 'L' movement
        #(0,0,0),
        (1,0,1), # 'L' movement
        #(0,0,0),
        #(0,0,0),
        (1,0,1), # 'L' movement
        #(0,0,0),
        (1,0,1), # 'L' movement
        (1,0,1), # 'L' movement
        (1,0,1), # 'L' movement

        (1,1,0),#Bishop - 1 space diagonal excludes knight, 2 space excludes king as well
        (1,1,0), #2 space diagonal
        (0,1,0), #1 space diagonal
        #(0,0,0),
        (0,1,0), #1 space diagonal
        #(0,0,0),
        #(0,0,0),
        (0,1,0), #1 space diagonal
        #(0,0,0),
        (0,1,0), #1 space diagonal
        (1,1,0), #2 space diagonal
        (1,1,0) #2 space diagonal
    ]
    
    metadata = {
        "name": "reduced_chess_v2",
    }

    def __init__(self, render_mode:str|None = None, memory_length:int = 10, final_rank_wins = False):
        '''

        '''
        self.render_mode = render_mode
        self.final_rank_wins = final_rank_wins
            
        #Start variables
        self.possible_agents = ["white_pieces", "black_pieces"]

        # Initial Board
        self.board = [[None for i in range(0,5)] for j in range (0,5)]

        # Overvation vectors - used to keep track of the board without having to loop through it each time
        self.black_piece_locations =  np.zeros((5, 5, 3), dtype=int)
        self.white_piece_locations =  np.zeros((5, 5, 3), dtype=int)
        
        self.white_revealed_information =  np.zeros((5, 5, 3), dtype=int) # the piece exclusions
        self.black_revealed_information =  np.zeros((5, 5, 3), dtype=int)

        # Setup memory observations
        self.memory_length = memory_length
        self.memory = np.zeros((5, 5, memory_length), dtype=int)

        self.white_king = piece("white_pieces" , "king")
        self.white_knight = piece("white_pieces", "knight")
        self.white_bishop = piece("white_pieces", "bishop")
        
        self.black_king = piece("black_pieces", "king")
        self.black_knight = piece("black_pieces", "knight")
        self.black_bishop = piece("black_pieces", "bishop")

        #Tracking
        self.game_state = 0
        self.king_capture = 0
        self.final_rank = 0
        self.pieces_revealed = 0


    def reset(self, seed=None, options=None):
        self.agents = copy.copy(self.possible_agents)
        
        # seeding
        if seed is not None:
            random.seed(int(seed))
            self.np_random, self.np_random_seed = seeding.np_random(int(seed))

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        #Reset the game
        self.board = [[None for i in range(0,5)] for j in range (0,5)]
        self.timestep = 0
        # Overvation vectors
        self.black_piece_locations =  np.zeros((5, 5, 3), dtype=int)
        self.white_piece_locations =  np.zeros((5, 5, 3), dtype=int)
        
        self.white_revealed_information =  np.zeros((5, 5, 3), dtype=int)
        self.black_revealed_information =  np.zeros((5, 5, 3), dtype=int)

        #Randomize piece locations
        
        white_random_positions = random.sample([1,2,3],  3)
        black_random_positions = random.sample([1,2,3],  3)
        
        # White pieces
        self.white_king.reset()
        self.white_king.pos = (white_random_positions[0], 0)
        self.white_king.prev_pos = (white_random_positions[0], 0)
        self.board[white_random_positions[0]][0] = self.white_king
        self.white_piece_locations[white_random_positions[0]][0][0] = 1

        self.white_knight.reset()
        self.white_knight.pos = (white_random_positions[1], 0)
        self.white_knight.prev_pos = (white_random_positions[1], 0)
        self.board[white_random_positions[1]][ 0] = self.white_knight
        self.white_piece_locations[white_random_positions[1]][0][1] = 1

        self.white_bishop.reset()
        self.white_bishop.pos = (white_random_positions[2], 0)
        self.white_bishop.prev_pos = (white_random_positions[2], 0)
        self.board[white_random_positions[2]][0] = self.white_bishop
        self.white_piece_locations[white_random_positions[2]][0][2] = 1
        
        # Black pieces
        self.black_king.reset()
        self.black_king.pos = (black_random_positions[0], 4)
        self.black_king.prev_pos = (black_random_positions[0], 4)
        self.board[black_random_positions[0]][4] = self.black_king
        self.black_piece_locations[black_random_positions[0]][4][0] = 1

        self.black_knight.reset()
        self.black_knight.pos = (black_random_positions[1], 4)
        self.black_knight.prev_pos = (black_random_positions[1], 4)
        self.board[black_random_positions[1]][4] = self.black_knight
        self.black_piece_locations[black_random_positions[1]][4][1] = 1

        self.black_bishop.reset()
        self.black_bishop.pos = (black_random_positions[2], 4)
        self.black_bishop.prev_pos = (black_random_positions[2], 4)
        self.board[black_random_positions[2]][4] = self.black_bishop
        self.black_piece_locations[black_random_positions[2]][4][2] = 1


        #Tracking
        self.game_state = 0
        self.king_capture = 0
        self.pieces_revealed = 0
        self.final_rank = 0

        #Human render
        #Load images at correct size
        if self.render_mode == "human":
            self.first_display = True
            self.previously_moved_piece = None
            # Get the directory of the current Python file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Initialize Pygame
            pygame.init()

            # Screen size
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 500, 500
            self.cell_width, self.cell_height = self.SCREEN_WIDTH/5, self.SCREEN_HEIGHT/5
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Reduced Chess")
            rows, cols = 5,5
            
            #Normal Pieces
            self.black_king_image = pygame.image.load(current_dir+"/black_king.png")
            self.black_knight_image = pygame.image.load(current_dir+"/black_knight.png")
            self.black_bishop_image = pygame.image.load(current_dir+"/black_bishop.png")
            
            self.white_king_image = pygame.image.load(current_dir+"/white_king.png")
            self.white_knight_image = pygame.image.load(current_dir+"/white_knight.png")
            self.white_bishop_image = pygame.image.load(current_dir+"/white_bishop.png")
            
            self.black_king_image = pygame.transform.scale(self.black_king_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.black_knight_image = pygame.transform.scale(self.black_knight_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.black_bishop_image = pygame.transform.scale(self.black_bishop_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.white_king_image = pygame.transform.scale(self.white_king_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.white_knight_image = pygame.transform.scale( self.white_knight_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.white_bishop_image = pygame.transform.scale(self.white_bishop_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  

            self.black_king.image = self.black_king_image
            self.black_knight.image = self.black_knight_image
            self.black_bishop.image = self.black_bishop_image
            self.white_king.image = self.white_king_image
            self.white_knight.image = self.white_knight_image
            self.white_bishop.image = self.white_bishop_image

            #Hidden Pieces
            self.black_king_hidden_image = pygame.image.load(current_dir+"/black_king_hidden.png")
            self.black_knight_hidden_image = pygame.image.load(current_dir+"/black_knight_hidden.png")
            self.black_bishop_hidden_image = pygame.image.load(current_dir+"/black_bishop_hidden.png")
            
            self.white_king_hidden_image = pygame.image.load(current_dir+"/white_king_hidden.png")
            self.white_knight_hidden_image = pygame.image.load(current_dir+"/white_knight_hidden.png")
            self.white_bishop_hidden_image = pygame.image.load(current_dir+"/white_bishop_hidden.png")
            
            self.black_king_hidden_image = pygame.transform.scale(self.black_king_hidden_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.black_knight_hidden_image = pygame.transform.scale(self.black_knight_hidden_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.black_bishop_hidden_image = pygame.transform.scale(self.black_bishop_hidden_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.white_king_hidden_image = pygame.transform.scale(self.white_king_hidden_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.white_knight_hidden_image = pygame.transform.scale( self.white_knight_hidden_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
            self.white_bishop_hidden_image = pygame.transform.scale(self.white_bishop_hidden_image, (self.SCREEN_WIDTH/rows, self.SCREEN_HEIGHT/cols))  
  
            self.black_king.hidden_image = self.black_king_hidden_image
            self.black_knight.hidden_image = self.black_knight_hidden_image
            self.black_bishop.hidden_image = self.black_bishop_hidden_image
            self.white_king.hidden_image = self.white_king_hidden_image
            self.white_knight.hidden_image = self.white_knight_hidden_image
            self.white_bishop.hidden_image = self.white_bishop_hidden_image

            self.first_display = True
            


        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()


    def step(self, action):
        """
        Function to execute the action for the current agent
        """
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            #Handle agents moving on game end
            self._was_dead_step(action)
            return
        

        # Execute actions
        '''
        Actions are
        [0-31] - every individual piece movement
        [0-7] - 8 king actions
        [8-19] - 12 bishop actions
        [20-31] - 12 knight actions
        
        For each piece, the action order will be leftmost movement to rightmost, top to bottom 

        For example 0 would be diagonal top left for the king and 7 would be diagonal bottom right
        '''
        
        # Check if the move is legal
        mask = self.get_action_mask(agent)

        if mask[action] == 0:
            #action illegal
            return 

        '''
        Note that the we only push the positions of the current players pieces before the movement.
        Thus the last slice in memory (before observation) will always be the opposite player's prior 
        piece positions.
        '''
        layout = self.get_piece_layout(agent)
        self.push_to_memory(layout)
        # Make the move
        self.move_piece(agent, action)
        if self.render_mode == "human":
            selected_piece, _, _ = self.translate_action(agent, action)
            self.previously_moved_piece = selected_piece

        # Check terminal conditions
        # king capture
        if not self.black_king.is_alive:
            #terminate
            self.terminations = {a: True for a in self.agents}
            #Tracking
            self.game_state = 1
            self.king_capture = 1
            self.pieces_revealed = self.count_revealed_pieces()

            #rewards
            self.rewards["black_pieces"] -= 1 
            self.rewards["white_pieces"] += 1 

        elif not self.white_king.is_alive:
            #terminate 
            self.terminations = {a: True for a in self.agents}
            #Tracking
            self.game_state = -1
            self.king_capture = 1
            self.pieces_revealed = self.count_revealed_pieces()

            #rewards
            self.rewards["black_pieces"] += 1 
            self.rewards["white_pieces"] -= 1 
        #King on final rank
        elif self.black_king.pos[1] == 0 and self.final_rank_wins:
            #terminate
            self.terminations = {a: True for a in self.agents}
            #Tracking
            self.game_state = -1
            self.final_rank = 1
            self.pieces_revealed = self.count_revealed_pieces()

            #rewards
            self.rewards["black_pieces"] += 1 
            self.rewards["white_pieces"] -= 1 

        elif self.white_king.pos[1] == 4 and self.final_rank_wins:
            #terminate 
            self.terminations = {a: True for a in self.agents}
            #Tracking
            self.game_state = 1
            self.final_rank = 1
            self.pieces_revealed = self.count_revealed_pieces()

            #rewards
            self.rewards["black_pieces"] -= 1 
            self.rewards["white_pieces"] += 1 

        #truncate after 100 steps
        if self.timestep >100:
            self.terminations = {a: True for a in self.agents}
            self.truncations = {a: True for a in self.agents}
        self.timestep += 1



        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()


        #if self.render_mode == "human":
        #    self.render()


        
    
    def render(self, frame_delay = 500, return_image_arr = False):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    #Todo check what I should do here
                    placebo = False
            if self.first_display == True:
                #setup checkered board
                for i in range (0,5):
                    for j in range (0,5):
                        # will use 4-j for height to keep consistent with chess board representation
                        # pygame draws from top left
                        rect = pygame.Rect(i*self.cell_width, (4-j)*self.cell_height, self.SCREEN_WIDTH/5, self.SCREEN_HEIGHT/5)

                        #XOR to alternate colors correctly
                        if((i%2 == 0) ^ (j%2== 0)):
                            pygame.draw.rect(self.screen, self.light_square_rgb, rect) 
                        else:
                            pygame.draw.rect(self.screen, self.dark_square_rgb, rect) 
                #setup initial piece locations
                for row in self.board:
                    for selected_piece in row:
                        if selected_piece != None:
                            self.draw_piece_pygame(selected_piece)

                self.first_display = False
                pygame.display.flip()
            else:
                selected_piece = self.previously_moved_piece
                #replace the checkered board at the previous piece location
                self.remove_piece_pygame(selected_piece)
                #redraw rectancle at new pos
                self.draw_sqaure(selected_piece)
                #draw the piece at the new location
                self.draw_piece_pygame(selected_piece)
                pygame.display.flip()

            pygame.time.delay(frame_delay)
            if return_image_arr:
                array = pygame.surfarray.array3d(self.screen) 
                return np.transpose(array, (1, 0, 2))


    def draw_sqaure(self, selected_piece:piece):
        if self.render_mode == "human":
            (x,y) = selected_piece.pos
            # pygame draws from top left
            rect = pygame.Rect(x*self.cell_width, (4-y)*self.cell_height, self.SCREEN_WIDTH/5, self.SCREEN_HEIGHT/5)

            #XOR to alternate colors correctly
            if((x%2 == 0) ^ (y%2== 0)):
                pygame.draw.rect(self.screen, self.light_square_rgb, rect) 
            else:
                pygame.draw.rect(self.screen, self.dark_square_rgb, rect) 

    def draw_piece_pygame(self, selected_piece:piece):
        if self.render_mode == "human":
            (x,y) = selected_piece.pos
            if selected_piece.owner == "white_pieces":
                flag = sum(self.white_revealed_information[x][y]) == 2
            else:
                flag = sum(self.black_revealed_information[x][y]) == 2
            if flag:
                #Other pieces have been excluded
                self.screen.blit(selected_piece.image, (x*self.cell_width,(4-y)*self.cell_height))
            else:
                self.screen.blit(selected_piece.hidden_image, (x*self.cell_width,(4-y)*self.cell_height))

    def remove_piece_pygame(self,selected_piece):
        if self.render_mode == "human":
            (x,y) = selected_piece.prev_pos
            rect = pygame.Rect(x*self.cell_width, (4-y)*self.cell_height, self.SCREEN_WIDTH/5, self.SCREEN_HEIGHT/5)
            #XOR to alternate colors correctly
            if((x%2 == 0) ^ (y%2== 0)):
                pygame.draw.rect(self.screen, self.light_square_rgb, rect) 
            else:
                pygame.draw.rect(self.screen, self.dark_square_rgb, rect) 


    def close(self):
        if self.render_mode == "human":
            pygame.time.delay(5000)
            pygame.quit()

        super().close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            "observation": Box(low=0, high=1, shape=(5,5, 1+3+3+3+self.memory_length), dtype=np.int8),
            "action_mask": MultiBinary(32),
        })
    

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent:str):
        return Discrete(len(self.action_to_delta_position_mapping))
    

    def translate_action(self, agent:str, action:int)->tuple[piece, int, int]:
        if agent == "white_pieces":
            #King
            if action <8:
                selected_piece = self.white_king
            elif action<16:
                selected_piece = self.white_knight
            else:
                selected_piece = self.white_bishop
            x, y = 1,1
        else:
            #King
            if action <8:
                selected_piece = self.black_king
            elif action<16:
                selected_piece = self.black_knight
            else:
                selected_piece = self.black_bishop
            x, y = -1,-1 # multiplier for board (inverted on y axis for black pieces)
        

        delta_x,delta_y = self.action_to_delta_position_mapping[action]

        return selected_piece, delta_x*x, delta_y*y

    # Helper function to get action mask
    def get_action_mask(self, agent:str, debug = False) -> list[int]:
        action_mask = []
        for action in range (0,len(self.action_to_delta_position_mapping)):
            selected_piece, delta_x, delta_y = self.translate_action(agent, action)
            if not selected_piece.is_alive:
                if debug:
                    print("dead piece")
                action_mask.append(0)
            elif (selected_piece.pos[0]+delta_x >=0 and selected_piece.pos[0]+delta_x <5 and
                selected_piece.pos[1]+delta_y >=0 and selected_piece.pos[1]+delta_y <5):
                #We are inside the board, make sure we aren't capturing our own piece
                captured_piece = self.board[selected_piece.pos[0]+delta_x][selected_piece.pos[1]+delta_y]
                if(captured_piece == None):
                    action_mask.append(1)
                elif(captured_piece.owner == selected_piece.owner):
                    if debug:
                        print("same piece")
                    action_mask.append(0)
                else:
                    action_mask.append(1)
            else:
                if debug:
                    print("out of board")
                action_mask.append(0)
        
        return np.array(action_mask, dtype=np.int8)
    
    def push_to_memory(self, item):

        self.memory = np.roll(self.memory , shift=-1, axis=2)  #shift left
        self.memory [:, :, -1] = item                #insert at the end

    def get_piece_layout(self, agent):
        if agent == 'white_pieces':
            layout = np.bitwise_or.accumulate(self.white_piece_locations, axis=2)
            return layout[:, :, -1].copy()
        else:
            layout = np.bitwise_or.accumulate(self.black_piece_locations, axis=2)
            return layout[:, :, -1].copy()
        
    def move_piece(self, agent, action):
        selected_piece, delta_x, delta_y = self.translate_action(agent, action)
        temp_x = selected_piece.pos[0]
        temp_y = selected_piece.pos[1]



        selected_piece.pos = (selected_piece.pos[0] + delta_x, selected_piece.pos[1] + delta_y)
        # Check if it is a Capture
        if self.board[selected_piece.pos[0]][selected_piece.pos[1]] != None:
            #we have a piece there
            captured_piece = self.board[selected_piece.pos[0]][selected_piece.pos[1]]
            captured_piece.is_alive = False
            #Update the observation information
            if agent== "white_pieces":
                self.black_piece_locations[selected_piece.pos[0]][selected_piece.pos[1]] = [0,0,0]
                self.black_revealed_information[selected_piece.pos[0]][selected_piece.pos[1]] = [0,0,0]
            else:
                self.white_piece_locations[selected_piece.pos[0]][selected_piece.pos[1]] = [0,0,0]
                self.white_revealed_information[selected_piece.pos[0]][selected_piece.pos[1]] = [0,0,0]

        # Set new pos on board
        self.board[selected_piece.pos[0]][selected_piece.pos[1]] = selected_piece
        # Set old pos to nothing
        self.board[temp_x][temp_y] = None
        #Update previous pos
        selected_piece.prev_pos = (temp_x, temp_y)

        # Update observation information
        
        #Update the revealed info
        revealed_info = np.array(self.action_to_piece_exclusion_mapping[action])


        if agent== "white_pieces":
            #piece location

            self.white_piece_locations[selected_piece.pos[0]][selected_piece.pos[1]] = \
                self.white_piece_locations[selected_piece.prev_pos[0]][selected_piece.prev_pos[1]]
            self.white_piece_locations[selected_piece.prev_pos[0]][selected_piece.prev_pos[1]] = [0,0,0]

            #revealed information
            #move it
            # update it with and "or" logical gate for newly revealed information
            old_info = self.white_revealed_information[selected_piece.prev_pos[0]][selected_piece.prev_pos[1]]
            new_info = old_info | revealed_info
            self.white_revealed_information[selected_piece.pos[0]][selected_piece.pos[1]] = new_info
            self.white_revealed_information[selected_piece.prev_pos[0]][selected_piece.prev_pos[1]] = [0,0,0]


        else:
            #piece location

            self.black_piece_locations[selected_piece.pos[0]][selected_piece.pos[1]] = \
                self.black_piece_locations[selected_piece.prev_pos[0]][selected_piece.prev_pos[1]]
            self.black_piece_locations[selected_piece.prev_pos[0]][selected_piece.prev_pos[1]] = [0,0,0]

            #revealed information
            #move it
            # update it with and "or" logical gate for newly revealed information
            old_info = self.black_revealed_information[selected_piece.prev_pos[0]][selected_piece.prev_pos[1]]
            new_info = old_info | revealed_info
            self.black_revealed_information[selected_piece.pos[0]][selected_piece.pos[1]] = new_info
            self.black_revealed_information[selected_piece.prev_pos[0]][selected_piece.prev_pos[1]] = [0,0,0]

        if not selected_piece.revealed and sum(new_info) == 2:
            selected_piece.revealed = True
    def observe(self, agent:str):
        '''
        provide obs as binary np array of grid positions (5x5) board size, 
        
        We will provide the following

        5x5x1 for enemy piece locations
        5x5x3 for enemy public information (we will provide a signal when a move is made which excludes a piece type, 
                for example (+1,+1) is impossible for a knight so (0,1,0) would be observed for that piece)

        5x5x3 for the agent piece locations (3 types of pieces)
        5x5x3 for the agent's public information

        5x5x(history_length) for the recent match history, each player pushes the state of their pieces before their move
            thus the last observation in always the opponents previous layout

        *Note that for the black pieces we will invert the board over the "y" axis and we will thus be able 
            to use self play
        '''
        observation = np.zeros((5,5, 1+3+3+3+self.memory_length), dtype=np.int8)
        if agent == "white_pieces":
            #enemy piece locations
            observation[:,:,0] = self.get_piece_layout(agent="black_pieces")
            #enemy public information
            observation[:,:,1:4] = self.black_revealed_information.copy()
            #agent piece information
            observation[:,:,4:7] = self.white_piece_locations.copy()
            #agent public information
            observation[:,:,7:10] = self.white_revealed_information.copy()
            #history
            observation[:,:,10:] = self.memory.copy() 
        else:
            #enemy piece locations
            observation[:,:,0] = self.get_piece_layout(agent="white_pieces")
            #enemy public information
            observation[:,:,1:4] = self.white_revealed_information.copy()
            #agent piece information
            observation[:,:,4:7] = self.black_piece_locations.copy()
            #agent public information
            observation[:,:,7:10] = self.black_revealed_information.copy()
            #history
            observation[:,:,10:] = self.memory.copy() 
            #invert
            observation = observation[::-1, ::-1, :]

        action_mask = self.get_action_mask(self.agent_selection)

        return {
            "observation": observation,
            "action_mask": action_mask
        }
    
    def count_revealed_pieces(self):
        count = 0
        if self.white_king.revealed: count+=1
        if self.white_knight.revealed: count+=1
        if self.white_bishop.revealed: count+=1
        if self.black_king.revealed: count+=1
        if self.black_knight.revealed: count+=1
        if self.black_bishop.revealed: count+=1
        
        return count