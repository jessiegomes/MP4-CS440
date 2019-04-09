import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def createStateRepresentation(state):
        snake_x = state[0]
        snake_y = state[1]
        food_x = state[3]
        food_y = state[4]
        body = state[2]

        ## wall left or right
        if (snake_y == 40):
            adjoining_walls_x = 1
        elif (snake_y == 480):
            adjoining_walls_x = 2
        else:
            adjoining_walls_x = 0
        ## wall up or down
        if (snake_x == 40):
            adjoining_walls_y = 1
        elif (snake_x == 480):
            adjoining_walls_y = 2
        else:
            adjoining_walls_y = 0
        ## food direction left or right
        if (food_y < snake_y):
            food_dir_x = 1
        elif (food_y > snake_y):
            food_dir_x = 2
        else:
            food_dir_x = 0
        ## food direction up or down
        if (food_x < snake_x):
            food_dir_y = 1
        elif (food_x > snake_x):
            food_dir_y = 2
        else:
            food_dir_y = 0
        ## is body on top
        new_x = snake_x - 40
        new_y = snake_y
        if (new_x, new_y) in body:
            adjoining_body_top = 1
        else:
             adjoining_body_top = 0
        ## is body on bottom
        new_x = snake_x + 40
        new_y = snake_y
        if (new_x, new_y) in body:
            adjoining_body_bottom = 1
        else:
             adjoining_body_bottom = 0
        ## is body on left
        new_x = snake_x
        new_y = snake_y - 40
        if (new_x, new_y) in body:
            adjoining_body_left = 1
        else:
             adjoining_body_left = 0
        ## is body on right
        new_x = snake_x
        new_y = snake_y + 40
        if (new_x, new_y) in body:
            adjoining_body_right = 1
        else:
             adjoining_body_right = 0
        return (adjoining_walls_x, adjoining_walls_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)


    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''

        utility_per_action = np.zeros(4)
        for i in range(actions):
            if i == 0:    #Right 
                
            elif i == 1:  #Left

            elif i == 2:  #Down

            elif i == 3:  #Up


        return self.actions[0]
