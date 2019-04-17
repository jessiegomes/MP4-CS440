import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.first = True
        self.track_actions = []
        self.s = None
        self.a = None

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
        self.first = True
        self.track_actions = []

    def makeState(self, state):
        snake_x = state[0]
        snake_y = state[1]
        food_x = state[3]
        food_y = state[4]
        body = state[2]

        ## wall left or right
        if (snake_x == 40):
            adjoining_walls_x = 1
        elif (snake_x == 480):
            adjoining_walls_x = 2
        else:
            adjoining_walls_x = 0
        ## wall up or down
        if (snake_y == 40):
            adjoining_walls_y = 1
        elif (snake_y == 480):
            adjoining_walls_y = 2
        else:
            adjoining_walls_y = 0
        ## food direction left or right
        if (food_x < snake_x):
            food_dir_x = 1
        elif (food_x > snake_x):
            food_dir_x = 2
        else:
            food_dir_x = 0
        ## food direction up or down
        if (food_y < snake_y):
            food_dir_y = 1
        elif (food_y > snake_y):
            food_dir_y = 2
        else:
            food_dir_y = 0
        ## is body on top
        new_x = snake_x
        new_y = snake_y - 40
        if (new_x, new_y) in body:
            adjoining_body_top = 1
        else:
             adjoining_body_top = 0
        ## is body on bottom
        new_x = snake_x
        new_y = snake_y + 40
        if (new_x, new_y) in body:
            adjoining_body_bottom = 1
        else:
             adjoining_body_bottom = 0
        ## is body on left
        new_x = snake_x - 40
        new_y = snake_y
        if (new_x, new_y) in body:
            adjoining_body_left = 1
        else:
             adjoining_body_left = 0
        ## is body on right
        new_x = snake_x + 40
        new_y = snake_y
        if (new_x, new_y) in body:
            adjoining_body_right = 1
        else:
             adjoining_body_right = 0
        return (adjoining_walls_x, adjoining_walls_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)

    # # TAKEN FROM MOVE(...) IN SNAKE.PY
    # def check_alive(self, state, action):
    #     track_head = None
    #     if len(state[2]) == 1:
    #         track_head = state[2][0]
    #     state[2].append((state[0], state[1]))
    #     if action == 0:
    #         state[1] -= 40
    #     elif action == 1:
    #         state[1] += 40
    #     elif action == 2:
    #         state[0] -= 40
    #     elif action == 3:
    #         state[0] += 40

    #     # check body length less than points
    #     # if len(state[2]) > points:
    #     # del(state[2][0])

    #     if len(state[2]) >= 1:
    #         for seg in state[2]:
    #             if state[0] == seg[0] and state[1] == seg[1]:
    #                 return False

    #     if len(state[2]) == 1:
    #         if track_head == (state[0], state[1]):
    #             return False

    #     if (state[0] < 40 or state[1] < 40 or
    #         state[0] + 40 > 520 or state[1] + 40 > 520):
    #         return False

    #     return True

    def compute_q(self, prev_state, prev_action, state, dead, points):
        prev = self.makeState(prev_state)
        #a
        alpha = self.C / (self.C + self.N[prev[0]][prev[1]][prev[2]][prev[3]][prev[4]][prev[5]][prev[6]][prev[7]][prev_action])

        #R(s)
        if points - self.points > 0:
            R = 1
        elif dead:
            R = -1
        else:
            R = -0.1

        s_ = self.makeState(state)
        q_U = self.Q[s_[0]][s_[1]][s_[2]][s_[3]][s_[4]][s_[5]][s_[6]][s_[7]][0]
        q_D = self.Q[s_[0]][s_[1]][s_[2]][s_[3]][s_[4]][s_[5]][s_[6]][s_[7]][1]
        q_L = self.Q[s_[0]][s_[1]][s_[2]][s_[3]][s_[4]][s_[5]][s_[6]][s_[7]][2]
        q_R = self.Q[s_[0]][s_[1]][s_[2]][s_[3]][s_[4]][s_[5]][s_[6]][s_[7]][3]
        a_prime = max(q_U, q_D, q_L, q_R)

        #Q_val
        Q_val = self.Q[prev[0]][prev[1]][prev[2]][prev[3]][prev[4]][prev[5]][prev[6]][prev[7]][prev_action]
        return ( Q_val + alpha * (R + self.gamma*a_prime - Q_val) )


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
        curr_state = self.makeState(state)

        if dead:
            # print("GAME OVER")
            # print(state[0], state[1])
            # print("POINTS: ", points)
            # print("ACTIONS: ", self.track_actions)
            prev_state = self.makeState(self.s)
            self.Q[prev_state[0]][prev_state[1]][prev_state[2]][prev_state[3]][prev_state[4]][prev_state[5]][prev_state[6]][prev_state[7]][self.a] = self.compute_q(self.s, self.a, state, dead, points)
            self.reset()
            return 

        if (self.s != None and self.a != None):
            prev_state = self.makeState(self.s)
            new_q = self.compute_q(self.s, self.a, state, dead, points)
            self.Q[prev_state[0]][prev_state[1]][prev_state[2]][prev_state[3]][prev_state[4]][prev_state[5]][prev_state[6]][prev_state[7]][self.a] = new_q


        ## choosing which action to pick
        utility = np.zeros(4)
        for i in range(4):
            N_val = self.N[curr_state[0]][curr_state[1]][curr_state[2]][curr_state[3]][curr_state[4]][curr_state[5]][curr_state[6]][curr_state[7]][i]
            Q_val = self.Q[curr_state[0]][curr_state[1]][curr_state[2]][curr_state[3]][curr_state[4]][curr_state[5]][curr_state[6]][curr_state[7]][i]
            if N_val < self.Ne:
                utility[i] = 1
            else:
                utility[i] = Q_val
        idxs = np.argwhere(utility == np.amax(utility))
        action = idxs[len(idxs)-1]
        self.track_actions.append(action)

        ## updating N(s,a)
        self.N[curr_state[0]][curr_state[1]][curr_state[2]][curr_state[3]][curr_state[4]][curr_state[5]][curr_state[6]][curr_state[7]][action] += 1

        self.s = state
        self.a = action
        self.points = points
        return action
