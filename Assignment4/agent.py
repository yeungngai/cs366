import random

import numpy as np

from action_value_table import ActionValueTable

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3  # agents actions
GAMMA = 0.99
STEP_SIZE = 0.25
EPSILON = 0.1


class QLearningAgent():
    '''
    Implement your code for a Q-learning agent here. We have provided code implementing the action-value
    table in action_value_table.py. Here, you will implement the get_action(), get_greedy_action() and update() methods.
    '''

    def __init__(self, dimension):
        self.actions = [UP, DOWN, LEFT, RIGHT]
        num_actions = len(self.actions)
        self.values = ActionValueTable(dimension, num_actions)
        self.gamma = GAMMA
        self.step_size = STEP_SIZE
        self.epsilon = EPSILON

    def update(self, state, action, reward, next_state, done):
        '''
        This function will update the values stored in self.value using Q-learning. 

        HINT: Use self.values.get_value and self.values.set_value
        HINT: Remember to add a special case to handle the terminal state

        parameters:
            state : (list) a list of type [bool, int, int] where the first entry is whether the agent
            posseses the key, and the next two entries are the row and column position of the agent in 
            the maze
            
            action : (int) the action taken at state

            reward : float

        returns:
            action : (int) a epsilon-greedy action for state 
        '''

        # Terminal state
        if (int(reward) == 1) and (int(state[0]) == 1):
            done = True
            print(done)
            return done

        oldQval = float(self.values.get_value(state, action))

        maxQval = np.max([float(self.values.get_value(next_state, 0)), float(self.values.get_value(next_state, 1)),
                          float(self.values.get_value(next_state, 2)), float(self.values.get_value(next_state, 3))])

        tempDiff = reward + self.gamma * maxQval - oldQval
        newQval = oldQval + self.step_size*tempDiff

        self.values.set_value(state, action, newQval)


    def get_action(self, state):
        '''
        This function returns an action from self.actions given a state. 

        Implement this function using an epsilon-greedy policy. 

        HINT: use np.random.rand() to generate random numbers
        HINT: If more than one action has maximum value, treat them all as the greedy action. In other words,
        if there are b greedy actions, each should have probability epsilon/b + |A|, where |A| is 
        the number of actions in this state.

        parameters:
            state : (list) a list of type [bool, int, int] where the first entry is whether the agent
            posseses the key, and the next two entries are the row and column position of the agent in 
            the maze

        returns:
            action : (int) a epsilon-greedy action for state 

        '''

        if np.random.rand() > self.epsilon:
            # Get Q values of the four directions
            q_vals = [float(self.values.get_value(state, 0)), float(self.values.get_value(state, 1)),
                      float(self.values.get_value(state, 2)), float(self.values.get_value(state, 3))]

            # get the max Q value
            maxQ = np.max(q_vals)
            maxQvalIndex = []
            for i in range(len(self.actions)):
                if q_vals[i] == maxQ:
                    maxQvalIndex.append(i)

            # Has multiple max Q values
            if len(maxQvalIndex) > 1:
                return np.random.choice(maxQvalIndex)
            else:
                # Only one max Q value action
                return q_vals.index(np.max(q_vals))

        else:
            # Take Random Action
            return np.random.randint(len(self.actions))



    def get_greedy_action(self, state):
        '''
        This function returns an action from self.actions given a state. 

        Implement this function using a greedy policy, i.e. return the action with the highest value
        HINT: If more than more than one action has maximum value, uniformly randomize amongst them

        parameters:
            state : (list) a list of type [bool, int, int] where the first entry is whether the agent
            posseses the key, and the next two entries are the row and column position of the agent in 
            the maze

        returns:
            action : (int) a greedy action for state 
        '''

        q_vals = [float(self.values.get_value(state, 0)), float(self.values.get_value(state, 1)),
                  float(self.values.get_value(state, 2)), float(self.values.get_value(state, 3))]

        maxQ = np.max(q_vals)
        maxQvalIndex = []
        for i in range(len(self.actions)):
            if q_vals[i] == maxQ:
                maxQvalIndex.append(i)

        # Has multiple max Q values
        if len(maxQvalIndex) > 1:
            return np.random.choice(maxQvalIndex)
        else:
            # Only one max Q value action
            return q_vals.index(np.max(q_vals))
