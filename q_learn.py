import numpy as np
import random
import math
from config import Hyper, Constants

class Q_learn:
    def __init__(self, no_actions):
        # The state is a combination of cell id and whether it is empty, a breadcrumb or an obstacle.
        # The state of a cell can change from breadcrumb to empty for the same cell id
        # As a result our q table will exist in 3 dimensions; 
        # (state cell_id, state space index, actions).
        # The state space index is a number between 0 and 2**(N + 1), where N is in the number of breadcrumbs.
        # The upper limit of this range 2**(N + 1) is the total number of unique environments where each 
        # breadcrumb cell have 1 of 2 states, empty or breadcrumb.
        # When a breadcrumb cell changes state from breadcrumb to empty, the state space index changes
        # accordingly.
        # Each breadcrumb is given an index of between 0 to N-1. This index is used to update the 
        # state space index. 
        self.no_cells = Hyper.N * Hyper.N
        self.no_actions = no_actions
        self.no_indexes = pow(2, Hyper.no_breadcrumbs + 2)
        self.state_space_index = 0
        self.Q_table = np.zeros((self.no_cells, self.no_indexes, no_actions), dtype=np.float)

    def reset(self):
        # By setting the state space index to zero, the q table will be reset with all the
        # breadcrumb cells in their original state
        self.state_space_index = 0

    def update(self, old_cell_id, new_cell_id, action, reward):
        alpha = Hyper.alpha
        gamma = Hyper.gamma
        q_old = self.Q_table[old_cell_id, self.state_space_index, action]
        q_max = self.get_max_q(new_cell_id)
        q_val = q_old + alpha * (reward + gamma * q_max - q_old)
        self.Q_table[old_cell_id, self.state_space_index, action] = q_val 

    def get_max_q(self, cell_id):
        actions = self.get_actions_for_cell_id(cell_id)
        q_max = np.max(actions) 
        return q_max 

    def get_actions_for_cell_id(self, cell_id):
        actions = self.Q_table[cell_id, self.state_space_index, :]
        return actions

    def get_action_for_max_q(self, cell_id):
        actions = self.get_actions_for_cell_id(cell_id)
        # For greedy policy get the index of the maximum value
        # in the actions array. 
        # If more than 1 index is returned, choose 1 randomly
        _actions = np.where(actions == np.amax(actions))
        _action = np.random.choice(_actions[0], 1).item()
        return _action

    def get_available_action_for_max_q(self, cell_id, available_actions):
        _action = -1
        q_max = -100
        for action in available_actions:
            q_val = self.Q_table[cell_id, self.state_space_index, action]
            # Get the maximum value of the available actions
            if q_val > q_max:
                q_max = q_val

        _actions = []
        for action in available_actions:
            q_val = self.Q_table[cell_id, self.state_space_index, action]
            # find all of the available actions for the maximum value
            if q_val == q_max:
                q_max = q_val        
                _actions.append(action)

        # Of all the available actions selected with the maximum Q value, choose 1 at random
        _action = np.random.choice(_actions, 1).item()
        return _action

    def update_Q_table_index(self, breadcrumb_id):
        # The agent is located on a breadcrumb
        # The index of the Q table needs to change for the new state
        self.state_space_index += pow(2, breadcrumb_id)
