
import random
import numpy as np
from config import Hyper, Constants
class Policy():
    
    def __init__(self):
        self.epsilon = Hyper.init_epsilon
        
    def get(self, cell_id, Q):
        # Sample an action from the policy, given a state
        # The action returned here is the numerical representation
        is_greedy = random.random() > self.epsilon
        if is_greedy:
            action = Q.get_action_for_max_q(cell_id)
        else:
            action = random.randint(0, 3)
        
        return action

    # This method is the same as the above get method EXCEPT
    # one of the available actions might be ghost instead of one of (up, down, left, right)
    def get_with_available_actions(self, cell_id, Q, available_actions):
        is_greedy = random.random() > self.epsilon
        if is_greedy:
            action = Q.get_available_action_for_max_q(cell_id, available_actions)
        else:
            action = random.choice(available_actions)
        
        return action


    def update_epsilon(self):
        # called for each episode
        if self.epsilon > Hyper.epsilon_threshold:
            self.epsilon *= Hyper.decay

        return self.epsilon
        