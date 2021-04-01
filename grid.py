import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sn
import sys
from policy import Policy
from collections import namedtuple
from config import Hyper, Constants
from q_learn import Q_learn


class Pacman_grid:
    def __init__(self):
        self.no_cells = Hyper.N * Hyper.N
        #self.results = np.zeros((2, int(Hyper.total_episodes / 100) + 1), dtype=np.int16)
        self.results = np.zeros((2, Hyper.total_episodes), dtype=np.int16)
        self.no_episodes = 0
        self.setup_display_dict()
        self.setup_env()
        self.setup_reward_dict()
        self.setup_action_dict()
        self.Q = Q_learn(self.no_actions)
        self.policy = Policy()
        self.timesteps_per_episode = []
        self.rewards_per_episode = []

    def setup_env(self):
        self.state_position_dict = {(i * Hyper.N + j):(i, j) for i in range(Hyper.N) for j in range(Hyper.N)}
        self.position_state_dict = {v: k for k, v in self.state_position_dict.items()}
        self.env = np.zeros((Hyper.N, Hyper.N), dtype = np.int8)
        self.env_counter = np.zeros((Hyper.N, Hyper.N), dtype = np.int16)
        # Borders are obstacles
        self.env[0, :] = self.env[-1, :] = self.env[:, 0] = self.env[:, -1] = Constants.OBSTACLE
        arr_temp = np.nonzero(self.env == Constants.OBSTACLE)
        self.border_cells_coords = [(arr_temp[0][i], arr_temp[1][i]) for i in range(len(arr_temp[0]))]
        self.env_dict = {i:[] for i in range(self.no_cells)}
        low_lim = -1
        high_lim = Hyper.N
        for cell_id in range(self.no_cells):
            if cell_id > high_lim - 1:
                low_lim += Hyper.N
                high_lim += Hyper.N
            actions = self.get_actions_for_cell_id(cell_id, low_lim, high_lim)
            self.env_dict[cell_id].append(np.array(actions))
        # Start cell in the middle

        _, i, j = self.get_start_cell_coords()
        self.env[i, j] = Constants.START

        # Replace empty cells with obstacles. 
        #no_obstacles = Hyper.N - 2
        self.populate_env_with_obstacles()
        # Replace empty cells with breadcrumbs. 
        self.populate_env_with_breadcrumbs()
        self.orig_env = np.copy(self.env)

    def get_actions_for_cell_id(self, cell_id, low_lim, high_lim):
        # These actions are to enable the ghost to move around the grid
        # from one cell to the next
        actions = []
        up = cell_id - Hyper.N
        if up > 0:
            actions.append(up)
        down = cell_id + Hyper.N
        if down < self.no_cells:
            actions.append(down)
        left = cell_id - 1
        if left > low_lim:
            actions.append(left)
        right = cell_id + 1
        if right < high_lim:
            actions.append(right)
        return actions


    def populate_env_with_obstacles(self):
        # selecting obstacles in random locations risks the possibility of
        # insoluble games with a breadcrumb inaccessible surrounded by obstacles.
        # To rectify this, obstacles are set from a list of coordinates
        for cell_id in Constants.OBSTACLE_CELL_IDS:
            coord = self.state_position_dict[cell_id]
            self.env[coord[0], coord[1]] = Constants.OBSTACLE

    def populate_env_with_breadcrumbs(self):
        # selecting obstacles in random locations risks the possibility of
        # insoluble games with a breadcrumb inaccessible surrounded by obstacles.
        # To rectify this, obstacles are set from a list of coordinates
        no_breadcrumbs = 0
        for cell_id in Constants.BREADCRUMB_CELL_IDS:
            no_breadcrumbs += 1
            if no_breadcrumbs > Hyper.no_breadcrumbs:
                break   
            coord = self.state_position_dict[cell_id]
            self.env[coord[0], coord[1]] = Constants.BREADCRUMB

        arr_temp = np.nonzero(self.env == Constants.BREADCRUMB)
        self.id_breadcrumb_coords = {i : (arr_temp[0][i], arr_temp[1][i]) for i in range(len(arr_temp[0]))}
        self.breadcrumb_coords_id = {v: k for k, v in self.id_breadcrumb_coords.items()} 

    def populate_env_with_random_breadcrumbs(self):
        # Keep a record of the breadcrumb coordinates
        # This can be used to calculate the index of the Q table
        self.populate_env_with_state(Constants.BREADCRUMB, Hyper.no_breadcrumbs)
        arr_temp = np.nonzero(self.env == Constants.BREADCRUMB)
        self.id_breadcrumb_coords = {i : (arr_temp[0][i], arr_temp[1][i]) for i in range(len(arr_temp[0]))}
        self.breadcrumb_coords_id = {v: k for k, v in self.id_breadcrumb_coords.items()} 

    def populate_env_with_state(self, state, limit):
        # This method is needed as sometimes the empty cells returned have duplicates
        count = 0
        while count < limit:
            no_cells = limit - count
            empty_coord = self.get_empty_cells(no_cells)
            self.env[empty_coord[0], empty_coord[1]] = state
            count = np.count_nonzero(self.env == state)

    def get_empty_cells(self, n_cells):
        empty_cells_coord = np.where( self.env == Constants.EMPTY)
        selected_indices = np.random.choice( np.arange(len(empty_cells_coord[0])), n_cells )
        selected_coordinates = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]
        
        if n_cells == 1:
            return np.asarray(selected_coordinates).reshape(2,)
        
        return selected_coordinates

    def get_start_cell_coords(self):
        # Start cell in the middle
        start_cell_id = int((self.no_cells - 1) / 2)     # N must be an odd number
        i, j = self.state_position_dict[start_cell_id]
        return start_cell_id, i, j

    def setup_reward_dict(self):
        self.reward_dict = {
            Constants.EMPTY: Constants.EMPTY_REWARD,
            Constants.BREADCRUMB: Constants.BREADCRUMB_REWARD,
            Constants.OBSTACLE: Constants.OBSTACLE_REWARD,
            Constants.GHOST: Constants.GHOST_REWARD,
            Constants.START: Constants.EMPTY_REWARD
            }

    def setup_action_dict(self):
        _Action = namedtuple('Action', 'name index delta_i delta_j')
        up = _Action('up', Constants.UP, -1, 0)    
        down = _Action('down', Constants.DOWN, 1, 0)    
        left = _Action('left', Constants.LEFT, 0, -1)    
        right = _Action('right', Constants.RIGHT, 0, 1)
        self.index_to_actions = {} 
        if Hyper.is_ghost:
            ghost = _Action('ghost', Constants.GHOST, 0, 0)
            for action in [up, down, left, right, ghost]:
                self.index_to_actions[action.index] = action
        else:
            for action in [up, down, left, right]:
                self.index_to_actions[action.index] = action

        self.no_actions = len(self.index_to_actions)

    def setup_display_dict(self):
        self.dict_map_display={ Constants.EMPTY: Constants.EMPTY_X,
                                Constants.BREADCRUMB: Constants.BREADCRUMB_X,
                                Constants.OBSTACLE: Constants.OBSTACLE_X,
                                Constants.START: Constants.START_X,
                                Constants.GHOST: Constants.GHOST_X,
                                Constants.AGENT: Constants.AGENT_X}

    def reset(self):
        # reset the breadcrumb indexes on the Q matrix
        self.Q.reset()
        # set the grid to how it was before
        self.env = np.copy(self.orig_env)
        # put agent in the start cell of the environment
        start_cell_id, i, j = self.get_start_cell_coords()
        self.env[i, j] = Constants.AGENT
        self.agent_cell_id = start_cell_id
        self.time_step = 0
        self.total_reward_per_episode = 0
        self.done = False
        self.breadcrumb_cnt = 0
        self.prev_state = Constants.START
        """ if self.no_episodes > 0 and self.no_episodes % 100 == 0:
            self.result_index += 1 """
        self.no_episodes += 1

        if Hyper.is_ghost:
            self.set_ghost()

    def set_ghost(self):
        # Choose a random start location for the ghost on the border cells
        idx = random.randint(0, len(self.border_cells_coords) - 1)
        self.ghost_cell_coords = self.border_cells_coords[idx]
        self.ghost_cell_id = self.position_state_dict[self.ghost_cell_coords[0], self.ghost_cell_coords[1]]
        self.prev_ghost_cell_id = self.ghost_cell_id
        self.prev_ghost_cell_state = Constants.OBSTACLE
        self.env[self.ghost_cell_coords[0], self.ghost_cell_coords[1]] = Constants.GHOST

    def move_ghost(self):
        (i, j) = self.state_position_dict[self.ghost_cell_id]
        self.env[i, j] = self.prev_ghost_cell_state
        temp = self.env_dict[self.ghost_cell_id]
        self.ghost_cell_id = np.random.choice(temp[0])
        (i, j) = self.state_position_dict[self.ghost_cell_id]
        if self.env[i, j] == Constants.AGENT:
            self.prev_ghost_cell_state = self.prev_state
        else:
            self.prev_ghost_cell_state = self.env[i, j]
        self.env[i, j] = Constants.GHOST

    def get_available_actions_including_ghost(self):
        # Check if an up, down, left, right action needs to be replaced by a ghost action
        available_actions = []
        agent_coords = self.state_position_dict[self.agent_cell_id]
        ghost_coords = self.state_position_dict[self.ghost_cell_id]
        action = self.get_action_for_state(Constants.UP, agent_coords, ghost_coords)
        available_actions.append(action)
        action = self.get_action_for_state(Constants.DOWN, agent_coords, ghost_coords)
        available_actions.append(action)
        action = self.get_action_for_state(Constants.LEFT, agent_coords, ghost_coords)
        available_actions.append(action)
        action = self.get_action_for_state(Constants.RIGHT, agent_coords, ghost_coords)
        available_actions.append(action)
        return available_actions

    def get_action_for_state(self, action, agent_coords, ghost_coords):
        coord = np.zeros(2)
        coord[0] = agent_coords[0] + self.index_to_actions[action].delta_i
        coord[1] = agent_coords[1] + self.index_to_actions[action].delta_j
        if ghost_coords[0] == coord[0] and ghost_coords[1] == coord[1]:
            # If the action moves onto a cell containing the ghost,
            # replace the previous action with the ghost action
            action = Constants.GHOST    
        return action


    def step(self, episode):
        self.time_step += 1
        # Q Learning algorithm code takes place here
        action = self.policy.get(self.agent_cell_id, self.Q)
        new_cell_id = self.get_cell_id_for_action(action)
        reward = self.get_reward(new_cell_id)
        self.total_reward_per_episode += reward
        self.Q.update(self.agent_cell_id, new_cell_id, action, reward)
        self.agent_step(new_cell_id)
        if Hyper.show_step:
            self.print_curr_grid(f"Environment for step {self.time_step}")
   
        if self.time_step > 1000:
            print("Too many timesteps")
            #self.results[Constants.LOSE_CELL, self.result_index] += 1
            self.results[Constants.LOSE_CELL, episode] += 1
            self.done = True
            return self.done

        self.done = self.breadcrumb_cnt == Hyper.no_breadcrumbs
        if self.done:
            #self.results[Constants.WIN_CELL, self.result_index] += 1
            self.results[Constants.WIN_CELL, episode] += 1

        return self.done

    def ghost_step(self, episode):
        self.time_step += 1
        self.move_ghost()
        # Q Learning algorithm code takes place here
        available_actions = self.get_available_actions_including_ghost()
        action = self.policy.get_with_available_actions(self.agent_cell_id, self.Q, available_actions)
        new_cell_id = self.get_cell_id_for_action(action)
        reward = self.get_reward(new_cell_id)
        self.total_reward_per_episode += reward
        self.Q.update(self.agent_cell_id, new_cell_id, action, reward)
        self.agent_step(new_cell_id)
        if Hyper.show_step:
            self.print_curr_grid(f"Environment for step {self.time_step}")
   
        if Hyper.is_ghost and self.ghost_cell_id == self.agent_cell_id:
            if Hyper.print_episodes:
                print("You lost to the ghost!")
            self.results[Constants.LOSE_CELL, episode] += 1
            self.done = True
            return self.done

        if self.time_step > 5000:
            # This is a safeguard check, it shouldn't happen.
            # It does prevent a possible infinite loop.
            print("Too many timesteps")
            self.results[Constants.LOSE_CELL, episode] += 1
            self.done = True
            return self.done

        self.done = self.breadcrumb_cnt == Hyper.no_breadcrumbs
        if self.done:
            self.results[Constants.WIN_CELL, episode] += 1

        return self.done

    def check_if_cell_breadcrumb(self, cell_id):
        i, j = self.state_position_dict[cell_id]
        state = self.env[i, j]
        is_breadcrumb = state == Constants.BREADCRUMB
        return is_breadcrumb

    def get_reward(self, cell_id):
        i, j = self.state_position_dict[cell_id]
        state = self.env[i, j]
        reward = self.reward_dict[state]
        return reward

    def agent_step(self, new_cell_id):
        # check if the new cell location is on an obstacle
        # if it is, do not change the environment or move the agent
        i, j = self.state_position_dict[new_cell_id]
        if self.env[i, j] == Constants.OBSTACLE:
            return

        # When the Pacman agent moves from the start cell, 
        # that cell will be empty
        i, j = self.state_position_dict[self.agent_cell_id]
        if self.prev_state == Constants.BREADCRUMB:
            # When the Pacman agent leaves the breadcrumb cell, 
            # it will change state to empty in the grid
            # and become empty in the Q table
            breadcrumb_id = self.breadcrumb_coords_id[i, j]
            self.Q.update_Q_table_index(breadcrumb_id)
            self.breadcrumb_cnt += 1 

        # The Pacman agent leaves the current cell, which will be then always be empty
        self.env[i, j] = Constants.EMPTY
        
        self.agent_cell_id = new_cell_id
        i, j = self.state_position_dict[self.agent_cell_id]
        self.prev_state = self.env[i, j]
        self.env[i, j] = Constants.AGENT
        # increment the number of steps counter for the current cell.
        self.env_counter[i, j] += 1

    def get_cell_id_for_action(self, action):
        # to move the agent, get the coordinates of the current cell
        # change one of the coordinates, and return the cell_id of the new cell
        _action = self.index_to_actions[action]
        if _action.index == Constants.GHOST:
            new_cell_id = self.ghost_cell_id
            return new_cell_id

        i, j = self.state_position_dict[self.agent_cell_id]
        i += _action.delta_i
        j += _action.delta_j
        new_cell_id = self.position_state_dict[i, j]
        return new_cell_id

    def print_orig_grid_to_txt(self, caption):
        # Print the original grid to the text file
        env_filename = f"images/env_lr{Hyper.alpha}_discount_rate{Hyper.gamma}_bc{Hyper.no_breadcrumbs}".replace(".","") + ".txt"
        if Hyper.is_ghost:
            env_filename = env_filename.replace("images/", "images/ghost_")
        sys.stdout = open(env_filename,'wt')
        self.print_grid(caption, self.orig_env)

    def print_curr_grid(self, caption):
        # Print the latest grid for diagnostic purposes
        self.print_grid(caption, self.env)

    def print_grid(self, caption, env):
        # Use characters rather than integers to make it easier to interpret the grid
        print(caption)
        lower = 0
        higher = Hyper.N - 1
        for i in range(Hyper.N):
            line = ''
            for j in range(Hyper.N):
                state_id = env[i,j]
                line += self.dict_map_display[state_id] + " "
            line += f"    cells {lower} - {higher}"
            print(line)
            lower += Hyper.N
            higher += Hyper.N

    def print_episode_results(self, episodes):
        caption = f"Completed environment after {episodes} episodes and {self.time_step} timesteps, total reward: {self.total_reward_per_episode} with epsilon: {self.policy.epsilon}"
        print(caption)

    def save_episode_stats(self):
        self.timesteps_per_episode.append(self.time_step)
        self.rewards_per_episode.append(self.total_reward_per_episode)

    def print_results(self):
        hm_filename = f"images/hm_lr{Hyper.alpha}_discount_rate{Hyper.gamma}_bc{Hyper.no_breadcrumbs}".replace(".","") + ".jpg"
        rw_filename = f"images/rw_lr{Hyper.alpha}_discount_rate{Hyper.gamma}_bc{Hyper.no_breadcrumbs}".replace(".","") + ".jpg"
        rw_ma_filename = f"images/rw_ma_lr{Hyper.alpha}_discount_rate{Hyper.gamma}_bc{Hyper.no_breadcrumbs}".replace(".","") + ".jpg"
        ts_filename = f"images/ts_lr{Hyper.alpha}_discount_rate{Hyper.gamma}_bc{Hyper.no_breadcrumbs}".replace(".","") + ".jpg"
        res_filename = f"images/res_lr{Hyper.alpha}_discount_rate{Hyper.gamma}_bc{Hyper.no_breadcrumbs}".replace(".","") + ".jpg"
        if Hyper.is_ghost:
            hm_filename = hm_filename.replace("images/", "images/ghost_")
            rw_filename = rw_filename.replace("images/", "images/ghost_")
            rw_ma_filename = rw_ma_filename.replace("images/", "images/ghost_")
            ts_filename = ts_filename.replace("images/", "images/ghost_")
            res_filename = res_filename.replace("images/", "images/ghost_")
        self.print_orig_grid_to_txt("Initial Environment")
        x_label_text = f"Episode # (learning rate = {Hyper.alpha}, discount factor = {Hyper.gamma})"
        _ = sn.heatmap(data=self.env_counter)
        plt.title("Number of steps per cell")
        plt.xlabel(x_label_text)
        plt.savefig(hm_filename)

        fig = plt.figure()
        fig.add_subplot(111)
        
        episodes = np.arange(1, len(self.timesteps_per_episode)+1)
        plt.title(f"Number of timesteps per episode for {Hyper.no_breadcrumbs} breadcrumbs")
        plt.plot(episodes, self.timesteps_per_episode)
        plt.ylabel('Steps')
        plt.xlabel(x_label_text)
        plt.savefig(ts_filename)

        fig = plt.figure()
        fig.add_subplot(111)
        x_label_text = f"Episode # (learning rate = {Hyper.alpha}, discount factor = {Hyper.gamma})"
        episodes = np.arange(1, len(self.rewards_per_episode)+1)
        plt.title(f"Value of rewards per episode for {Hyper.no_breadcrumbs} breadcrumbs")
        plt.plot(episodes, self.rewards_per_episode)
        plt.ylabel('Rewards')
        plt.xlabel(x_label_text)
        plt.savefig(rw_filename)

        self.moving_average_rewards = self.get_moving_average_rewards()
        episodes = np.arange(Hyper.total_episodes)
        fig = plt.figure()
        fig.add_subplot(111)
        x_label_text = f"Episode # (learning rate = {Hyper.alpha}, discount factor = {Hyper.gamma})"
        plt.title(f"Moving Average rewards per hundred episodes")
        plt.plot(episodes, self.moving_average_rewards, "b-")
        plt.ylabel('Results per 100')
        plt.xlabel(x_label_text)
        plt.savefig(rw_ma_filename)

        self.moving_average_results = self.get_moving_average_results()
        episodes = np.arange(Hyper.total_episodes)
        fig = plt.figure()
        fig.add_subplot(111)
        x_label_text = f"Episode # (learning rate = {Hyper.alpha}, discount factor = {Hyper.gamma})"
        plt.title(f"Moving Average per hundred episodes % wins/losses: Wins in blue, losses in red")
        plt.plot(episodes, self.moving_average_results[Constants.WIN_CELL], "b-", episodes, self.moving_average_results[Constants.LOSE_CELL], "r-")
        plt.ylabel('Results per 100')
        plt.xlabel(x_label_text)
        plt.savefig(res_filename)

    def get_moving_average_rewards(self):
        moving_average_rewards = np.zeros((Hyper.total_episodes), dtype=np.float)
        limit = 100
        for i in range(self.no_episodes):
            episodes = i + 1
            if i < limit:
                reward_total = sum(self.rewards_per_episode[0:episodes])
                moving_average_rewards[i] = reward_total / episodes
            else:
                reward_total = sum(self.rewards_per_episode[(episodes - limit):i])
                moving_average_rewards[i] = reward_total / limit
        return moving_average_rewards

    def get_moving_average_results(self):
        moving_average_results = np.zeros((2, Hyper.total_episodes), dtype=np.float)
        limit = 100
        for i in range(self.no_episodes):
            episodes = i + 1
            if i < limit:
                win_total = sum(self.results[Constants.WIN_CELL, 0:episodes])
                lose_total = sum(self.results[Constants.LOSE_CELL, 0:episodes])
                moving_average_results[Constants.WIN_CELL, i] = win_total / episodes * 100
                moving_average_results[Constants.LOSE_CELL, i] = lose_total / episodes * 100
            else:
                win_total = sum(self.results[Constants.WIN_CELL, (episodes - limit):i])
                lose_total = sum(self.results[Constants.LOSE_CELL, (episodes - limit):i])
                moving_average_results[Constants.WIN_CELL, i] = win_total / limit * 100
                moving_average_results[Constants.LOSE_CELL, i] = lose_total / limit * 100 
        return moving_average_results


