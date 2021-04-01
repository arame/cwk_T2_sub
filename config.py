import os


class Hyper:
    total_episodes = 500
    # N is the number of cells in the side of the grid.
    # If you change it, remember to change the cell locations
    # for the breadcrumbs and obstacles. It needs to be an odd number.
    N = 7
    gamma = 0.51
    alpha = 0.8
    init_epsilon = 0.95
    decay = 0.995
    epsilon_threshold = 0.001
    no_breadcrumbs = 10     # Can vary from 1 to 17 for a 7 x 7 grid
    is_ghost = True         # Set to True for stochastic environment
    show_step = False       # Usually set to false to reduce excessive output.
    print_episodes = True   # Usually set to true
    images_folder = "images"

    [staticmethod]   
    def display():
        if os.path.isdir(Hyper.images_folder) == False:
            os.mkdir(Hyper.images_folder)
        print("The Hyperparameters")
        print("-------------------")
        print(f"Threshold for exploitation (epsilon) = {Hyper.init_epsilon}")
        print(f"epsilon decay = {Hyper.decay}")
        print(f"minimum value of epsilon = {Hyper.epsilon_threshold}")
        print(f"learning rate (alpha) = {Hyper.alpha}")
        print(f"discount factor (gamma) = {Hyper.gamma}")
        print(f"total number of breadcrumbs {Hyper.no_breadcrumbs}")

class Constants:
    EMPTY = 0
    BREADCRUMB = 1
    OBSTACLE = 2
    START = 3
    GHOST = 4
    AGENT = 9
    EMPTY_X = "."
    BREADCRUMB_X = "b"
    OBSTACLE_X = "X"
    START_X = "S"
    AGENT_X = "A"
    GHOST_X = "G"
    EMPTY_REWARD = -1
    BREADCRUMB_REWARD = 10
    OBSTACLE_REWARD = -100
    GHOST_REWARD = -500
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    GHOST = 4
    OBSTACLE_CELL_IDS = [8, 11, 16, 25, 30, 32, 39]
    # The breadcrumb cells selected depends on Hyper.no_breadcrumbs value
    BREADCRUMB_CELL_IDS = [17, 12, 40, 38, 15, 23, 31, 22, 10, 19, 29, 36, 18, 9, 33, 37, 26]
    WIN_CELL = 0
    LOSE_CELL = 1