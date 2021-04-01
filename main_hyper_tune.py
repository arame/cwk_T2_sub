from grid import Pacman_grid
from config import Hyper, Constants
import optuna


# This code is the same as in main.py except
# it is using the Optuna library to 
# tune the hyperparameters
# Instead of producing graphs, it produces statistics


def objective(trial):
    Hyper.print_episodes = False
    Hyper.gamma = trial.suggest_float("Hyper.gamma", 0.01, 0.99)
    Hyper.alpha = trial.suggest_float("Hyper.alpha", 0.01, 0.99)
    Hyper.init_epsilon = trial.suggest_float("Hyper.init_epsilon", 0.9, 1)
    Hyper.epsilon_threshold = trial.suggest_float("Hyper.epsilon_threshold", 0.001, 0.1)
    Hyper.decay = trial.suggest_float("Hyper.decay", 0.990, 0.999)
    pacman_grid = Pacman_grid()
    for i in range(Hyper.total_episodes):
        pacman_grid.reset()
        done = False
        while done == False:
            if Hyper.is_ghost:
                done = pacman_grid.ghost_step()
            else:
                done = pacman_grid.step()
            pacman_grid.policy.update_epsilon()
        episodes = i + 1
        if Hyper.print_episodes:
            pacman_grid.print_episode_results(episodes)
        pacman_grid.save_episode_stats()

    # Average the rewards at the end to even out any outlier results caused
    # by the stochastic environment.
    sample = 100
    last_sample = -1 * sample
    reward_for_last_sample_episode = sum(pacman_grid.rewards_per_episode[last_sample:]) / sample
    return reward_for_last_sample_episode
    
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
# Note n_trials=10000. This is a lot of trials and will take a long time to run, but the results will be better
study.optimize(objective, n_trials=10000)
print("Number of finished trials: ", len(study.trials))
print(study.best_params)
print(study.best_value)     
    