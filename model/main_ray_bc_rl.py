from datetime import datetime
import jax
import json
import os
from ray import tune
from ray.air import RunConfig

from rnn_bc_rl import make_train

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

##
# CONFIG
##

# Training config
trial_space = {
    "LR": 2.5e-4,
    "NUM_ENVS": 32,
    "NUM_ENVS_EVAL": 100,
    "NUM_STEPS": 200,
    "NUM_EVAL_STEPS": 200,
    'freq_save': 10,
    "TOTAL_TIMESTEPS": 5e5,
    "NUM_EPOCHS": 201,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "WEIGHT_BC": 0,
    "WEIGHT_RL": tune.grid_search([1]), #, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    "MAX_GRAD_NORM": 0.5,
    "ANNEAL_LR": True,
    "DEBUG": False,
    "params": {'maze_size': 13,
               'rf_size': 3},
    "key": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "full_obs": False,
    "expert_expe_num": '20231214_190953', # expert (full obs)
    "load_path": '/data/draco/cleain/imitation_gap_minigrid/logs_rl'
}

def train(config):
    # Ckeckpoint path
    current_time = datetime.now() 
    date_string = current_time.strftime("%Y%m%d_%H%M%S")

    log_folder = f"logs_bc_rl/{date_string}"
    os.makedirs(log_folder, exist_ok='True')

    config['log_folder'] = log_folder

    # Save training config
    training_args = config

    with open(os.path.join(log_folder, 'args.json'), 'w') as json_file:
        json.dump(training_args, json_file, indent=4)

    ##
    # TRAINING
    ##

    training = make_train(config)
    _ = training.train()

# local_dir = f"ray_logs_bc_rl/w_bc_{trial_space['WEIGHT_BC']}_w_rl_{trial_space['WEIGHT_RL']}"
# if not os.path.exists(local_dir):
#     os.makedirs(local_dir)

train = tune.with_resources(train, {"gpu": 0.05})
tuner = tune.Tuner(train, param_space=trial_space, run_config=RunConfig(local_dir='/data/draco/cleain/imitation_gap_minigrid/ray_results'))
tuner.fit()