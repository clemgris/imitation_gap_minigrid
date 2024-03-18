from datetime import datetime
import jax
import json
import os
from ray import tune
from ray.air import RunConfig

from rnn_ppo_bc import make_train

import pickle
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

##
# CONFIG
##

# Training config
trial_space = {
    "LR": 2.5e-4,
    "NUM_ENVS": 64,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 5e8,
    "UPDATE_EPOCHS": 1,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ANNEAL_LR": True,
    "DEBUG": False,
    "params": {'maze_size': 13,
               'rf_size': 3},
    "WEIGHT_BC": 0,
    "WEIGHT_RL": 1, # tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    "KEY": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "full_obs": False,
    "expert_expe_num": '20231214_190953', # Full obs expert
    # "expert_expe_num": '20231214_192147', # Limited obs expert
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
    train_jit = jax.jit(make_train(config))
    out = train_jit()

    ##
    # SAVING
    ##

    with open(os.path.join(config['log_folder'], f'training_metrics_{config["UPDATE_EPOCHS"]}.pkl'), "wb") as json_file:
        pickle.dump(out['metric'], json_file)

    # Save model weights
    with open(os.path.join(config['log_folder'], f'params_{config["UPDATE_EPOCHS"]}.pkl'), 'wb') as f:
        pickle.dump(out['runner_state_rl'][0].params, f)

# for WEIGHT_RL in [0,1,2,3,4,5,6,7,8,9,10,15]:
#     trial_space['WEIGHT_RL'] = WEIGHT_RL

current_time = datetime.now() 
date_string = current_time.strftime("%Y%m%d")

train = tune.with_resources(train, {"gpu": 0.5})
tuner = tune.Tuner(train, param_space=trial_space, run_config=RunConfig(local_dir=f'/data/draco/cleain/imitation_gap_minigrid/ray_results_{date_string}'))
tuner.fit()