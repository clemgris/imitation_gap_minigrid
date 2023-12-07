from datetime import datetime
import jax
import json
import os 

from rnnbc import make_train

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

##
# CONFIG
##

# Training config
config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 32,
    "NUM_ENVS_EVAL": 100,
    "NUM_STEPS": 200,
    "NUM_EVAL_STEPS": 200,
    'freq_save': 10,
    "TOTAL_TIMESTEPS": 5e5,
    "NUM_EPOCHS": 101,
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
    "key": 123,
    "full_obs": True,
    "expert_expe_num": '20231205_100708',
    "load_path": '/data/draco/cleain/imitation_gap_minigrid/logs_rl'
}

# Ckeckpoint path
current_time = datetime.now() 
date_string = current_time.strftime("%Y%m%d_%H%M%S")

log_folder = f"logs_bc/{date_string}"
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

# with jax.disable_jit(): # DEBUG
training_dict = training.train()
