from datetime import datetime
import jax
import json
import pickle
import time

from rnn_ppo_bc import make_train

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##
# CONFIG
##

# Training config
config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 64,
    "NUM_STEPS": 128, #64, 32, 128
    "TOTAL_TIMESTEPS": 5e6, #5e8, #5e9, #5e6, 5e8
    "UPDATE_EPOCHS": 1,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01, # 0
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ANNEAL_LR": True,
    "DEBUG": False,
    "params": {'maze_size': 13,
               'rf_size': 3},
    "WEIGHT_BC": 1,
    "WEIGHT_RL": 0,
    "KEY": 123,
    "full_obs": False, #False, True
    "expert_expe_num": '20231214_190953', # Full obs expert
    # "expert_expe_num": '20231214_192147', # Limited obs expert
    "load_path": '/data/draco/cleain/imitation_gap_minigrid/logs_rl'
}

print(f'Available devices: {jax.devices()}')

# Ckeckpoint path
current_time = datetime.now() 
date_string = current_time.strftime("%Y%m%d_%H%M%S")

log_folder = f"logs_ppo_bc/{date_string}"
os.makedirs(log_folder, exist_ok='True')

config['log_folder'] = log_folder

# Save training config
training_args = config

with open(os.path.join(log_folder, 'args.json'), 'w') as json_file:
    json.dump(config, json_file, indent=4)

##
# TRAINING
##

start= time.time()

# with jax.disable_jit(): # DEBUG
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

print('Training time', time.time() - start)