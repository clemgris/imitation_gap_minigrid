from datetime import datetime
import jax
import json
import pickle

from rnn_ppo import make_train

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

##
# CONFIG
##

# Training config
config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 64,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 5e9,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
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
    "is_expert": False
}

print(f'Available devices: {jax.devices()}')

# Ckeckpoint path
current_time = datetime.now() 
date_string = current_time.strftime("%Y%m%d_%H%M%S")

log_folder = f"logs_rl/{date_string}"
os.makedirs(log_folder, exist_ok='True')

config['log_folder'] = log_folder

# Save training config
training_args = config

with open(os.path.join(log_folder, 'args.json'), 'w') as json_file:
    json.dump(config, json_file, indent=4)

##
# TRAINING
##
# with jax.disable_jit(): # DEBUG
rng = jax.random.PRNGKey(123)
train_jit = jax.jit(make_train(config))
out = train_jit(rng)

##
# SAVING
##

with open(os.path.join(config['log_folder'], f'training_metrics_{config["UPDATE_EPOCHS"]}.pkl'), "wb") as json_file:
    pickle.dump(out['metric'], json_file)

# Save model weights
with open(os.path.join(config['log_folder'], f'params_{config["UPDATE_EPOCHS"]}.pkl'), 'wb') as f:
    pickle.dump(out['runner_state'][0].params, f)