import jax
import jax.numpy as jnp
import json
import pickle
from tqdm import trange

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('../')
sys.path.append('./')

from environment.maze import MetaMaze
from model.rnn_policy import ScannedRNN, ActorCriticRNN


rng = jax.random.PRNGKey(123)

expe_num = '20231128_144348'

# Load config
with open(f'/data/draco/cleain/imitation_gap_minigrid/logs_rl/{expe_num}/args.json', 'r') as file:
    config = json.load(file)

# Load params
with open(f'/data/draco/cleain/imitation_gap_minigrid/logs_rl/{expe_num}/params_4.pkl', 'rb') as file:
    params = pickle.load(file)

# Def env
env = MetaMaze(**config['params'])
env_params = env.default_params

# Def network
network = ActorCriticRNN(env.action_space(env_params).n, config=config)

lengths = []
returns = []

n_eval = 3
for _ in trange(n_eval):

    tt = 0
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Create the Pendulum-v1 environment
    env = MetaMaze(**config['params'])
    env_params = env.default_params

    obs, env_state = env.reset(key_reset, env_params)
    done = False
    hstate = ScannedRNN.initialize_carry((1, 128))

    while not done and tt < 200:

        rng, _rng = jax.random.split(rng)

        # SELECT ACTION
        ac_in = (obs[jnp.newaxis,jnp.newaxis, :], jnp.array([done])[jnp.newaxis,jnp.newaxis, :])
        hstate, pi, value = network.apply(params, hstate, ac_in)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        # STEP ENV
        rng, _rng = jax.random.split(rng)

        # Update the environment
        _, env_state, reward, done, info = env.step(_rng, env_state, action[0], env_params)

        # Get the observation
        obs = env.get_obs(env_state, env_params, config['is_expert'])

        tt += 1

    lengths.append(tt)
    returns.append(reward.item())

# Save evaluation
save_eval_path = f'/data/draco/cleain/imitation_gap_minigrid/logs_rl/{expe_num}/eval_{n_eval}.json'
with open(save_eval_path, 'w') as file:
    json.dump({'l': lengths,
               'r': returns}, file)