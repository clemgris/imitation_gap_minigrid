import jax
import jax.numpy as jnp
import json
import pickle
import argparse

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('../')
sys.path.append('./')

from environment.maze import MetaMaze
from environment.wrappers import LogWrapper
from model.rnn_policy import ScannedRNN, ActorCriticRNN

parser = argparse.ArgumentParser(description="Agent evaluation")
parser.add_argument('--type', '-t', choices=['rl', 'bc', 'bc_rl', 'ppo_bc'], help='Agent trained with BC or RL')
parser.add_argument('--expe_num', '-expe', type=str, help='Number of the experiment')
parser.add_argument('--epochs', '-e', type=int, help='Number of training epochs')
parser.add_argument('--n_evals', '-n', type=int, help='Number of evaluation environments')

if __name__ == "__main__":

    args = parser.parse_args()
    rng = jax.random.PRNGKey(42)

    logs = f'logs_{args.type}'
    expe_num = args.expe_num
    n_epochs = args.epochs

    # Load config
    with open(f'/data/draco/cleain/imitation_gap_minigrid/{logs}/{expe_num}/args.json', 'r') as file:
        config = json.load(file)

    if 'is_expert' in config.keys():
        config['full_obs'] = config['is_expert']

    # Load params
    with open(f'/data/draco/cleain/imitation_gap_minigrid/{logs}/{expe_num}/params_{n_epochs}.pkl', 'rb') as file:
        params = pickle.load(file)

    # Def env
    print('Init environment')
    env = MetaMaze(**config['params'])
    env = LogWrapper(env)
    env_params = env.default_params

    # Def network
    network = ActorCriticRNN(env.action_space(env_params).n)

    lengths = []
    returns = []

    n_eval = args.n_evals

    # Init environment
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, n_eval)
    _, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    init_rnn_state = ScannedRNN.initialize_carry((n_eval, 128))


    init_visiting_map = jnp.zeros((config['params']['maze_size'] + 2, config['params']['maze_size'] + 2), dtype=int)

    runner_state = (
        params,
        env_state,
        jnp.zeros((n_eval), dtype=bool),
        init_rnn_state,
        init_rnn_state,
        _rng,
        init_visiting_map
    )

    # EVAL NETWORK
    def _eval_step(runner_state, unused):

        params, env_state, done, rnn_state, expert_rnn_state, rng, visiting_map = runner_state

        visiting_map_step = jax.vmap(lambda arr, x, y : arr.at[(x, y)].add(1), in_axes=(None, 0, 0)
                                )(init_visiting_map, env_state.env_state.pos[:, 0], env_state.env_state.pos[:, 1])
        
        visiting_map = visiting_map_step.sum(axis=0) + visiting_map

        # Get the imitator obs
        obsv = jax.vmap(
            env.get_obs, in_axes=(0, None, None)
        )(env_state.env_state, env_params, config['full_obs'])

        rng, _rng = jax.random.split(rng)

        # Sample imitator action
        ac_in = (obsv[jnp.newaxis, :], done[jnp.newaxis, :])
        rnn_state, action_dist, _ = network.apply(params, rnn_state, ac_in)
        imitator_action = action_dist.sample(seed=_rng).squeeze(0)

        # Update the environment with the imitator action
        rng_step = jax.random.split(_rng, n_eval)

        _, env_state, _, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, imitator_action, env_params)
        
        runner_state = (params, env_state, done, rnn_state, expert_rnn_state, rng, visiting_map)

        return runner_state, info

    runner_state, eval_metric = jax.lax.scan(_eval_step, runner_state, None, 200)

    valid = eval_metric['returned_episode']

    # Save evaluation
    save_eval_path = f'/data/draco/cleain/imitation_gap_minigrid/{logs}/{expe_num}/eval_{n_eval}.json'
    with open(save_eval_path, 'w') as file:
        json.dump({'l': eval_metric['returned_episode_lengths'][valid].tolist(),
                'r': eval_metric['returned_episode_returns'][valid].tolist()
                }, file)
        
    # Save evaluation
    save_eval_path = f'/data/draco/cleain/imitation_gap_minigrid/{logs}/{expe_num}/eval_{n_eval}.json'
    with open(save_eval_path, 'w') as file:
        json.dump({'l': eval_metric['returned_episode_lengths'][valid].tolist(),
                'r': eval_metric['returned_episode_returns'][valid].tolist()
                }, file)
        
    save_eval_path = f'/data/draco/cleain/imitation_gap_minigrid/{logs}/{expe_num}/visiting_{n_eval}.pkl'
    with open(save_eval_path, 'wb') as file:
        pickle.dump(runner_state[-1], file)