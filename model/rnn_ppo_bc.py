import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState

import json
import os
import pickle
import sys
sys.path.append('./')
sys.path.append('../')

from environment.maze import MetaMaze
from environment.wrappers import FlattenObservationWrapper, LogWrapper

from model.rnn_policy import ScannedRNN, ActorCriticRNN

class TransitionBC(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.array
    obs: jnp.ndarray

class TransitionRL(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["BATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"]
    )
    env = MetaMaze(**config['params'])
    env_params = env.default_params
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    # DEFINE EXPERT AGENT
    with open(os.path.join(config['load_path'], config['expert_expe_num'], 'params_4.pkl'), 'rb') as file:
        expert_params = pickle.load(file)
    with open(os.path.join(config['load_path'], config['expert_expe_num'], 'args.json'), 'rb') as file:
        expert_config = json.load(file)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train():
        
        rng = jax.random.PRNGKey(config['KEY'])

        # EXPERT NETWORK
        expert_network = ActorCriticRNN(env.action_space(env_params).n)

        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_rnn_state = ScannedRNN.initialize_carry((config["NUM_ENVS"], 128))
        network_params = network.init(_rng, init_rnn_state, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        _, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        obsv = jax.vmap(env.get_obs, in_axes=(0, None, None))(env_state.env_state, env_params, config['full_obs'])
        init_rnn_state = ScannedRNN.initialize_carry((config["NUM_ENVS"], 128))

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            runner_state_bc, runner_state_rl = runner_state

            # COLLECT TRAJECTORIES FROM EXPERT POLICY
            def _env_step_bc(runner_state_bc, unused):
                train_state, env_state, done, expert_rnn_state, imitator_rnn_state, rng = runner_state_bc

                # Get the imitator obs
                obsv = jax.vmap(
                    env.get_obs, in_axes=(0, None, None)
                    )(env_state.env_state,
                      env_params,
                      config['full_obs']
                    )

                # Get the expert obs
                expert_obsv = jax.vmap(
                    env.get_obs, in_axes=(0, None, None)
                    )(env_state.env_state,
                      env_params,
                      expert_config['is_expert']
                    )

                rng, _rng = jax.random.split(rng)

                # Sample expert action
                ac_in_expert = (expert_obsv[jnp.newaxis, :], done[jnp.newaxis, :])
                expert_rnn_state, expert_action_dist, _ = expert_network.apply(expert_params, expert_rnn_state, ac_in_expert)
                expert_action = expert_action_dist.sample(seed=_rng).squeeze(0)

                # Extract imitator RNN state
                ac_in_imitator = (obsv[jnp.newaxis, :], done[jnp.newaxis, :])
                imitator_rnn_state, _, _ = network.apply(train_state.params, imitator_rnn_state, ac_in_imitator)

                transition = TransitionBC(done,
                                          expert_action,
                                          obsv
                                          )

                # Update the environment with the expert action
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                _, env_state, _, done, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step,
                    env_state,
                    expert_action,
                    env_params
                    )

                runner_state_bc = (train_state, env_state, done, expert_rnn_state, imitator_rnn_state, rng)

                return runner_state_bc, transition

            # COLLECT TRAJECTORIES FROM IMITATOR
            def _env_step_rl(runner_state_rl, unused):
                train_state, env_state, last_obs, last_done, rnn_state, rng = runner_state_rl
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
                rnn_state, pi, value = network.apply(train_state.params, rnn_state, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                # Update the environment
                _, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                # Get the observation
                obsv = jax.vmap(
                    env.get_obs, in_axes=(0, None, None)
                )(env_state.env_state, env_params, config['full_obs'])

                transition = TransitionRL(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state_rl = (train_state, env_state, obsv, done, rnn_state, rng)
                return runner_state_rl, transition

            # TRAJECTORY FROM EXPERT POLICY
            initial_rnn_state_bc = runner_state_bc[-2]
            runner_state_bc, traj_batch_bc = jax.lax.scan(
                _env_step_bc, runner_state_bc, None, config["NUM_STEPS"]
            )
            train_state, env_state_bc, last_done_bc, expert_rnn_state_bc, imitator_rnn_state_bc, rng = runner_state_bc

            # TRAJECTORY FROM IMITATOR POLICY
            initial_rnn_state_rl = runner_state_rl[-2]
            runner_state_rl, traj_batch_rl = jax.lax.scan(
                _env_step_rl, runner_state_rl, None, config["NUM_STEPS"]
            )
            train_state, env_state_rl, last_obs_rl, last_done_rl, rnn_state_rl, rng = runner_state_rl

            # CALCULATE ADVANTAGE
            ac_in = (last_obs_rl[jnp.newaxis, :], last_done_rl[jnp.newaxis, :])
            _, _, last_val = network.apply(train_state.params, rnn_state_rl, ac_in)
            last_val = last_val.squeeze(0)
            def _calculate_gae(traj_batch, last_val, last_done_rl):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    return (gae, value, done), gae
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done_rl), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value
            advantages, targets = _calculate_gae(traj_batch_rl, last_val, last_done_rl)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_rnn_state_bc, init_rnn_state_rl, traj_batch_bc, traj_batch_rl, advantages, targets = batch_info

                    def _loss_fn(params, init_rnn_state_bc, init_rnn_state_rl, traj_batch_bc, traj_batch_rl, advantages, targets):

                        def _loss_fn_bc(params, init_rnn_state, traj_batch):
                            _, action_dist, _ = network.apply(
                                params, init_rnn_state[0], (traj_batch.obs, traj_batch.done)
                                )
                            log_prob = action_dist.log_prob(traj_batch.expert_action)

                            total_loss = - log_prob.mean()

                            return total_loss

                        def _loss_fn_rl(params, init_rnn_state, traj_batch, gae, targets):
                            # RERUN NETWORK
                            _, pi, value = network.apply(
                                params, init_rnn_state[0], (traj_batch.obs, traj_batch.done)
                                )
                            log_prob = pi.log_prob(traj_batch.action)

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )

                            # CALCULATE ACTOR LOSS
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        bc_loss = _loss_fn_bc(params, init_rnn_state_bc, traj_batch_bc)
                        rl_loss, (value_loss, loss_actor, entropy) = _loss_fn_rl(params, init_rnn_state_rl, traj_batch_rl, advantages, targets)

                        total_loss = (
                            config['WEIGHT_BC'] * bc_loss
                            + config['WEIGHT_RL'] * rl_loss
                        )

                        return total_loss, (bc_loss, rl_loss, value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_rnn_state_bc, init_rnn_state_rl, traj_batch_bc, traj_batch_rl, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                train_state,
                init_rnn_state_bc,
                init_rnn_state_rl,
                traj_batch_bc,
                traj_batch_rl,
                advantages,
                targets,
                rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_rnn_state_bc, init_rnn_state_rl, traj_batch_bc, traj_batch_rl, advantages, targets)


                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                train_state, total_loss = _update_minbatch(train_state, shuffled_batch)

                update_state = (
                    train_state,
                    init_rnn_state_bc,
                    init_rnn_state_rl,
                    traj_batch_bc,
                    traj_batch_rl,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_rnn_state_rl = initial_rnn_state_rl[None, :]  # TBH
            init_rnn_state_bc = initial_rnn_state_bc[None, :]  # TBH
            update_state = (
                train_state,
                init_rnn_state_bc,
                init_rnn_state_rl,
                traj_batch_bc,
                traj_batch_rl,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            valid = jnp.array(traj_batch_rl.info['returned_episode'])
            returns = jnp.array(traj_batch_rl.info['returned_episode_returns'])
            lengths = jnp.array(traj_batch_rl.info['returned_episode_lengths'])
            metric = {
                'r': jnp.where(valid, returns, jnp.zeros_like(returns)).sum() / valid.sum(),
                'l': jnp.where(valid, lengths, jnp.zeros_like(lengths)).sum() / valid.sum()
                      }
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state_bc = (train_state, env_state_bc, last_done_bc, expert_rnn_state_bc, imitator_rnn_state_bc, rng)
            runner_state_rl = (train_state, env_state_rl, last_obs_rl, last_done_rl, rnn_state_rl, rng)
            return (runner_state_bc, runner_state_rl), metric

        rng, _rng = jax.random.split(rng)
        runner_state_bc = (
            train_state,
            env_state,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_rnn_state, # Expert rnn state
            init_rnn_state, # Imitator rnn state
            _rng,
        )

        rng, _rng = jax.random.split(rng)
        runner_state_rl = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_rnn_state, # Imitator rnn state
            _rng,
        )

        (runner_state_bc, runner_state_rl), metric = jax.lax.scan(
            _update_step, (runner_state_bc, runner_state_rl), None, config["NUM_UPDATES"]
        )
        return {"runner_state_rl": runner_state_rl, "metric": metric}

    return train