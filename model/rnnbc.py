from flax.training.train_state import TrainState
import json
import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple

import os
import optax
import pickle

import sys
sys.path.append('./')

from environment.maze import MetaMaze
from environment.wrappers import FlattenObservationWrapper, LogWrapper
from rnn_policy import ActorCriticRNN, ScannedRNN


class Transition(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.array
    obs: jnp.ndarray


class make_train:
    
    def __init__(self, 
                 config):

        self.config = config
        self.config["NUM_UPDATES"] = (
            self.config["TOTAL_TIMESTEPS"] // self.config["NUM_STEPS"] // self.config["NUM_ENVS"]
        )

        # Device
        self.devices = jax.devices()
        print(f'Available devices: {self.devices}')
        
        # Random key
        self.rng = jax.random.PRNGKey(self.config['key'])

        # DEF ENV
        self.env = MetaMaze(**self.config['params'])
        self.env_params = self.env.default_params
        self.env = FlattenObservationWrapper(self.env)
        self.env = LogWrapper(self.env)
        
        # DEFINE EXPERT AGENT
        with open(os.path.join(self.config['load_path'], self.config['expert_expe_num'], 'params_4.pkl'), 'rb') as file:
            self.expert_params = pickle.load(file)
        with open(os.path.join(self.config['load_path'], self.config['expert_expe_num'], 'args.json'), 'rb') as file:
            self.expert_config = json.load(file)

    # SCHEDULER
    def linear_schedule(self, count):
        frac = (
            1.0
            - (count // (self.config["NUM_EPOCHS"]))
            / self.config["NUM_UPDATES"]
        )
        return self.config["LR"] * frac

    def train(self,):

        # EXPERT NETWORK
        expert_network = ActorCriticRNN(self.env.action_space(self.env_params).n)
        
        # INIT NETWORK
        network = ActorCriticRNN(self.env.action_space(self.env_params).n)

        init_x = (
            jnp.zeros(
                (1, self.config["NUM_ENVS"], *self.env.observation_space(self.env_params).shape)
            ),
            jnp.zeros((1, self.config["NUM_ENVS"])),
        )
        init_rnn_state = ScannedRNN.initialize_carry((self.config["NUM_ENVS"], 128))
        
        network_params = network.init(self.rng, init_rnn_state, init_x)
        
        if self.config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=self.linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(self.config["lr"], eps=1e-5),
            )

        train_state = TrainState.create(apply_fn=network.apply,
                                        params=network_params,
                                        tx=tx,
                                        )

        # TRAIN LOOP
        def _update_epoch(cary, unused):

            train_state, rng = cary

            # UPDATE NETWORK
            def _update_step(cary, unused):
                
                train_state, rng = cary

                # Init environment
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, self.config["NUM_ENVS"])
                _, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(reset_rng, self.env_params)
                init_rnn_state = ScannedRNN.initialize_carry((self.config["NUM_ENVS"], 128))

                runner_state = (
                    train_state,
                    env_state,
                    jnp.zeros((self.config["NUM_ENVS"]), dtype=bool),
                    init_rnn_state,
                    _rng,
                )

                # COLLECT TRAJECTORIES FROM EXPERT POLICY
                def _env_step(runner_state, unused):
                    train_state, env_state, done, expert_rnn_state, rng = runner_state
                    
                    # Get the imitator obs
                    obsv = jax.vmap(
                        self.env.get_obs, in_axes=(0, None, None)
                    )(env_state.env_state, self.env_params, self.config['full_obs'])
                    
                    # Get the expert obs
                    expert_obsv = jax.vmap(
                        self.env.get_obs, in_axes=(0, None, None)
                    )(env_state.env_state, self.env_params, self.expert_config['is_expert'])

                    rng, _rng = jax.random.split(rng)

                    # Sample expert action
                    ac_in = (expert_obsv[jnp.newaxis, :], done[jnp.newaxis, :])
                    expert_rnn_state, expert_action_dist, _ = expert_network.apply(self.expert_params, expert_rnn_state, ac_in)
                    expert_action = expert_action_dist.sample(seed=_rng).squeeze(0)

                    transition = Transition(done,
                                            expert_action,
                                            obsv
                                            )
                    
                    # Update the environment with the expert action
                    rng_step = jax.random.split(_rng, self.config["NUM_ENVS"])

                    _, env_state, _, done, _ = jax.vmap(
                        self.env.step, in_axes=(0, 0, 0, None)
                    )(rng_step, env_state, expert_action, self.env_params)
                    
                    runner_state = (train_state, env_state, done, expert_rnn_state, rng)
                    
                    return runner_state, transition
                
                _, traj_batch = jax.lax.scan(_env_step, runner_state, None, self.config["NUM_STEPS"])

                # BACKPROPAGATION
                def _update(train_state, traj_batch):

                    def _loss_fn(params, traj_batch): # DISCRETE ACTION SPACE
                        init_rnn_state = ScannedRNN.initialize_carry((self.config["NUM_ENVS"], 128))
                        _, action_dist, _ = network.apply(params, init_rnn_state, (traj_batch.obs, traj_batch.done))
                        log_prob = action_dist.log_prob(traj_batch.expert_action)

                        total_loss = - log_prob.mean()

                        return total_loss

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
                    total_loss, grads = grad_fn(train_state.params, traj_batch)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss
                
                train_state, total_loss = _update(train_state, traj_batch)

                return (train_state, rng), total_loss
            
            (train_state, rng), metric = jax.lax.scan(_update_step, (train_state, rng), None, self.config["NUM_UPDATES"])
                        
            return (train_state, rng), metric
        
        # EVALUATION LOOP
        def _evaluate_epoch(cary):

            train_state, rng = cary

            # Init environment
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, self.config["NUM_ENVS_EVAL"])
            _, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(reset_rng, self.env_params)
            init_rnn_state = ScannedRNN.initialize_carry((self.config["NUM_ENVS_EVAL"], 128))

            runner_state = (
                train_state,
                env_state,
                jnp.zeros((self.config["NUM_ENVS_EVAL"]), dtype=bool),
                init_rnn_state,
                init_rnn_state,
                _rng,
            )

            # EVAL NETWORK
            def _eval_step(runner_state, unused):

                train_state, env_state, done, rnn_state, expert_rnn_state, rng = runner_state

                # Get the imitator obs
                obsv = jax.vmap(
                    self.env.get_obs, in_axes=(0, None, None)
                )(env_state.env_state, self.env_params, self.config['full_obs']) # False

                # Get the expert obs
                expert_obsv = jax.vmap(
                    self.env.get_obs, in_axes=(0, None, None)
                )(env_state.env_state, self.env_params, self.expert_config['is_expert'])

                rng, _rng = jax.random.split(rng)

                # Sample expert action
                ac_in = (expert_obsv[jnp.newaxis, :], done[jnp.newaxis, :])
                expert_rnn_state, expert_action_dist, _ = expert_network.apply(self.expert_params, expert_rnn_state, ac_in)
                expert_action = expert_action_dist.sample(seed=_rng).squeeze(0)

                # Sample imitator action
                ac_in = (obsv[jnp.newaxis, :], done[jnp.newaxis, :])
                rnn_state, action_dist, _ = network.apply(train_state.params, rnn_state, ac_in)
                imitator_action = action_dist.sample(seed=_rng).squeeze(0)

                # Update the environment with the imitator action
                rng_step = jax.random.split(_rng, self.config["NUM_ENVS_EVAL"])

                _, env_state, _, done, info = jax.vmap(
                    self.env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, imitator_action, self.env_params)
                
                runner_state = (train_state, env_state, done, rnn_state, expert_rnn_state, rng)

                log_prob = action_dist.log_prob(expert_action).mean()
                info['log_prob'] = - log_prob
                
                return runner_state, info

            _, eval_metric = jax.lax.scan(_eval_step, runner_state, None, self.config['NUM_EVAL_STEPS'])

            return eval_metric
        
        metrics = {}
        rng, _rng = jax.random.split(self.rng)

        for epoch in range(self.config["NUM_EPOCHS"]):

            rng, _rng = jax.random.split(_rng)
            metrics[epoch] = {}
            
            # Training
            (train_state, rng), train_metric = _update_epoch((train_state, rng), None)
            metrics[epoch]['train'] = train_metric

            train_message = f"Epoch | {epoch} | Train | loss | {jnp.array(train_metric).mean():.4f}"
            print(train_message)

            # Validation            
            val_metric = _evaluate_epoch((train_state, rng))
            metrics[epoch]['validation'] = val_metric

            val_message = f'Epoch | {epoch} | Val | '
            for key in ['returned_episode_lengths', 'returned_episode_returns', 'log_prob']:
                if key == 'log_prob':
                    val_message += f" {key} | {jnp.array(val_metric[key]).mean():.4f} | "
                else:
                    val_message += f" {key} | {jnp.array(val_metric[key])[val_metric['returned_episode']].mean():.4f} | "

            print(val_message)

            if (epoch % self.config['freq_save'] == 0) or (epoch == self.config['NUM_EPOCHS'] - 1):
                past_log_metric = os.path.join(self.config['log_folder'], f'training_metrics_{epoch - self.config["freq_save"]}.pkl')
                past_log_params = os.path.join(self.config['log_folder'], f'params_{epoch - self.config["freq_save"]}.pkl')
                
                if os.path.exists(past_log_metric):
                    os.remove(past_log_metric)

                if os.path.exists(past_log_params):
                    os.remove(past_log_params)

                # Checkpoint
                with open(os.path.join(self.config['log_folder'], f'training_metrics_{epoch}.pkl'), "wb") as json_file:
                    pickle.dump(metrics, json_file)

                # Save model weights
                with open(os.path.join(self.config['log_folder'], f'params_{epoch}.pkl'), 'wb') as f:
                    pickle.dump(train_state.params, f)
        
        return {"train_state": train_state, "metrics": metrics}