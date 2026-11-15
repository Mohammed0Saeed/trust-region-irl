import os
import shutil
import json
from copy import deepcopy
import logging
import time
import tree
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

# Same policy and critic as trirl_ppo_spr (the local copies are identical to it)
from trust_region_irl.algorithms.ppo_retraining_curriculum.general_properties import GeneralProperties
from trust_region_irl.algorithms.ppo_retraining_curriculum.policy import get_policy
from trust_region_irl.algorithms.ppo_retraining_curriculum.critic import get_critic

from trust_region_irl.algorithms.ppo_retraining.ppo_retraining import PPO_RETRAINING

rlx_logger = logging.getLogger("rl_x")

class Batch:
    def __init__(self, states, next_states, actions, rewards, values, terminations, log_probs, advantages, returns):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.terminations = terminations
        self.log_probs = log_probs
        self.advantages = advantages
        self.returns = returns

class PPO_RETRAINING_CURRICULUM:
    def __init__(self, config, train_env, eval_env, run_path, writer, reward_function=None):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer

        # Retraining always saves its policy (best.model / latest.model), so evaluate_model.py --retrained can load it.
        # config.runner.save_model can't be used: the runner refuses it in test mode, which retraining runs in.
        self.save_model = True
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * (self.total_timesteps // self.batch_size)
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon

        # PPO as in trirl_ppo, without the reward correction: trust-region (KL) loss and absorbing-state handling
        self.init_eta = config.algorithm.init_eta
        self.const_eta = config.algorithm.const_eta
        self.handle_absorbing_states = config.algorithm.handle_absorbing_states
        self.env_reward_frac = config.algorithm.env_reward_frac
        self.H_terminal = jnp.sum(jnp.log(self.train_env.single_action_space.high - self.train_env.single_action_space.low))

        # Success-threshold curriculum, same as trirl_ppo (only when environment.curriculum_success=True)
        self.curriculum_success = config.environment.get("curriculum_success", False)
        if self.curriculum_success:
            threshold_values = [float(v) for v in config.environment.success_threshold_values.split(",")]
            threshold_fractions = [float(f) for f in config.environment.success_threshold_fractions.split(",")]
            if len(threshold_values) != len(threshold_fractions):
                raise ValueError("success_threshold_values and success_threshold_fractions must have the same number of entries")
            if abs(sum(threshold_fractions) - 1.0) > 1e-6:
                raise ValueError(f"success_threshold_fractions must sum to 1, got {sum(threshold_fractions)}")
            self.success_threshold_values = jnp.array(threshold_values, dtype=jnp.float32)
            # Env step at which each threshold after the first one starts
            self.success_threshold_boundaries = jnp.array(np.cumsum(threshold_fractions)[:-1] * self.total_timesteps, dtype=jnp.float32)
            self.final_success_threshold = threshold_values[-1]  # evaluation always uses the final threshold

        # Use the saved reward function
        self.reward_function = reward_function

        # Fixed reward theta (no theta curriculum): the corrected-reward theta stored in the loaded TRIRL checkpoint,
        # so theta and the encoder of the reward come from the same point of TRIRL training
        if not (hasattr(config.runner, "load_model") and config.runner.load_model):
            raise ValueError("runner.load_model must point to a TRIRL checkpoint to read the reward theta from")
        checkpoint_path = os.path.abspath(config.runner.load_model)
        self.reward_model_path = checkpoint_path  # written next to the saved policy (retraining_info.json)
        unpack_dir = os.path.join(os.path.dirname(checkpoint_path), "tmp_reward_theta")
        if os.path.exists(unpack_dir):
            shutil.rmtree(unpack_dir)
        shutil.unpack_archive(checkpoint_path, unpack_dir, "zip")
        try:
            # Restore every leaf as a numpy array (no target tree is available here)
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            metadata = checkpointer.metadata(unpack_dir)
            metadata = getattr(metadata, "item_metadata", metadata)
            metadata = getattr(metadata, "tree", metadata)
            restore_args = jax.tree_util.tree_map(lambda _: orbax.checkpoint.RestoreArgs(restore_type=np.ndarray), metadata)
            checkpoint = checkpointer.restore(unpack_dir, restore_args=restore_args)
            self.theta = jnp.array(np.asarray(checkpoint["corrected_reward"]["params"]["params"]["theta"]))
        finally:
            shutil.rmtree(unpack_dir)
        rlx_logger.info(f"Loaded the corrected-reward theta (shape {self.theta.shape}) from {checkpoint_path}.")

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")

        if self.nr_parallel_seeds > 1:
            raise ValueError(
                "Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.train_env.reset(reset_key, False)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.best_model_file_name = "best.model"
            self.best_eval_return = -np.inf
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            policy_state = self.policy_state
            critic_state = self.critic_state

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry,
                                                       multi_learning_iteration_step):
                policy_state, critic_state, env_state, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, env_state, key = learning_iteration_carry

                    current_global_iteration = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step

                    # Fixed reward theta for the whole training (no theta curriculum)
                    current_theta = self.theta

                    # Success-threshold curriculum: pick the threshold from the env steps done before this rollout
                    if self.curriculum_success:
                        steps_done = current_global_iteration * self.batch_size
                        stage = jnp.searchsorted(self.success_threshold_boundaries, steps_done, side="right")
                        env_state = env_state.replace(success_threshold=jnp.full_like(env_state.success_threshold, self.success_threshold_values[stage]))

                    # Acting
                    def single_rollout(single_rollout_carry, _):
                        policy_state, critic_state, env_state, key = single_rollout_carry

                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                        log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(
                            2.0 * jnp.pi) - action_logstd).sum(1)
                        processed_action = self.get_processed_action(action)
                        value = self.critic.apply(critic_state.params, observation).squeeze(-1)
                        env_state = self.train_env.step(env_state, processed_action)
                        transition = (observation, env_state.actual_next_observation, action, env_state.reward, value,
                                      env_state.terminated, log_prob, action_mean, env_state.info)

                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)

                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (policy_state, critic_state, env_state, key), transition

                    # Start from env_state (not learning_iteration_carry) so the curriculum threshold set above is used
                    single_rollout_carry, batch = jax.lax.scan(single_rollout, (policy_state, critic_state, env_state, key), None,
                                                               self.nr_steps)
                    policy_state, critic_state, env_state, key = single_rollout_carry
                    states, next_states, actions, env_rewards, values, terminations, log_probs, old_action_means, infos = batch

                    # Learnt reward (already scaled by entropy_coef in the reward function), no reward correction
                    learnt_rewards = self.reward_function(current_theta, states, actions, next_states, terminations, log_probs)
                    if self.handle_absorbing_states:
                        # Reward of the absorbing state, evaluated at the next state (as in trirl_ppo)
                        learnt_rewards_absorbing_state = self.reward_function(current_theta, next_states, actions, next_states, terminations, log_probs)
                    else:
                        learnt_rewards_absorbing_state = jnp.asarray(0.0)

                    rewards = self.env_reward_frac * env_rewards + (1 - self.env_reward_frac) * learnt_rewards
                    rewards_absorbing_state = (1 - self.env_reward_frac) * learnt_rewards_absorbing_state  # the environment pays no reward in the absorbing state

                    # Calculating advantages and returns
                    def calculate_gae_advantages(critic_state, next_states, rewards, values, terminations):
                        def compute_advantages(carry, t):
                            prev_advantage = carry[0]
                            advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                            return (advantage,), advantage

                        next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
                        delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                        init_advantages = delta[-1]
                        _, advantages = jax.lax.scan(compute_advantages, (init_advantages,),
                                                     jnp.arange(self.nr_steps - 2, -1, -1))
                        advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                        returns = advantages + values
                        return advantages, returns

                    def calculate_gae_advantages_absorbing(critic_state, next_states, rewards, rewards_next_state, values, terminations):
                        """
                        Correctly handle absorbing state value and entropy (instead of setting to 0.0)
                        """
                        def compute_advantages(carry, t):
                            prev_advantage = carry[0]
                            advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                            return (advantage,), advantage

                        next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
                        terminal_tail = (self.gamma / (1.0 - self.gamma)) * (rewards_next_state + self.entropy_coef * self.H_terminal)
                        delta = rewards + self.gamma * next_values * (1.0 - terminations) + (terminations * terminal_tail) - values
                        init_advantages = delta[-1]
                        _, advantages = jax.lax.scan(compute_advantages, (init_advantages,),
                                                     jnp.arange(self.nr_steps - 2, -1, -1))
                        advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                        returns = advantages + values
                        return advantages, returns

                    if self.handle_absorbing_states:
                        advantages, returns = calculate_gae_advantages_absorbing(critic_state, next_states, rewards, rewards_absorbing_state, values, terminations)
                    else:
                        advantages, returns = calculate_gae_advantages(critic_state, next_states, rewards, values, terminations)

                    # Compute eta (linear schedule), as in trirl_ppo
                    global_step = (current_global_iteration + 1) * self.batch_size

                    def linear_schedule_eta(global_step):
                        if self.const_eta:
                            return self.init_eta
                        else:
                            fraction = 1.0 - jnp.clip(global_step, min=None, max=self.total_timesteps) / self.total_timesteps
                            return self.init_eta * fraction

                    eta = linear_schedule_eta(global_step)

                    # Optimizing
                    def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b, old_mean_b, old_logstd_b, eta):
                        # Policy loss
                        action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                        action_std = jnp.exp(action_logstd)
                        new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(
                            2.0 * jnp.pi) - action_logstd
                        new_log_prob = new_log_prob.sum(1)
                        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                        entropy = self.entropy_coef * entropy  # scaling to reduce critic loss (has no effect on solution since the same is also applied to the reward function)

                        logratio = new_log_prob - log_prob_b
                        ratio = jnp.exp(logratio)
                        approx_kl_div = (ratio - 1) - logratio
                        clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                        pg_loss1 = -advantage_b * ratio
                        pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        pg_loss = jnp.maximum(pg_loss1, pg_loss2)

                        entropy_loss = entropy.sum(1)

                        # Critic loss
                        new_value = self.critic.apply(critic_params, state_b)
                        critic_loss = 0.5 * (new_value - return_b) ** 2

                        # Trust Region Loss
                        # Analytical KL (across all samples)
                        old_logstd_b = jnp.expand_dims(old_logstd_b, 0)  # shape (1, as_shape)
                        old_action_std = jnp.exp(old_logstd_b)
                        tr_loss_maha = 0.5 * jnp.sum(((old_mean_b - action_mean) / old_action_std) ** 2, axis=1)
                        tr_loss_cov_part = 0.5 * jnp.sum(2.0 * (jnp.log(old_action_std) - jnp.log(action_std)) + (action_std / old_action_std) ** 2 - 1.0, axis=1)
                        trust_region_loss = tr_loss_maha + tr_loss_cov_part

                        # Combine losses
                        loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss + eta * trust_region_loss

                        # Create metrics
                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "loss/trust_region_loss": trust_region_loss,
                            "policy_ratio/approx_kl": approx_kl_div,
                            "policy_ratio/clip_fraction": clip_fraction,
                        }

                        return loss, (metrics)

                    # Gradients must only flow through the predicted ones.
                    # The log standard deviation is a state-independent parameter, so the rollout's value is the
                    # parameter before this update (shape (1, as_shape) -> (as_shape,))
                    old_action_means = jax.lax.stop_gradient(old_action_means)
                    old_action_logstd = jax.lax.stop_gradient(policy_state.params["params"]["policy_logstd"])[0]

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)
                    batch_action_means = old_action_means.reshape((-1,) + self.as_shape)

                    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0, 0, None, None), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices[:, :self.nr_minibatches * self.minibatch_size]
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state = carry

                        minibatch_advantages = batch_advantages[minibatch_indices]
                        minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (
                                    jnp.std(minibatch_advantages) + 1e-8)

                        (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_returns[minibatch_indices],
                            minibatch_advantages,
                            batch_action_means[minibatch_indices],
                            old_action_logstd,
                            eta
                        )

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)

                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                        carry = (policy_state, critic_state)

                        return carry, (metrics)

                    init_carry = (policy_state, critic_state)
                    carry, (optimization_metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    policy_state, critic_state = carry

                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (
                                jnp.var(returns) + 1e-8)
                    optimization_metrics["policy/std_dev"] = jnp.mean(
                        jnp.exp(policy_state.params["params"]["policy_logstd"]))

                    optimization_metrics["correction/eta"] = eta
                    optimization_metrics["reward/learnt_mean"] = jnp.mean(learnt_rewards)
                    optimization_metrics["reward/learnt_std"] = jnp.std(learnt_rewards)

                    # Logging
                    combined_metrics = {**infos, **optimization_metrics}
                    if self.curriculum_success:
                        combined_metrics["env/success_threshold"] = env_state.success_threshold
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int(
                            (self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        metrics["steps/nr_env_steps"] = global_step
                        metrics[
                            "steps/nr_updates"] = combined_learning_iteration_step.item() * self.nr_epochs * self.nr_minibatches
                        is_last_train_update_before_eval = self.evaluation_active and (
                                    learning_iteration_step + 1 == self.nr_updates_per_multi_learning_iteration)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_train_update_before_eval)

                    combined_learning_iteration_step = (
                                                                   multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    jax.debug.callback(callback,
                                       (combined_metrics, learning_iteration_step, combined_learning_iteration_step,
                                        parallel_seed_id))

                    return (policy_state, critic_state, env_state, key), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration,
                                                           (policy_state, critic_state, env_state, subkey),
                                                           jnp.arange(self.nr_updates_per_multi_learning_iteration))
                policy_state, critic_state, env_state, key = learning_iteration_carry

                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, eval_env_state = single_eval_rollout_carry

                        eval_action_mean, _ = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        eval_action = eval_action_mean
                        eval_processed_action = self.get_processed_action(eval_action)
                        eval_env_state = self.eval_env.step(eval_env_state, eval_processed_action)

                        return (policy_state, eval_env_state), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.eval_env.reset(reset_keys, True)
                    if self.curriculum_success:
                        # Fixed final threshold, so eval/episode_return and the best-model choice are comparable over training
                        eval_env_state = eval_env_state.replace(success_threshold=jnp.full_like(eval_env_state.success_threshold, self.final_success_threshold))
                    single_eval_rollout_carry, _ = jax.lax.scan(single_eval_rollout, (policy_state, eval_env_state),
                                                                jnp.arange(self.horizon))
                    _, eval_env_state = single_eval_rollout_carry

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    if "rollout/is_success" in eval_env_state.info:
                        eval_metrics["eval/is_success"] = jnp.mean(eval_env_state.info["rollout/is_success"])

                    if "rollout/task_err" in eval_env_state.info:
                        eval_metrics["eval/task_err"] = jnp.mean(eval_env_state.info["rollout/task_err"])

                    def callback(metrics_and_global_step):
                        metrics, combined_learning_iteration_step = metrics_and_global_step
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    combined_learning_iteration_step = (multi_learning_iteration_step + 1) * self.nr_updates_per_multi_learning_iteration
                    jax.debug.callback(callback, (eval_metrics, combined_learning_iteration_step))

                    # Saving
                    if self.save_model:
                        # Fetch episode_return from eval_metrics, fallback to -inf if not evaluating
                        eval_return_for_save = (eval_metrics["eval/episode_return"]
                                                if self.evaluation_active else jnp.asarray(-jnp.inf))

                        def save_with_check(policy_state, critic_state, eval_return):
                            self.save(policy_state, critic_state)  # always save latest.model

                            if self.evaluation_active:
                                current_return = float(np.asarray(eval_return).reshape(-1)[0])

                                # Check if the current return is GREATER than the best return
                                if current_return > self.best_eval_return:
                                    self.best_eval_return = current_return
                                    self.save(policy_state, critic_state, file_name=self.best_model_file_name)
                                    rlx_logger.info(
                                        f"[save-best] new best eval/episode_return={current_return:.4f} -> {self.best_model_file_name}")

                        jax.debug.callback(save_with_check, policy_state, critic_state, eval_return_for_save)

                return (policy_state, critic_state, env_state, key), None

            jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, critic_state, env_state, key),
                         jnp.arange(self.nr_multi_learning_and_eval_save_iterations))

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")

    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_console:
            self.log_console(name, value)

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)

    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")

    def end_logging(self, wandb_commit=True):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache, commit=wandb_commit)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    def save(self, policy_state, critic_state, file_name=None):
        file_name = file_name or self.latest_model_file_name
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)

        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)

        # Which TRIRL reward this policy was retrained on, and the threshold setup (read by evaluate_model.py --retrained)
        retraining_info = {
            "reward_model": self.reward_model_path,
            "curriculum_success": bool(self.curriculum_success),
            "success_threshold": float(self.config.environment.get("success_threshold", float("nan"))),
            "success_threshold_values": self.config.environment.get("success_threshold_values", ""),
            "success_threshold_fractions": self.config.environment.get("success_threshold_fractions", ""),
        }
        with open(f"{self.save_path}/tmp/retraining_info.json", "w") as f:
            json.dump(retraining_info, f, indent=2)
        with open(f"{self.save_path}/retraining_info.json", "w") as f:
            json.dump(retraining_info, f, indent=2)

        # shutil.make_archive automatically appends '.zip' to the target file
        shutil.make_archive(f"{self.save_path}/{file_name}", "zip", f"{self.save_path}/tmp")

        # We REMOVED the os.rename line that was deleting the .zip extension
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            # Note: Added .zip here so wandb uploads the correctly named archive
            wandb.save(f"{self.save_path}/{file_name}.zip", base_path=self.save_path)

    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = PPO_RETRAINING(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]

        shutil.rmtree(checkpoint_dir)

        return model

    def test(self, episodes):
        self.test_ppo(episodes)


    def test_ppo(self, episodes):
        def visualize():
            rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

            @jax.jit
            def rollout(env_state, key):
                # Use deterministic action mean for evaluation
                action_mean, _ = self.policy.apply(self.policy_state.params, env_state.next_observation)
                action = action_mean
                processed_action = self.get_processed_action(action)
                env_state = self.eval_env.step(env_state, processed_action)
                return env_state, key

            self.key, subkey = jax.random.split(self.key)
            reset_keys = jax.random.split(subkey, self.nr_envs)
            env_state = self.eval_env.reset(reset_keys, True)

            while True:
                env_state, self.key = rollout(env_state, self.key)
                if self.render:
                    env_state = self.eval_env.render(env_state)

        def evaluate():
            rlx_logger.info("Running and saving episodes.")
            nr_steps = 1000
            batch = Batch(
                states=np.zeros((1, nr_steps, self.nr_envs) + self.os_shape),
                next_states=np.zeros((1, nr_steps, self.nr_envs) + self.os_shape),
                actions=np.zeros((1, nr_steps, self.nr_envs) + self.as_shape),
                rewards=np.zeros((1, nr_steps, self.nr_envs)),
                terminations=np.zeros((1, nr_steps, self.nr_envs)),

                values=np.zeros((1, nr_steps, self.nr_envs)),
                log_probs=np.zeros((1, nr_steps, self.nr_envs)),
                advantages=np.zeros((1, nr_steps, self.nr_envs)),
                returns=np.zeros((1, nr_steps, self.nr_envs)),
            )

            episode_return = jnp.zeros((self.nr_envs))
            self.key, subkey = jax.random.split(self.key)
            reset_keys = jax.random.split(subkey, self.nr_envs)
            env_state = self.eval_env.reset(reset_keys, True)

            for step in range(nr_steps):
                batch.states[0, step] = env_state.next_observation

                # step
                action_mean, _ = self.policy.apply(self.policy_state.params, env_state.next_observation)
                action = action_mean
                processed_action = self.get_processed_action(action)
                env_state = self.eval_env.step(env_state, processed_action)

                batch.next_states[0, step] = env_state.actual_next_observation
                batch.actions[0, step] = processed_action
                batch.rewards[0, step] = env_state.reward
                batch.terminations[0, step] = env_state.terminated
                episode_return += env_state.reward

                if self.render:
                    env_state = self.eval_env.render(env_state)

            mean_episode_return = episode_return.mean()
            rlx_logger.info(f"Mean Episode Return: {mean_episode_return}")

            def flatten_and_prune(arr):
                flat = arr.reshape(-1, arr.shape[-1])
                nan_mask = ~np.isnan(flat).any(axis=1)
                return flat[nan_mask]

            exp_states = flatten_and_prune(batch.states)
            exp_actions = flatten_and_prune(batch.actions)
            exp_next_states = flatten_and_prune(batch.next_states)
            exp_absorbing = flatten_and_prune(batch.terminations).flatten()
            exp_rewards = flatten_and_prune(batch.rewards).flatten()

            print(f"save path: {self.save_path}")
            print(f"states shape: {exp_states.shape}")
            print(f"actions shape: {exp_actions.shape}")
            print(f"rewards shape: {exp_rewards.shape}")
            print(f"absorbing shape: {exp_absorbing.shape}")

            os.makedirs(self.save_path, exist_ok=True)
            np.savez(f"{self.save_path}/expert_dataset_PPO_retrained_{self.nr_envs}", states=exp_states,
                     actions=exp_actions,
                     next_states=exp_next_states, absorbing=exp_absorbing, rewards=exp_rewards)

        visualize()
        # evaluate() # Uncomment this line and comment visualize() if you want to generate the dataset instead

    def general_properties(self):
        return GeneralProperties