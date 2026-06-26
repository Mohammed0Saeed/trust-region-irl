#!/bin/sh

## TRIRL PPO FB
#python experiment.py \
#    --algorithm.name="trirl_ppo_fb.flax_full_jit" \
#    --algorithm.data_path="/home/kukoboshi/PycharmProjects/trust-region-irl/trirl_dataset/rl_expert/expert_dataset_pusht_50_episodes_20hz.npz" \
#    --algorithm.total_timesteps=100e6 \
#    --algorithm.entropy_coef=0.001 \
#    --algorithm.clip_range=0.2 \
#    --algorithm.env_reward_frac=0.0 \
#    --algorithm.nr_steps=10 \
#    --algorithm.nr_epochs=20 \
#    --algorithm.minibatch_size=512 \
#    --algorithm.learning_rate_disc=1e-04 \
#    --algorithm.learning_rate=4e-04 \
#    --algorithm.epsilon=0.2 \
#    --algorithm.init_eta=30.0 \
#    --algorithm.gp_lambda=0.05 \
#    --algorithm.evaluation_and_save_frequency=2129920 \
#    --environment.name="franka_pusht" \
#    --environment.nr_envs=4096 \
#    --environment.seed=0 \
#    --runner.mode="train" \
#    --runner.track_tb=True \
#    --runner.track_console=True \
#    --runner.track_wandb=True \
#    --runner.save_model=True \
#    --runner.wandb_entity="s-mohddheia-tu-darmstadt" \
#    --runner.project_name="trust_region_irl" \
#    --runner.exp_name="pusht_ppo_fb_100m"

## TRIRL PPO FB
python experiment.py \
    --algorithm.name="trirl_ppo_fb.flax_full_jit" \
    --algorithm.data_path="/home/kukoboshi/PycharmProjects/trust-region-irl/trirl_dataset/rl_expert/expert_dataset_Ant-v5_30_PPO.npz" \
    --algorithm.total_timesteps=20e6 \
    --algorithm.entropy_coef=0.001 \
    --algorithm.clip_range=0.2 \
    --algorithm.env_reward_frac=0.0 \
    --algorithm.nr_steps=10 \
    --algorithm.nr_epochs=20 \
    --algorithm.minibatch_size=512 \
    --algorithm.learning_rate_disc=1e-04 \
    --algorithm.learning_rate=4e-04 \
    --algorithm.epsilon=0.2 \
    --algorithm.init_eta=30.0 \
    --algorithm.gp_lambda=0.05 \
    --algorithm.evaluation_and_save_frequency=2129920 \
    --environment.name="ant_mjx" \
    --environment.nr_envs=4096 \
    --environment.seed=0 \
    --runner.mode="train" \
    --runner.track_tb=True \
    --runner.track_console=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="s-mohddheia-tu-darmstadt" \
    --runner.project_name="trust_region_irl" \
    --runner.exp_name="ant_mjx_ppo_fb_50m"

# Test Policy
#python experiment.py \
#    --algorithm.name="trirl_ppo_fb.flax_full_jit" \
#    --algorithm.data_path="/home/kukoboshi/PycharmProjects/trust-region-irl/trirl_dataset/rl_expert/expert_dataset_pusht_50_episodes.npz" \
#    --environment.name="franka_pusht" \
#    --environment.render=True \
#    --environment.nr_envs=1 \
#    --environment.seed=0 \
#    --runner.mode="test" \
#    --runner.load_model="runs/trust_region_irl/pusht_trirl/1782026789/models/latest.model.zip" \
#    --runner.nr_test_episodes=10 \
#    --runner.track_tb=False \
#    --runner.track_wandb=False \
#    --runner.save_model=False \
#    --runner.project_name="trust_region_irl" \
#    --runner.exp_name="pusht_trirl"

#python experiment.py \
#    --algorithm.name="trirl_ppo_fb.flax_full_jit" \
#    --environment.name="franka_pusht" \
#    --environment.render=True \
#    --environment.nr_envs=1 \
#    --runner.mode="test" \
#    --runner.load_model="runs/trust_region_irl/pusht_ppo_fb_100m/1782405066/models/latest.model.zip" \
#    --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False
#
#

#python experiment.py \
#    --algorithm.name="trirl_ppo_fb.flax_full_jit" \
#    --environment.name="ant_mjx" \
#    --environment.render=True \
#    --environment.nr_envs=1 \
#    --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False

