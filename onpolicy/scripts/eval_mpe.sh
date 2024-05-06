#!/bin/bash

logs_folder="out3"
mkdir -p $logs_folder
# Run the script
# script to iterate through different hyperparameters
env_names=("MPE" "GraphMPE")
scenarios=("navigation" "navigation_graph")
models=("vanilla3" "graph3")
seeds=(0)

args_models=()
args_env_names=()
args_scenarios=()
args_seeds=()
# iterate through all combos and make a list
for i in ${!models[@]}; do
    for j in ${!seeds[@]}; do
        args_models+=(${models[$i]})
        args_env_names+=(${env_names[$i]})
        args_scenarios+=(${scenarios[$i]})
        args_seeds+=(${seeds[$j]})
    done
done

# execute the script with different params
python -u onpolicy/scripts/eval_mpe.py --use_valuenorm --use_popart \
--project_name "compare_3" \
--env_name "${args_env_names[$SLURM_ARRAY_TASK_ID]}" \
--algorithm_name "rmappo" \
--seed "${args_seeds[$SLURM_ARRAY_TASK_ID]}" \
--experiment_name "${args_models[$SLURM_ARRAY_TASK_ID]}" \
--scenario_name "${args_scenarios[$SLURM_ARRAY_TASK_ID]}" \
--num_agents=3 --num_obstacles=5 \
--n_training_threads 1 --n_rollout_threads 1 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs False \
--auto_mini_batch_size --target_mini_batch_size 128 \
--model_dir "./onpolicy" \
--save_gifs True \
--render_eval True \
--use_render True
&> $logs_folder/out_${args_models[$SLURM_ARRAY_TASK_ID]}_${args_seeds[$SLURM_ARRAY_TASK_ID]}