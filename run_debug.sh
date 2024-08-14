#!/bin/bash
#SBATCH --job-name=single_gpu_pose_est
#SBATCH --nodelist=gcpl4-eu-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=l4-24g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=48G
#SBATCH --output=./joblogs/shapenet/debug_vis_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/shapenet/debug_bis_%j.error     # Redirect stderr to a separate error log file
#SBATCH --time=24:00:00

JOB_START_TIME=$(date)
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}" 
echo "Running on node: $(hostname)"
echo "Starting on:     ${JOB_START_TIME}" 


# cuda
export CUDA_HOME=/opt/modules/nvidia-cuda-12.3
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-12.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-12.3/bin:$PATH


source ~/.bashrc
micromamba activate hyper # pose_est


# single gpu to see what is proper learning rate, using in hala
export PYTHONPATH="$PYTHONPATH:/home/qi_ma/neuips_2024/Weights_Space_Learning"
# ####### exp0: use small vit for image encoder with 32x32x32 of a tiny model
# python main.py --config-name=train_plane

python main.py --config-name=train_omniobject_apple

# python -m experiments.omni_pose_regression \
#     output_dir=output_exps/final/learnable/pretrain_tiny_1gpu_learnable_doreg_large_lr \
#     net.learnable_volume=1 \
#     batch_size=8 \
#     epochs=150 \
#     data.samples_num=24 \
#     data=omniobject_object  \
#     net=transformer_3D_tiny \
#     lr_image_encoder=1e-4 \
#     lr_volume_encoder=1e-5 \
#     lr_pose_encoder=1e-4 \
#     lr_translation_decoder=1e-4 \
#     lr_rotation_decoder=1e-4 \
#     lr_volume=5e-4 \
#     reg=1 \
#     workers=1 \
#     seed=123 \
#     reg=1 \
#     data.data_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/omni_weight_dataset_filter \
#     net.pretrain_encoder_path=/home/qi_ma/neuips_2024/Weights_Space_Learning/checkpoints/best_model_tiny.pth \
#     use_wandb=true \

# python -m experiments.omni_pose_regression \
#     output_dir=output_exps/final/learnable/small_baseline_1gpu_learnable_doreg_alpha_round0 \
#     net.learnable_volume=1 \
#     batch_size=8 \
#     epochs=150 \
#     data.samples_num=24 \
#     data=omniobject_object  \
#     net=transformer_3D_small \
#     lr_image_encoder=1e-4 \
#     lr_volume_encoder=1e-4 \
#     lr_pose_encoder=1e-4 \
#     lr_translation_decoder=1e-4 \
#     lr_rotation_decoder=1e-4 \
#     lr_volume=2e-4 \
#     reg=1 \
#     workers=1 \
#     seed=123 \
#     reg=1 \
#     data.data_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/omni_weight_dataset_filter \
#     use_wandb=true \