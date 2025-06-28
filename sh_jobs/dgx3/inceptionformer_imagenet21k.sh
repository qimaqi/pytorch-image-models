#!/bin/bash
#SBATCH --job-name=run_2d_unet
#SBATCH --output=sbatch_log/run_2d_unet_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu07,bmicgpu08,bmicgpu09,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexander.eins.qi@gmail.com


# pip install torch-geometric
export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

source /insait/qimaqi/miniconda3/bin/activate 
conda activate timm
cd /insait/qimaqi/workspace/pytorch-image-models

./distributed_train.sh 1 --data-dir /insait/imagenet21k/imagenet21k_resized --model hiera_inception_former_tiny_w0 --sched cosine --epochs 150 --warmup-epochs 5 --lr 1e-3 --reprob 0.5 --remode pixel --batch-size 512 --amp -j 1 --num-classes 21000 --amp-dtype float16
