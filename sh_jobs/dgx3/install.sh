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

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/ConvPixelFormer/luna_exps
source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda create -n timm python=3.10

