#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=test
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
###SBATCH --partition=gpu
#SBATCH --gres=gpu:p1080:1

module load pytorch/python3.6/0.3.0_4
# module load python/intel/2.7.12 pytorch/0.2.0_1 protobuf/intel/3.1.0 
# module load torchvision/0.1.8 
python Ex6.py