#!/bin/bash
#SBATCH --qos=long
##SBATCH --account=pintxo
##SBATCH --partition=pintxo
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1         
#SBATCH --nodes=1         
#SBATCH --cpus-per-task=4    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=48G         # memory per cpu-core (4G is default)
##SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="hybrid"
#SBATCH --gres=gpu:1       # number of gpus per node
#BATCH --constraint=a100-sxm4


pwd; hostname; date

module load Miniforge3
conda activate /scratch/vperez/conda-envs/deepmd-jax

python -u train_hybrid.py > log.txt

date
