#!/bin/bash
#SBATCH --qos=regular
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1         
#SBATCH --nodes=1         
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=24G         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="md"
#SBATCH --gres=gpu:1        # number of gpus per node



pwd; hostname; date

module load Miniforge3
conda activate /scratch/vperez/conda-envs/deepmd-jax

export PYTHONNOUSERSITE=1
export PYTHONPATH="/scratch/vperez/deepmd-jax:$PYTHONPATH"
export PLUMED_KERNEL="/scratch/vperez/conda-envs/deepmd-jax/lib/libplumedKernel.so"

/scratch/vperez/conda-envs/deepmd-jax/bin/python plumed_md.py > log_plumed.txt
# /scratch/vperez/conda-envs/deepmd-jax/bin/python
date
