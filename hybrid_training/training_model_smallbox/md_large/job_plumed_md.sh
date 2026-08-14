#!/bin/bash
#SBATCH --qos=xlong
##SBATCH --account=pintxo
##SBATCH --partition=pintxo
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1         
#SBATCH --nodes=1         
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=96G         # memory per cpu-core (4G is default)
##SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="hybrid_md"
#SBATCH --gres=gpu:4        # number of gpus per node
#SBATCH --constraint=a100-sxm4


pwd; hostname; date

module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

module load Miniforge3
conda activate /scratch/vperez/conda-envs/deepmd-jax

export PYTHONNOUSERSITE=1
export PYTHONPATH="/scratch/vperez/deepmd-jax:$PYTHONPATH"
export PLUMED_KERNEL="/scratch/vperez/conda-envs/deepmd-jax/lib/libplumedKernel.so"


export LD_LIBRARY_PATH="/scratch/vperez/conda-envs/deepmd-jax/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/scratch/vperez/conda-envs/deepmd-jax/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/scratch/vperez/conda-envs/deepmd-jax/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"


/scratch/vperez/conda-envs/deepmd-jax/bin/python plumed_md.py > log_plumed.txt
# /scratch/vperez/conda-envs/deepmd-jax/bin/python
date
