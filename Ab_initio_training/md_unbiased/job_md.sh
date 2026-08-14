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

cd /scratch/vperez/ForVera/ExampleTrainDeePMD_Jax_2

export PYTHONPATH=/scratch/vperez/deepmd-jax:$PYTHONPATH

/scratch/vperez/conda-envs/deepmd-jax/bin/python -u plumed_md.py > log_md.txt

date
