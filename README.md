# Hybrid-CaCO3

## Project Overview

This repository contains a hybrid deep learning and ab initio framework for modeling calcium carbonate systems, with a focus on CaCO3 ion pairing and molecular dynamics in water and related phases. The project combines a custom `deepmd_jax` implementation with hybrid training routines that can incorporate observable-based ab initio corrections during training.

Key components:
- `deepmd_jax/`: custom DeepMD-style JAX package providing model training, evaluation, and data handling.
- `hybrid_training/`: hybrid training examples and scripts for CaCO3 models.
- `Ab_initio_training/`: ab initio trajectory and dataset material used for hybrid observable training.
- `training-set/`: training dataset directories used by the model training scripts. This dataset is obtained from the external repository `https://github.com/AMLS-PRG/CaCO3-SCAN-ML`.

## Repository Structure

- `deepmd_jax/`: main Python package for model training and testing.
- `hybrid_training/training_model_smallbox/`: training script, evaluation script, SLURM batch example, and related data for the small-box hybrid model.
- `Ab_initio_training/`: ab initio MD trajectories, data conversion scripts, and biasing inputs used by hybrid training.
- `training-set/`: local copy of the CaCO3-SCAN-ML training dataset for different chemical species and phases.

## Requirements

This project is designed to run in a Python environment with JAX and scientific Python libraries. A minimal environment should include:

- Python 3.10 or 3.11
- JAX and JAXLIB
- NumPy
- SciPy
- Matplotlib
- Optax
- Flax
- jax-md (if using MD utilities)

Because the repository is built around a custom local Python package, the simplest option is to use a Conda environment and run scripts from the repository root.

## Installation

1. Create a Conda environment:

```bash
conda create -n deepmd-jax python=3.11 -y
conda activate deepmd-jax
```

2. Install required Python packages:

```bash
pip install numpy scipy matplotlib jax jaxlib optax flax jax-md
```

3. Set the project root on `PYTHONPATH` so `hybrid_training` scripts can import the local package:

```bash
cd /scratch/vperez/Hybrid-CaCO3
export PYTHONPATH="$PWD/deepmd_jax:$PYTHONPATH"
```

If you prefer not to use `PYTHONPATH`, run scripts from the root directory and keep `deepmd_jax` sibling to the script location so Python can resolve the import path automatically.

## Usage

### 1. Train the hybrid model

The primary training script is located at:

```bash
hybrid_training/training_model_smallbox/train_hybrid.py
```

This script uses the local `deepmd_jax` package and a hybrid observable-based training setup. A sample SLURM batch script is provided at:

```bash
hybrid_training/training_model_smallbox/job.sh
```

To run training directly:

```bash
cd /scratch/vperez/Hybrid-CaCO3/hybrid_training/training_model_smallbox
python train_hybrid.py > log.txt
```

To run with SLURM (if available):

```bash
cd /scratch/vperez/Hybrid-CaCO3/hybrid_training/training_model_smallbox
sbatch job.sh
```

### 2. Evaluate the trained model

The evaluation script is:

```bash
hybrid_training/training_model_smallbox/predictions.py
```

It loads `model.pkl` and tests it against the training data directories:

```bash
cd /scratch/vperez/Hybrid-CaCO3/hybrid_training/training_model_smallbox
python predictions.py
```

This writes predicted energies and forces to:

- `pred_force.txt`
- `pred_energy.txt`

### 3. Data sources and hybrid training inputs

The hybrid training script references:

- `../../training-set/*`: training datasets for CaCO3, water, carbonate, and other species.
- `/Hybrid-CaCO3/Ab_initio_training/md_small_2ns/dataset`: observable training dataset for hybrid correction.

Update these paths in `train_hybrid.py` if your local dataset paths differ.

## Jupyter Notebooks

This project includes several Jupyter notebooks for visualization and analysis:

### Training Analysis Notebooks

- **`hybrid_training/training_model_smallbox/plotting.ipynb`**  
  Parses the training log file (`log.txt`) and generates plots of training metrics including:
  - RMSE loss for energy, forces, and observables
  - Effective sample size (ESS)
  - Reweighted vs. unreweighted observables
  - Target value convergence

- **`hybrid_training/training_model_bigbox/plotting.ipynb`**  
  Similar analysis notebook for the large-box hybrid model training, following the same structure as the small-box version.

### Molecular Dynamics Visualization Notebooks

- **`hybrid_training/training_model_smallbox/molecular_dynamics/graficas.ipynb`**  
  Analyzes molecular dynamics trajectories from PLUMED COLVAR files:
  - Parses collective variables (distances, biases)
  - Computes free energy surfaces (FES) using reweighting
  - Generates probability distributions and free energy plots
  - Uses statistical methods (Gaussian filters, interpolation)

- **`hybrid_training/training_model_bigbox/molecular_dynamics/graficas.ipynb`**  
  Similar MD analysis for the large-box model molecular dynamics simulations.



## Notes

- The repository does not currently include a packaged `setup.py` at the top level. Scripts are intended to be run from the repository tree with `deepmd_jax` available as a local package.
- The sample `job.sh` activates a specific Conda environment located at `/scratch/vperez/conda-envs/deepmd-jax`. Adjust this path for your own environment.
- The project uses hybrid training options such as `hybrid=True`, `obs_train_data_path`, `obs_temperature`, and `obs_target` inside `train_hybrid.py`.

## Recommended workflow

1. Activate Conda environment and set `PYTHONPATH`.
2. Confirm dataset directories exist under `training-set/`.
3. Run training in `hybrid_training/training_model_smallbox`.
4. Run evaluation with `python predictions.py`.
5. Inspect logs and output files in the same folder.

## Contacts and further development

Use the local scripts and notebooks in `hybrid_training/training_model_smallbox/` and `Ab_initio_training/` as the main entry points for further hybrid model development.

