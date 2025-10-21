#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --job-name=gin_optuna
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n19


# set max wallclock time
#SBATCH --time=48:00:00
#SBATCH --output=/home/ivanam/BIO-Info/bio_info/multi_seminar2/optuna/logs/gin_optuna_%j.log
#SBATCH --error=/home/ivanam/BIO-Info/bio_info/multi_seminar2/optuna/logs/gin_optuna_%j.err


module load python/miniconda3.10 
eval "$(conda shell.bash hook)"
conda activate bio_inf
#conda install -c conda-forge rdkit

PYTHON_EXECUTABLE=$(which python)

${PYTHON_EXECUTABLE} /home/ivanam/BIO-Info/bio_info/multi_seminar2/optuna/hyperparam_gin.py

echo "All files done."