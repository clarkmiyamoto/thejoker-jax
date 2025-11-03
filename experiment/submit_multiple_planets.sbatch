#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx8000,a100,v100
#SBATCH --mem=200GB
#SBATCH --job-name=MultiplePlanets
#SBATCH --mail-type=END
#SBATCH --mail-user=cm6627@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

OVERLAY_FILE=/scratch/cm6627/hemcee_env/overlay-15GB-500K.ext3:ro
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif

PYTHON_FILE=multiple_planets.py
export PYTHONUNBUFFERED=1

singularity exec --nv \
  --overlay $OVERLAY_FILE \
  $SINGULARITY_IMAGE \
  /bin/bash -c "source /ext3/env.sh; python $PYTHON_FILE --move=hmc_walk"