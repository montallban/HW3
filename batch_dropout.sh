#!/bin/bash

#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results/bmi_shoulder_dropout_%04a_stdout.txt
#SBATCH --error=results/error_bmi_shoulder_dropout__%04a_.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=bmi_dropout
#SBATCH --mail-user=michael.montalbano@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/mcmontalbano/HW3
#SBATCH --array=0-479
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate
# Change this line to start an instance of your experiment
python base_lxreg.py -exp_index $SLURM_ARRAY_TASK_ID -epochs 1000 -dropout_input 'on' -dropout 0.1


