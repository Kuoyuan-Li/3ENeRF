#!/bin/bash

#---------------------------------------------------#
#@Author: Kuoyuan Li
#@Email:  kuoyuanl@student.unimelb.edu.au
#@Create: 15:45 27/01/2023
#@Version: v0.0
#---------------------------------------------------#

# Partition for the job:
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="deca-deform"

# The project ID which this job should run under:
#SBATCH --account="punim1485"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# The amount of memory in megabytes per process in the job:
#SBATCH --mem-per-cpu=8G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=1-00:0:00

module load anaconda3/2021.11
export CONDA_ENVS_PATH=/data/gpfs/projects/punim1485/kuoyuanl/anaconda3/envs
eval "$(conda shell.bash hook)"
conda activate DECA

# Load required modules
module load cuda/11.1.1
# module load python/3.7.4

echo "Starting the process--------------------------------------"

# Run the job from the directory where it was launched (default)

# The job command(s):
python demos/DECA_canonical_construct.py -i ./../kuoyuan_home_glass/rgb/ds -s ./../kuoyuan_home_glass/deca_canonical_out --saveNeededOnly True --saveOrigVis True

echo "Finishing the training-------------------------------------"
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
