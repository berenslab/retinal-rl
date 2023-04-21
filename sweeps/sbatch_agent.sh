#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=24         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti     # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/%j.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sacha.sokoloski@uni-tuebingen.de  # Email to which notifications will be sent

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID 
echo -e "---------------------------------\n"

# Run code with values specified in task array
echo "-------- PYTHON OUTPUT ----------"
singularity exec --nv -B `pwd` --pwd `pwd` retinal_rl.sif wandb agent alex404/retinal-rl/$1 --count $2
echo "---------------------------------"
