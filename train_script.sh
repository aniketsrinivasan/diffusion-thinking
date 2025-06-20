#!/bin/bash
#SBATCH --time=160:00:00
#SBATCH --mem=108GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=ALL
#SBATCH --gres=gpu:1
#SBATCH --nodefile=nodefile.txt

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
SBATCH -o outputs/JOB%j-sbatch.out # File to which STDOUT will be written
SBATCH -e outputs/JOB%j-sbatch-err.out # File to which STDERR will be written

# email notifications: Get email when your job starts, stops, fails, completes...
# Set email address
#SBATCH --mail-user=$SBATCH_EMAIL
# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=END,FAIL
source .venv/bin/activate

# Diagnostic commands
echo "CUDA version: $(nvcc --version)"
nvidia-smi

python train.py