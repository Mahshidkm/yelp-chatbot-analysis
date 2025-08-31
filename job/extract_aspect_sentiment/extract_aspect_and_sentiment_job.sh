#!/bin/bash
#SBATCH --job-name=aspect_analysis              # Job name
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks (usually 1 for Python scripts)
#SBATCH --cpus-per-task=8                       # CPUs to allocate per task
#SBATCH --mem=64G                              # Total memory limit
#SBATCH --gpus=4                               # Total GPUs
#SBATCH --time=02:00:00                         # Time limit (hh:mm:ss)
#SBATCH --partition=normal                      # Partition to submit the job
#SBATCH --account=mscbdt2024                    # Account name
#SBATCH --output=%x-%j.out                      # Standard output file
#SBATCH --error=%x-%j.err                       # Standard error file

# Load the Java environment (if necessary)
VENV_PATH=~/mahshid_env
export JAVA_HOME=$VENV_PATH/java/jdk-11.0.2
export PATH=$JAVA_HOME/bin:$PATH

# Activate your virtual environment
source $VENV_PATH/bin/activate

# Load the CUDA module
module load cuda/12.2.2                          # Ensure the correct version

# Run the script with Accelerate
accelerate launch --num_processes=4 --mixed_precision=fp16 my_project/src/aspect_sentiment_restaurants_reviews.py