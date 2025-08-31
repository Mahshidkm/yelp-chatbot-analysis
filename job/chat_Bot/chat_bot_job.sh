#!/bin/bash
#SBATCH --job-name=Chat_Bot_job          # Name of your job
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-gpu=16                # CPU cores per GPU
#SBATCH --gpus-per-node=1                # Number of GPUs for the task
#SBATCH --time=01:00:00                  # Total run time limit (HH:MM:SS)
#SBATCH --partition=normal              # The partition where you submit
#SBATCH --account=mscbdt2024     # Specify your project group account

# Define the path to your virtual environment
VENV_PATH=~/mahshid_env

# Set Java environment (assuming it's installed in mahshid_env/java)
export JAVA_HOME=$VENV_PATH/java/jdk-11.0.2
export PATH=$JAVA_HOME/bin:$PATH

# Activate your virtual environment
source $VENV_PATH/bin/activate

# Get compute node IP
NODE_IP=$(hostname -i)

echo "Running on node: $NODE_IP"
echo "Web interface will be available at: http://${NODE_IP}:5000"

# Execute your Python script
python my_project/run.py