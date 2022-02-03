#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0-01:00:00
#SBATCH --output="./logs/%x.out"
#SBATCH --error="./logs/%x.err"
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL

source /space/calico/1/users/Harsha/venvs/recon-venv/bin/activate
export PYTHONPATH=/space/calico/1/users/Harsha/photo-reconstruction
# export LD_LIBRARY_PATH=/usr/pubsw/packages/CUDA/10.1/lib64:/usr/pubsw/packages/CUDA/10.2/lib64:/usr/pubsw/packages/CUDA/11.1/lib64

echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi
echo 'End time:' `date`