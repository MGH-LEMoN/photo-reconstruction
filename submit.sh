#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output="./logs/propagate/%x.out"
#SBATCH --error="./logs/propagate/%x.err"
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL

source /space/calico/1/users/Harsha/venvs/recon-venv/bin/activate
export PYTHONPATH=/space/calico/1/users/Harsha/photo-reconstruction

echo 'Start time:' `date`
echo 'Node:' $HOSTNAME
echo "$@"
start=$(date +%s)
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    "$@"
fi
end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"