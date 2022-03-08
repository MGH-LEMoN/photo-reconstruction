#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=dgx-a100,rtx8000,rtx6000,lcnrtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output="./logs/hcp_recon/%x.out"
#SBATCH --error="./logs/hcp_recon/%x.err"
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL

source /space/calico/1/users/Harsha/venvs/recon-venv/bin/activate
export PYTHONPATH=/space/calico/1/users/Harsha/photo-reconstruction

echo 'Start time:' `date`
start=$(date +%s)
echo 'Node:' $HOSTNAME
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi
end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"