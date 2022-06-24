#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000,lcnrtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
##SBATCH --cpus-per-task=1
##SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output="./logs/hcp-recon-20220601/%x.out"
#SBATCH --error="./logs/hcp-recon-20220601/%x.err"
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
    python "$@"
fi
end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
