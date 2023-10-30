#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000,rtx6000,lcnrtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
##SBATCH --cpus-per-task=1
##SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output="./logs/4diana-hcp-recons/%x.out"
#SBATCH --error="./logs/4diana-hcp-recons/%x.err"
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL

source /usr/local/freesurfer/nmr-dev-env-bash

echo 'Start time:' `date`
echo 'Node:' $HOSTNAME
echo "$@"
start=$(date +%s)

"$@"

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
