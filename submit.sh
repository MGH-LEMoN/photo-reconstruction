#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000,lcnrtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
##SBATCH --cpus-per-task=1
##SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output="./logs/mgh-recon/%x.out"
#SBATCH --error="./logs/mgh-recon/%x.err"
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL

source /usr/local/freesurfer/nmr-dev-env-bash

echo 'Start time:' `date`
echo 'Node:' $HOSTNAME
echo "$@"
start=$(date +%s)

fspython "$@"

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
