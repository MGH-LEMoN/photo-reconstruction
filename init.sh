#!/bin/bash

export PYTHONPATH=/space/calico/1/users/Harsha/photo-reconstruction
if [ -d $PYTHONPATH ] 
then
    echo "Directory $PYTHONPATH exists." 
else
    # clone repository
    git clone https://github.com/hvgazula/photo-reconstruction
fi

export UW_photo_recon=$PYTHONPATH/data/UW_photo_recon
if [ -d  $UW_photo_recon ] 
then
    echo "Directory $UW_photo_recon exists." 
else
    # link data
    ln -s /cluster/vive/UW_photo_data data/
fi

if [[ "$VIRTUAL_ENV" != "" ]]
then
    echo "Environment already activated."
else
    # activate environment
    source /space/calico/1/users/Harsha/venvs/recon-venv/bin/activate
fi

echo "Go to Makefile for more information."

