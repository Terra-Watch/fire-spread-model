#!/bin/bash
# vm/launch_train.sh PROJECT NAME JOBS_FILE "ZONES" HOURS : one spot L4 VM running vm/train_startup.sh. Data and results are in the jalari bucket.
PRJ=$1; NAME=$2; JOBS=$3; ZONES=$4; HOURS=${5:-24}; D=/Users/kireeti/Desktop/Projects/TerraWatch/wildfire-datasets-v3/DECISIONS.md
ACC=$(grep -E "^\| $PRJ \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}')
T=gs://terrawatch-data-jalari/firespread_v1
for Z in $ZONES; do
  gcloud compute instances create $NAME --account="$ACC" --project=$PRJ --zone=$Z --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
    --provisioning-model=SPOT --instance-termination-action=STOP --max-run-duration=${HOURS}h --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
    --boot-disk-size=150GB --boot-disk-type=pd-balanced --scopes=storage-rw --labels=purpose=wildfire,owner=kireeti \
    --metadata=install-nvidia-driver=True,bucket=${BKT:-$T/train},out=${OUTP:-$T/train_out},dataset=${DSET:-$T/dataset},ver=${VER:-v2} --metadata-from-file=jobs=$JOBS,startup-script=$(dirname "$0")/train_startup.sh >/dev/null 2>&1
  gcloud compute instances list --account="$ACC" --project=$PRJ --filter="name=$NAME" --format="value(name,zone.basename(),status)" 2>&1 | grep -q $NAME && { echo "TRAIN_VM_CREATED $PRJ $NAME in $Z"; exit 0; }
done
echo "TRAIN_VM_FAILED $PRJ $NAME"; exit 1
