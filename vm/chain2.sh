#!/bin/bash
# jalari has one L4: when the rescoring VM (seed 1, and failed attempts at seed 2) has stopped, run seed 2 (4 training jobs, then their rescoring).
cd "$(dirname "$0")/.."; D=../wildfire-datasets-v3/DECISIONS.md
A=$(grep -E "^\| jalari \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}'); J=gs://terrawatch-data-jalari/firespread_v1
ZONES="asia-south1-a asia-south1-b asia-south1-c asia-southeast1-b asia-southeast1-c asia-east1-a asia-east1-c us-central1-a us-central1-b us-central1-c europe-west4-a europe-west4-b"
while true; do s=$(gcloud compute instances list --account="$A" --project=jalari --filter="name=spread-resc-j" --format="value(status)" 2>/dev/null); [ "$s" = TERMINATED ] && break; sleep 120; done
echo "$(date -u +%H:%M) rescoring VM stopped: launching seed 2"
BKT=$J/train OUTP=$J/train_out_v4 DSET=$J/dataset VER=v4 bash vm/launch_train.sh jalari spread-train-c /tmp/jobs_j2.txt "$ZONES" 8 | tail -1
echo CHAIN2_DONE
