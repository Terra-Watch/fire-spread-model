#!/bin/bash
# jalari has one L4: when the seed-2 VM (spread-train-c) has stopped, run the layer-group ablation and the learning curve (spread-abl).
cd "$(dirname "$0")/.."; D=../wildfire-datasets-v3/DECISIONS.md
A=$(grep -E "^\| jalari \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}'); J=gs://terrawatch-data-jalari/firespread_v1
ZONES="asia-south1-a asia-south1-b asia-south1-c asia-southeast1-b asia-southeast1-c asia-east1-a asia-east1-c us-central1-a us-central1-b us-central1-c europe-west4-a europe-west4-b"
while true; do s=$(gcloud compute instances list --account="$A" --project=jalari --filter="name=spread-train-c" --format="value(status)" 2>/dev/null); [ "$s" = TERMINATED ] && break; sleep 120; done
n=$(gcloud storage ls $J/train_out_v4/done/ --account="$A" --project=jalari 2>/dev/null | grep -c done/); echo "$(date -u +%H:%M) seed-2 VM stopped (done markers in the bucket: $n)"
BKT=$J/train OUTP=$J/train_out_v4 DSET=$J/dataset VER=v4 bash vm/launch_train.sh jalari spread-abl /tmp/jobs_abl.txt "$ZONES" 10 | tail -1
echo CHAIN3_DONE
