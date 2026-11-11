#!/bin/bash
# Launches the next job set of a project's L4 as soon as the VM before it has stopped (one L4 per project). Log: chain.log next to this script's caller.
cd "$(dirname "$0")/.."; D=../wildfire-datasets-v3/DECISIONS.md
acc() { grep -E "^\| $1 \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}'; }
R=gs://kireeti-wildfire-507012/firespread_v1; J=gs://terrawatch-data-jalari/firespread_v1
ZONES="asia-south1-a asia-south1-b asia-south1-c asia-southeast1-b asia-southeast1-c asia-east1-a asia-east1-c us-central1-a us-central1-b us-central1-c europe-west4-a europe-west4-b"
state() { gcloud compute instances list --account="$(acc $1)" --project=$1 --filter="name=$2" --format="value(status)" 2>/dev/null; }
done_ace=0; done_c=0; done_j=0
while [ $((done_ace + done_c + done_j)) -lt 3 ]; do
  if [ $done_ace = 0 ]; then s=$(state ace-sequencer-478304-h4 mlg-clean-abl); if [ "$s" = TERMINATED ]; then
     echo "$(date -u +%H:%M) ace ablation VM stopped: launching the baseline models"; BKT=$R/train OUTP=$R/train_out_v4 DSET=$R/dataset VER=v4 bash vm/launch_train.sh ace-sequencer-478304-h4 spread-base-a /tmp/jobs_base.txt "$ZONES" 14 | tail -1; done_ace=1; fi; fi
  if [ $done_c = 0 ]; then s=$(state causal-bongo-508008-e5 spread-train-v4); if [ "$s" = TERMINATED ]; then
     echo "$(date -u +%H:%M) causal training VM stopped: launching the pooled-metric rescoring (seed 0)"; BKT=$R/train OUTP=$R/train_out_v4 DSET=$R/dataset VER=v4 bash vm/launch_train.sh causal-bongo-508008-e5 spread-resc-c /tmp/jobs_rescore_c.txt "$ZONES" 4 | tail -1; done_c=1; fi; fi
  if [ $done_j = 0 ]; then s=$(state jalari spread-train-b); if [ "$s" = TERMINATED ]; then
     echo "$(date -u +%H:%M) jalari training VM stopped: launching the pooled-metric rescoring (seeds 1, 2)"; BKT=$J/train OUTP=$J/train_out_v4 DSET=$J/dataset VER=v4 bash vm/launch_train.sh jalari spread-resc-j /tmp/jobs_rescore_j.txt "$ZONES" 5 | tail -1; done_j=1; fi; fi
  sleep 180
done
echo CHAIN_DONE
