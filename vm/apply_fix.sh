#!/bin/bash
# vm/apply_fix.sh PART [reset] -- puts the corrected startup script (data/vm_pack/startup_partN.sh) into the builder VM's metadata; with `reset` the VM is rebooted, which re-runs the script (every step skips finished work).
set -u
PART=$1; MODE=${2:-}; FS=/Users/kireeti/Desktop/Projects/TerraWatch/fire-spread; D=/Users/kireeti/Desktop/Projects/TerraWatch/wildfire-datasets-v3/DECISIONS.md
case $PART in 0|1) PRJ=ace-sequencer-478304-h4;; 2|3) PRJ=elated-terminal-484811-h0;; 4) PRJ=jalari;; 5) PRJ=causal-bongo-508008-e5;; esac
ACC=$(grep -E "^\| $PRJ \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}')
[ -z "$ACC" ] && for A in $(gcloud auth list --format='value(account)' 2>/dev/null); do [ -n "$(gcloud projects list --account="$A" --filter="projectId=$PRJ" --format='value(projectId)' 2>/dev/null | head -1)" ] && ACC=$A; done
red() { sed -E 's/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+/<email>/g'; }
Z=$(gcloud compute instances list --account="$ACC" --project=$PRJ --filter="name=wf-build-$PART" --format="value(zone.basename())" 2>/dev/null)
gcloud compute instances add-metadata wf-build-$PART --zone=$Z --account="$ACC" --project=$PRJ --metadata-from-file=startup-script=$FS/data/vm_pack/startup_part$PART.sh 2>&1 | red | tail -1
[ "$MODE" = reset ] && gcloud compute instances reset wf-build-$PART --zone=$Z --account="$ACC" --project=$PRJ 2>&1 | red | tail -1
echo "FIX_APPLIED part $PART $MODE"
