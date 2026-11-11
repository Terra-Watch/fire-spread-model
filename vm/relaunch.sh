#!/bin/bash
# vm/relaunch.sh PROJECT PART NPARTS [OUT_BUCKET_URL] -- re-renders the startup script, stops the builder, updates its metadata and starts it again; the disk keeps the finished work, every step skips what is done.
set -u
PRJ=$1; PART=$2; NP=$3; B=gs://kireeti-wildfire-507012/firespread_v1; OUT=${4:-$B}
FS=/Users/kireeti/Desktop/Projects/TerraWatch/fire-spread; P=$FS/data/vm_pack
ACC=$(grep -E "^\| $PRJ \| " /Users/kireeti/Desktop/Projects/TerraWatch/wildfire-datasets-v3/DECISIONS.md | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}')
[ -z "$ACC" ] && for A in $(gcloud auth list --format='value(account)' 2>/dev/null); do [ -n "$(gcloud projects list --account="$A" --filter="projectId=$PRJ" --format='value(projectId)' 2>/dev/null | head -1)" ] && ACC=$A; done
red() { sed -E 's/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+/<email>/g'; }
sed -e "s|@BUCKET@|$B|" -e "s|@OUT@|$OUT|" -e "s|@PART@|$PART|" -e "s|@NPARTS@|$NP|" $FS/vm/startup.sh.tmpl > $P/startup_part$PART.sh
Z=$(gcloud compute instances list --account="$ACC" --project=$PRJ --filter="name=wf-build-$PART" --format="value(zone.basename())" 2>/dev/null | tail -1)
gcloud compute instances stop wf-build-$PART --zone=$Z --account="$ACC" --project=$PRJ --quiet 2>&1 | red | tail -1
gcloud compute instances add-metadata wf-build-$PART --zone=$Z --account="$ACC" --project=$PRJ --metadata-from-file=startup-script=$P/startup_part$PART.sh 2>&1 | red | tail -1
gcloud compute instances start wf-build-$PART --zone=$Z --account="$ACC" --project=$PRJ 2>&1 | red | tail -1
echo "RELAUNCHED $PRJ part $PART"
