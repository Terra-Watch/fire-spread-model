#!/bin/bash
# vm/launch.sh PROJECT PART NPARTS [OUT_BUCKET_URL] -- creates the builder VM for one share of the big case set (the pack is already staged in the research bucket; the VM's service account has access to it).
set -u
PRJ=$1; PART=$2; NP=$3; B=gs://kireeti-wildfire-507012/firespread_v1; OUT=${4:-$B}   # a VM whose service account can only read the research bucket writes to its own project bucket
FS=/Users/kireeti/Desktop/Projects/TerraWatch/fire-spread; P=$FS/data/vm_pack
ACC=$(grep -E "^\| $PRJ \| " /Users/kireeti/Desktop/Projects/TerraWatch/wildfire-datasets-v3/DECISIONS.md | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}')
[ -z "$ACC" ] && for A in $(gcloud auth list --format='value(account)' 2>/dev/null); do [ -n "$(gcloud projects list --account="$A" --filter="projectId=$PRJ" --format='value(projectId)' 2>/dev/null | head -1)" ] && ACC=$A; done   # projects not in DECISIONS.md: find the logged-in account that sees it
red() { sed -E 's/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+/<email>/g'; }
sed -e "s|@BUCKET@|$B|" -e "s|@OUT@|$OUT|" -e "s|@PART@|$PART|" -e "s|@NPARTS@|$NP|" $FS/vm/startup.sh.tmpl > $P/startup_part$PART.sh
for Z in asia-south1-c asia-south1-a asia-south1-b; do
  echo "== $PRJ part $PART: trying $Z"
  gcloud compute instances create wf-build-$PART --account="$ACC" --project=$PRJ --zone=$Z --machine-type=n2-standard-16 \
    --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=120GB --boot-disk-type=pd-balanced \
    --scopes=storage-rw --labels=purpose=wildfire,owner=kireeti --max-run-duration=17h --instance-termination-action=STOP \
    --metadata-from-file=startup-script=$P/startup_part$PART.sh 2>&1 | red | tail -3
  gcloud compute instances list --account="$ACC" --project=$PRJ --filter="name=wf-build-$PART" --format="value(name,zone.basename(),status)" 2>&1 | red | grep -q wf-build-$PART && { echo "VM_CREATED $PRJ part $PART in $Z"; break; }
done
echo LAUNCH_DONE
