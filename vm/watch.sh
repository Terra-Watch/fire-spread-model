#!/bin/bash
# One line per builder per check: step, latest progress, failures; exits when all six parts are done or a VM stopped without finishing.
D=/Users/kireeti/Desktop/Projects/TerraWatch/wildfire-datasets-v3/DECISIONS.md
acc() { grep -E "^\| $1 \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}'; }
ACCB=$(sed -n '146p' $D | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]+' | head -1)
ACCE=""; for A in $(gcloud auth list --format='value(account)' 2>/dev/null); do [ -n "$(gcloud projects list --account="$A" --filter="projectId=elated-terminal-484811-h0" --format='value(projectId)' 2>/dev/null | head -1)" ] && ACCE=$A; done
# part: project, vm-account, log location, log reader account
declare -A PRJ=( [0]=ace-sequencer-478304-h4 [1]=ace-sequencer-478304-h4 [2]=elated-terminal-484811-h0 [3]=elated-terminal-484811-h0 [4]=jalari [5]=causal-bongo-508008-e5 )
declare -A VMACC=( [0]=$(acc ace-sequencer-478304-h4) [1]=$(acc ace-sequencer-478304-h4) [2]=$ACCE [3]=$ACCE [4]=$(acc jalari) [5]=$(acc causal-bongo-508008-e5) )
R=gs://kireeti-wildfire-507012/firespread_v1/out; E=gs://elated-terminal-wildfire-stage1/firespread_v1/out
declare -A LOG=( [0]=$R/part0 [1]=$R/part1 [2]=$E/part2 [3]=$E/part3 [4]=$R/part4 [5]=$R/part5 )
declare -A RDR=( [0]=$ACCB [1]=$ACCB [2]=$ACCE [3]=$ACCE [4]=$ACCB [5]=$ACCB )
sleep ${1:-600}
while true; do
  done_n=0; line=""
  for k in 0 1 2 3 4 5; do
    if [[ " ${SKIP:-} " == *" $k "* ]]; then line="$line | p$k paused"; continue; fi
    ST=$(gcloud compute instances list --account="${VMACC[$k]}" --project=${PRJ[$k]} --filter="name=wf-build-$k" --format="value(status)" 2>/dev/null | tail -1)
    T=$(gcloud storage cat ${LOG[$k]}/job.log --account="${RDR[$k]}" 2>/dev/null | tail -n 600)
    if echo "$T" | grep -q PART_DONE; then done_n=$((done_n+1)); line="$line | p$k DONE"; continue; fi
    if [ -n "$ST" ] && [ "$ST" != "RUNNING" ]; then echo "BUILDER $k NOT RUNNING (status $ST) and not finished"; echo "$T" | tail -8 | cut -c1-200; exit 1; fi
    stp=$(echo "$T" | grep -E '^===' | tail -1 | sed -E 's/.*scripts\/([0-9a-z_]+)\.py.*/\1/'); prog=$(echo "$T" | grep -E '^[0-9]+ of [0-9]+' | tail -1)
    line="$line | p$k ${ST:-?} ${stp:-setup} [${prog:-}] fails $(echo "$T" | grep -c '^fail')"
  done
  echo "$line"
  NS=$(echo ${SKIP:-} | wc -w); [ $done_n -eq $((6-NS)) ] && { echo "ALL ACTIVE PARTS DONE (paused: ${SKIP:-none})"; exit 0; }
  sleep ${2:-900}
done
