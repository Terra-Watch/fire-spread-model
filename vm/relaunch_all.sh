#!/bin/bash
cd /Users/kireeti/Desktop/Projects/TerraWatch/fire-spread
tar czf data/vm_pack/code.tgz src scripts pyproject.toml
ACCB=$(sed -n '146p' /Users/kireeti/Desktop/Projects/TerraWatch/wildfire-datasets-v3/DECISIONS.md | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]+' | head -1)
gcloud storage cp data/vm_pack/code.tgz gs://kireeti-wildfire-507012/firespread_v1/pack/ --account="$ACCB" --project=space-debris-507012 2>&1 | tail -1
E_OUT=gs://elated-terminal-wildfire-stage1/firespread_v1
vm/relaunch.sh ace-sequencer-478304-h4 0 6 &
vm/relaunch.sh ace-sequencer-478304-h4 1 6 &
[ -z "${SKIP2:-}" ] && vm/relaunch.sh elated-terminal-484811-h0 2 6 $E_OUT &
vm/relaunch.sh elated-terminal-484811-h0 3 6 $E_OUT &
vm/relaunch.sh jalari 4 6 &
vm/relaunch.sh causal-bongo-508008-e5 5 6 &
wait
echo ALL_RELAUNCHED
