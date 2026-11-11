#!/bin/bash
cd /Users/kireeti/Desktop/Projects/TerraWatch/fire-spread
scripts/cloud/launch.sh ace-sequencer-478304-h4 1 6
scripts/cloud/launch.sh elated-terminal-484811-h0 2 6 gs://elated-terminal-wildfire-stage1/firespread_v1
scripts/cloud/launch.sh elated-terminal-484811-h0 3 6 gs://elated-terminal-wildfire-stage1/firespread_v1
scripts/cloud/launch.sh jalari 4 6
echo REST_LAUNCHED
