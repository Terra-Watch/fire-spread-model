#!/bin/bash
# Server-side copy of the cleaned spread parts 0, 1, 2 from the research bucket into the jalari bucket (same region). Never deletes.
exec >> /var/log/copy2.log 2>&1
set -x
shutdown -h +180
S=gs://kireeti-wildfire-507012/firespread_v1/dataset; D=gs://terrawatch-data-jalari/firespread_v1/dataset
for n in 0 1 2; do gcloud storage rsync -r $S/part_p${n}_v2 $D/part_p${n}_v2 && echo "part_p${n}_v2 COPIED" || echo "part_p${n}_v2 FAILED"; done
gcloud storage cp /var/log/copy2.log gs://terrawatch-data-jalari/firespread_v1/v2_logs/copy2.log
shutdown -h now
