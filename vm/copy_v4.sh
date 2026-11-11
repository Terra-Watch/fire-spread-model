#!/bin/bash
# Server-side copies of the final (v4) spread parts so both buckets hold all six: parts 0-2 research -> jalari, parts 3-5 jalari -> research. Never deletes.
exec >> /var/log/copy4.log 2>&1
set -x
shutdown -h +180
R=gs://kireeti-wildfire-507012/firespread_v1/dataset; J=gs://terrawatch-data-jalari/firespread_v1/dataset
for n in 0 1 2; do gcloud storage rsync -r $R/part_p${n}_v4 $J/part_p${n}_v4 && echo "p$n research->jalari COPIED" || echo "p$n FAILED"; done
for n in 3 4 5; do gcloud storage rsync -r $J/part_p${n}_v4 $R/part_p${n}_v4 && echo "p$n jalari->research COPIED" || echo "p$n FAILED"; done
gcloud storage cp /var/log/copy4.log $J/fill_logs/copy4.log
shutdown -h now
