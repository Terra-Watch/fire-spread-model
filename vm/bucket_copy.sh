#!/bin/bash
# Copies everything from the research bucket into the jalari bucket, server side (same region, no download, no egress). Safe to run again: rsync skips what is already there. Never deletes.
exec >> /var/log/copy.log 2>&1
set -x
shutdown -h +720
SRC=gs://kireeti-wildfire-507012; DST=gs://terrawatch-data-jalari
( while true; do sleep 300; gcloud storage cp /var/log/copy.log $DST/_copy_logs/copy.log >/dev/null 2>&1; done ) &
for top in $(gcloud storage ls $SRC/ | sed 's|.*/\([^/]*\)/$|\1|'); do
  echo "=== $top"; gcloud storage rsync -r $SRC/$top $DST/$top && echo "$top COPIED" || echo "$top FAILED"
done
for f in $(gcloud storage ls $SRC/ | grep -v '/$'); do gcloud storage cp $f $DST/ ; done
gcloud storage du -s -h $SRC > /tmp/s.txt 2>&1; gcloud storage du -s -h $DST >> /tmp/s.txt 2>&1; cat /tmp/s.txt
gcloud storage cp /var/log/copy.log $DST/_copy_logs/copy.final.log
shutdown -h now
