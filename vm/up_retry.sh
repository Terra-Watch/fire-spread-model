#!/bin/bash
# Retry the upload of one v2 part (a silent stall of gcloud storage cp hit part 1): up to 3 tries of 70 minutes each, then free the disk.
cd "$(dirname "$0")/.."; N=$1; P=part_p$N; B=gs://kireeti-wildfire-507012/firespread_v1/dataset
ACC=$(sed -n '146p' ../wildfire-datasets-v3/DECISIONS.md | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]+' | head -1)
for t in 1 2 3; do
  timeout 4200 gcloud storage cp -r data/dataset/${P}_v2 $B/ --account="$ACC" --project=space-debris-507012 >/dev/null 2>&1
  S=$(gcloud storage ls -l $B/${P}_v2/spatial.npy --account="$ACC" --project=space-debris-507012 2>/dev/null | head -1 | awk '{print $1}')
  L=$(wc -c < data/dataset/${P}_v2/spatial.npy | tr -d " ")
  [ "$S" = "$L" ] && { mkdir -p data/dataset/v2_small; rsync -a --exclude spatial.npy --exclude 'labels_*' data/dataset/${P}_v2/ data/dataset/v2_small/${P}_v2/; rm -rf data/dataset/$P data/dataset/${P}_v2; echo "$P uploaded (try $t)"; exit 0; }
  echo "$P try $t failed (remote ${S:-none} of $L bytes)"
done
echo "$P UPLOAD FAILED"; exit 1
