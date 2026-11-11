#!/bin/bash
# One part at a time: download part_pN from the research bucket, rebuild the detection layers from clean detections (54b), audit v2, upload v2, free the disk. Keeps the small files of v2 in data/dataset/v2_small/part_pN_v2 for the merge.
cd "$(dirname "$0")/.."; B=gs://kireeti-wildfire-507012/firespread_v1/dataset; PY=.venv/bin/python
ACC=$(sed -n '146p' ../wildfire-datasets-v3/DECISIONS.md | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]+' | head -1)
for N in ${PARTS:-0 1 2 3 4 5}; do
  P=part_p$N; [ -f data/dataset/v2_small/${P}_v2/v2.json ] && { echo "$P already done"; continue; }
  echo "== $P download"; rm -rf data/dataset/${P}_v2; [ "$KEEP" = "$N" ] || rm -rf data/dataset/$P
  [ "$KEEP" = "$N" ] || gcloud storage cp -r $B/$P data/dataset/ --account="$ACC" --project=space-debris-507012 >/dev/null 2>&1 || { [ "$KEEP" = "$N" ] || { echo "$P DOWNLOAD FAILED"; continue; }; }
  echo "== $P v2"; $PY scripts/54b_reexport_detection_layers.py data/dataset/$P > /tmp/v2_$P.log 2>&1 || { echo "$P V2 FAILED"; tail -3 /tmp/v2_$P.log; continue; }; tail -1 /tmp/v2_$P.log
  echo "== $P audit"; $PY scripts/55_audit_part.py data/dataset/${P}_v2 2>&1 | tail -2
  echo "== $P upload"; gcloud storage cp -r data/dataset/${P}_v2 $B/ --account="$ACC" --project=space-debris-507012 >/dev/null 2>&1 && echo "$P uploaded" || { echo "$P UPLOAD FAILED"; continue; }
  mkdir -p data/dataset/v2_small; rsync -a --exclude spatial.npy --exclude 'labels_*' data/dataset/${P}_v2/ data/dataset/v2_small/${P}_v2/
  rm -rf data/dataset/$P data/dataset/${P}_v2; echo "$P DONE"
done
echo V2_ALL_DONE
