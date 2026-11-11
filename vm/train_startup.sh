#!/bin/bash
# GPU training VM for the image models (U-Net ResNet-50, FireSenseNet). Reads instance metadata: bucket (gs:// prefix with code.tgz, sample_train.parquet and parts/part_*_v2 folders), out (gs:// results prefix), jobs (one 60_train.py argument line per line, e.g. "--model unet --horizon 24 --out runs/unet24 --epochs 30 --sample sample_train.parquet").
# Each job's run folder is uploaded as soon as it ends; finished jobs (marker in the bucket) are skipped; the VM powers itself off at the end. Spot VM: a restart resumes the queue.
exec >> /var/log/train.log 2>&1
set -x
md() { curl -s -f -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true; }
B=$(md bucket); OUT=$(md out); JOBS=$(md jobs); NAME=$(hostname)
[ -n "$B" ] && [ -n "$OUT" ] && [ -n "$JOBS" ] || { echo "missing metadata"; shutdown -h now; exit 0; }
( while true; do sleep 300; gcloud storage cp /var/log/train.log $OUT/logs/$NAME.log >/dev/null 2>&1; done ) &
mkdir -p /opt/fs && cd /opt/fs
[ -d fire-spread ] || { mkdir fire-spread && gcloud storage cp $B/code.tgz . && tar xzf code.tgz -C fire-spread; }
DS=$(md dataset); VER=$(md ver); VER=${VER:-v2}; mkdir -p fire-spread/data/dataset && for n in 0 1 2 3 4 5; do gcloud storage cp -r "$DS/part_p${n}_${VER}" fire-spread/data/dataset/ ; done; gcloud storage cp $B/sample_train.parquet fire-spread/data/
python3 -m pip install -q segmentation-models-pytorch pandas pyarrow scipy scikit-learn numpy
cd fire-spread; nvidia-smi -L
PARTS=$(ls -d data/dataset/part_*_${VER} | tr '\n' ' '); i=0; [ $(echo $PARTS | wc -w) -eq 6 ] || { echo "only $(echo $PARTS | wc -w) parts for $VER"; gcloud storage cp /var/log/train.log $OUT/logs/$NAME.final.log; shutdown -h now; exit 0; }
while IFS= read -r line; do
  [ -z "$line" ] && continue; i=$((i+1)); tag=$(echo "$line" | md5sum | cut -c1-8)
  if gcloud storage ls $OUT/done/$tag >/dev/null 2>&1; then echo "skip job $i"; continue; fi
  echo "=== job $i: $line"
  rs=$(echo "$line" | sed -nE 's#.*--rescore runs/([^ ]+).*#\1#p'); [ -n "$rs" ] && [ ! -d runs/$rs ] && { mkdir -p runs; gcloud storage cp -r $OUT/$rs runs/; }   # a rescoring job needs the trained model first
  python3 -u scripts/60_train.py $(echo "$line" | sed 's#--sample sample_train.parquet#--sample data/sample_train.parquet#') --parts $PARTS || echo "job $i FAILED"
  outdir=$(echo "$line" | sed -E 's/.*--out ([^ ]+).*/\1/')
  gcloud storage cp -r $outdir $OUT/ && echo done | gcloud storage cp - $OUT/done/$tag
done <<< "$JOBS"
gcloud storage cp /var/log/train.log $OUT/logs/$NAME.final.log
shutdown -h now
