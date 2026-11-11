#!/bin/bash
# CPU VM: rebuild the detection layers (54b) of the given spread parts from clean detections, audit, upload part_pN_v2 to the research bucket. Metadata: parts (e.g. "3 4 5"). Parts run in parallel. Powers off at the end.
exec >> /var/log/v2.log 2>&1
set -x
shutdown -h +480
md() { curl -s -f -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true; }
LG=gs://terrawatch-data-jalari/firespread_v1/v2_logs; PARTS=$(md parts); PK=gs://kireeti-wildfire-507012/firespread_v1/pack_v2; B=gs://terrawatch-data-jalari/firespread_v1/dataset   # code pack from the research bucket; parts read from and written to the jalari bucket
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-pip python3-venv
mkdir -p /opt/w && cd /opt/w && gcloud storage cp $PK/v2pack.tgz . && tar xzf v2pack.tgz
python3 -m venv venv && ./venv/bin/pip install -q numpy pandas pyarrow scipy pyproj rasterio shapely numba httpx joblib
cd /opt/w/fire-spread
( while true; do sleep 300; gcloud storage cp /var/log/v2.log $LG/v2_cloud.log >/dev/null 2>&1; done ) &
one() {
  N=$1; P=part_p$N
  mkdir -p data/dataset; rm -rf data/dataset/$P; gcloud storage cp -r $B/$P data/dataset/ || { echo "$P DOWNLOAD FAILED"; return; }
  ../venv/bin/python scripts/54b_reexport_detection_layers.py data/dataset/$P > /var/log/v2_$P.log 2>&1 || { echo "$P V2 FAILED"; tail -3 /var/log/v2_$P.log; return; }
  tail -1 /var/log/v2_$P.log
  ../venv/bin/python scripts/55_audit_part.py data/dataset/${P}_v2 2>&1 | tail -1
  gcloud storage cp -r data/dataset/${P}_v2 $B/ && echo "$P V2 UPLOADED" && echo ok | gcloud storage cp - $LG/done_$P
}
PIDS=''; for N in $PARTS; do one $N & PIDS="$PIDS $!"; done; wait $PIDS   # not a bare wait: it would also wait for the log loop
gcloud storage cp /var/log/v2.log $LG/v2_cloud.final.log
shutdown -h now
