#!/bin/bash
# CPU VM: fills the empty burn-history, NDVI and Sentinel-2 layers (scripts/57_fill_gaps.py) of the given cleaned spread parts and uploads part_pN_v3. Metadata: parts ("3 4 5"), src (gs:// folder holding part_pN_v2 and gappack.tgz), workers.
exec >> /var/log/fill.log 2>&1
set -x
shutdown -h +900
md() { curl -s -f -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true; }
PARTS=$(md parts); SRC=$(md src); W=$(md workers); NAME=$(hostname); INP=$(md inp); INP=${INP:-v2}; OUTV=v3; [ "$INP" = v3 ] && OUTV=v4
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-pip python3-venv
mkdir -p /opt/w && cd /opt/w && gcloud storage cp $SRC/gappack.tgz . && tar xzf gappack.tgz
python3 -m venv venv && ./venv/bin/pip install -q numpy pandas pyarrow scipy pyproj rasterio shapely numba httpx joblib
cd /opt/w/fire-spread; mkdir -p data/dataset
( while true; do sleep 300; gcloud storage cp /var/log/fill.log $SRC/fill_logs/$NAME.log >/dev/null 2>&1; done ) &
for N in $PARTS; do
  P=part_p$N
  rm -rf data/dataset/${P}_${INP} data/dataset/${P}_${OUTV}
  gcloud storage cp -r $SRC/${P}_${INP} data/dataset/ || { echo "$P DOWNLOAD FAILED"; continue; }
  WORKERS=${W:-8} PYTHONPATH=src ../venv/bin/python -u scripts/57_fill_gaps.py data/dataset/${P}_${INP} > /var/log/fill_$P.log 2>&1 || { echo "$P FILL FAILED"; tail -3 /var/log/fill_$P.log; }
  tail -2 /var/log/fill_$P.log
  gcloud storage cp -r data/dataset/${P}_${OUTV} $SRC/ && echo "$P ${OUTV} UPLOADED"
  gcloud storage cp /var/log/fill_$P.log $SRC/fill_logs/fill_$P.log
  rm -rf data/dataset/${P}_${INP} data/dataset/${P}_${OUTV}
done
gcloud storage cp /var/log/fill.log $SRC/fill_logs/$NAME.final.log
shutdown -h now
