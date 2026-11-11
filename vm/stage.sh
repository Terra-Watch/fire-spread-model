#!/bin/bash
# Uploads the pack once to gs://kireeti-wildfire-507012/firespread_v1/pack/ (the research bucket; only this new prefix is written). Run with the bucket owner's account.
FS=/Users/kireeti/Desktop/Projects/TerraWatch/fire-spread; P=$FS/data/vm_pack; B=gs://kireeti-wildfire-507012/firespread_v1
ACCB=$(sed -n '146p' /Users/kireeti/Desktop/Projects/TerraWatch/wildfire-datasets-v3/DECISIONS.md | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]+' | head -1)
red() { sed -E 's/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+/<email>/g'; }
export CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=False
for f in code.tgz parquet.tgz tiles.txt forest.npy.gz osm_lines.geojsonseq.gz; do
  for try in 1 2 3; do
    gcloud storage cp $P/$f $B/pack/ --account="$ACCB" --project=space-debris-507012 2>&1 | red | tail -1
    gcloud storage ls $B/pack/$f --account="$ACCB" --project=space-debris-507012 >/dev/null 2>&1 && { echo "STAGED $f"; break; }
  done
done
gcloud storage cp $P/worldpop/*.tif $B/pack/worldpop/ --account="$ACCB" --project=space-debris-507012 2>&1 | red | tail -1
echo STAGE_DONE
