#!/bin/bash
# Packs the code and the static data the VMs need into data/vm_pack/.
cd /Users/kireeti/Desktop/Projects/TerraWatch/fire-spread
P=data/vm_pack
tar czf $P/code.tgz src scripts pyproject.toml
tar czf $P/parquet.tgz data/cases.parquet data/events.parquet data/pilot_cases.parquet data/big_cases.parquet data/detections
gzip -c ../wildfire-dataset/dataset/forest_union_0p001.npy > $P/forest.npy.gz
gzip -c data/osm_lines.geojsonseq > $P/osm_lines.geojsonseq.gz
mkdir -p $P/worldpop && cp ../wildfire-dataset/inputs_raw/worldpop/*.tif $P/worldpop/
ls -la $P | awk '{print $5/1e6" MB", $9}'
echo PACK_DONE
