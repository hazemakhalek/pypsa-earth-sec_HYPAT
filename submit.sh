#!/bin/bash

conda activate pypsa-earth

cp config.bright_ref.yaml config.yaml
snakemake --profile slurm all

cp config.bright.yaml config.yaml
snakemake --profile slurm all

NEXTCLOUD_URL="https://tubcloud.tu-berlin.de/remote.php/webdav/BRIGHT/results/"
USERNAME="cpschau"
PASSWORD=$(get_nextcloud_password)

# Upload the file to Nextcloud via WebDAV
tar -czf results_241031.tar.gz /results/241031/
curl -u "$USERNAME:$PASSWORD" -T "results_241031.tar.gz" "$NEXTCLOUD_URL"
