#!/bin/bash

conda activate pypsa-earth

# rm pypsa-earth/networks/elec.nc
cp config.pypsa-earth_low.yaml config.pypsa-earth.yaml
cp config.bright_ref_low.yaml config.yaml
snakemake --profile slurm all
cp config.bright_low.yaml config.yaml
snakemake --profile slurm all

rm pypsa-earth/networks/elec.nc
cp config.pypsa-earth_med.yaml config.pypsa-earth.yaml
cp config.bright_ref_med.yaml config.yaml
snakemake --profile slurm all
cp config.bright_med.yaml config.yaml
snakemake --profile slurm all

rm pypsa-earth/networks/elec.nc
cp config.pypsa-earth_high.yaml config.pypsa-earth.yaml
cp config.bright_ref_high.yaml config.yaml
snakemake --profile slurm all
cp config.bright_high.yaml config.yaml
snakemake --profile slurm all

rm pypsa-earth/networks/elec.nc
cp config.pypsa-earth_vhigh.yaml config.pypsa-earth.yaml
cp config.bright_ref_vhigh.yaml config.yaml
snakemake --profile slurm all
cp config.bright_vhigh.yaml config.yaml
snakemake --profile slurm all

# NEXTCLOUD_URL="https://tubcloud.tu-berlin.de/remote.php/webdav/BRIGHT/results/"
# USERNAME="cpschau"
# PASSWORD=$(get_nextcloud_password)

# # Upload the file to Nextcloud via WebDAV
# tar -czf results_241031.tar.gz /results/241031/
# curl -u "$USERNAME:$PASSWORD" -T "results_241031.tar.gz" "$NEXTCLOUD_URL"
