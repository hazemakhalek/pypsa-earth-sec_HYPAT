#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --error='job-%j-error.out'
#SBATCH --output='job-%j-out.out'
#SBATCH --export=NONE
#SBATCH --nodelist=CB-HPC-07
#SBATCH --chdir=/nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec
echo $HOSTNAME

module purge
module load Anaconda3
module load Java
source activate /nfs/home/edd32710/.conda/envs/pypsa-earth

export GRB_LICENSE_FILE=/nfs/home/edd32710/gurobi.lic


which python3
python3 -c "import sys;print(sys.path)"

# -----------------------------------------------
# h2_delivery = none
# -----------------------------------------------
rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
    # -----------
    # Q0 - Cons
    # -----------
python modify_config_files_none.py
python modify_config_files_Q0_2050.py
python prepare_2050_runs.py
cp config.pypsa-earth_conservative_2050.yaml config.pypsa-earth.yaml
cp config_2050_cons.yaml config.yaml
snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Cons
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Cons
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Real
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_realistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_real.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Real
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Real
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Opt
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_optimistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_opt.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Opt
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Opt
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary


# # -----------------------------------------------
# # h2_delivery = monthly
# # -----------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Cons
#     # -----------
# python modify_config_files_monthly.py
# python modify_config_files_Q0_2050.py
# python prepare_2050_runs.py
# cp config.pypsa-earth_conservative_2050.yaml config.pypsa-earth.yaml
# cp config_2050_cons.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Cons
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Cons
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Real
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_realistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_real.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Real
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Real
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Opt
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_optimistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_opt.yaml config.yaml
# snakemake -j 32 make_summary
#     -----------
#     Q1 - Opt
#     -----------
# python modify_config_files_Q1.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Opt
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary


# # -----------------------------------------------
# # h2_delivery = weekly
# # -----------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Cons
#     # -----------
# python modify_config_files_weekly.py
# python modify_config_files_Q0_2050.py
# python prepare_2050_runs.py
# cp config.pypsa-earth_conservative_2050.yaml config.pypsa-earth.yaml
# cp config_2050_cons.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Cons
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Cons
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Real
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_realistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_real.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Real
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Real
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Opt
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_optimistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_opt.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Opt
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Opt
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary




# # -----------------------------------------------
# # h2_delivery = daily
# # -----------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Cons
#     # -----------
# python modify_config_files_daily.py
# python modify_config_files_Q0_2050.py
# python prepare_2050_runs.py
# cp config.pypsa-earth_conservative_2050.yaml config.pypsa-earth.yaml
# cp config_2050_cons.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Cons
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Cons
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Real
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_realistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_real.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Real
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Real
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Opt
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_optimistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_opt.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Opt
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Opt
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary


# # -----------------------------------------------
# # h2_delivery = hourly
# # -----------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Cons
#     # -----------
# python modify_config_files_hourly.py
# python modify_config_files_Q0_2050.py
# python prepare_2050_runs.py
# cp config.pypsa-earth_conservative_2050.yaml config.pypsa-earth.yaml
# cp config_2050_cons.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Cons
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Cons
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_cons.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Real
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_realistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_real.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Real
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Real
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_real.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # ---------------------------------------------------------
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec.nc
# rm /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/networks/elec_s.nc
#     # -----------
#     # Q0 - Opt
#     # -----------
# python modify_config_files_Q0_2050.py
# cp config.pypsa-earth_optimistic_2050.yaml config.pypsa-earth.yaml
# cp config_2050_opt.yaml config.yaml
# snakemake -j 32 make_summary
#     # -----------
#     # Q1 - Opt
#     # -----------
# python modify_config_files_Q1.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q1.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary
#     # -----------
#     # Q2_rest - Opt
#     # -----------
# python modify_config_files_Q2_rest.py
# cp config_2050_opt.yaml config.yaml
# cp /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports_Q_rest.csv /nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/data/export_ports.csv
# snakemake -j 32 make_summary