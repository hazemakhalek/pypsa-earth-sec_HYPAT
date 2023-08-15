#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:40:26 2023

@author: haz43975
"""
import pypsa
import snakemake
import pandas as pd
import os
import ruamel.yaml

def extract_res(n):
        
    res_caps = pd.DataFrame(index=n.buses[n.buses.carrier=='AC'].index, columns=technologies)
        
    for tech in technologies:
        res_ind=n.generators[n.generators.carrier==tech].index
        res_caps_vals= n.generators.loc[res_ind, "p_nom_opt"]
        res_caps_vals.index = res_caps_vals.index.to_series().apply(lambda x: x.split(" ")[0])
        res_caps[tech] = res_caps_vals


    ex_q_30 = int(n_path.split('_')[-1].split('e')[0])
    ex_q_50 = ex_quantities_2050[ex_q_30]

    res_caps.to_csv(parent_dir+ "/" + n_path.split("/")[-1].\
                    replace(".nc", ".csv").replace("elec", "res_caps_elec").replace('_'+str(ex_q_30)+'export', '_'+str(ex_q_50)+'export'))

    
def extract_electrolyzers(n):
    
    # parent_dir = n_path.split("/")[0] + "/{}/optimal_capacities".format(run)

    ex_q_30 = int(n_path.split('_')[-1].split('e')[0])
    ex_q_50 = ex_quantities_2050[ex_q_30]

    elec_ind =n.links[n.links.carrier=='H2 Electrolysis'].index
    n.links.loc[elec_ind].p_nom_opt.to_csv(parent_dir+ "/" +n_path.split("/")[-1]\
                    .replace(".nc", ".csv").replace("elec", "electrolyzer_caps_elec").replace('_'+str(ex_q_30)+'export', '_'+str(ex_q_50)+'export'))

def extract_pipelines(n):
    
    # parent_dir = n_path.split("/")[0] + "/{}/optimal_capacities".format(run)

    ex_q_30 = int(n_path.split('_')[-1].split('e')[0])
    ex_q_50 = ex_quantities_2050[ex_q_30]

    ppl_ind =n.links[n.links.carrier=='H2 pipeline'].index
    n.links.loc[ppl_ind].p_nom_opt.to_csv(parent_dir+ "/" +n_path.split("/")[-1].\
                    replace(".nc", ".csv").replace("elec", "pipeline_caps_elec").replace('_'+str(ex_q_30)+'export', '_'+str(ex_q_50)+'export'))
    
    
if __name__ == "__main__":
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    technologies= ['csp', 'rooftop-solar', 'solar', 'onwind', 'onwind2', 'offwind', 'offwind2']
    
    # ex_quantities =   [0, 1, 10, 50, 200, 1000]

    # ex_quantities_2050 = {0:0, 1:10, 10:100, 50:500, 200:1000, 1000:3000}



    file_paths_2030 = ['/nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/config_2030_cons.yaml',
                '/nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/config_2030_opt.yaml',
                '/nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/config_2030_real.yaml']

    file_paths_2050 = ['/nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/config_2050_cons.yaml',
                '/nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/config_2050_opt.yaml',
                '/nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth-sec/config_2050_real.yaml']


    # Read the YAML file
    yaml = ruamel.yaml.YAML()

    for file_path in file_paths_2030:
        with open(file_path, 'r') as file:
            yaml_content = yaml.load(file)

        ex_quantities = yaml_content['export']['h2export_all_quantities']

    for file_path in file_paths_2050:
        with open(file_path, 'r') as file:
            yaml_content_2050 = yaml.load(file)

        run = yaml_content_2050['run'].replace('2050', '2030')


    folder_path = 'results/{}/postnetworks/'.format(run)

    # List all files in the folder
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    cluster_info = {}
    rate_info = {}

    # Extract clusters and rate data from 2030 runs:
    for file_name in file_names:
        cluster_info[file_name.split("_")[-2]] = int(file_name.split("_")[-9])
        rate_info[file_name.split("_")[-2]] = file_name.split("_")[-3]

    
    # Change the existing_params in config files of 2050 runs:
    for file_path in file_paths_2050:
        with open(file_path, 'r') as file:
            yaml_content_2050 = yaml.load(file)

        ex_quantities_2050 = yaml_content_2050['export']['h2export_all_quantities']
        

        yaml_content_2050 ['custom_data']['existing_params']['run'] = run
        yaml_content_2050 ['custom_data']['existing_params']['H'] = 3
        yaml_content_2050 ['custom_data']['existing_params']['year'] = 2030

        if yaml_content_2050 ['scenario']['demand'][0] == 'BS':
            yaml_content_2050 ['custom_data']['existing_params']['demand'] = 'BS'
            yaml_content_2050 ['custom_data']['existing_params']['clusters'] = cluster_info['BS']
            yaml_content_2050 ['custom_data']['existing_params']['rate'] = rate_info['BS']

        elif yaml_content_2050 ['scenario']['demand'][0] == 'AP':
            yaml_content_2050 ['custom_data']['existing_params']['demand'] = 'AP'
            yaml_content_2050 ['custom_data']['existing_params']['clusters'] = cluster_info['AP']
            yaml_content_2050 ['custom_data']['existing_params']['rate'] = rate_info['AP']

        elif yaml_content_2050 ['scenario']['demand'][0] == 'NZ':
            yaml_content_2050 ['custom_data']['existing_params']['demand'] = 'NZ'
            yaml_content_2050 ['custom_data']['existing_params']['clusters'] = cluster_info['NZ']
            yaml_content_2050 ['custom_data']['existing_params']['rate'] = rate_info['NZ']
        
            # Write the updated YAML back to the file
        with open(file_path, 'w') as file:
            yaml.dump(yaml_content_2050, file)


    ex_quantities_2050 = dict(zip(ex_quantities, ex_quantities_2050))

    n_paths = []
    for item in file_names:
        n_paths.append(folder_path + item)

    parts = folder_path.split('/') 
    parent_dir = '/'.join(parts[:2]) + '/optimal_capacities'
    os.makedirs(parent_dir, exist_ok=True)

    for n_path in n_paths:
        n = pypsa.Network(n_path)
        clusters = n_path.split("_")[-9]

        extract_res(n)
        extract_electrolyzers(n)
        extract_pipelines(n)