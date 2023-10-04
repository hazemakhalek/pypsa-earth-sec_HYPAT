# -*- coding: utf-8 -*-
""""This script is currently configured for the Brazilian Enertile data. Before running, make sure to adjust:

1. The folder structure and paths. Parent directory is data_preprocessing, which contains the script and one sub-driectory called data.
Within the data folder there is another sub-directory, called enertile. In Enertile you find the folders, which corrrespond to the pathways e.g. 2030_Low.
Within a pathway directory you can find the technology sub-directories, where the technology names are spelled with capital letters (PVR, SOPV, SOST, WINDOFFSHORE, WINDONSHORE).
The technology folder contains one csv file for the hourly availability timeseries and another one containing other e.g. the costs. The filenames contain the numerical scenario id,
which also has to be specified in the configuration dictionary scen_dict in line 114. All together one examplary path could be:
data_preprocessing/data/enertile/2030_Low/PVR/respotentialhourdata179.csv

2. Alphabetical country code (search BR, BRA)

3. Names of pypsa network clusters, especially if your network comprises AC and DC nodes (search _AC)

4. Configuration dictionaries (scen_dict, i_r_dict)
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import pypsa
from itertools import product

#%%
def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    if w.sum() == 0:
        return 0
    else:
        return (d * w).sum() / w.sum()

def prepare_dataframe(co, tech, year, scenario_1, scenario_2, timestep, interest_rate, step=0):
    global res_t, installable
    
    if scenario_1 == 'High':
        df = pd.read_csv('/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/data_preprocessing/data/enertile/2030_2050_{}/{}/respotentialhourdata{}.csv'.format( scenario_1, tech.upper(), scenario_2))
        dfp = pd.read_csv('/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/data_preprocessing/data/enertile/2030_2050_{}/{}/respotentials{}.csv'.format(scenario_1, tech.upper(), scenario_2))
    else:
        df = pd.read_csv('/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/data_preprocessing/data/enertile/{}_{}/{}/respotentialhourdata{}.csv'.format(year, scenario_1, tech.upper(), scenario_2))
        dfp = pd.read_csv('/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/data_preprocessing/data/enertile/{}_{}/{}/respotentials{}.csv'.format(year, scenario_1, tech.upper(), scenario_2))

    if tech in ['windonshore','windoffshore']:
        if step == 0:
            df = df[df.step==0]
            dfp = dfp[dfp.step==0]
        else:
            df = df[df.step!=0]
            dfp = dfp[dfp.step!=0]
    
    regions = gpd.read_file('/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth/resources/bus_regions/regions_onshore_elec_s_30.geojson')
    regions = regions.rename({'name':'region'}, axis=1)[['region']]
    # df=df[df.hour<10]
    df=df[df['region'].str.contains('BR')]
    df['region']=df.region.str.replace(r'BRA_(0_)?', 'BR.', regex=True).str.replace('_1$', '_1_AC', regex=True)
    df['tech'] = df['tech'].str.replace(tech, tech_dict[tech])
    df = df[df.simyear==year]
    #%%
    missing=list(set(regions.region)-set(df.region.unique()))
    # df_miss=pd.DataFrame(list(product(missing, range(10))), columns=['region', 'hour'])
    df_miss=pd.DataFrame(list(product(missing, range(8760))), columns=['region', 'hour'])
    df_miss['simyear'] = year
    df_miss['tech'] = tech_dict[tech]
    df_miss['step'] = 0
    df_miss['value'] = 0
    df = pd.concat([df, df_miss])
    #%%
    df=df.set_index(['region', 'step'])

    #%%
    dfp=dfp[dfp['region'].str.contains('BR')]
    dfp['region']=dfp.region.str.replace(r'BRA_(0_)?', 'BR.', regex=True).str.replace('_1$', '_1_AC', regex=True)
    dfp = dfp[dfp.simyear==year]
    dfp_missing = pd.DataFrame(data=missing, columns=['region'])
    dfp_missing['potstepsizeMW'] = 0
    dfp_missing['simyear'] = year
    dfp_missing['tech'] = tech_dict[tech]
    dfp_missing['step'] = 0
    dfp_missing['flh'] = 0
    dfp_missing['installedcapacity'] = 0
    dfp_missing['annualcostEuroPMW'] = dfp['annualcostEuroPMW'].mean()
    dfp_missing['variablecostEuroPMWh'] = dfp['variablecostEuroPMWh'].mean()
    dfp_missing['investmentEuroPKW'] = dfp['investmentEuroPKW'].mean()
    dfp_missing['interestrate'] = dfp['interestrate'].mean()
    dfp_missing['lifetime'] = dfp['lifetime'].mean()
    dfp_missing['scenarioid'] = dfp['scenarioid'].mean()
    dfp_missing['fixedomEuroPKW'] = dfp['fixedomEuroPKW'].mean()
    
    dfp = pd.concat([dfp, dfp_missing])
    dfp.rename(columns={'region':'Generator', 'potstepsizeMW':'p_nom_max'}
                            , inplace=True)
    dfp=dfp.groupby(['Generator', 'step']).mean(numeric_only=True)#['potstepsizeMW']
        # return df
    
    #%%    
    df['intsall_cap']=dfp['p_nom_max']
    df=df.reset_index().groupby(['region', 'hour']).apply(
        w_avg, 'value', 'intsall_cap').reset_index().rename(columns={0: 'value'})
    
    #%%
    flh=dfp.groupby(['Generator']).apply(w_avg, 'flh', 'p_nom_max')
    installable=dfp.groupby(['Generator']).agg({'p_nom_max':np.sum, 
                    'annualcostEuroPMW': np.mean, 'fixedomEuroPKW': np.mean,
                        'installedcapacity': np.sum, 'lifetime': np.mean})
    res_t = pd.pivot_table(df, values='value', index='hour', columns='region')
    res_t.index=pd.date_range(start='01-01-2013 00:00:00', end='31-12-2013 23:00:00', freq='h')
    # res_t.index=pd.date_range(start='01-01-2013 00:00:00', end='01-01-2013 09:00:00', freq='h')
    res_t = res_t.multiply(flh)#.resample(timestep).mean()
    


    if step == 0:
        res_t.to_csv('/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/data_preprocessing/data/enertile/postprocessed/{0}_{1}_{2}_potential.csv'.format(tech_dict[tech].replace(' ', '-'), year, interest_rate, step))
        installable.to_csv('/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/data_preprocessing/data/enertile/postprocessed/{0}_{1}_{2}_installable.csv'.format(tech_dict[tech].replace(' ', '-'), year, interest_rate, step))
    else:
        res_t.to_csv('/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/data_preprocessing/data/enertile/postprocessed/{0}{3}_{1}_{2}_potential.csv'.format(
            tech_dict[tech].replace(' ', '-'), year, interest_rate, str(step+1)))
        installable.to_csv('/nfs/home/edd32710/projects/HyPAT/Brazil_cas/pypsa-earth-sec/data_preprocessing/data/enertile/postprocessed/{0}{3}_{1}_{2}_installable.csv'.format(
            tech_dict[tech].replace(' ', '-'), year, interest_rate, str(step+1)))



scen_dict = {'Low':{
                2030:{
                    'sopv': '339', 'pvr': '204', 'sost': '220', 'windonshore':'316', 'windoffshore':'224'
                },
                2050:{
                    'sopv': '338', 'pvr': '203', 'sost': '219', 'windonshore':'315', 'windoffshore':'223'
                }
    },
            'Medium':{
                2030:{
                    'sopv': '341', 'pvr': '206', 'sost': '222', 'windonshore':'318', 'windoffshore':'226'
                },
                2050:{
                    'sopv': '342', 'pvr': '207', 'sost': '223', 'windonshore':'319', 'windoffshore':'227' 
                }
            },
            'High':{
                2030:{
                    'sopv': '340', 'pvr': '205', 'sost': '221', 'windonshore':'317', 'windoffshore':'225' 
                },
                2050:{
                    'sopv': '340', 'pvr': '205', 'sost': '221', 'windonshore':'317', 'windoffshore':'225' 
                }
            }
}
i_r_dict = {'Low':{
                2030:'0.071',
                2050:'0.045'
    },
            'Medium':{
                2030:'0.076',
                2050:'0.086'
            },
            'High':{
                2030:'0.175',
                2050:'0.175'
            }
}


tech_dict={'sopv': 'solar', 'pvr': 'rooftop solar', 'sost': 'csp', 'windonshore':'onwind', 'windoffshore':'offwind'}

for i_1, pathway in enumerate(scen_dict.keys()):
    print('Preprocessing enertile data of {} pathway ({}/3)'.format(pathway, i_1+1))
    
    scenario = scen_dict[pathway]

    for i_2, year in enumerate(scenario.keys()):
        print('Year: {} ({}/2)'.format(year, i_2+1))
        
        scenario_y = scenario[year]
        
        for i_3, technology in enumerate(scenario_y.keys()):
            print('Technology: {} ({}/{})'.format(technology, i_3+1, len(scenario_y.keys())))

            scenario_y_t = scenario_y[technology]
            interest_rate_s_y = i_r_dict[pathway][year]
            if 'wind' in technology:
                steps=[0,1]
            else:
                steps=[0]

            for i_4, step in enumerate(steps):
                print('Step: {} ({}/{})'.format(step+1, i_4+1, len(steps)))
                prepare_dataframe('BR', technology, year, pathway, scenario_y_t, '3H', interest_rate_s_y, step)
