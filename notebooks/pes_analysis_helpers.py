import pandas as pd
import pypsa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import geopandas as gpd
import cartopy.crs as ccrs
import os
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse

def sets_path_to_root(root_directory_name):  # Imported from pypsa-africa
    """
    Search and sets path to the given root directory (root/path/file).

    Parameters
    ----------
    root_directory_name : str
        Name of the root directory.
    n : int
        Number of folders the function will check upwards/root directed.

    """
    import os

    repo_name = root_directory_name
    n = 8  # check max 8 levels above. Random default.
    n0 = n

    while n >= 0:
        n -= 1
        # if repo_name is current folder name, stop and set path
        if repo_name == os.path.basename(os.path.abspath(".")):
            repo_path = os.getcwd()  # os.getcwd() = current_path
            os.chdir(repo_path)  # change dir_path to repo_path
            print("This is the repository path: ", repo_path)
            print("Had to go %d folder(s) up." % (n0 - 1 - n))
            break
        # if repo_name NOT current folder name for 5 levels then stop
        if n == 0:
            print("Cant find the repo path.")
        # if repo_name NOT current folder name, go one dir higher
        else:
            upper_path = os.path.dirname(os.path.abspath("."))  # name of upper folder
            os.chdir(upper_path)


def calc_expansion(n, carrier=None):
    '''''''returns expansion of generation and link components in MW'''''''    
    gens = n.generators.groupby('carrier').sum()
    inv_gens = gens.p_nom_opt - gens.p_nom

    links = n.links.groupby('carrier').sum()
    inv_links = links.p_nom_opt - links.p_nom

    inv = pd.concat([inv_gens, inv_links])

    if carrier != None:
        try:
            inv = inv.loc[carrier]
        except:
            print('carrier not existing')
    return inv


def calc_loads(n):
    '''''''in TWh'''''''
    loads_t = n.loads_t.p
    loads_t = loads_t[loads_t.columns.drop(list(loads_t.filter(regex='emissions')))]
    return loads_t.sum().sum() / 1e6 * n.snapshot_weightings.iloc[0,0]


def calc_generation(n, tech='all'):
    '''''''in TWh'''''''
    gen_t = n.generators_t.p[n.generators.loc[n.generators.bus.str.contains('1_AC$')].index].sum()
    gen_agg = pd.DataFrame(data={'bus':gen_t.index,'generation':gen_t.values})
    gen_agg['carrier'] = n.generators.loc[gen_agg.bus, 'carrier'].values
    gen_agg = gen_agg.groupby('carrier').sum() / 1e6 * n.snapshot_weightings.iloc[0,0]

    hydro_gen = n.storage_units_t.p.sum().sum() / 1e6 * n.snapshot_weightings.iloc[0,0]
    gen_agg.loc['hydro'] = [hydro_gen]

    link_gen_pps = n.links.filter(regex=('OCGT|biomass EOP|CHP'), axis=0)#n.links.loc[n.links.bus1.str.contains('_AC$', regex=True)]
    #link_gen_pps = n.links.loc[n.links.bus1.str.contains('_AC$', regex=True)]
    link_gen = -n.links_t.p1[link_gen_pps.index] / 1e6 * n.snapshot_weightings.iloc[0,0]
    link_gen.columns = link_gen_pps.carrier
    link_gen = link_gen.rename({'biomass EOP':'biomass', 'OCGT':'OCGT', 'urban central gas CHP':'gas_CHP', 'urban central solid biomass CHP':'biomass_CHP'}, axis=1)
    link_gen = link_gen.T.groupby(link_gen.T.index).sum().T
    for c in link_gen.columns:
        if c in gen_agg.index:
            gen_agg.loc[c] += link_gen[c].sum()
        else:
            gen_agg.loc[c] = {'generation':link_gen[c].sum()}

    res_idx = ['solar', 'rooftop-solar', 'onwind', 'onwind2', 'offwind', 'offwind2', 'csp', 'hydro', 'ror']
    if tech == 'all':
        return gen_agg
    elif tech == 'res':
        return gen_agg.loc[res_idx]
    else:
        return gen_agg.loc[tech]
    
    
def calc_util(n, tech):
    '''''Calculated capacity factor for generation technology'''''
    tech_index = n.generators.loc[n.generators.carrier == tech].index

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(tech_index)),
        index=n.snapshots,
        columns=tech_index,
    )
    
    gen_tech = (n.generators_t.p.filter(regex='{}$'.format(tech)) * weightings).sum()
    installed_cap = n.generators.p_nom_opt.filter(regex='{}$'.format(tech))
    year_avail = (n.generators_t.p_max_pu.filter(regex='{}$'.format(tech)) * weightings ).sum()
    
    max_gen_solar=(installed_cap * year_avail)
    
    return (gen_tech/max_gen_solar).mean()


def calc_additional_res(n, n_ref, tech):
    '''''''in GW'''''''

    res_index = n.generators.filter(regex='{}$'.format(tech), axis=0).index

    res_cap = n.generators.p_nom_opt[res_index].sum()
    res_cap_ref = n_ref.generators.p_nom_opt[res_index].sum()

    return (res_cap - res_cap_ref) / 1e3


def calc_additional_elec(n, n_ref):
    '''''''in GW'''''''

    elec_cap = n.links.filter(like='Electrolysis', axis=0).p_nom_opt.sum()
    res_cap_ref = n_ref.links.filter(like='Electrolysis', axis=0).p_nom_opt.sum()

    return (elec_cap - res_cap_ref) / 1e3


def calc_elec_cf(n):
    '''''Calculated capacity factor for electrolysis'''''
    elec_cap = n.links.filter(like='Electrolysis', axis=0).p_nom_opt.sum() *8760
    elec_output = n.links_t.p0.filter(like='Electrolysis').sum().sum() * n.snapshot_weightings.iloc[0,0]

    return elec_output / elec_cap

def calculate_spec_prod_costs(n):
    '''Takes a solved sector-coupled PyPSA network and outputs specific productio costs for one kg hydrogen at export nodes'''
    h2_prod_ex = n.links_t['p0'].filter(like='H2 export') #Amounts of H2 produced and provided for export
    h2_prod_ex.columns = h2_prod_ex.columns.str.strip(' export') #Modify column names to match with MP dataframe
    marginal_prices = n.buses_t.marginal_price.loc[:, h2_prod_ex.columns] #Marginal H2 prices at export nodes
    prod_costs = (h2_prod_ex * marginal_prices).sum(axis=0) #Product of production ampunts and prices yields total production costs for each export node
    spec_prod_costs = prod_costs / h2_prod_ex.sum(axis=0) # Division by total nodal production amount yields specific production costs of hydrogen for each export node
    spec_prod_costs = spec_prod_costs * 33.3 / 1000 #Calculate costs per kg assuming 1 kg H2 equals 33.3 KWh
    return spec_prod_costs

def calc_wap_h2_exp(n):
    '''Takes a solved sector-coupled PyPSA network and outputs specific productio costs for one kg exported hydrogen'''
    h2_prod_ex = n.links_t['p0'].filter(like='H2 export') #Amounts of H2 produced and provided for export

    h2_prod_ex.columns = h2_prod_ex.columns.str.strip(' export') #Modify column names to match with MP dataframe
    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(h2_prod_ex.T)),
        index=n.snapshots,
        columns=h2_prod_ex.columns,
    )
    h2_prod_ex = h2_prod_ex * weightings
    marginal_prices = n.buses_t.marginal_price.loc[:, h2_prod_ex.columns] #Marginal H2 prices at export nodes
    prod_costs = (h2_prod_ex * marginal_prices).sum().sum() #Product of production ampunts and prices yields total production costs for each export node
    spec_prod_costs = prod_costs / h2_prod_ex.sum().sum() # Division by total nodal production amount yields specific production costs of hydrogen for each export node
    spec_prod_costs = spec_prod_costs * 33.3 / 1000 #Calculate costs per kg assuming 1 kg H2 equals 33.3 KWh
    return spec_prod_costs


def calc_wap_h2(n, agg=True):
    '''Takes a solved sector-coupled PyPSA network and outputs specific productio costs for one kg exported hydrogen'''
    d_h2 = n.loads_t['p'].filter(like='H2').drop('H2 export load', axis=1) #Amounts of H2 demanded at export node

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(d_h2.T)),
        index=n.snapshots,
        columns=d_h2.columns,
    )
    d_h2 = d_h2 * weightings
    d_h2.columns = n.loads.loc[d_h2.columns, 'bus']
    d_h2 = d_h2.groupby(d_h2.columns, axis=1).sum()

    marginal_prices = n.buses_t.marginal_price.loc[:, d_h2.columns] #Marginal H2 prices at export nodes
    h2_costs = (d_h2 * marginal_prices).sum()
    
    if agg:
        return h2_costs.sum() / d_h2.sum().sum() * 33.3 / 1000
    else:
        return h2_costs / d_h2.sum() * 33.3 / 1000

def calc_curtailment(n):
    '''''Returns curtailment in TWh'''''
    gen_ts = n.generators_t
    max_gen = gen_ts.p_max_pu * 3
    gen = gen_ts.p[max_gen.columns] * 3
    p_nom_max = n.generators.loc[max_gen.columns].p_nom_opt

    curtailment = ((p_nom_max * max_gen) - gen)
    agg_curtailment =  curtailment.sum().sum() / 1e6

    return agg_curtailment / (gen.sum().sum() / 1e6) * 100

def plot_expansion(n):
    color_dict = {
        'CCGT':'#ee8340',
        'biomass':"green",
        'coal':'k',
        'oil':'#B5A642',
        'onwind':"dodgerblue",
        'offwind':'#6895dd',
        'solar':"orange",
        'rooftop-solar':'#ffef60',
        'OCGT':'wheat',
        'hydro':'b',
        'nuclear':'r',
        'gas':'brown',
        'residential rural solar thermal':'coral',
        'services rural solar thermal':'coral',
        'residential urban decentral solar thermal':'coral',
        'services urban decentral solar thermal':'coral',
        'urban central solar thermal':'coral',
        'csp':'coral'
        }

    gens = n.generators
    gens.loc[gens.carrier.str.contains('solar thermal'), 'carrier'] = 'csp'
    gens.loc[gens.carrier.str.contains('solar thermal'), 'carrier'] = 'csp'
    gens.loc[gens.carrier.str.contains('offwind'), 'carrier'] = 'offwind'
    gens.loc[gens.carrier.str.contains('onwind'), 'carrier'] = 'onwind'
    gen_grouped = gens.groupby('carrier', as_index=False).sum(numeric_only =True)

    OCGT_exp = n.links.filter(like='OCGT', axis=0).sum().p_nom_opt
    gen_grouped.loc[gen_grouped.carrier == 'OCGT', 'p_nom_opt'] = gen_grouped.loc[gen_grouped.carrier == 'OCGT', 'p_nom'] + OCGT_exp


    stors = n.storage_units
    stors_grouped = stors.groupby('carrier', as_index=False).sum(numeric_only =True)

    techs = pd.concat([gen_grouped.set_index('carrier')[['p_nom', 'p_nom_opt']].divide(1e3), stors_grouped.set_index('carrier')[['p_nom', 'p_nom_opt']].divide(1e3)])
    techs.loc['hydro'] = techs.loc['hydro'] + techs.loc['ror']
    techs.drop('ror', inplace=True)
    techs['color'] = techs.reset_index().carrier.apply(lambda c: color_dict[c]).values
    techs.sort_index(inplace=True)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 7)
    techs.p_nom_opt.plot.bar(ax=ax, color=techs.color, alpha=0.5)
    techs.p_nom.plot.bar(ax=ax, color=techs.color)

    ax.set_xlabel('Generation technology')
    ax.set_ylabel('Capacity in GW')

# def plot_h2_exports(n, sample_rate='W', aggregated=True):
#     links = n.links
#     links_t = n.links_t

#     exp_h2_links = n.links.loc[links.bus1 == 'H2 export bus']
#     exp_h2_links_ts = links_t.p0[exp_h2_links.index] * n.snapshot_weightings.iloc[0,0] / 1e3
#     exp_h2_links_ts = exp_h2_links_ts.resample(sample_rate).sum()

#     if aggregated:
#         exp_h2_links_ts = exp_h2_links_ts.sum(axis=1)

#     fig, ax = plt.subplots(1, 1)

#     exp_h2_links_ts.plot(ax=ax).legend(bbox_to_anchor=(1.1, 1.1))

#     ax.set_ylabel('Export quantity in GWh for chosen sample rate')
#     ax.set_xlabel('Snapshot')
#     if aggregated == False:
#         print('\n\nHighest H2 delivery from {}'.format(exp_h2_links_ts.sum().sort_values(ascending=False).idxmax()))



def plot_h2_production(n, sample_rate='W', aggregated=True):
    links = n.links
    links_t = n.links_t

    exp_h2_links = n.links.loc[links.bus1 == 'H2 export bus'].index.str.rstrip('export') + 'Electrolysis'
    exp_h2_links_prod_ts = links_t.p1[exp_h2_links] * -n.snapshot_weightings.iloc[0,0] / 1e3
    exp_h2_links_prod_ts = exp_h2_links_prod_ts.resample(sample_rate).sum()

    fig, ax = plt.subplots(1, 1)

    if aggregated:
        exp_h2_links_prod_ts = exp_h2_links_prod_ts.sum(axis=1)

    exp_h2_links_prod_ts.plot(ax=ax).legend(bbox_to_anchor=(1.1, 1.1))

    ax.set_ylabel('Production quantity in GWh for chosen sample rate')
    ax.set_xlabel('Snapshot')
    if aggregated == False:
        print('\n\nHighest H2 production quantity in {}'.format(exp_h2_links_prod_ts.sum().sort_values(ascending=False).idxmax()))

def calc_local_h2_share(n):
    links = n.links
    links_t = n.links_t

    exp_h2_links = n.links.loc[links.bus1 == 'H2 export bus']
    exp_h2_links_ts = links_t.p0[exp_h2_links.index] * n.snapshot_weightings.iloc[0,0] / 1e3
    exp_h2_links_exp = exp_h2_links_ts.sum().sum()

    exp_h2_links = n.links.loc[links.bus1 == 'H2 export bus'].index.str.rstrip('export') + 'Electrolysis'
    exp_h2_links_prod_ts = links_t.p1[exp_h2_links] * -n.snapshot_weightings.iloc[0,0] / 1e3
    exp_h2_links_prod = exp_h2_links_prod_ts.sum().sum()

    return  exp_h2_links_exp / exp_h2_links_prod

def plot_h2_mps(n, sample_rate='W', only_export=True, include_export_bus=False):
    links = n.links
    buses_t = n.buses_t

    if only_export:
        exp_h2_links = n.links.loc[links.bus1 == 'H2 export bus'].index.str.rstrip(' export')
        mp_ts = buses_t.marginal_price[exp_h2_links]
    else:
        mp_ts = buses_t.marginal_price.filter(regex='H2$')

    if include_export_bus:
            exp_bus = n.buses.filter(like='H2 export', axis=0).index
            mp_ts = pd.concat([mp_ts, buses_t.marginal_price[exp_bus]], axis=1)
            
    mp_ts = mp_ts.resample(sample_rate).mean()
    fig, ax = plt.subplots(1, 1)

    mp_ts.plot(ax=ax).legend(bbox_to_anchor=(1.1, 1.1))

    ax.set_ylabel('Average MP for H2')
    ax.set_xlabel('Snapshot')
    print('\n\nHighest mean hydrogen price in {}'.format(mp_ts.mean().sort_values(ascending=False).idxmax()))


def plot_elec_mps(n, sample_rate='W', only_export=True):
    links = n.links
    buses_t = n.buses_t

    if only_export:
        exp_h2_links = n.links.loc[links.bus1 == 'H2 export bus'].index.str.rstrip(' H2 export')
        mp_ts = buses_t.marginal_price[exp_h2_links]
    else:
        mp_ts = buses_t.marginal_price.filter(regex='_AC$')
    mp_ts = mp_ts.resample(sample_rate).mean()
    fig, ax = plt.subplots(1, 1)

    mp_ts.plot(ax=ax).legend(bbox_to_anchor=(1.1, 1.1))

    ax.set_ylabel('Average MP for electricity')
    ax.set_xlabel('Snapshot')
    print('\n\nHighest mean electricity price in {}'.format(mp_ts.mean().sort_values(ascending=False).idxmax()))


def get_fossil_emissions(n):
    '''''in t'''''   
    AC_index = n.buses[n.buses.carrier == 'AC'].index

    # vars_conv_gens = get_var(n, "Store", "e").loc[sns[-1], co2_atmosphere]
    conv_gens = list(n.carriers[n.carriers.co2_emissions > 0].index)
    conv_index = n.generators[n.generators.carrier.isin(conv_gens)].index    
    # vars_conv_gens = n.generators_t.p[conv_index]
    convs = n.generators[n.generators.carrier.isin(conv_gens)]
    conv_index = convs[convs['bus'].isin(AC_index)].index

    conv_gen = n.generators_t.p[conv_index].sum().sum() * n.snapshot_weightings.iloc[0, 0] / 1e6
    print('\nFossil generation amounts to {} TWh'.format(conv_gen))

    n.generators.loc[n.generators.carrier.isin(conv_gens), "emissions"] = 0
    n.generators.loc[conv_index, "emissions"] = n.generators.loc[
    conv_index, "carrier"
    ].apply(lambda x: n.carriers.loc[x].co2_emissions)
    n.generators.emissions = n.generators.emissions.fillna(0)

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(conv_index)),
        index=n.snapshots,
        columns=conv_index,
    )

    emission_factors = pd.DataFrame(
        np.outer(
            [1.0] * len(n.snapshot_weightings["generators"]),
            n.generators.loc[conv_index, "emissions"],
        ),
        index=n.snapshots,
        columns=conv_index,
    )

    return n.generators_t.p[conv_index], weightings, emission_factors, conv_index

def calc_curtailment(n):
    gen_ts = n.generators_t
    max_gen = gen_ts.p_max_pu * 3
    gen = gen_ts.p[max_gen.columns] * 3
    p_nom_max = n.generators.loc[max_gen.columns].p_nom_opt

    curtailment = ((p_nom_max * max_gen) - gen)
    agg_curtailment =  curtailment.sum().sum() / 1e6

    return agg_curtailment / (gen.sum().sum() / 1e6) * 100

def calc_batt_capa(n):
    '''in TWh'''
    return n.stores.filter(like='battery', axis=0).e_nom_opt.sum() / 1e6

def calc_batt_charge_capa(n):
    '''in GW'''
    return n.links.filter(regex='battery charge', axis=0).p_nom_opt.sum() / 1e3

def calc_batt_discharge_capa(n):
    '''in GW'''
    return n.links.filter(regex='battery discharge', axis=0).p_nom_opt.sum() / 1e3

def calc_h2_demand(n):
    return n.loads_t.p.filter(like='H2').multiply(n.snapshot_weightings.iloc[0,0]).divide(1e6).sum().sum()

def calc_ac_demand(n):
    return n.loads_t.p.filter(regex='_AC$').multiply(n.snapshot_weightings.iloc[0,0]).divide(1e6).sum().sum()

def calc_line_capa(n):
    return (n.lines.s_nom_opt * n.lines.length).sum()

def calc_elec_capa(n):
    return n.links.filter(like='Electrolysis', axis=0).p_nom_opt.sum() / 1e3

def calc_solar_capa(n):
    return n.generators.loc[n.generators.carrier == 'solar', 'p_nom_opt'].sum() / 1e3

def calc_onwind_capa(n):
    return n.generators.loc[n.generators.carrier.str.contains('onwind'), 'p_nom_opt'].sum() / 1e3

def calc_uhs_capa(n):
    '''in TWH'''
    return n.stores.filter(like='H2 Store', axis=0).e_nom_opt.sum() / 1e6

def calc_export_shares(n, absolute=True):
    ex_qs = n.links_t.p0.filter(like='H2 export').sum() * 3
    if absolute==False:
        ex_qs_rel = ex_qs / ex_qs.sum()
        return ex_qs_rel
    else:
        return ex_qs / 1e6

def calc_demand(n):
    loads = n.loads_t.p.filter(regex='_AC').sum().sum() *3 / 1e6
    return loads


def calc_wap_elec(n, agg=True):
    '''Takes a solved sector-coupled PyPSA network and returns electricity price per MWh for system or single nodes'''
    d_elec = n.loads_t['p'].filter(regex='AC$') #Amounts of electricity consumed

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(d_elec.T)),
        index=n.snapshots,
        columns=d_elec.columns,
    )
    d_elec = d_elec * weightings
    marginal_prices = n.buses_t.marginal_price.loc[:, d_elec.columns] #Marginal elec prices at nodes
    d_costs = (d_elec * marginal_prices).sum() #Product of demand and prices yields average total value of electricity consumed at nodes
    if agg:
        return d_costs.sum() / d_elec.sum().sum()
    else:
        return d_costs / d_elec.sum()


def calc_elec_mix(n):
    generation = calc_generation(n) 
    mix = generation.loc[generation.generation > 0]
    mix = mix.sort_values(by='generation', ascending=False)
    mix = mix/mix.sum()
    return mix.squeeze()


def calc_energy_mix(n):
    gens = n.generators_t.p / 1e6 * 3
    gens.columns = n.generators.loc[gens.columns, 'carrier'].str.strip()#.apply(rename_techs_tyndp)
    gens = gens.T.groupby(gens.columns).sum().T

    gens_hydro = n.storage_units_t.p_dispatch / 1e6 * 3
    gens_hydro.columns = n.storage_units.loc[gens_hydro.columns, 'carrier'].str.strip()#.apply(rename_techs_tyndp)
    gens_hydro = gens_hydro.T.groupby(gens_hydro.columns).sum().T

    gen_stor = n.stores_t.p.filter(regex='biomass|oil|biogas|gas Store') / 1e6 * 3
    gen_stor.columns = n.stores.loc[gen_stor.columns,'carrier'].str.strip()#.apply(rename_techs_tyndp)

    gen_stor = gen_stor.T.groupby(gen_stor.columns).sum().T#.drop('CCS', axis=1)

    gens = pd.concat([gens, gens_hydro, gen_stor.clip(lower=0)]).fillna(0)
    gens = gens.T.groupby(gens.T.index).sum().T
    gens = gens.groupby(gens.index).sum()
    gens = gens.sum()
    
    return gens.squeeze()

def calc_pipeline_cap(n):
    pipelines = n.links.filter(like='H2 pipeline', axis=0)
    pipeline_capa = pipelines.p_nom_opt * pipelines.length
    return pipeline_capa.sum()

def calc_emissions(n):
    emissions = n.stores_t.p.filter(like='atmosphere').squeeze().sum() * n.snapshot_weightings['generators'].iloc[0]
    return emissions

def create_summary_df(run_name):
    summary = pd.DataFrame()
    path = os.getcwd()+'/pypsa-earth-sec/results/{}/postnetworks'.format(run_name)
    exp_ports_all = []

    for f in os.listdir(path):
        n_path = path+'/{}'.format(f)
        q = int(f.split('_')[-1].split('e')[0])
        scen = f.split('_')[-2]
        year = f.split('_')[-4]

        n = pypsa.Network(n_path)

        #system costs
        system_costs = n.objective
        system_costs_corr = system_costs + n.objective_constant
        if q == 0:
            system_costs_no_exports= system_costs
            system_costs_no_exports_corr = system_costs_corr

        #average costs of exported hydrogen
        hydrogen_costs = (system_costs - system_costs_no_exports) / (q*1e9) * 33.3
        hydrogen_costs_corr = (system_costs_corr - system_costs_no_exports_corr) / (q*1e9) * 33.3

        #local markt prices
        elec_wap = calc_wap_elec(n)
        h2_wap = calc_wap_h2(n)
        
        #energy demand
        demand = calc_demand(n) + q
        ac_demand = calc_ac_demand(n)
        h2_demand = calc_h2_demand(n)

        #curtailment
        curtailment = calc_curtailment(n)
        
        exp_ports = n.links.filter(like='export', axis=0).index
        #export port shares and export h2 price according to MP
        if q == 0:
            hydrogen_wap_exp = 0
            exp_shares = pd.Series(np.zeros(len(exp_ports)), index=exp_ports)
            exp_ports_all = exp_ports
        else:
            hydrogen_wap_exp = calc_wap_h2_exp(n)
            exp_shares = pd.Series(np.zeros(len(exp_ports_all)), index=exp_ports_all)
            exp_shares_val = calc_export_shares(n)
            exp_shares.loc[exp_shares_val.index] = exp_shares_val.values
        exp_dict = {}

        for exp_p in exp_ports_all:
            exp_dict[exp_p+' share'] = exp_shares[exp_p]
            node = exp_p.split(' ')[0]
            exp_dict[exp_p+ ' solar cap'] = n.generators.loc[(n.generators.bus == node) & (n.generators.carrier == 'solar'), 'p_nom_opt'].item()
            exp_dict[exp_p+' onwind cap'] = n.generators.loc[(n.generators.bus == node) & (n.generators.carrier == 'onwind'), 'p_nom_opt'].item()
            exp_dict[exp_p+' onwind2 cap'] = n.generators.loc[(n.generators.bus == node) & (n.generators.carrier == 'onwind2'), 'p_nom_opt'].item()
            exp_dict[exp_p+' offwind cap'] = n.generators.loc[(n.generators.bus == node) & (n.generators.carrier == 'offwind'), 'p_nom_opt'].item()
            exp_dict[exp_p+' offwind2 cap'] = n.generators.loc[(n.generators.bus == node) & (n.generators.carrier == 'offwind2'), 'p_nom_opt'].item()
            exp_dict[exp_p+' csp cap'] = n.generators.loc[(n.generators.bus == node) & (n.generators.carrier == 'csp'), 'p_nom_opt'].item()
            exp_dict[exp_p+' rooftop-solar cap'] = n.generators.loc[(n.generators.bus == node) & (n.generators.carrier == 'rooftop-solar'), 'p_nom_opt'].item()

            if exp_p in exp_ports:
                exp_dict[exp_p+' electrolyzer cap'] = n.links.filter(like=node, axis=0).loc[n.links.carrier == 'H2 Electrolysis', 'p_nom_opt'].item()
                exp_dict[exp_p+' electrolyzer cf'] = n.links_t.p0.filter(like=node).filter(like='Electrolysis').sum().sum()*3 / (exp_dict[exp_p+' electrolyzer cap'] * 8760)

        #renewable capacities
        solar_cap = n.generators.loc[n.generators.carrier == 'solar', 'p_nom_opt'].sum()
        onwind_cap = n.generators.loc[n.generators.carrier == 'onwind', 'p_nom_opt'].sum()
        onwind2_cap = n.generators.loc[n.generators.carrier == 'onwind2', 'p_nom_opt'].sum()
        offwind_cap = n.generators.loc[n.generators.carrier == 'offwind', 'p_nom_opt'].sum()
        offwind2_cap = n.generators.loc[n.generators.carrier == 'offwind2', 'p_nom_opt'].sum()
        roof_solar_cap = n.generators.loc[n.generators.carrier == 'rooftop-solar', 'p_nom_opt'].sum()
        csp_cap = n.generators.loc[n.generators.carrier == 'csp', 'p_nom_opt'].sum()

        #electrolyzer capacities and capacity factor
        elec_cap = n.links.filter(like='Electrolysis', axis=0).p_nom_opt.sum()
        elec_cf = n.links_t.p0.filter(like='Electrolysis').sum().sum()*3 / (elec_cap * 8760)

        #pipeline capa
        pipeline_cap = calc_pipeline_cap(n)

        #storage capacities
        battery_cap = calc_batt_capa(n)
        uhs_cap = calc_uhs_capa(n)

        #electricity and energy mix
        elec_mix = calc_elec_mix(n).sort_index().to_string() #relative
        ener_mix = calc_energy_mix(n).sort_index().to_string() #TWh

        #emissions
        emissions = calc_emissions(n) /1e6 #Mt
        emissions_mp = n.buses_t.marginal_price.filter(like='atmosphere').squeeze().mean()
        
        #Create row for summary df
        summary_general = pd.DataFrame(data={
                    #general
                    'network':[n_path],
                    'year':[int(year)],
                    'scenario':[scen],
                    'export_quantity':[int(q)],

                    #hydrogen export related
                    'system_costs':[system_costs],
                    'system_costs_add_ex':[system_costs_corr],
                    'exp_h2_cost_norm':[hydrogen_costs],
                    'exp_h2_cost_norm_add_ex':[hydrogen_costs_corr],
                    'exp_h2_cost_mp':[hydrogen_wap_exp],
                    'electrolyzer_cap':elec_cap,
                    'electrolyzer_cf':elec_cf,
                    'uhs_cap':uhs_cap,
                    'pipeline_cap':pipeline_cap,

                    #RES related
                    'solar_cap': solar_cap,
                    'onwind_cap':onwind_cap,
                    'onwind2_cap':onwind2_cap,
                    'offwind_cap':offwind_cap,
                    'offwind2_cap':offwind2_cap,
                    'roof_solar_cap':roof_solar_cap,
                    'csp_cap':csp_cap,
                    'battery_cap':battery_cap,
                    'electricity_mix_rel':elec_mix,
                    'energy_mix_abs':ener_mix,
                    'curtailment':curtailment,

                    #local markets related
                    'elec_wap':elec_wap,
                    'h2_wap':h2_wap,
                    'demand':demand,
                    'ac_demand':[ac_demand],
                    'h2_demand':[h2_demand],
                    'costs-demand-ratio':[system_costs / demand / 1e6],

                    #emission related
                    'emissions':emissions,
                    'emissions_mp':emissions_mp,
                    })
        summary_exp = pd.DataFrame(data={k:v for (k,v) in exp_dict.items()}, index=summary_general.index)
        summary_n = pd.concat([summary_general, summary_exp], axis=1)
        summary = pd.concat([summary, summary_n]).fillna(0)
        sort_dict = {'BS':0, 'AP':1, 'NZ':2}
        summary = summary.sort_values(['year', 'scenario', 'export_quantity']).sort_values(by='scenario', key= lambda k: k.map(sort_dict), kind='mergesort')
    try:
        os.mkdir(os.getcwd()+'/pypsa-earth-sec/outputs/{}'.format(run_name))
        os.mkdir(os.getcwd()+'/pypsa-earth-sec/outputs/{}/tables'.format(run_name))
        summary.to_csv(os.getcwd()+'/pypsa-earth-sec/outputs/{}/tables/summary.csv'.format(run_name))
        os.mkdir(os.getcwd()+'/pypsa-earth-sec/outputs/{}/plots'.format(run_name))
    except:
        summary.to_csv(os.getcwd()+'/pypsa-earth-sec/outputs/{}/tables/summary.csv'.format(run_name))
    return summary

def get_summary_df(run_name, update_table=True):
    try:
        df = pd.read_csv(os.getcwd()+'/pypsa-earth-sec/outputs/{}/tables/summary.csv'.format(run_name))
        if update_table == True:
            print('Updating existing summary dataframe for given run {}'.format(run_name))
            df = create_summary_df(run_name)
    except:
        print('Creating summary dataframe for given run {}'.format(run_name))
        df = create_summary_df(run_name)
    return df

# def get_networks(run_name):
#     networks = {}
#     path = os.getcwd()+'/pypsa-earth-sec/results/{}/postnetworks'.format(run_name)
#     for f in os.listdir(path):
#         n_path = path+'/{}'.format(f)
#         q = int(f.split('_')[-1].split('e')[0])
#         scen = f.split('_')[-2]
#         year = int(f.split('_')[-4])
#         n = pypsa.Network(n_path)

#         networks[year] = {}
#         networks[year][scen] = {}
#         networks[year][scen][q] = n
#     return networks

def get_networks(summary):
    summary = summary.reset_index()
    networks = {}
    for y in summary.year.unique():
        networks[y] = {}
        for s in summary.scenario.unique():
            networks[y][s] = {}
            summary_y_s = summary.loc[(summary.year == y) & (summary.scenario == s)]
            for q in summary_y_s.export_quantity.unique():
                n_path = summary_y_s.loc[summary_y_s.export_quantity == q, 'network'].item()
                n = pypsa.Network(n_path)
                networks[y][s][q] = n
    return networks


def get_network(df, year=2030, scenario='AP', quantity=0):
    df = df.reset_index()
    path = df.loc[(df.year == year) & (df.scenario == scenario) & (df.export_quantity == quantity), 'network'].item()
    n = pypsa.Network(path)
    return n


def plot_average_h2_costs(summary):
    summary = summary.reset_index()
    for y in summary.year.unique():
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)    
        costs_y = summary.loc[summary.year == y]
        color_dict = {'BS': 'red', 'AP':'blue', 'NZ':'green'}
        scen_dict = {'BS': 'Conservative', 'AP':'Realistic', 'NZ':'Optimistic'}
        for s in summary.scenario.unique():
            costs_y_s = costs_y.loc[costs_y.scenario == s]
            if y == 2030:
                h2_costs = costs_y_s.exp_h2_cost_norm
            else:
                h2_costs = costs_y_s.exp_h2_cost_norm_add_ex
            ax.plot(costs_y_s.export_quantity, h2_costs, label=scen_dict[s], color=color_dict[s], linestyle='dashed', alpha=0.8, linewidth=0.8)
            ax.scatter(costs_y_s.export_quantity.iloc[1:], h2_costs.iloc[1:], color=color_dict[s], s=30)
        
        ax.set_xlabel('Export quantity steps in TWh', fontsize=14)
        ax.set_ylabel('Average normalized costs of\nexported hydrogen in €/kg', fontsize=14)
        ax.set_xticks(costs_y_s.export_quantity.iloc[1:])

        tick_labels = costs_y_s.export_quantity.iloc[1:].astype(str)
        tick_labels.iloc[1] = '\n' + tick_labels.iloc[1]
        ax.set_xticklabels(tick_labels)
        ax.set_xlim((0, costs_y.export_quantity.max()))
        
        ax.legend(loc='lower right')
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        run_name = summary.network.iloc[0].split('/')[-3]
        plt.savefig(os.getcwd()+'/pypsa-earth-sec/outputs/{}/plots/export_average_h2_costs_{}.png'.format(run_name, str(y)))

def plot_h2_mp_exp(summary):
    summary = summary.reset_index()
    for y in summary.year.unique():
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)    
        costs_y = summary.loc[summary.year == y]
        color_dict = {'BS': 'red', 'AP':'blue', 'NZ':'green'}
        scen_dict = {'BS': 'Conservative', 'AP':'Realistic', 'NZ':'Optimistic'}
        for s in summary.scenario.unique():
            costs_y_s = costs_y.loc[costs_y.scenario == s]
            h2_costs = costs_y_s.exp_h2_cost_mp
            
            ax.plot(costs_y_s.export_quantity, h2_costs, label=scen_dict[s], color=color_dict[s], linestyle='dashed', alpha=0.8, linewidth=0.8)
            ax.scatter(costs_y_s.export_quantity.iloc[1:], h2_costs.iloc[1:], color=color_dict[s], s=30)
        
        ax.set_xlabel('Export quantity steps in TWh', fontsize=14)
        ax.set_ylabel('Weighted average price of\nexported hydrogen in €/kg', fontsize=14)
        ax.set_xticks(costs_y_s.export_quantity.iloc[1:])

        tick_labels = costs_y_s.export_quantity.iloc[1:].astype(str)
        tick_labels.iloc[1] = '\n' + tick_labels.iloc[1]
        ax.set_xticklabels(tick_labels)
        ax.set_xlim((0, costs_y.export_quantity.max()))
        
        ax.legend(loc='lower right')
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        run_name = summary.network.iloc[0].split('/')[-3]
        plt.savefig(os.getcwd()+'/pypsa-earth-sec/outputs/{}/plots/export_h2_mp_{}.png'.format(run_name, str(y)))

def plot_electrolyzer_caps(elec_cap):
    df = elec_cap.reset_index()
    for y in df.year.unique():
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)

        df_y = df.loc[df.year == y].drop('year', axis=1)
        df_y['electrolyzer_cap'] = df_y.filter(like='electrolyzer cap').sum(axis=1)
        df_y.rename({'electrolyzer_cap':'Electrolyzer capacity in other nodes'}, inplace=True, axis=1)
        df_y.set_index(['scenario', 'export_quantity']).plot.bar(stacked=True, ax=ax, cmap='spring_r')

        ax.set_ylabel('Electrolyzer capacity in MW')
        ax.legend(bbox_to_anchor=(1,1))

def plot_electrolyzer_cf(summary):
    df = summary.filter(like='cf')
    df.reset_index(inplace=True)
    for y in df.year.unique():
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)

        df_y = df.loc[df.year == y].drop('year', axis=1)
        
        df_y.rename({'electrolyzer_cf':'Overall electrolyzer capacity factor'}, inplace=True, axis=1)
        df_y.set_index(['scenario', 'export_quantity']).plot.bar(ax=ax, cmap='spring_r')

        ax.set_ylabel('Electrolyzer capacity factor')
        ax.legend(bbox_to_anchor=(1,1))


def plot_h2_exports(ns, summary, sample_rate='W', scenario='AP'):
    summary = summary.reset_index()
    summary = summary.loc[summary.export_quantity > 0]
    ex_quantities={2030: list(summary.loc[summary.year ==2030, 'export_quantity'].unique()),
                    2050: list(summary.loc[summary.year ==2050, 'export_quantity'].unique())} # , 2050: [10, 100, 500, 1000, 3000]
    scen_dict={'BS':'Conservative', 'AP':'Realistic', 'NZ':'Optimistic'}

    fig, ax = plt.subplots(5, len(summary.year.unique()), figsize=(12, 10))

    keys_list = list(ex_quantities.keys())

    exp_h2_dict = {}

    for year, quantities in ex_quantities.items():
        position_x =  keys_list.index(year)
        for (idx,q) in enumerate(quantities):
            position_y = quantities.index(q)
            n = ns[year][scenario][q]

            links = n.links
            links_t = n.links_t

            exp_h2_links = n.links.loc[links.bus1 == 'H2 export bus']
            exp_h2_links_ts = links_t.p0[exp_h2_links.index] * n.snapshot_weightings.iloc[0,0] / 1e3
            exp_h2_links_ts = exp_h2_links_ts.resample(sample_rate).sum()      # .sum() .mean()

            exp_h2_dict['{} {} {} TWh'.format(year, scen_dict[scenario], q)] = exp_h2_links_ts

            #exp_h2_links_ts.rename(columns=ports, inplace=True)

            if len(summary.year.unique()) > 1:
                exp_h2_links_ts.plot(ax=ax[position_y, position_x])

                # Set the title
                ax[position_y, position_x].set_title('{} {} {} TWh'.format(year, scen_dict[scenario], q))
                ax[position_y, position_x].legend(bbox_to_anchor=(1, 2))
                if idx > 0:
                    ax[position_y, position_x].legend().set_visible(False)  # Hide the legend for the first subplot
                ax[position_y, position_x].set_xlabel('')  # Hide the x-label
            else:
                exp_h2_links_ts.plot(ax=ax[position_y])

                # Set the title
                ax[position_y].set_title('{} {} {} TWh'.format(year, scen_dict[scenario], q))
                # ax[position_y].legend(bbox_to_anchor=(1, 2))
                if idx > 0:
                    ax[position_y].legend().set_visible(False)  # Hide the legend for the first subplot
                ax[position_y].set_xlabel('')  # Hide the x-label 

    if len(summary.year.unique()) > 1:
        handles, labels = ax[position_y, position_x].get_legend_handles_labels()
    else:
        handles, labels = ax[position_y].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.95), ncol=4)

    # Modify the legend labels
    # new_labels = ['Pecem (BR.6)', 'Aratu (BR.5)', 'Itaguai (BR.19)', 'Rio Grande (BR.21)']
    # new_labels = exp_h2_links_ts.columns.to_list()
    # fig.legend(new_labels, bbox_to_anchor=(0.8, 0.95), ncol=4)


    # Set the y-label for the whole figure
    fig.text(0.05, 0.5, 'Hydrogen delivery to export locations with weekly resampling [GWh]', va='center', rotation='vertical', fontdict={'fontsize': 14})


    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3)  # Adjust the width spacing between subplots
    plt.subplots_adjust(hspace=0.6)  # Adjust the height spacing between subplots

    # plt.savefig('../outputs/Hydrogen_delivery_{}.png'.format(scen_dict[s]), bbox_inches='tight')
    #plt.savefig('../outputs_{}/Hydrogen_delivery_{}.png'.format(run[2050], scen_dict[s]), bbox_inches='tight')
    return(None)


def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale) ** 0.5, **kw) for s in sizes]
def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0] * (
            72.0 / fig.dpi
        )

    ellipses = []
    if not dont_resize_actively:

        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2.0 * radius * dist

        fig.canvas.mpl_connect("resize_event", update_width_height)
        ax.callbacks.connect("xlim_changed", update_width_height)
        ax.callbacks.connect("ylim_changed", update_width_height)

    def legend_circle_handler(
        legend, orig_handle, xdescent, ydescent, width, height, fontsize
    ):
        w, h = 2.0 * orig_handle.get_radius() * axes2pt()
        e = Ellipse(
            xy=(0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
            width=w,
            height=w,
        )
        ellipses.append((e, orig_handle.get_radius()))
        return e

    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}

def plot_h2_infra(network):

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Mercator()})

    fig.set_size_inches(10.5, 9)
    ex_qs = [network.links_t.p0.filter(like='export').sum().sum()/1e6*network.snapshot_weightings.iloc[0,0]]

    link_colors = ['blueviolet', 'mediumspringgreen', 'yellow',  'blue', 'cyan', 'red']

    for i, q in enumerate(ex_qs):

    # assign_location(n)
        n = network#s[2050]['AP'][q]
        if q > 200:
            bus_size_factor = 1e10#1e10
            linewidth_factor = 1e4
            elec_leg = 50
            pip_leg = 20
        else:
            bus_size_factor = 1e9#1e10
            linewidth_factor = 1e2
            elec_leg = 5
            pip_leg = 1
        # MW below which not drawn
        line_lower_threshold = 1e2
        bus_color = 'm'
        link_color = 'c'

        n.links.loc[:, "p_nom_opt"] = n.links.loc[:, "p_nom_opt"]
        # n.links.loc[n.links.carrier == "H2 Electrolysis"].p_nom_opt

        # Drop non-electric buses so they don't clutter the plot
        n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

        elec = n.links.index[n.links.carrier == "H2 Electrolysis"]

        bus_sizes = (
            n.links.loc[elec, "p_nom_opt"].groupby(n.links.loc[elec, "bus0"]).sum()
            / bus_size_factor
        )

        # make a fake MultiIndex so that area is correct for legend
        bus_sizes.index = pd.MultiIndex.from_product([bus_sizes.index, ["electrolysis"]])

        n.links = n.links.filter(like='H2 pipeline', axis=0)

        #n.links.drop(n.links.index[n.links.carrier != "H2 pipeline"], inplace=True)

        link_widths = n.links.p_nom_opt / linewidth_factor
        link_widths[n.links.p_nom_opt < line_lower_threshold] = 0.0

        n.links.bus0 = n.links.bus0.str.replace(" H2", "")
        n.links.bus1 = n.links.bus1.str.replace(" H2", "")

        print(link_widths.sort_values())

        print(n.links[["bus0", "bus1"]])

        n_buses = n.buses
        n_buses = n_buses.loc[n_buses.carrier == 'AC']
        pos = n_buses[['x', 'y']].describe()
        span_x = pos.loc['max', 'x'] - pos.loc['min', 'x']
        span_y = pos.loc['max', 'y'] - pos.loc['min', 'y']
        
        #link_color = [float(q)]*len(n.links)
        n.plot(
            bus_sizes=bus_sizes*1e5,
            bus_colors={"electrolysis": bus_color},
            link_colors=link_color,
            link_widths=link_widths,
            branch_components=["Link"],
            color_geomap={"ocean": "lightblue", "land": "gainsboro"},
            ax=ax,
            boundaries=(pos.loc['min', 'x'] - span_x*0.15, pos.loc['max', 'x'] + span_x*0.15, pos.loc['min', 'y'] - span_y*0.15, pos.loc['max', 'y'] + span_y*0.15),
            #boundaries=(-75, -33, -35, 6), # Brazil
            #boundaries=(11, 26, -29, -15), # Namibia
            #boundaries=(21, 41, 42, 55), # Ukraine
            link_cmap='Dark2',
            #color_geomap='no_export'
        )


        handles = make_legend_circles_for(
            [elec_leg*1e9, 0.2*elec_leg*1e9], scale=bus_size_factor/1e9, facecolor=bus_color
        )
        labels = ["{} GW".format(s) for s in (elec_leg, int(0.2*elec_leg))]
        l2 = ax.legend(
            handles,
            labels,
            loc="lower right",
            #bbox_to_anchor=(0.01, 1.01),
            labelspacing=0.8,
            framealpha=1.0,
            title="Electrolyzer capacity",
            handler_map=make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False),
        )
        ax.add_artist(l2)

        handles2 = []
        labels = []

        for s in (pip_leg, int(0.1*pip_leg)):
            handles2.append(
                plt.Line2D([0], [0], color=link_color, linewidth=s * 1e3 / linewidth_factor)#*1e3
            )
            labels.append("{} GW".format(s))
        l1_1 = ax.legend(
            handles2,
            labels,
            loc="upper right",
            #bbox_to_anchor=(0.32, 1.01),
            framealpha=1,
            labelspacing=0.8,
            handletextpad=1.5,
            title="H2 pipeline capacity",
        )
        ax.add_artist(l1_1)

def plot_res_caps(df):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    df = df.filter(regex='cap$').drop(df.filter(regex='_AC').columns, axis=1).drop('battery_cap', axis=1)
    df = df.rename(lambda c: c.split('_')[0], axis=1)
    colors={
            'solar':'gold',
            'onwind':'blue',
            'onwind2':'royalblue',
            'offwind':'lightblue',
            'offwind2':'dodgerblue',
            'roof':'orange',
            'csp':'coral'
        }
    df.plot.bar(stacked=True, ax=ax, color=df.columns.map(colors))
    ax.set_ylabel('Installed capacity in MW')

def plot_res_caps_exp_nodes(df, technology='solar'):
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(20, 6)
    for (i_s, s) in enumerate(df.index.get_level_values(1).unique()):
        df_s = df.reset_index()
        df_s = df_s.loc[df_s.scenario == s].set_index(['scenario', 'export_quantity'])
        df_s_exp = df_s.filter(regex='cap$').filter(regex='_AC')
        #.filter(regex=technology)
        df_s_exp = df_s_exp.rename(lambda c: c.split(' ')[3], axis=1)
        df_s_exp = df_s_exp.T.groupby(df_s_exp.T.index).sum().T
        colors={
            'solar':'gold',
            'onwind':'blue',
            'onwind2':'royalblue',
            'offwind':'lightblue',
            'offwind2':'dodgerblue',
            'rooftop-solar':'orange',
            'csp':'coral'
        }
        if i_s > 0:
            df_s_exp.plot.bar(stacked=True, ax=ax[i_s], color=df_s_exp.columns.map(colors))
        else:
            df_s_exp.plot.bar(stacked=True, ax=ax[i_s], color=df_s_exp.columns.map(colors))
        ax[i_s].set_ylabel('Installed capacity in MW')

def plot_uhs_caps(ns):
    df = pd.DataFrame()
    for y in ns.keys():
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        ns_y = ns[y]
        for (j, s) in enumerate(ns_y.keys()):
            ns_y_s = ns_y[s]
            for (i, q) in enumerate(ns_y_s.keys()):
                n = ns_y_s[q]
                n_uhs = n.stores.filter(like='H2 Store', axis=0)
                n_uhs = n_uhs[n_uhs.e_nom_opt > 0].e_nom_opt.to_frame().T
                n_uhs.index = [i]

                df_n = pd.DataFrame(data={'scenario':s, 'export_quantity':q}, index=[i])
                df_n = pd.concat([df_n, n_uhs], axis=1)
                df = pd.concat([df, df_n])
            if j > 0:
                df.set_index(['scenario', 'export_quantity']).plot.bar(stacked=True, ax=ax, cmap='spring_r', legend=False)
            else:
                df.set_index(['scenario', 'export_quantity']).plot.bar(stacked=True, ax=ax, cmap='spring_r', legend=True)
            ax.set_ylabel('Hydrogen storage capacity in MWh')
            
def plot_elec_mix(df):
    df = df.reset_index()
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(20, 6)

    #demand = df.loc[df.scenario == s].set_index(['year', 'scenario', 'export_quantity']).ac_demand
    for (idx_s,s) in enumerate(df.scenario.unique()):
        df_s = df.loc[df.scenario == s].set_index(['year', 'scenario', 'export_quantity'])[['electricity_mix_rel']]
        df_s_tech = df_s.electricity_mix_rel.apply(lambda m: pd.Series(m.split('\n')[1:]).apply(lambda s: s.split(' ')[0]))
        df_s_shares = df_s.electricity_mix_rel.apply(lambda m: pd.Series(m.split('\n')[1:]).apply(lambda s: float(s.split(' ')[-1])*100))
        # mix_dict = {}
        # for q in df_s_shares.iterrows():
            
        # elec_mix = pd.DataFrame(index=df_s_shares.index, columns=df_s_tech.iloc[0].values)
        df_s_shares.columns = df_s_tech.iloc[0].values
        df_s_shares = df_s_shares[df_s_shares > 0.5].dropna(axis=1)
        df_s_shares = df_s_shares.T.sort_values(by=(df_s_shares.index.get_level_values(0)[0], s, 0), ascending=False).T

        # #df_s_key = pd.Series(summary_res.electricity_mix_rel.iloc[0].split('\n')).apply(lambda s: s.split(' ')[0])
        # df_s_value = pd.Series(summary_res.electricity_mix_rel.iloc[0].split('\n')).apply(lambda s: s.split(' ')[-1])
        # df_s_value = df_s_value.iloc[1:].astype(float)
        # df_s_value.index = df_s_key.iloc[1:].values
        #mix_s = pd.DataFrame(index=df_s.index, data=df_s_value.to_dict())
        #mix_s = mix_s * demand
        colors={
            'solar':'gold',
            'onwind':'steelblue',
            'onwind2':'royalblue',
            'offwind':'lightblue',
            'offwind2':'cyan',
            'rooftop-solar':'orange',
            'csp':'coral',
            'biomass':'green',
            'hydro':'midnightblue',
            'ror':'slateblue',
            'nuclear':'greenyellow',
            'coal':'brown',
            'OCGT':'red',
            'CCGT':'darkred',
            'oil':'grey',
            'gas_CHP':'crimson',
            'biomass_CHP':'lawngreen',
        }
        
        df_s_shares.plot.bar(stacked=True, ax=ax[idx_s], color=df_s_shares.columns.map(colors))
    
        ax[idx_s].set_ylabel('Electricity share [%]')
        
        h, l = ax[idx_s].get_legend_handles_labels()
        #ax[idx_s].legend(bbox_to_anchor=(0.6,1.3), ncol=3, handles=h[:int(len(h)/3)], labels=l[:int(len(l)/3)])
        ax[idx_s].legend(bbox_to_anchor=(1,1.3), ncol=3, handles=h, labels=l)

def plot_energy_mix(df):
    df = df.reset_index()
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(20, 6)

    #demand = df.loc[df.scenario == s].set_index(['year', 'scenario', 'export_quantity']).ac_demand
    for (idx_s,s) in enumerate(df.scenario.unique()):
        df_s = df.loc[df.scenario == s].set_index(['year', 'scenario', 'export_quantity'])[['energy_mix_abs']]
        df_s_tech = df_s.energy_mix_abs.apply(lambda m: pd.Series(m.split('\n')[1:]).apply(lambda s: s.split('   ')[0]))
        df_s_shares = df_s.energy_mix_abs.apply(lambda m: pd.Series(m.split('\n')[1:]).apply(lambda s: float(s.split('   ')[-1])))
        df_s_shares.columns = df_s_tech.iloc[0].values
        df_s_shares = df_s_shares[df_s_shares > 0.5].dropna(axis=1)
        df_s_shares = df_s_shares.T.sort_values(by=(df_s_shares.index.get_level_values(0)[0], s, 0), ascending=False).T

        # #df_s_key = pd.Series(summary_res.electricity_mix_rel.iloc[0].split('\n')).apply(lambda s: s.split(' ')[0])
        # df_s_value = pd.Series(summary_res.electricity_mix_rel.iloc[0].split('\n')).apply(lambda s: s.split(' ')[-1])
        # df_s_value = df_s_value.iloc[1:].astype(float)
        # df_s_value.index = df_s_key.iloc[1:].values
        #mix_s = pd.DataFrame(index=df_s.index, data=df_s_value.to_dict())
        #mix_s = mix_s * demand
        colors={
            'solar':'gold',
            'onwind':'steelblue',
            'onwind2':'royalblue',
            'offwind':'lightblue',
            'offwind2':'cyan',
            'rooftop-solar':'orange',
            'csp':'coral',
            'solar thermal':'lightcoral',
            'biomass':'green',
            'solid biomass':'green',
            'hydro':'midnightblue',
            'ror':'slateblue',
            'nuclear':'greenyellow',
            'coal':'brown',
            'OCGT':'red',
            'CCGT':'darkred',
            'oil':'grey',
            'biogas':'lawngreen',
            'gas':'crimson'
        }
        
        df_s_shares.plot.bar(stacked=True, ax=ax[idx_s], color=df_s_shares.columns.map(colors))

        ax[idx_s].set_ylabel('Energy share [MWh]')
        
        h, l = ax[idx_s].get_legend_handles_labels()
        #ax[idx_s].legend(bbox_to_anchor=(0.6,1.3), ncol=3, handles=h[:int(len(h)/3)], labels=l[:int(len(l)/3)])
        ax[idx_s].legend(bbox_to_anchor=(1,1.3), ncol=3, handles=h, labels=l)

def plot_mps(df):
    df = df.reset_index()

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    fig, axs = plt.subplots(len(df.year.unique()), 1)
    fig.set_size_inches(12, 4*len(df.year.unique()))

    color_dict = {'elec': '#005b7f', 'h2':'#b2d235', 'NZ':'#b2d235'}
    marker_dict = {'BS': '^', 'AP':'s', 'NZ':'o'}
    scen_dict = {'BS': 'Conservative', 'AP':'Realistic', 'NZ':'Optimistic'}

    for i, y in enumerate(df.year.unique()):
        
        if len(df.year.unique()) > 1:
            ax = axs[i]
        else:
            ax = axs
        df_y = df.loc[df.year == y]
        if (y == 2050):
            df_y = df_y.loc[df_y.export_quantity <= 3000]

        
        min_elec = df_y[['export_quantity', 'elec_wap']].groupby('export_quantity', as_index=False).min(numeric_only=True)
        max_elec = df_y[['export_quantity', 'elec_wap']].groupby('export_quantity', as_index=False).max(numeric_only=True)

        min_h2 = df_y[['export_quantity', 'h2_wap']].groupby('export_quantity').min(numeric_only=True)/33.3*1e3
        max_h2 = df_y[['export_quantity', 'h2_wap']].groupby('export_quantity').max(numeric_only=True)/33.3*1e3

        plots1 = []
        plots2 = []
        for s in df_y.scenario.unique():
            df_y_s = df_y.loc[df_y.scenario == s]
            df_y_s = df_y_s.set_index('export_quantity')
            ax.scatter(df_y_s.index, df_y_s.elec_wap, label=scen_dict[s], color=color_dict['elec'], marker=marker_dict[s])#, linestyle='dashed', linewidth=0.8)
            ax.scatter(df_y_s.index, df_y_s.h2_wap/33.3*1e3, label=scen_dict[s], color=color_dict['h2'], marker=marker_dict[s])#, linestyle='dashed', linewidth=0.8)


            ax.fill_between(min_elec.export_quantity, min_elec.elec_wap, max_elec.elec_wap, color=color_dict['elec'], alpha=.1)
            ax.fill_between(min_h2.index, min_h2.h2_wap, max_h2.h2_wap, color=color_dict['h2'], alpha=.1)
            ax.set_ylim((0, max(max_h2.h2_wap.max(), max_elec.elec_wap.max())))
            ax.tick_params(axis='both', which='major', labelsize=20)
            if (y == 2030):
                ax.set_xticklabels([])
                ax.legend(loc='upper left', fontsize=10, ncol=3)
                h, l = ax.get_legend_handles_labels()
                leg1 = ax.legend(handles=[i for i in h[0::2]], labels=[i for i in l[0::2]], loc='upper right', fontsize=18, title=r'\huge\textbf{Electricity}', bbox_to_anchor=(0.3, 1.6))
                ax.legend(handles=[i for i in h[1::2]], labels=[i for i in l[1::2]], loc='upper right', fontsize=18, title=r'\huge\textbf{Hydrogen}', bbox_to_anchor=(1, 1.6))
                ax.add_artist(leg1)
            else:
                ax.set_xlabel(r'\textbf{Export quantity [TWh]}', fontsize=20)

            ax.set_title(y, fontdict={'size':20})
            fig.text(0.05, 0.5, r'\textbf{Average prices [€/MWh]}', va='center', rotation='vertical', fontsize=20)