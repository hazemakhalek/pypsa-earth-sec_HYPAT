# -*- coding: utf-8 -*-
"""
Proposed code structure:
X read network (.nc-file)
X add export bus
X connect hydrogen buses (advanced: only ports, not all) to export bus
X add store and connect to export bus
X (add load and connect to export bus) only required if the "store" option fails

Possible improvements:
- Select port buses automatically (with both voronoi and gadm clustering). Use data/ports.csv?
"""


import logging
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pypsa
from helpers import locate_bus, override_component_attrs, three_2_two_digits_country, sets_path_to_root

logger = logging.getLogger(__name__)


def create_nested_dict_from_csv(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    df['two_digits_country'] = df.Code.apply(lambda i: three_2_two_digits_country(i))

    # Initialize an empty dictionary
    nested_dict = {}

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Get the key from the specified key_column
        key = row['two_digits_country']
        
        # Create a dictionary from the two sets of columns
        values_dict = {row['Q0']: row['P0'], 
                        row['Q1']: row['P1'],
                        row['Q2']: row['P2'],
                        row['Q3']: row['P3'],
                        row['Q4']: row['P4'],
                        row['Q5']: row['P5']}
        
        # Add the values dictionary as the value for the current key in the nested dictionary
        nested_dict[key] = values_dict

    return nested_dict


def select_ports(n):
    """This function selects the buses where ports are located"""

    ports = pd.read_csv(
        snakemake.input.export_ports,
        index_col=None,
        keep_default_na=False,
    ).squeeze()
    ports = ports[ports.country.isin(countries)]

    gadm_level = snakemake.config["sector"]["gadm_level"]

    ports["gadm_{}".format(gadm_level)] = ports[["x", "y", "country"]].apply(
        lambda port: locate_bus(
            port[["x", "y"]],
            port["country"],
            gadm_level,
            snakemake.input["shapes_path"],
            snakemake.config["clustering_options"]["alternative_clustering"],
        ),
        axis=1,
    )

    ports = ports.set_index("gadm_{}".format(gadm_level))

    # Select the hydrogen buses based on nodes with ports
    hydrogen_buses_ports = n.buses.loc[ports.index + " H2"]
    hydrogen_buses_ports.index.name = "Bus"

    return hydrogen_buses_ports



def add_export(n, hydrogen_buses_ports, export_h2, delivery_export_h2):
    country_shape = gpd.read_file(snakemake.input["shapes_path"])
    # Find most northwestern point in country shape and get x and y coordinates
    country_shape = country_shape.to_crs("EPSG:4326")

    # Get coordinates of the most western and northern point of the country and add a buffer of 2 degrees (equiv. to approx 220 km)
    x_export = country_shape.geometry.centroid.x.min() - 2
    y_export = country_shape.geometry.centroid.y.max() + 2

    # add export bus
    n.add(
        "Bus",
        "H2 export bus",
        carrier="H2",
        x=x_export,
        y=y_export,
    )

    # add export links
    logger.info("Adding export links")
    n.madd(
        "Link",
        names=hydrogen_buses_ports.index + " export",
        bus0=hydrogen_buses_ports.index,
        bus1="H2 export bus",
        p_nom_extendable=True,
        #p_nom_max = export_h2 / 100   
        # p_nom_max = export_h2 / 8760 / len(hydrogen_buses_ports) * 2
        # p_nom_max = export_h2 / 8760 
    )

    export_links = n.links[n.links.index.str.contains("export")]
    logger.info(export_links)

    # add store
    n.add(
        "Store",
        "H2 export store",
        bus="H2 export bus",
        e_nom_extendable=True,
        carrier="H2",
        e_initial=0,
        marginal_cost=0,
        capital_cost=0,
        e_cyclic=True,
        e_nom_max = export_h2 / delivery_export_h2 
    )

    # add load
    n.add(
        "Load",
        "H2 export load",
        bus="H2 export bus",
        carrier="H2",
        p_set=export_h2 / 8760,
    )

    return


if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        from helpers import mock_snakemake, sets_path_to_root

        snakemake = mock_snakemake(
            "add_export",
            simpl="",
            clusters="187",
            ll="c1.0",
            opts="Co2L",
            planning_horizons="2030",
            sopts="144H",
            discountrate=0.175,
            demand="BS",
            h2export='0', # [1000]
            export_delivery='daily'
        )
        sets_path_to_root("pypsa-earth-sec")

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    # countries = list(n.buses.country.unique())
    countries = snakemake.config["countries"]

    # get export delivery type 
    delivery_h2 = snakemake.config["export"]["export_delivery"]

    delivery_dic = {'none':1e-100, 'monthly':12, 'weekly':52, 'bidaily':156, 'daily':365, 'hourly':8760}

    delivery_export_h2 = delivery_dic[delivery_h2]

    # get export demand
    
    export_h2 = eval(snakemake.wildcards["h2export"]) * 1e6  # convert TWh to MWh
    logger.info(
        f"The yearly export demand is {export_h2/1e6} TWh resulting in an hourly average of {export_h2/8760:.2f} MWh"
    )

    # get hydrogen export buses/ports
    hydrogen_buses_ports = select_ports(n)

    # Reduce the values of p_nom_max for export regions according to H2-prosim requirements 
    # (percentage of installable capacity to be used from synthesis)
    if snakemake.config["custom_data"]["renewables"]:
        techs = snakemake.config["custom_data"]["renewables"]

        # get the location of buses from hydrogen_buses_ports
        buses_ports = list(hydrogen_buses_ports.location.values)
        

        # load the nested dictionary
        ee_reduction_ptx_dic = create_nested_dict_from_csv(snakemake.input.ee_reduction_ptx)

        for country in countries:
            ee_reduction_ptx = ee_reduction_ptx_dic[country][export_h2/1e6]
            for tech in techs:
                bus_ports_list = [str(value)+ " " + tech for value in buses_ports]
                n.generators.loc[bus_ports_list, 'p_nom_max'] = n.generators.loc[bus_ports_list, 'p_nom_max'] * (1 - ee_reduction_ptx)


    # add export value and components to network
    add_export(n, hydrogen_buses_ports, export_h2, delivery_export_h2)

    n.export_to_netcdf(snakemake.output[0])

    logger.info("Network successfully exported")
