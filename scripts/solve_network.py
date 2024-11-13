# -*- coding: utf-8 -*-
"""Solve network."""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from helpers import override_component_attrs
from pypsa.linopf import ilopf, network_lopf
from pypsa.linopt import define_constraints, get_var, join_exprs, linexpr
from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def add_land_use_constraint(n):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n)
    else:
        _add_land_use_constraint(n)


def _add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = (
            n.generators.loc[n.generators.carrier == carrier, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    grouping_years = snakemake.config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            set(
                [
                    i.split(sep=" ")[0] + " " + i.split(sep=" ")[1]
                    for i in existing.index
                ]
            )
        )

        previous_years = [
            str(y)
            for y in planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def prepare_network(n, solve_opts=None):
    # if snakemake.config["rescale_emissions"]:
    #     pass
    # n.carriers.co2_emissions = n.carriers.co2_emissions * 1e-6
    # n.global_constraints.at["CO2Limit", "constant"] = n.global_constraints.at["CO2Limit", "constant"] * 1e-6

    # modify electrolyzer capital costs
    n.links.loc[n.links.carrier == "H2 Electrolysis", "capital_cost"] *= float(
        snakemake.wildcards.electrolyzer_cc
    )

    if "lv_limit" in n.global_constraints.index:
        n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
        n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if solve_opts.get("load_shedding"):
        n.add("Carrier", "Load")
        n.madd(
            "Generator",
            n.buses.index,
            " load",
            bus=n.buses.index,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e2,  # Eur/kWh
            # intersect between macroeconomic and surveybased
            # willingness to pay
            # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                np.random.seed(174)
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            np.random.seed(123)
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if snakemake.config["foresight"] == "myopic":
        add_land_use_constraint(n)

    return n


def add_battery_constraints(n):
    chargers_b = n.links.carrier.str.contains("battery charger")
    chargers = n.links.index[chargers_b & n.links.p_nom_extendable]
    dischargers = chargers.str.replace("charger", "discharger")

    if chargers.empty or ("Link", "p_nom") not in n.variables.index:
        return

    link_p_nom = get_var(n, "Link", "p_nom")

    lhs = linexpr(
        (1, link_p_nom[chargers]),
        (
            -n.links.loc[dischargers, "efficiency"].values,
            link_p_nom[dischargers].values,
        ),
    )

    define_constraints(n, lhs, "=", 0, "Link", "charger_ratio")


def add_h2_network_cap(n, cap):
    h2_network = n.links.loc[n.links.carrier == "H2 pipeline"]
    if h2_network.index.empty or ("Link", "p_nom") not in n.variables.index:
        return
    h2_network_cap = get_var(n, "Link", "p_nom")
    subset_index = h2_network.index.intersection(h2_network_cap.index)
    diff_index = h2_network_cap.index.difference(subset_index)
    if len(diff_index) > 0:
        logger.warning(
            f"Impossible to set H2 cap for the following links: {diff_index}"
        )
    lhs = linexpr(
        (h2_network.loc[subset_index, "length"], h2_network_cap[subset_index])
    ).sum()
    # lhs = linexpr((1, h2_network_cap[h2_network.index])).sum()
    rhs = cap * 1000
    define_constraints(n, lhs, "<=", rhs, "h2_network_cap")


def H2_export_yearly_constraint(n):
    res = [
        "csp",
        "rooftop-solar",
        "solar",
        "onwind",
        "onwind2",
        "onwind3",
        "offwind",
        "offwind2",
        "ror",
    ]
    res_index = n.generators.loc[n.generators.carrier.isin(res)].index

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_index)),
        index=n.snapshots,
        columns=res_index,
    )
    res = join_exprs(
        linexpr((weightings, get_var(n, "Generator", "p")[res_index]))
    )  # single line sum

    load_ind = n.loads[n.loads.carrier == "AC"].index.intersection(
        n.loads_t.p_set.columns
    )

    load = (
        n.loads_t.p_set[load_ind].sum(axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    h2_export = n.loads.loc["H2 export load"].p_set * 8760

    lhs = res

    include_country_load = snakemake.config["policy_config"]["yearly"][
        "re_country_load"
    ]

    if include_country_load:
        elec_efficiency = (
            n.links.filter(like="Electrolysis", axis=0).loc[:, "efficiency"].mean()
        )
        rhs = (
            h2_export * (1 / elec_efficiency) + load
        )  # 0.7 is approximation of electrloyzer efficiency # TODO obtain value from network
    else:
        rhs = h2_export * (1 / 0.7)

    con = define_constraints(n, lhs, ">=", rhs, "H2ExportConstraint", "RESproduction")


def hydrogen_temporal_constraint(n, n_ref, time_period):
    res_techs = [
        "csp",
        "rooftop-solar",
        "solar",
        "onwind",
        "onwind2",
        "onwind3",
        "offwind",
        "offwind2",
        "ror",
    ]

    res_stor_techs = ["hydro"]

    allowed_excess = snakemake.config["policy_config"]["hydrogen"]["allowed_excess"]

    res_gen_index = n.generators.loc[n.generators.carrier.isin(res_techs)].index
    res_stor_index = n.storage_units.loc[
        n.storage_units.carrier.isin(res_stor_techs)
    ].index

    weightings_gen = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_gen_index)),
        index=n.snapshots,
        columns=res_gen_index,
    )

    res = linexpr((weightings_gen, get_var(n, "Generator", "p")[res_gen_index])).sum(
        axis=1
    )

    if not res_stor_index.empty:
        weightings_stor = pd.DataFrame(
            np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_stor_index)),
            index=n.snapshots,
            columns=res_stor_index,
        )
        res += linexpr(
            (weightings_stor, get_var(n, "StorageUnit", "p_dispatch")[res_stor_index])
        ).sum(axis=1)

    if time_period == "month":
        res = res.groupby(res.index.month).sum()
    elif time_period == "year":
        res = res.groupby(res.index.year).sum()

    electrolysis = get_var(n, "Link", "p")[
        n.links.index[n.links.index.str.contains("H2 Electrolysis")]
    ]
    weightings_electrolysis = pd.DataFrame(
        np.outer(
            n.snapshot_weightings["generators"], [1.0] * len(electrolysis.columns)
        ),
        index=n.snapshots,
        columns=electrolysis.columns,
    )

    elec_input = linexpr((-allowed_excess * weightings_electrolysis, electrolysis)).sum(
        axis=1
    )

    if time_period == "month":
        elec_input = elec_input.groupby(elec_input.index.month).sum()
    elif time_period == "year":
        elec_input = elec_input.groupby(elec_input.index.year).sum()

    if snakemake.config["policy_config"]["hydrogen"]["additionality"]:
        res_ref_gen = n_ref.generators_t.p[res_gen_index] * weightings_gen

        if not res_stor_index.empty:
            res_ref_store = (
                n_ref.storage_units_t.p_dispatch[res_stor_index] * weightings_stor
            )
            res_ref = pd.concat([res_ref_gen, res_ref_store], axis=1)
        else:
            res_ref = res_ref_gen

        if time_period == "month":
            res_ref = (
                res_ref.groupby(n_ref.generators_t.p.index.month).sum().sum(axis=1)
            )
        elif time_period == "year":
            res_ref = res_ref.groupby(n_ref.generators_t.p.index.year).sum().sum(axis=1)

        elec_input_ref = (
            -n_ref.links_t.p0.loc[
                :, n_ref.links_t.p0.columns.str.contains("H2 Electrolysis")
            ]
            * weightings_electrolysis
        )
        if time_period == "month":
            elec_input_ref = (
                -elec_input_ref.groupby(elec_input_ref.index.month).sum().sum(axis=1)
            )
        elif time_period == "year":
            elec_input_ref = (
                -elec_input_ref.groupby(elec_input_ref.index.year).sum().sum(axis=1)
            )

        for i in range(len(res.index)):
            lhs = res.iloc[i] + "\n" + elec_input.iloc[i]
            rhs = res_ref.iloc[i].sum() + elec_input_ref.iloc[i].sum()
            con = define_constraints(
                n, lhs, ">=", rhs, f"RESconstraints_{i}", f"REStarget_{i}"
            )

    else:
        for i in range(len(res.index)):
            lhs = res.iloc[i] + "\n" + elec_input.iloc[i]

            con = define_constraints(
                n, lhs, ">=", 0.0, f"RESconstraints_{i}", f"REStarget_{i}"
            )
    # else:
    #     logger.info("ignoring H2 export constraint as wildcard is set to 0")


def add_chp_constraints(n):
    electric_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric = n.links.index[electric_bool]
    heat = n.links.index[heat_bool]

    electric_ext = n.links.index[electric_bool & n.links.p_nom_extendable]
    heat_ext = n.links.index[heat_bool & n.links.p_nom_extendable]

    electric_fix = n.links.index[electric_bool & ~n.links.p_nom_extendable]
    heat_fix = n.links.index[heat_bool & ~n.links.p_nom_extendable]

    link_p = get_var(n, "Link", "p")

    if not electric_ext.empty:
        link_p_nom = get_var(n, "Link", "p_nom")

        # ratio of output heat to electricity set by p_nom_ratio
        lhs = linexpr(
            (
                n.links.loc[electric_ext, "efficiency"]
                * n.links.loc[electric_ext, "p_nom_ratio"],
                link_p_nom[electric_ext],
            ),
            (-n.links.loc[heat_ext, "efficiency"].values, link_p_nom[heat_ext].values),
        )

        define_constraints(n, lhs, "=", 0, "chplink", "fix_p_nom_ratio")

        # top_iso_fuel_line for extendable
        lhs = linexpr(
            (1, link_p[heat_ext]),
            (1, link_p[electric_ext].values),
            (-1, link_p_nom[electric_ext].values),
        )

        define_constraints(n, lhs, "<=", 0, "chplink", "top_iso_fuel_line_ext")

    if not electric_fix.empty:
        # top_iso_fuel_line for fixed
        lhs = linexpr((1, link_p[heat_fix]), (1, link_p[electric_fix].values))

        rhs = n.links.loc[electric_fix, "p_nom"].values

        define_constraints(n, lhs, "<=", rhs, "chplink", "top_iso_fuel_line_fix")

    if not electric.empty:
        # backpressure
        lhs = linexpr(
            (
                n.links.loc[electric, "c_b"].values * n.links.loc[heat, "efficiency"],
                link_p[heat],
            ),
            (-n.links.loc[electric, "efficiency"].values, link_p[electric].values),
        )

        define_constraints(n, lhs, "<=", 0, "chplink", "backpressure")


def add_co2_sequestration_limit(n, sns):
    co2_stores = n.stores.loc[n.stores.carrier == "co2 stored"].index

    if co2_stores.empty or ("Store", "e") not in n.variables.index:
        return

    vars_final_co2_stored = get_var(n, "Store", "e").loc[sns[-1], co2_stores]

    lhs = linexpr((1, vars_final_co2_stored)).sum()
    rhs = (
        n.config["sector"].get("co2_sequestration_potential", 5) * 1e6
    )  # TODO change 200 limit (Europe)

    name = "co2_sequestration_limit"
    define_constraints(
        n, lhs, "<=", rhs, "GlobalConstraint", "mu", axes=pd.Index([name]), spec=name
    )


def set_h2_colors(n):
    blue_h2 = get_var(n, "Link", "p")[
        n.links.index[n.links.index.str.contains("blue H2")]
    ]

    pink_h2 = get_var(n, "Link", "p")[
        n.links.index[n.links.index.str.contains("pink H2")]
    ]

    fuelcell_ind = n.loads[n.loads.carrier == "land transport fuel cell"].index

    other_ind = n.loads[
        (n.loads.carrier == "H2 for industry")
        | (n.loads.carrier == "H2 for shipping")
        | (n.loads.carrier == "H2")
    ].index

    load_fuelcell = (
        n.loads_t.p_set[fuelcell_ind].sum(axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    load_other_h2 = n.loads.loc[other_ind].p_set.sum() * 8760

    load_h2 = load_fuelcell + load_other_h2

    weightings_blue = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(blue_h2.columns)),
        index=n.snapshots,
        columns=blue_h2.columns,
    )

    weightings_pink = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(pink_h2.columns)),
        index=n.snapshots,
        columns=pink_h2.columns,
    )

    total_blue = linexpr((weightings_blue, blue_h2)).sum().sum()

    total_pink = linexpr((weightings_pink, pink_h2)).sum().sum()

    rhs_blue = load_h2 * snakemake.config["sector"]["hydrogen"]["blue_share"]
    rhs_pink = load_h2 * snakemake.config["sector"]["hydrogen"]["pink_share"]

    define_constraints(n, total_blue, "==", rhs_blue, "blue_h2_share")

    define_constraints(n, total_pink, "==", rhs_pink, "pink_h2_share")


def constrain_smr_dispatch(n, n_ref):
    logger.info(
        "Adding constraint to limit p0 of SMR units to reference case dispatch."
    )

    # Identify SMR units in both networks
    smr_units_ref = n_ref.links.loc[n_ref.links.index.str.contains("SMR")]
    smr_units = n.links.loc[n.links.index.str.contains("SMR")]

    # Get the reference p0 values for each SMR unit at each timestep
    smr_ref_p0 = n_ref.links_t.p0.loc[:, smr_units_ref.index]

    # For each SMR unit, create a constraint that limits its p0 in 'n' to be at most the value from 'n_ref'
    for unit in smr_units.index:
        # Get the reference dispatch for the current SMR unit
        p0_ref = smr_ref_p0[unit]

        # Define the constraint
        # lhs = get_var(n, "Link", "p")[unit]
        lhs = linexpr((1, get_var(n, "Link", "p")[unit]))
        rhs = p0_ref  # .loc[t]
        logger.info("adding SMR constraint for {}".format(unit))
        # Add the constraint: p0 in 'n' must be less than or equal to the reference case
        define_constraints(n, lhs, "<=", rhs, f"smr_p0_limit_{unit}")


def fix_rooftopPV_utilityScale_ratio(n, target_ratio=2):
    logger.info(
        f"Adding constraint to ensure the total rooftop solar capacity is {target_ratio} times the total utility-scale solar capacity."
    )

    # Filter for rooftop and utility-scale solar generators
    rooftop_index = n.generators.loc[n.generators.carrier == "solar-rooftop"].index
    utility_index = n.generators.loc[n.generators.carrier == "solar"].index

    # Get the generator capacity variables (p_nom) for each generator type
    gen_var = get_var(n, "Generator", "p_nom")

    # Extract capacity values by index
    rooftop_total = gen_var.loc[rooftop_index]
    utility_total = gen_var.loc[utility_index]

    # Define the constraint using linexpr, ensuring alignment by combining after summing
    lhs = (
        linexpr((1, rooftop_total)).sum()
        + linexpr((-target_ratio, utility_total)).sum()
    )

    # RHS is zero
    rhs = 0

    # Add the constraint to the model
    define_constraints(n, lhs, "==", rhs, "utility_to_rooftop_ratio_constraint")


def extra_functionality(n, snapshots):
    add_battery_constraints(n)

    additionality = snakemake.config["policy_config"]["hydrogen"]["additionality"]
    ref_for_additionality = snakemake.config["policy_config"]["hydrogen"][
        "is_reference"
    ]
    temportal_matching_period = snakemake.config["policy_config"]["hydrogen"][
        "temporal_matching"
    ]

    if temportal_matching_period == "no_temporal_matching":
        logger.info("no h2 temporal constraint set")

    elif additionality:
        if ref_for_additionality:
            logger.info("preparing reference case for additionality constraint")
        else:
            logger.info(
                "setting h2 export to {}ly matching constraint with additionality".format(
                    temportal_matching_period
                )
            )
            hydrogen_temporal_constraint(n, n_ref, temportal_matching_period)
    elif not additionality:
        logger.info(
            "setting h2 export to {}ly matching constraint without additionality".format(
                temportal_matching_period
            )
        )
        hydrogen_temporal_constraint(n, n_ref, temportal_matching_period)

    if snakemake.config["sector"]["hydrogen"]["network"]:
        if snakemake.config["sector"]["hydrogen"]["network_limit"]:
            add_h2_network_cap(
                n, snakemake.config["sector"]["hydrogen"]["network_limit"]
            )

    if snakemake.config["sector"]["hydrogen"]["set_color_shares"]:
        logger.info("setting H2 color mix")
        set_h2_colors(n)

    if (
        snakemake.config["policy_config"]["hydrogen"]["limit_SMR_dispatch"]
        and not snakemake.config["policy_config"]["hydrogen"]["is_reference"]
    ):
        constrain_smr_dispatch(n, n_ref)

    if snakemake.config["policy_config"]["renewables"][
        "ratio_rooftop_to_utility_solar"
    ]:
        fix_rooftopPV_utilityScale_ratio(
            n,
            float(
                snakemake.config["policy_config"]["renewables"][
                    "ratio_rooftop_to_utility_solar"
                ]
            ),
        )

    add_co2_sequestration_limit(n, snapshots)


def solve_network(n, config, opts="", **kwargs):
    solver_options = config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    cf_solving = config["solving"]["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if cf_solving.get("skip_iterations", False):
        network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=extra_functionality,
            **kwargs,
        )
    else:
        ilopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            track_iterations=track_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            extra_functionality=extra_functionality,
            **kwargs,
        )
    return n


def add_existing(n):
    if snakemake.wildcards["planning_horizons"] == "2050":
        directory = (
            "results/"
            + "Existing_capacities/"
            + snakemake.config["run"].replace("2050", "2030")
        )
        n_name = (
            snakemake.input.network.split("/")[-1]
            .replace(str(snakemake.config["scenario"]["clusters"][0]), "")
            .replace(str(snakemake.config["costs"]["discountrate"][0]), "")
            .replace("_presec", "")
            .replace(".nc", ".csv")
        )
        df = pd.read_csv(directory + "/electrolyzer_caps_" + n_name, index_col=0)
        existing_electrolyzers = df.p_nom_opt.values

        h2_index = n.links[n.links.carrier == "H2 Electrolysis"].index
        n.links.loc[h2_index, "p_nom_min"] = existing_electrolyzers

        # n_name = snakemake.input.network.split("/")[-1].replace(str(snakemake.config["scenario"]["clusters"][0]), "").\
        #     replace(".nc", ".csv").replace(str(snakemake.config["costs"]["discountrate"][0]), "")
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)

        for tech in snakemake.config["custom_data"]["renewables"]:
            # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
            existing_res = df.loc[tech]
            existing_res.index = existing_res.index.str.apply(lambda x: x + tech)
            tech_index = n.generators[n.generators.carrier == tech].index
            n.generators.loc[tech_index, tech] = existing_res


if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        from helpers import mock_snakemake, sets_path_to_root

        snakemake = mock_snakemake(
            "solve_network",
            simpl="",
            clusters="11",
            ll="v1.0",
            opts="Co2L",
            planning_horizons="2035",
            sopts="3H",
            discountrate=0.06,
            demand="BI",
            h2export="0",
            h2mp="endogenous",
            electrolyzer_cc=1.0,
        )

        sets_path_to_root("pypsa-earth-sec")

    logging.basicConfig(
        filename=snakemake.log.python, level=snakemake.config["logging_level"]
    )

    tmpdir = snakemake.config["solving"].get("tmpdir")
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]

    fn = getattr(snakemake.log, "memory", None)
    with memory_logger(filename=fn, interval=30.0) as mem:
        overrides = override_component_attrs(snakemake.input.overrides)
        n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

        if (
            snakemake.config["custom_data"]["add_existing"]
            and snakemake.wildcards.planning_horizons == "2050"
        ):
            add_existing(n)

        if (
            snakemake.config["policy_config"]["hydrogen"]["additionality"]
            and not snakemake.config["policy_config"]["hydrogen"]["is_reference"]
            and snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
            != "no_res_matching"
        ):
            n_ref_path = snakemake.config["policy_config"]["hydrogen"]["path_to_ref"]
            n_ref_path = (
                n_ref_path[:-3].replace("scenario", snakemake.wildcards["demand"])
                + f"_{snakemake.wildcards.electrolyzer_cc}ecc"
                + n_ref_path[-3:]
            )
            n_ref = pypsa.Network(n_ref_path)
        else:
            n_ref = None

        n = prepare_network(n, solve_opts)

        n = solve_network(
            n,
            config=snakemake.config,
            opts=snakemake.wildcards.opts.split("-"),
            solver_dir=tmpdir,
            solver_logfile=snakemake.log.solver,
        )

        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output[0])

    # logging output to the terminal
    print("Objective function: {}".format(n.objective))

    # logging output to file
    logger.info("Objective function: {}".format(n.objective))
    logger.info("Objective constant: {}".format(n.objective_constant))
    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
