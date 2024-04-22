"""
:Title:        get capacity existing
:Created:      April-2024
:Authors:      Johannes Burger (jburger@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

For Resilience optimization, based on David Bertschinger's Master thesis: retrieve existing capacity from nominal optimization
"""

# %% import
import pandas as pd
from pprint import pprint
from zen_garden.postprocess.results import Results
from pathlib import Path
import os
import shutil
import json


# %% Define the paths
#MODEL_PATH = os.path.abspath(os.path.join('/Users', 'jburger', 'Documents', 'GitHub', 'ZEN-resource', 'models'))
MODEL_PATH = 'F:\GitHub\ZEN-garden\data'
MODEL_Nominal = '02_CH_complete'
MODEL_N1 = f'{MODEL_Nominal}_ConvRes_n1'


def copy_existing_model():
    # copy the nominal model to the n1 folder
    origin = os.path.join(MODEL_PATH, MODEL_Nominal)
    destination = os.path.join(MODEL_PATH, MODEL_N1)
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(origin, destination)


def create_existing_capacity(results_nominal):
    cap_addition = results_nominal.get_df('capacity_addition')['none'].reset_index()  # only works if there are no scenarios
    technologies = cap_addition['technology'].copy().unique()
    start_year = results_nominal.get_system()['reference_year']

    for tech in technologies:
        df = cap_addition[cap_addition['technology'] == tech]
        path = Path(str([f for f in Path(os.path.join(MODEL_PATH, MODEL_N1)).rglob(tech)][0]) + '/capacity_existing.csv')

        df.drop(columns=['technology', 'capacity_type'], inplace=True)
        df = df[df['value'] != 0]
        if len(df.index) == 0:  # skip if no capacity exists
            continue
        df.rename(columns={'year': 'year_construction', 'value': 'capacity_existing'}, inplace=True)
        df['year_construction'] += start_year

        # clean up table, differentiate between node and edge
        if path.parent.parent.name == 'set_transport_technologies':
            df.rename(columns={'location': 'edge'}, inplace=True)
        else:
            df.rename(columns={'location': 'node'}, inplace=True)

        df.to_csv(path, index=False)


def create_nominal_flow(results_nominal):
    start_year = results_nominal.get_system()['reference_year']
    nom_flow_transport = results_nominal.get_df('flow_transport')['none'].reset_index()  # only works if there are no scenarios
    technologies = nom_flow_transport['technology'].copy().unique()

    for tech in technologies:
        df = nom_flow_transport[nom_flow_transport['technology'] == tech]
        path = Path(str([f for f in Path(os.path.join(MODEL_PATH, MODEL_N1)).rglob(tech)][0]) + '/nominal_flow_transport.csv')

        df.drop(columns='technology', inplace=True)
        df = df[df['value'] != 0]
        if len(df.index) == 0:  # skip if no flow exists
            continue
        # todo: the time_operation should be kept.
        #  Currently does not work though because the nominal flow is not indexed correctly
        df.rename(columns={'time_operation': 'year', 'value': 'nominal_flow_transport'}, inplace=True)
        df['year'] += start_year

        df.to_csv(path, index=False)

    nom_flow_conversion = results_nominal.get_df('flow_conversion_input')['none'].reset_index()  # only works if there are no scenarios
    technologies = nom_flow_conversion['technology'].copy().unique()

    for tech in technologies:
        df = nom_flow_conversion[nom_flow_conversion['technology'] == tech]
        path = Path(
            str([f for f in Path(os.path.join(MODEL_PATH, MODEL_N1)).rglob(tech)][0]) + '/nominal_flow_conversion_input.csv')

        df.drop(columns='technology', inplace=True)
        df = df[df['value'] != 0]
        if len(df.index) == 0:  # skip if no flow exists
            continue
        # todo: the time_operation should be kept.
        #  Currently does not work though because the nominal flow is not indexed correctly
        df.rename(columns={'time_operation': 'year'}, inplace=True)
        df['year'] += start_year
        df = df.pivot(index=['node', 'year'], columns='carrier', values='value').reset_index()

        df.to_csv(path, index=False)


def adjust_energy_system_file(results_nominal, target_year=None):
    with open(os.path.join(MODEL_PATH, MODEL_N1, 'energy_system', 'attributes.json'), 'r') as f:
        attributes = json.load(f)
    c_emissions = results_nominal.get_total('carbon_emissions_annual').iloc[target_year-results_nominal.get_system()['reference_year']]
    attributes.update({
        "carbon_emissions_annual_limit": {
            "default_value": c_emissions,
            "unit": "tons"},
        "min_co2_stored": {
            "default_value": 0,
            "unit": "kilotons"},
    })
    with open(os.path.join(MODEL_PATH, MODEL_N1, 'energy_system', 'attributes.json'), 'w') as f:
        json.dump(attributes, f, indent=4, sort_keys=True)
    # delete the min_co2_stored.csv file
    if os.path.exists(os.path.join(MODEL_PATH, MODEL_N1, 'energy_system', 'min_co2_stored.csv')):
        os.remove(os.path.join(MODEL_PATH, MODEL_N1, 'energy_system', 'min_co2_stored.csv'))


def adjust_system_file(target_year=None):
    system_file = os.path.join(MODEL_PATH, MODEL_N1, 'system.py')
    with open(system_file, 'r') as f:
        system = f.read()
    assert "include_n1_contingency_conversion" in system, "The system file does not contain the necessary variables."
    assert "include_n1_contingency_transport" in system, "The system file does not contain the necessary variables."
    assert "include_n1_contingency_import_export" in system, "The system file does not contain the necessary variables."
    if target_year:
        system = system.replace("['reference_year'] = ", f"['reference_year'] = {target_year}  # ")
    system = system.replace("['optimized_years'] = ", f"['optimized_years'] = 1  # ")
    system = system.replace("['include_n1_contingency_conversion'] = False", "['include_n1_contingency_conversion'] = True")
    system = system.replace("['include_n1_contingency_transport'] = False", "['include_n1_contingency_transport'] = True")
    system = system.replace("['include_n1_contingency_import_export'] = False", "['include_n1_contingency_import_export'] = True")
    system = system.replace("['use_capacities_existing'] = False", "['use_capacities_existing'] = True")
    system = system.replace("['load_lca_factors'] = True", "['load_lca_factors'] = False")

    with open(system_file, 'w') as f:
        f.write(system)


if __name__ == '__main__':
    target_year = 2050
    results_nominal = Results(os.path.join(MODEL_PATH, 'outputs', MODEL_Nominal))
    copy_existing_model()
    create_existing_capacity(results_nominal)
    create_nominal_flow(results_nominal)
    adjust_energy_system_file(results_nominal, target_year=target_year)
    adjust_system_file(target_year=target_year)
