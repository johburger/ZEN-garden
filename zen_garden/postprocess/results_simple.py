import os
import pandas as pd
import numpy as np
from zen_garden.postprocess.results import Results
from zen_garden.__main__ import run_module
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # %% Load the results
    # os.chdir('/Users/jburger/Documents/GitHub/ZEN-garden/tests/testcases/')

    def convert_timesteps(df, years = 10, timesteps_per_year=365):
        # Extract the original time indices
        time_indices = df['time_operation']
        # Convert to new time indices that reflect the chronological order
        new_time_indices = (time_indices // years) + (time_indices % years) * timesteps_per_year
        # Assign the new time indices to a new column
        df['sorted_set_time_steps_operation'] = new_time_indices
        # Sort the DataFrame based on the new time indices
        df = df.sort_values(by='sorted_set_time_steps_operation')
        return df

    os.chdir('F:/GitHub/ZEN-garden/data')
    model = '01_5_nodes_07_03'
    #run_module(args=['--config=./config.py', f'--dataset=./{model}'])
    results = Results(f'./outputs/{model}')
    capacity = results.get_total('capacity')
    capex = results.get_total('cost_capex')
    flow_full = results.get_full_ts('flow_transport')
    flow_full = flow_full.loc[(flow_full != 0).any(axis=1)]
    flow_full = flow_full.loc[:, (flow_full != 0).any(axis=0)]
    capacity_truck = capacity.loc['truck']
    capacity_truck = capacity_truck.loc[(capacity_truck != 0).any(axis=1)]
    operation = results.get_full_ts('operation_state_array')
    operation_series = results.get_df('operation_state_array')
    truck_series = operation_series['none'].iloc.obj['truck']
    edge = 'cement_178_CH-basel_export_CH'
    year = 0
    year_start = year*365
    year_end = year*365 + 365
    locals()[f"truck_series_{year}"] = truck_series.loc[edge][year_start:year_end]

    # Get Storage flows and level
    flow_storage_charge = results.get_df('flow_storage_charge')
    flow_storage_charge = flow_storage_charge['none']
    flow_storage_179_charge = flow_storage_charge.loc['co2_storage','cement_179_CH']
    flow_storage_179_charge = flow_storage_179_charge.reset_index()
    sorted_flow_storage_179_charge = convert_timesteps(flow_storage_179_charge, 10, 365)

    flow_storage_discharge = results.get_df('flow_storage_discharge')
    flow_storage_discharge = flow_storage_discharge['none']
    flow_storage_179_discharge = flow_storage_discharge.loc['co2_storage','cement_179_CH']
    flow_storage_179_discharge = flow_storage_179_discharge.reset_index()
    sorted_flow_storage_179_discharge = convert_timesteps(flow_storage_179_discharge, 10, 365)

    storage_level = results.get_df('storage_level')
    storage_level = storage_level['none']
    storage_level_179 = storage_level.loc['co2_storage','cement_179_CH']
    storage_level_179 = storage_level_179.reset_index().rename(columns={'time_storage_level':'time_operation'})
    sorted_storage_level_179 = convert_timesteps(storage_level_179, 10, 365)