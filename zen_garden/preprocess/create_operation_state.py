"""
:Title:         ZEN-GARDEN
:Created:       July-2024
:Authors:       Georg Henke (ghenke@ethz.ch)
:Organization:  Laboratory of Risk and Reliability Engineering, ETH Zurich

Function to create operation state array and save as csv file
"""
import copy
import logging
import math
import os
import json
import numpy as np
import pandas as pd
#from data.01_5_nodes_07_03.system import system


def calculate_haversine_distances_from_nodes(nodes):
    """
    Computes the distance in kilometers between two nodes by using their lon lat coordinates and the Haversine formula

    :return: dict containing all edges along with their distances
    """
    set_haversine_distances_of_edges = {}
    # read coords file
    df_coords_input = pd.read_csv('F:\GitHub\ZEN-garden_new\data/01_5_nodes_07_03\energy_system\set_nodes.csv')
    df_coords_input = df_coords_input[df_coords_input["node"].isin(nodes)]
    # convert coords from decimal degrees to radians
    df_coords_input["lon"] = df_coords_input["lon"] * np.pi / 180
    df_coords_input["lat"] = df_coords_input["lat"] * np.pi / 180
    # Radius of the Earth in kilometers
    radius = 6371.0
    edges = pd.read_csv('F:\GitHub\ZEN-garden_new\data/01_5_nodes_07_03\energy_system\set_edges.csv')
    edges = edges[(edges['node_from'].isin(nodes)) & (edges['node_to'].isin(nodes))].reset_index(drop=True)
    edge_dict = {f"{row['node_from']}-{row['node_to']}": (row['node_from'], row['node_to']) for _, row in edges.iterrows()}
    for edge, nodes in edge_dict.items():
        node_1, node_2 = nodes
        coords1 = df_coords_input[df_coords_input["node"] == node_1]
        coords2 = df_coords_input[df_coords_input["node"] == node_2]
        # Haversine formula
        dlon = coords2["lon"].squeeze() - coords1["lon"].squeeze()
        dlat = coords2["lat"].squeeze() - coords1["lat"].squeeze()
        a = np.sin(dlat / 2) ** 2 + np.cos(coords1["lat"].squeeze()) * np.cos(coords2["lat"].squeeze()) * np.sin(
            dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = radius * c
        set_haversine_distances_of_edges[edge] = distance
    multiplier = 1
    set_haversine_distances_of_edges = {key: value * multiplier for key, value in
                                        set_haversine_distances_of_edges.items()}
    return set_haversine_distances_of_edges

def simulate_operation(tech_type, locs, nodes, failure_rate_offset = 0.1):
    """
    Simulate the state of the system over one timestep.

    Parameters:
    - num_edges: Number of transport technologies.
    - failure_probabilities: Array of failure probabilities for each technology.
    - downtime: Downtime duration in time steps.
    - array: Current state of the system represented as a numpy array.
    - downtime_counters: Array of remaining downtimes for each technology.

    Returns:
    - array: Updated state of the system represented as a numpy array.
    - downtime_counters: Updated array of remaining downtimes for each technology.
    """
    #test = system['set_nodes']
    locs = locs[(locs['node_from'].isin(nodes)) & (locs['node_to'].isin(nodes))].reset_index(drop=True)
    locs = locs.drop(columns=['node_from', 'node_to'])
    times = list(range(0, 3649))

    fraction_of_year = 365 / 8760

    operation = pd.DataFrame(data=[[1] * len(locs)], columns=locs['edge'])
    downtime_counters = np.zeros(len(locs), dtype=int)

    path = f'F:\GitHub\ZEN-garden_new\data/01_5_nodes_07_03\set_technologies\set_transport_technologies/{tech_type}/attributes.json'
    with open(path, 'r') as file:
        data = json.load(file)
        failure_rate = data.get('failure_rate', {}).get('default_value')
        downtime = data.get('downtime', {}).get('default_value') * fraction_of_year
    distance = pd.read_csv(f'F:/GitHub/ZEN-garden_new/data/01_5_nodes_07_03/set_technologies/set_transport_technologies/{tech_type}/distance.csv')
    distance = distance[distance['edge'].isin(locs['edge'])]

    # Calculate the haversine distances and add missing edges to the DataFrame
    haversine_distances = calculate_haversine_distances_from_nodes(nodes)
    missing_distances = pd.DataFrame([(edge, haversine_distances[edge]) for edge in set(haversine_distances) - set(distance['edge'])],
                              columns=['edge', 'distance'])

    distance = pd.concat([distance, missing_distances], ignore_index=True)

    failure_probabilities = distance['distance'].multiply(failure_rate * fraction_of_year) + failure_rate_offset
    failure_probabilities.index = distance['edge']

    # TODO keine Failures back to back zulassen
    num_edges = len(locs)
    for timestep in range(0, len(times)-1):
        # Create a new row for the current timestep
        new_row = np.ones(num_edges, dtype=int)

        # Decrement the downtime counters
        downtime_counters = np.maximum(0, downtime_counters - 1)

        # Determine which technologies are currently in downtime
        in_downtime = downtime_counters > 0

        # Determine failures for technologies
        failures = np.random.rand(num_edges) < failure_probabilities

        # Update the new row and downtime counters for new failures
        new_row[failures | in_downtime] = 0
        downtime_counters[failures & ~in_downtime] = downtime

        # Append the new row to the operation array
        #operation = np.vstack([operation, new_row])
        operation = pd.concat([operation, pd.DataFrame([new_row], columns=locs['edge'])], ignore_index=True)

    #TODO als csv speichern
    operation_series = operation.T.stack()
    operation_series.index.names = ['edge', 'time']

    return operation_series

locs = pd.read_csv('F:/GitHub/ZEN-garden_new/data/01_5_nodes_07_03/energy_system/set_edges.csv')
nodes = [
    'waste_278_CH',
    'waste_410_CH',
    'cement_179_CH',
    'cement_178_CH',
    'basel_export_CH'
]

for i in range(2):
    operation_series_truck = simulate_operation('truck', locs, nodes)
    operation_series_truck.to_csv(f'F:/GitHub/ZEN-garden_new/data/01_5_nodes_07_03/set_technologies/set_transport_technologies/truck/operation_state_{i}.csv')
    operation_series_pipeline = simulate_operation('pipeline_lin', locs, nodes)
    operation_series_pipeline.to_csv(f'F:/GitHub/ZEN-garden_new/data/01_5_nodes_07_03/set_technologies/set_transport_technologies/pipeline_lin/operation_state_{i}.csv')

#operation_series_truck = simulate_operation('truck')
#operation_series_truck.to_csv('F:/GitHub/ZEN-garden_new/data/operation_state/operation_state.csv')

#operation_series_pipeline = simulate_operation('pipeline_lin')
#operation_series_pipeline.to_csv('F:/GitHub/ZEN-garden_new/data/operation_state/operation_state_pipeline.csv')