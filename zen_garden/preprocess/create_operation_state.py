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
from scipy.stats import linregress

from zen_garden.utils import InputDataChecks

from zen_garden.model.objects.energy_system import EnergySystem
class OperationState:
    """
    Class to create operation state array and save as csv file
    """
    def __init__(self, element, system, analysis, solver, energy_system):
        """ data input object to extract input data

        :param element: element for which data is extracted
        :param system: dictionary defining the system
        :param analysis: dictionary defining the analysis framework
        :param solver: dictionary defining the solver
        :param energy_system: instance of class <EnergySystem> to define energy_system
        :param unit_handling: instance of class <UnitHandling> to convert units """
        self.element = element
        self.system = system
        self.analysis = analysis
        self.solver = solver
        self.energy_system = energy_system
        self.technology = technology
        self.scenario_dict = None
        self.unit_handling = unit_handling
        # extract folder path
        self.folder_path = getattr(self.element, "input_path")
        # get names of indices
        self.index_names = self.analysis['header_data_inputs']
        # load attributes file
        self.attribute_dict = self.load_attribute_file()


    def simulate_operation(self, failure_rate_offset = 0.1):
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

        locs = self.energy_system.set_edges
        times = self.energy_system.set_base_time_steps
        downtime_transport = self.technology.downtime_transport.values[0]
        downtime_transport_scaled = downtime_transport * self.technology.calculate_fraction_of_year()
        #operation = np.ones((1, len(locs)), dtype=int)
        operation = pd.DataFrame(data=[[1] * len(locs)], columns=locs)
        downtime_counters = np.zeros(len(locs), dtype=int)
        failure_probabilities = self.technology.distance.array * self.technology.failure_rate_transport.array / self.technology.calculate_fraction_of_year()
        failure_probabilities += failure_rate_offset
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
            downtime_counters[failures & ~in_downtime] = downtime_transport_scaled

            # Append the new row to the operation array
            #operation = np.vstack([operation, new_row])
            operation = pd.concat([operation, pd.DataFrame([new_row], columns=locs)], ignore_index=True)

        #TODO als csv speichern
        operation_series = operation.T.stack()
        operation_series.index.names = ['edge', 'time']

        return operation_series