# ZEN-garden with N-1 contingency

The ZEN-garden project models energy systems and value chains, incorporating electricity, hydrogen, and CCUS (carbon capture, storage, and utilization) value chains. This branch is updated to perform resilience analysis through the inclusion of the N-1 contingency.

## Enhancements in This Branch
This branch includes additional code elements to facilitate in resilience analysis:

### `energy_system.py`
- **Added index:**
  - `set_failure_states`: Defines a set of failure states for N-1 contingency. The data for these states are sourced from `extract_input_data.py` within the `extract_set_technologies_existing` constraint.

### `transport_technology.py`
- **Added parameters:**
  - `nominal_flow_transport`: Extracts nominal flow transport from input data.
  - `failure_rate_transport`: Extracts failure rate transport from input data.
  - `downtime_transport`: Extracts downtime transport from input data.
  - `operation_probability_transport`: Calculates operation probability for transport technologies from the `calculate_operation_probability()` function.
- **Updated variable:**
  - `flow_transport`: Includes the `set_failure_states` index.
  - `flow_transport_loss`: Includes the `set_failure_states` index.

### `conversion_technology.py`
- **Added parameters:**
  - `nominal_flow_conversion_input`: Extracts nominal flow conversion input from input data.
  - `failure_rate_conversion`: Extracts failure rate conversion from input data.
  - `downtime_conversion`: Extracts downtime conversion from input data.
  - `operation_probability_conversion`: Calculates operation probability for conversion technologies from the `calculate_operation_probability()` function.
- **Updated variable:**
  - `flow_conversion_input`: Includes the "set_failure_states" index.
  - `flow_conversion_output`: Includes the "set_failure_states" index.

### `technology.py`
- **Added constraints:**
  - `n1_contingency_transport`: Limits transport flow to nominal flow times operation probability for N-1 contingency.
  - `n1_contingency_conversion`: Limits conversion input flow to nominal flow times operation probability for N-1 contingency.
  - `n1_contingency_no_failure_transport`: Sets transport flow in "no failure state" to nominal flow.
  - `n1_contingency_no_failure_conversion`: Sets conversion input flow in "no failure state" to nominal flow.
- **Updated constraints:**
  - `constraint_opex_yearly_block`
  - `constraint_carbon_emissions_technology_total_block`
  - `constraint_capacity_factor_block`

### `carrier.py`
- **Added constraints:**
  - `constraint_nodal_energy_balance_failure_states`: Ensures node- and time-dependent energy balance for each carrier in N-1 contingency.
- **Updated constraints:**
  - `constraint_availability_import_yearly_block`
  - `constraint_availability_export_yearly_block`
  - `constraint_cost_carrier_block`
  - `constraint_carbon_emissions_carrier_block`

## Notebooks for CSV Handling

- `01_get_existing_capacity.ipynb`: Retrieves the existing capacities of all technologies.
- `02_get_nominal_flow_transport.ipynb`: Retrieves the nominal flow of transport for all technologies.
- `03_get_nominal_flow_conversion.ipynb`: Retrieves the nominal flow for all conversion technologies.
- `04_get_carbon_emissions_annual_limit_adjustment.ipynb`: Retrieves the expected carbon noz stored for the adjustment of the annual carbon emissions limit.

## For Analysis

- `plot.py`: Utilized to plot a map with nodes, flow transport, and flow conversion.

## How to Use N-1 Contingency

### Cost Optimal Solution
Create a dataset for the cost-optimal solution (`data_cost_optimal`), follow these guidelines:
- Avoid using storage technologies as they cannot fail.
- Set the carbon emissions annual limit to manage the carbon budget.
- Set prices to infinite for carbon emissions budget overshoot (`price_carbon_emissions_budget_overshoot`) and annual overshoot (`price_carbon_emissions_annual_overshoot`).

#### Configuration Parameters
```json
{
  "carbon_emissions_budget": {
    "default_value": "inf",
    "unit": "megatons"
  },
  "carbon_emissions_cumulative_existing": {
    "default_value": 0,
    "unit": "megatons"
  },
  "price_carbon_emissions": {
    "default_value": 0,
    "unit": "Euro/tons"
  },
  "price_carbon_emissions_budget_overshoot": {
    "default_value": "inf",
    "unit": "kiloEuro/tons"
  },
  "price_carbon_emissions_annual_overshoot": {
    "default_value": "inf",
    "unit": "kiloEuro/tons"
  }
}
```

`Run optimization of data_cost_optimal`

### N-1 Contingency Analysis

Copy `data_cost_optimal` to `data_n1` and ensure the following attributes are correctly set for transport and conversion technologies:

#### Transport Technologies Attributes
- nominal_flow_transport: allways inf
- failure_rate: the rate of failure for a technology per hour and km
- downtime: downtime when failure occurs
```json
{
  "nominal_flow_transport": {
    "default_value": "inf",
    "unit": "tons/hour"
  },
  "failure_rate": {
    "default_value": 0.000000014840183,
    "unit": "1/km/hour"
  },
  "downtime": {
    "default_value": 480,
    "unit": "hour"
  }
}
```

#### Conversion Technologies Attributes
- nominal_flow_conversion input: allways inf needs to be specified for every input
- failure_rate: the rate of failure for a technology per hour the rate of failure for a technology per hour and km (set to 0 if technology should not be included in failure analysis, this can also be done for specified edges through the csv)
- downtime: downtime when failure occurs
```json
  {
    "nominal_flow_conversion_input": {
        "flue_gas": {
          "default_value": "inf",
          "unit": "tons/tons"
        },
        "heat": {
          "default_value": "inf",
          "unit": "tons/hour"
        },
        "electricity": {
          "default_value": "inf",
          "unit": "tons/hour"
        }
    },
    "failure_rate": {
      "default_value": 0,
      "unit": "1/year"
    },
    "downtime": {
      "default_value": 0,
      "unit": "hour"
    }
  }
 ```

### Notebook Usage for Data Management

Utilize the following Jupyter notebooks to automatically create csv with needed input data.

- `01_get_existing_capacity.ipynb`: Retrieves the existing capacities of all technologies.
- `02_get_nominal_flow_transport.ipynb`: Retrieves the nominal flow of transport for all technologies.
- `03_get_nominal_flow_conversion.ipynb`: Retrieves the nominal flow for all conversion technologies.
- 
Configure the data paths in the notebooks as follows:

- Input path: `output/data_cost_optimal`
- Export path: `data_n1`

### System Configuration for N-1 Contingency

In the `system.py` file, update the system configuration to include N-1 contingency features:

```python
# N-1 contingency
system['include_n1_contingency_transport'] = True
system['include_n1_contingency_conversion'] = True
system['include_n1_contingency_import_export'] = True
```

It is advisable to enable all three options (`True`) to maximize the assessment of the system's resilience under N-1 contingency scenarios.

`Run optimization of data_n1`


## How to Use Adjustment of `carbon_emissions_annual_limit_adjustment`

To analyze how the system behaves when the annual carbon budget is adjusted between 0 and 100% of the expected carbon not stored, follow these steps:

### Data Preparation
Copy `data_n1` to `data_n1_add_0` and set the `capacity_addition_max` for all technologies to 0, particularly for those that might be used to surpass failure thresholds, reducing the chance of system infeasibility.

   ```json
   {
     "capacity_addition_max": {
       "default_value": 0,
       "unit": "kilotons/year"
     }
   }
   ```

This configuration ensures that carbon emissions resulting from failures cannot be stored, enabling us to measure the impacts of such failures and calculate expected carbon non-resilience metrics. 
To prevent system infeasibility, include emergency storages that maintain the carbon budget without actual carbon storage, albeit at a significantly higher cost. 
This approach should be applied to every type of CO2 carrier, ensuring the system's ability to manage carbon budgets effectively under various failure scenarios.

```json
[
  {
    "capacity_addition_min": {
      "default_value": 0,
      "unit": "tons/hour"
    }
  },
  {
    "capacity_addition_max": {
      "default_value": "inf",
      "unit": "tons/hour"
    }
  },
  {
    "capacity_existing": {
      "default_value": 0,
      "unit": "tons/hour"
    }
  },
  {
    "capacity_limit": {
      "default_value": "inf",
      "unit": "tons/hour"
    }
  },
  {
    "capacity_limit_country": {
      "default_value": "inf",
      "unit": "GW"
    }
  },
  {
    "capacity_limit_super": {
      "default_value": "inf",
      "unit": "GW"
    }
  },
  {
    "capacity_addition_unbounded": {
      "default_value": 0,
      "unit": "tons/hour"
    }
  },
  {
    "min_load": {
      "default_value": 0,
      "unit": "1"
    }
  },
  {
    "max_load": {
      "default_value": 1,
      "unit": "1"
    }
  },
  {
    "lifetime": {
      "default_value": 50,
      "unit": "1"
    }
  },
  {
    "opex_specific_variable": {
      "default_value": 3000,
      "unit": "Euro/tons"
    }
  },
  {
    "reference_carrier": {
      "default_value": ["dummy_carrier"]
    }
  },
  {
    "input_carrier": {
      "default_value": ["co2_liquid_16bar"]
    }
  },
  {
    "output_carrier": {
      "default_value": ["dummy_carrier"]
    }
  },
  {
    "carbon_intensity": {
      "default_value": -1,
      "unit": "tons/tons"
    }
  },
  {
    "construction_time": {
      "default_value": 0,
      "unit": "1"
    }
  },
  {
    "capacity_investment_existing": {
      "default_value": 0,
      "unit": "tons/hour"
    }
  },
  {
    "opex_specific_fixed": {
      "default_value": 0,
      "unit": "Euro/tons/hour"
    }
  },
  {
    "max_diffusion_rate": {
      "default_value": "inf",
      "unit": "1"
    }
  },
  {
    "conversion_factor": [
      {
        "co2_liquid_16bar": {
          "default_value": 1,
          "unit": "tons/tons"
        }
      }
    ]
  },
  {
    "capex_specific": {
      "default_value": 0,
      "unit": "Euro/tons/hour"
    }
  },
  {
    "nominal_flow_conversion_input": [
      {
        "co2_liquid_16bar": {
          "default_value": "inf",
          "unit": "tons/hour"
        }
      }
    ]
  },
  {
    "failure_rate": {
      "default_value": 0,
      "unit": "1/year"
    }
  },
  {
    "downtime": {
      "default_value": 48,
      "unit": "hour"
    }
  }
]
```
`Run optimization of data_n1_add_0`

### Carbon Budget adjusted Analysis

Copy `data_cost_optimal` to `data_n1_adjusted` and ensure the following attributes are correctly set for transport and conversion technologies:

#### Energy System Attributes
* carbon_emissions_annual_limit_adjustment: allways 0 (csv gets crate further down)
* carbon_emissions_annual_limit_adjustment_factor: between 0 and 1 to define how much more budget there is be between 0 and 100% of expected carbon not stored

```json
  {
    "carbon_emissions_annual_limit_adjustment": {
      "default_value": 0,
      "unit": "kilotons"
    }
  },
  {
    "carbon_emissions_annual_limit_adjustment_factor": {
      "default_value": 0,
      "unit": "1"
    }
  }
```
### Notebook Usage for Data Management

Utilize the following Jupyter notebook to automatically create csv with needed input data.

- `04_get_carbon_emissions_annual_limit_adjustment.ipynb`: Retrieves the expected carbon noz stored for the adjustment of the annual carbon emissions limit.
Configure the data paths in the notebooks as follows:

- Input path: `output/data_n1_addition_0`
- Export path: `data_n1_adjusted`

### System Configuration for adjusted carbon budget

In the `system.py` file, update the system configuration to include the adjustment of the carbon emissions annual limit:

```python
# N-1 contingency
system['include_carbon_emissions_annual_limit_adjustment'] = True
```
`Run optimization of data_n1_adjusted`

## Needs a Fix

In the future, the following points should be addressed to improve the system:

### `technology.py`
- **n1_contingency_transport**: Fix the index for the return of the constraint.
- **n1_contingency_conversion**: Fix the index for the return of the constraint.
- **n1_contingency_no_failure_transport**: Fix the index for the return of the constraint.
- **n1_contingency_no_failure_conversion**: Fix the index for the return of the constraint.

### `transport_technology.py`
- **nominal_flow_transport**: Change the timestep from yearly to operational.

### `conversion_technology.py`
- **nominal_flow_conversion_input**: Change the timestep from yearly to operational.

