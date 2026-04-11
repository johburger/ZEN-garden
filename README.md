# ZEN-garden


![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FZEN-universe%2FZEN-garden%2Fmain%2Fpyproject.toml)

[![Static Badge](https://img.shields.io/badge/ZEN--garden_version-2.6.13-%23627313?labelColor=%23215CAF)](https://github.com/ZEN-universe/ZEN-garden) 

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ZEN-universe/ZEN-models/data_structure_check.yml?branch=food_energy_CH)](https://github.com/ZEN-universe/ZEN-models/actions)

<img src="https://github.com/ZEN-universe/ZEN-garden/assets/114185605/d6a9aca9-74b0-4a82-8295-43e6a78b8450" alt="drawing" width="200"/>

Welcome to the ZEN-garden! ZEN-garden is an optimization framework for energy transition pathways. 
It is currently used to model the electricity system, carbon capture, transport, and storage (CCTS) value chains, and the food-energy nexus. 
However, it is designed to be modular and flexible, and can be extended to model other types of energy systems, value chains or other network-based systems. 

This branch is an adapted version, suitable for modelling the food-energy nexus, specifically the 
integration of food production and renewable energy generation on agricultural land in Switzerland. 
It includes specific functionalities and modifications to enable this focus, which are not compatible 
with the main branch of ZEN-garden.

ZEN-garden is developed by the [Reliability and Risk Engineering Laboratory](https://www.rre.ethz.ch/) at ETH Zurich.
<hr style="height: 5px; background-color: black;">

## Quick Start
To get started with ZEN-garden, you can follow the instructions in the [installation guide](https://zen-garden.readthedocs.io/en/latest/files/quick_start/installation.html). 
Since this branch includes specific modifications, you will need to clone this repository and 
switch to the `Food_Energy_res1ha_CH` branch to use this dataset and model.

## Documentation
Please refer to the documentation of the ZEN-garden framework [on Read-the-Docs](https://zen-garden.readthedocs.io/en/latest/). 
The additional constraints and input parameters are described in the Supplementary Information of 
the associated publication (see below).

___
# Food and energy production on agricultural land in Switzerland

## 1. Description

### Purpose
This dataset supports a spatially explicit multi-objective optimisation study of Switzerland's 
agricultural land, evaluating synergies and trade-offs between food production and renewable 
electricity generation (solar PV and wind power). The dataset is intended for use with the 
ZEN-garden optimisation framework (Development_agriculture branch) to construct Pareto 
frontiers between food metabolisable energy supply and electricity generation under various 
dietary and land-use scenarios.

### Associated publication
Oeuvray, P., Burger, J., Nöhl, J., Becattini, V., and Mazzotti, M., Integrating food production 
and energy generation on agricultural land: a multi-objective optimisation analysis for 
Switzerland, 2026, *In review*.
The link to the publication will be added here once available.

The dataset used for the publication can be found here: http://hdl.handle.net/20.500.11850/797942

### Date
Creation date: 2026-03-27

## 2. Dataset Summary
A structured summary of key dataset attributes.

| Attribute                      | Description                               |
|--------------------------------|-------------------------------------------|
| **Spatial Scope**              | Switzerland, subsets of agricultural land |
| **Number of Nodes**            | 464 in each sample                        |
| **Temporal Scope**             | 2022                                      |
| **Number of Investment Years** | n.a.                                      |
| **Number of Time Steps**       | 1                                         |
| **Number of Technologies**     | 134                                       |
| **Number of Energy Carriers**  | 60                                        |


## 3. Framework Compatibility

This model does not run with the main branch of ZEN-garden! It has specific changes 
and functionalities to enable the integration of food and energy production on 
agricultural land, which are not compatible with the energy system focus of the general 
ZEN-garden framework. 

The model is designed to run with the Food_energy_CH branch of ZEN-garden, which is 
available here: https://github.com/johburger/ZEN-garden/tree/Food_energy_CH.

## 4. Comments

- **Objective function**: Food production (metabolisable energy) is maximised subject to an epsilon-constraint on minimum electricity generation, which is progressively tightened to construct the Pareto frontier. The minimum electricity generation level is controlled by the parameter `min_energy_production`.
- **Temporal scope**: No investment years or multi-period planning are considered — the model represents a single representative year (2022).
- **Food composition constraint**: The animal-rich scenario enforces minimum production proportional to current shares for all foodstuffs; the plant-rich scenario applies this constraint only to plant-based foodstuffs. The proportional minimum share for each foodstuff is set with `min_item_production`.
- **Land-use change constraint**: The maximum allowable share of agricultural land that can change activity is set with `activity_change_limit`.
- **Technology sets**: The datasets include 90 conversion technologies and 44 transport technologies. Conversion technologies cover: land-use transitions (e.g. `al_to_ar`, `na_to_none`), exclusive crop production (e.g. `cereals_excl`, `potato_excl`), agrivoltaic systems (`apv_*`), combined wind–crop systems (`wind_*`), exclusive solar and wind (`solar`, `wind`), crop-to-feed allocation processes (e.g. `grass_to_roughagefodder`, `oilseeds_to_concentratesfodder`), and animal and plant food processes (e.g. `milk_stream`, `cereals_stream`). Transport technologies handle spatial aggregation of crop production, feed categories, food energy and protein carriers, and electricity across nodes.
- **Carriers**: The dataset includes 60 carriers in total. These comprise 5 land-use carriers (e.g., `land_na`), 9 usable crop production carriers (e.g., `cereals_usableproduction`) and 11 intermediate crop carriers linking exclusive or combined production systems to downstream processes (e.g., `cereals_usableproduction_single`). Feed-related carriers include 5 by-product feed carriers (e.g., `cerealsbyproduct_fodder`) and the 3 aggregated feed category carriers (`roughagefodder`, `concentratesfodder`, `otherfodder`). Food output is tracked via 14 metabolisable energy carriers (e.g., `potato_energy`) and 12 protein carriers (e.g., `potato_protein`). The remaining carrier is `electricity`.

## 5. Contributors

- Pauline Oeuvray, poeuvray@ethz.ch
- Johannes Burger, jburger@ethz.ch
- Julian Nöhl, jnoehl@ethz.ch
- Viola Becattini, viola.becattini@esc.ethz.ch
- Marco Mazzotti, marco.mazzotti@ipe.mavt.ethz.ch