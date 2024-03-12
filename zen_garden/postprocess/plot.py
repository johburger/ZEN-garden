"""=====================================================================================================================
Title:        Unimportant notes
Created:      January-2024
Authors:      Johannes Burger (jburger@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Space for looking at data temporarily or just writing something down that can be deleted immediately
====================================================================================================================="""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import utm
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import os
from zen_garden.postprocess.results import Results
import numpy as np

# insert results path here:
results = Results('../../data/outputs/04_CCTS_WTE_Cement_01_optimal_design')

# define plot parameters

# should capacity be plotted, if false flows are plotted
capacity = True

# which scenario
scenario = None
#scenario = results.scenarios[6]

#Are there failure states and which one should be plotted
failure = False
failure_number = 4

# time step
time_step = 8

# should coversion technologies be plotted and which should be ignored.
plot_conversion_technologies = False
conversion_technologies_ignore =['heat_pump', 'emitter_wte']
# adjust square width
square_width = 0.63
# adjust line height
line_spacing = 0.03


# plot parameters
resize = True
legend= True
title = True
flow_annotations = False
fontsize = 6


def create_directed_graph(ax, edges, nodes):
    """create directed Graph from edges and connect them to the nodes"""
    G = nx.MultiDiGraph(ax=ax)
    posDict = {}
    max_flow = edges['flow'].max()
    for tech, edge, flow in zip(edges['technology'], edges['edge'], edges['flow']):
        node_from, node_to = edge.split("-")
        # add nodes
        if node_from not in list(G.nodes):
            position = (nodes[nodes['node'] == node_from]['lon'], nodes[nodes['node'] == node_from]['lat'])
            posDict[node_from] = position
            G.add_node(node_from, pos=position)
        if node_to not in list(G.nodes):
            position = (nodes[nodes['node'] == node_to]['lon'], nodes[nodes['node'] == node_to]['lat'])
            posDict[node_to] = position
            G.add_node(node_to, pos=position)
        # adjust weight
        weight = 1.4* flow/max_flow

        # change color for technologies
        linestyle = "solid"
        if tech == 'flowline':
            color = 'olive'
        elif tech == 'truck_batch':
            color = 'blue'
        elif tech == 'train_batch':
            color = 'purple'
        else:
            color = 'black'

        # add dashed arrow for failed transport technology
        if failure:
            failure_tech, failure_edge = failure_state.split(': ')
            if tech == failure_tech and edge == failure_edge:
                linestyle = ':'

        # add edges
        G.add_edge(node_from, node_to, technology=tech, color=color, label=tech.split('_')[0], weight=weight, linestyle=linestyle, flow=flow)
    return G


def offset_position(posA, posB, offset=0.02, shorten_value=1):
    """
    Apply an offset perpendicular to the line from posA to posB and shorten the arrow.
    """
    # Calculate the original direction vector and length
    direction = np.array(posB) - np.array(posA)
    length = np.linalg.norm(direction)

    # Normalize the direction vector
    direction = direction / length

    # Calculate the new length and the new end position
    new_length = length - shorten_value
    new_posB = np.array(posA) + direction * new_length

    # Apply the perpendicular offset
    dx, dy = direction[0], direction[1]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    perp_dir = np.array([-dy, dx]) / norm
    new_posA = np.array(posA) + perp_dir * offset
    new_posB = new_posB + perp_dir * offset

    return new_posA.tolist(), new_posB.tolist()


def draw_square_with_text(ax, x, y, width, height, node, technologies, flows, line_spacing, facecolor='white'):
    """
    Function to draw a square and annotate it
    """
    # Draw a square
    square = mpatches.Rectangle((x, y), width, height, facecolor=facecolor, edgecolor='black')
    ax.add_patch(square)
    node_name = node.replace('_', ' ')

    # Set fixed line spacing
    line_spacing = line_spacing # Adjust line spacing if needed
    current_y = y + height - line_spacing  # Start at top of box

    # Annotate the square with the node name in bold
    ax.text(x + 0.05 * width, current_y, node_name, ha='left', va='center', fontsize=fontsize-2, weight='bold')

    # Annotate the square with each technology and its flow
    for tech, flow in zip(technologies, flows):
        color = 'black'
        current_y -= line_spacing  # Move down to the next line
        # Remove underscores from technology names and split the name and value
        if failure:
            failure_tech, failure_node = failure_state.split(': ')
            if tech == failure_tech and node == failure_node:
                color = "red"
        tech_name = tech.replace('_', ' ')
        # Annotate with the name to the left
        ax.text(x + 0.05 * width, current_y, tech_name + ":", ha='left', va='center', fontsize=fontsize-2, color=color)
        # Annotate with the value to the right
        ax.text(x + .95 * width, current_y, f"{flow} kt/year", ha='right', va='center', fontsize=fontsize-2 , color=color)


# get nodes information
nodes = pd.read_csv(f"{results.get_analysis(scenario=scenario)['dataset']}/energy_system/set_nodes.csv")
nodes = nodes[nodes['node'].isin(results.get_system(scenario=scenario)['set_nodes'])]
nodes['utm_all'] = nodes.apply(lambda x: list(utm.from_latlon(x['lat'], x['lon'], 32, 'N')[0:2]), axis=1)
nodes['utm_lon'] = nodes['utm_all'].apply(lambda x: x[0])
nodes['utm_lat'] = nodes['utm_all'].apply(lambda x: x[1])
geodata = gpd.GeoDataFrame(nodes, crs='epsg:32632', geometry=gpd.points_from_xy(nodes['utm_lon'], nodes['utm_lat']))
geodata = geodata.to_crs(crs='epsg:4326')  # 'epsg:2056' correct CRS for Swiss map. crs='epsg:3035' CRS for Europe
nodes['lon'] = geodata['geometry'].apply(lambda k: k.x)
nodes['lat'] = geodata['geometry'].apply(lambda k: k.y)

# get swiss map
CH_json_url = 'https://labs.karavia.ch/swiss-boundaries-geojson/geojson/2020/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson'
CH_shapefile = gpd.read_file(CH_json_url)

# initiate plot
fig, ax = plt.subplots(figsize=(6, 4), dpi=300, layout="constrained")
CH_shapefile.plot(ax=ax, facecolor='linen', edgecolor='lightgrey', alpha=0.6, linewidth=0.5)

#import capacites of lows
if capacity:
    input_edges = results.get_total('capacity', scenario=scenario)
    input_nodes = results.get_total('capacity', scenario=scenario)
else:
    input_edges = results.get_total('flow_transport', scenario=scenario)
    input_nodes = results.get_total('flow_conversion_output', scenario=scenario)

years = input_edges.columns
years = [i - results.get_system(scenario=scenario)['reference_year'] for i in years] if years[0] > 100 else years
edges = input_edges[input_edges.sum(axis=1) >= 0.1]  # flow cut-off at 1 tons/year
nodes_info = input_nodes[input_nodes.sum(axis=1) >= 0.1]  # flow cut-off at 1 tons/year


year = years[time_step]
e = edges[year][edges[year] != 0].reset_index()
n = nodes_info[year][nodes_info[year] != 0].reset_index()

# clean up e and n
if failure:
    failure_state = e["failure_states"].unique()[failure_number]
    e = e[e["failure_states"] == failure_state].drop(columns="failure_states")
    n = n[n["failure_states"] == failure_state].drop(columns="failure_states").drop(columns="carrier")
if capacity:
    e = e.drop(columns="capacity_type")
    e = e[e.technology.isin(results.get_system(scenario=scenario)['set_transport_technologies'])]
    n = n[n.technology.isin(results.get_system(scenario=scenario)['set_conversion_technologies'])].drop(columns="capacity_type")
    n =n.rename(columns={'location': 'node'})
for tech in conversion_technologies_ignore:
    n = n.drop(n[n['technology'] == tech].index)

e.columns = ['technology', 'edge', 'flow']
# create directed graph and get information
G_with_capture = create_directed_graph(ax, e, nodes)
technologies = nx.get_edge_attributes(G_with_capture, "technology")
edge_colors = nx.get_edge_attributes(G_with_capture, "color")
position = nx.get_node_attributes(G_with_capture, "pos")
labels = nx.get_edge_attributes(G_with_capture, 'label')
weights = nx.get_edge_attributes(G_with_capture, "weight")
linestyles = nx.get_edge_attributes(G_with_capture, "linestyle")
flows = nx.get_edge_attributes(G_with_capture, "flow")

# add edges
for edge in list(G_with_capture.edges):
    posA = [x.values[0] for x in position[edge[0]]]
    posB = [x.values[0] for x in position[edge[1]]]

    # Apply offset if there are multiple technologies for the same edge and shorten arrow
    offset_value = 0.0
    shorten_value = 0.03
    tech = e['technology'].unique()

    if len(tech) > 1:
        position_tech = np.where(tech == technologies[edge])[0][0]
        offset_value = 0.015 * np.linspace(-1, 1, len(tech))[position_tech]

    posA, posB = offset_position(posA, posB, offset_value, shorten_value)

    style_kwds = {'arrowstyle': mpatches.ArrowStyle.Simple(head_width=max(5 * weights[edge],3), head_length=10, tail_width=weights[edge])}
    arrow = mpatches.FancyArrowPatch(posA=posA, posB=posB, color=edge_colors[edge], label=labels[edge], **style_kwds, linestyle=linestyles[edge])
    ax.add_patch(arrow)

    # add annotations for the actual flow values on the graph
    if flow_annotations:
        flow = flows[edge]
        if capacity:
            flow_kt_per_year = round(flow * 8.76, 1)
        else:
            flow_kt_per_year = round(flow / 1000, 1)

        # offset text
        offset_text = 0.035
        ha = 'left'

        # if two arrow between to nodes adjust offset of text depending on position
        if len(tech) > 1:
             position_tech = np.where(tech == technologies[edge])[0][0]
             if posA[1] > posB[1]:  # if the edge is going from left to right
                 offset_text = abs(offset_text) *  np.linspace(-1, 1, len(tech))[position_tech]
             if posA[1] < posB[1]:  # if the edge is going from left to right
                 offset_text = -abs(offset_text) * np.linspace(-1, 1, len(tech))[position_tech]

             if offset_text < 0:
                 ha = 'right'
        mid_point = np.mean([posA, posB], axis=0)
        plt.text(mid_point[0] + offset_text, mid_point[1], f"{flow_kt_per_year} kt/year", fontsize=fontsize - 2, ha=ha)  # Adjust fontsize as needed

#add nodes
node_patches = {}
for j, node in nodes.iterrows():
    pos = (node['lon'], node['lat'])
    size = 0.03
    circle = mpatches.Circle(xy=pos, radius=size, facecolor='orange', edgecolor='black',
                             linewidth=0.2)  # , label=sector)
    ax.add_patch(circle)
    ax.annotate(' '.join(node['node'].split('_')[:-1]), pos, fontsize=fontsize, ha='center', va='center', weight='bold')

# resize
if resize:
    padding = 0.5  # This adds some space around the items of interest
    x_min = min(nodes['lon']) - padding
    x_max = max(nodes['lon']) + padding
    y_min = min(nodes['lat']) - padding/2
    y_max = max(nodes['lat']) + padding/2

    # Set the plot limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

# add conversion technology information
if plot_conversion_technologies:
    # Get the current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Define the starting position for the squares (right side of the plot)
    right_edge = xlim[1] + 0.01 * (xlim[1] - xlim[0])  # This should be adjusted to push the squares to the edge
    square_width = square_width * (xlim[1] - xlim[0])  # This makes the square width smaller
    y_pos = ylim[1]
    line_spacing = line_spacing * (ylim[1] - ylim[0])
    # Draw squares and annotate with values from n
    # Iterate through nodes and their technologies
    for idx, node in enumerate(n['node'].unique()):
        node_data = n[n['node'] == node]

        # Get the list of technologies and their flows for the current node
        technologies = node_data['technology'].tolist()
        if capacity:
            flows = [f"{round(val * 8.76, 1)}" for val in node_data[year].tolist()]
        else:
            flows = [f"{round(val / 1000, 1)}" for val in node_data[year].tolist()]

        # Calculate the box height based on fixed line spacing instead of dynamic content
        num_lines = len(technologies) + 2 # One line for node name, rest for technologies
        box_height = num_lines * line_spacing  # Adjust as needed for padding


        # Calculate the height of the box based on the number of technologies
        # Calculate the starting y position for the current node's box
        y_pos = y_pos - (box_height + 0.02)

        # Draw the box and annotate it
        draw_square_with_text(ax, right_edge, y_pos, square_width, box_height, node, technologies, flows, line_spacing)

    ax.set_xlim(xlim[0], right_edge + square_width + 0.02)

# add legend
if legend:
    # Define the legend handles
    truck_handle = mpatches.FancyArrowPatch((0, 0), (0, 0), color='blue')
    train_handle = mpatches.FancyArrowPatch((0, 0), (0, 0), color='purple')
    flowline_handle = mpatches.FancyArrowPatch((0, 0), (0, 0), color='olive')
    flowline_failure_handle = mpatches.FancyArrowPatch((0, 0), (0, 0), color='olive', linestyle=':')
    node_handle = Line2D([0], [0], marker='o', color='w', label='Node', markerfacecolor='orange', markeredgecolor='orange')

    # Create the legend
    # Initialize the legend handles list with the handles that are always present
    legend_handles = [flowline_handle, train_handle, truck_handle]
    legend_labels = ['Flowline', 'Train', 'Truck']

    # Add the flowline failure handle only if 'failed' is True
    if failure:
        legend_handles.append(flowline_failure_handle)
        legend_labels.append('Flowline with failure')

    # Always add the node handle
    legend_handles.append(node_handle)
    legend_labels.append('Node')

    # Plot the legend
    ax.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize=fontsize)

if title:
    # Build the title string
    title_str = f"{results.name}"  # Assuming 'name' is an attribute or method returning '03_example_CCTS_N-1_03_CT_infisbal_adjusted'
    if capacity:
        title_str += " | Capacity\n"
    else:
        title_str += " | Flow\n"

    if failure:
        title_str += f" Failure State: {failure_state} | "

    title_str += f"Year: {year + results.get_system(scenario=scenario)['reference_year']}"  # Adjust the year based on the reference year of the system

    # Set the plot title
    plt.title(title_str, fontsize=fontsize)

plt.show()
