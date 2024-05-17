import os
import pandas as pd
import numpy as np
from zen_garden.postprocess.results import Results
from zen_garden.__main__ import run_module


if __name__ == '__main__':
    # %% Load the results
    # os.chdir('/Users/jburger/Documents/GitHub/ZEN-garden/tests/testcases/')
    os.chdir('F:/GitHub/ZEN-garden/data')
    model = '01_5_nodes_rev'
    run_module(args=['--config=./config.py', f'--dataset=./{model}'])
    results = Results(f'./outputs/{model}')
    capacity = results.get_total('capacity')
    capex = results.get_total('cost_capex')