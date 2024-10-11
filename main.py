import sys
import os

import numpy as np
import pandas as pd 

from lib.solver import load_and_run_true_data_experiment
from lib.data import CoPresenceReader
from config_files.template_generation.generate_cfg import generate_cfg

if __name__ == '__main__':
    np.random.seed(420)  # reproducible seed for runnign the script in shells
    shell = int(sys.argv[1])
    print(f'shell: {shell}')        
    
    num_walker_subsets = 31

    loss = 'KL_divergence_forward'  # or 'KL_divergence_forward', 'mse', 'KL_divergence_backward
    model_type = 'SteadyStateSoftmaxMarkovChain'  # or 'SteadyStateSoftmaxMarkovChain', 'MultipleSteadyStateSoftmaxMarkovChain
    dataset = 'LyonSchool'  # or 'LyonSchool', 'InVS15', 'SFHH, Thiers13
    num_optimisation_trials = 3
    epochs = 2500 
    M = 3

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'datasets',
                                'co-presence',
                                f'tij_pres_{dataset}.dat')
    
    data = pd.read_csv(data_path, 
                        sep=' ', 
                        header=None,
                        names=['timestamp', 'node1', 'node2'])
    
    all_nodes = np.union1d(data['node1'].unique(), data['node2'].unique())
    node_subsets = [np.random.choice(all_nodes, 3, replace=False) for _ in range(num_walker_subsets)]




    if shell == 0:
        N = 12
        model_type = 'SteadyStateSoftmaxMarkovChain'
    elif shell == 1:
        N = 15
        model_type = 'SteadyStateSoftmaxMarkovChain'
    elif shell == 2:
        N = 18
        model_type = 'SteadyStateSoftmaxMarkovChain'
    elif shell == 3:
        N = 12
        model_type = 'MultipleSteadyStateSoftmaxMarkovChain'
    elif shell == 4:
        N = 15
        model_type = 'MultipleSteadyStateSoftmaxMarkovChain'
    elif shell == 5:
        N = 18
        model_type = 'MultipleSteadyStateSoftmaxMarkovChain'


    for i in range(num_walker_subsets):
        cfg = generate_cfg(N, M, loss, model_type, dataset, num_optimisation_trials, epochs, node_subsets[i], i)
        load_and_run_true_data_experiment(cfg)
        del cfg

