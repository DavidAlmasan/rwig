import os
from colorama import Fore, Back, Style

from ml_collections.config_dict import ConfigDict
import numpy as np
import torch 
import networkx as nx
from discreteMarkovChain import markovChain


N, M = 18, 3 # Number of underlying states, Number of walkers IN THE SUBSET OF NODES

loss = 'KL_divergence_forward'  # or 'KL_divergence_forward', 'mse', 'KL_divergence_backward
model_type = 'SteadyStateSoftmaxMarkovChain'  # or 'SteadyStateSoftmaxMarkovChain', 'MultipleSteadyStateSoftmaxMarkovChain
dataset = 'LyonSchool'  # or 'LyonSchool', 'InVS15', 'SFHH, Thiers13
"""--------------------Experiment configuration--------------------"""
CFG = ConfigDict()
CFG.NAME = f'[{dataset}][N={N}][M={M}][{loss}][{model_type}]'
CFG.VERBOSE = True
CFG.SEQUENCE_ALIGNMENT_HISTOGRAM_NAME = F'{CFG.NAME}_sequence_alignment_histogram.png'
CFG.GROUND_TRUTH_SEQUENCE_FIGURE_NAME = F'{CFG.NAME}_ground_truth_sequence.png'
CFG.GENERATED_SEQUENCE_FIGURE_NAME = F'{CFG.NAME}_generated_sequence.png'
CFG.SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../'
                                'experiments',
                                 CFG.NAME)
os.makedirs(CFG.SAVE_PATH, exist_ok=True)
if CFG.VERBOSE:
    print(f'Figures and metrics will be saved in {CFG.SAVE_PATH}')

CFG.NUM_TRIALS = 10
### DATA
CFG.DATA = ConfigDict()
## Parameters
CFG.DATA.TRUE_N = N  # True underlying states
CFG.DATA.TRUE_M = M  # True number of walkers
# Seq length doesnt matter if CFG.DATA.COMPUTE_STEADY_STATE_RWIG_DISTRIBUTION = True
CFG.DATA.WALKER_AGNOSTIC = False  # False = labelled graphs, True = unlabelled graphs
CFG.DATA.COMPUTE_STEADY_STATE_RWIG_DISTRIBUTION = True
CFG.DATA.NUM_SAMPLES_IN_SEQUENCE = [100, 1000, 10000]

def create_random_policy(N):
    P = np.sort(np.random.rand(N))
    P = P / P.sum()
    P = P.repeat(N).reshape(N, N)
    return torch.tensor(P, dtype=torch.double)

print(Fore.RED + f'Steady states need to be sorted from the experiment CFG file.')
print(Style.RESET_ALL)


##Initial states setup
# All walkers start from the first state
CFG.DATA.STEADY_STATES = None
CFG.DATA.DATASET = dataset
CFG.DATA.CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../'
                                'datasets',
                                'co-presence',
                                f'tij_pres_{dataset}.dat')
CFG.DATA.NODES = [1426, 1427, 1428]   # None for all walkers, int for sampling some nodes, list for specific nodes
### MODEL
CFG.MODEL = ConfigDict()
## Parameters
CFG.MODEL.N = CFG.DATA.TRUE_N  # Number of underlying states
CFG.MODEL.M = CFG.DATA.TRUE_M  # True number of walkers
CFG.MODEL.MULTIPLE_POLICIES = True if model_type == 'MultipleSteadyStateSoftmaxMarkovChain' else False
CFG.MODEL.PARAMETER_DICT = {
    'N': CFG.MODEL.N,
    'M': CFG.MODEL.M,
}

### Solver
CFG.SOLVER = ConfigDict()
## Parameters
CFG.SOLVER.OPTIMIZER = 'Adam'
CFG.SOLVER.LR = 0.05
CFG.SOLVER.EPOCHS = 2500
CFG.SOLVER.LOSS_TYPE = loss
CFG.SOLVER.TOLERANCE = 1e-10
CFG.SOLVER.CHECKPOINT_EVERY_EPOCH = 1

CFG.SOLVER.PARAMETER_DICT = {
    'optimizer': CFG.SOLVER.OPTIMIZER,
    'lr': CFG.SOLVER.LR,
    'epochs': CFG.SOLVER.EPOCHS,
    'loss_type': CFG.SOLVER.LOSS_TYPE,
    'save_path': CFG.SAVE_PATH,
    'tolerance': CFG.SOLVER.TOLERANCE,
    'checkpoint_every_epoch': CFG.SOLVER.CHECKPOINT_EVERY_EPOCH,
}