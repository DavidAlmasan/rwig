import os
from colorama import Fore, Back, Style

from ml_collections.config_dict import ConfigDict
import numpy as np
import torch 
import networkx as nx
from discreteMarkovChain import markovChain


N, M = 2, 3
loss = 'KL_divergence_backward'  # or 'KL_divergence_forward', 'mse', 'KL_divergence_backward
model_type = 'MultipleSteadyStateSoftmaxMarkovChain'  # or 'SteadyStateSoftmaxMarkovChain', 'MultipleSteadyStateSoftmaxMarkovChain

"""--------------------Experiment configuration--------------------"""
CFG = ConfigDict()
CFG.NAME = f'[N={N}][M={M}][{loss}][{model_type}]'
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

CFG.NUM_TRIALS = 20
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

if model_type == 'MultipleSteadyStateSoftmaxMarkovChain':
    # True multiple policy setup
    CFG.DATA.POLICY = [create_random_policy(CFG.DATA.TRUE_N) for _ in range(CFG.DATA.TRUE_M)]
    print(f'True policy stochasticity check: {all([P.sum(axis=1).sum() == CFG.DATA.TRUE_N for P in CFG.DATA.POLICY])}')
    steady_state_vectors = []


    for P in CFG.DATA.POLICY:
        # calculate the steady state 
        mc = markovChain(P.numpy().T)  # Transpose P to get column-stochastic
        mc.computePi('linear')  # Compute stationary distribution
        pi = torch.tensor(mc.pi, dtype=torch.double)
        steady_state_vectors.append(pi)

    steady_state_vectors = sorted(steady_state_vectors, key=lambda x: x[0]) # Sort by the first element


else:
    P = create_random_policy(CFG.DATA.TRUE_N)
    print(f'True policy stochasticity check: {P.sum(axis=1).sum() == CFG.DATA.TRUE_N}')

    steady_state_vectors = []

    # calculate the steady state 
    mc = markovChain(P.numpy().T)  # Transpose P to get column-stochastic
    mc.computePi('linear')  # Compute stationary distribution
    pi = torch.tensor(mc.pi, dtype=torch.double)

    pi = torch.tensor(sorted(pi))
    steady_state_vectors = [pi] * CFG.DATA.TRUE_M

for i, pi in enumerate(steady_state_vectors):
    print(f'Stationary distribution for walker {i}: {pi.numpy()}')

##Initial states setup
# All walkers start from the first state
CFG.DATA.STEADY_STATES = torch.stack(steady_state_vectors).t().double()

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
CFG.SOLVER.EPOCHS = 10000
CFG.SOLVER.LOSS_TYPE = loss
CFG.SOLVER.TOLERANCE = 1e-10
CFG.SOLVER.CHECKPOINT_EVERY_EPOCH = 100

CFG.SOLVER.PARAMETER_DICT = {
    'optimizer': CFG.SOLVER.OPTIMIZER,
    'lr': CFG.SOLVER.LR,
    'epochs': CFG.SOLVER.EPOCHS,
    'loss_type': CFG.SOLVER.LOSS_TYPE,
    'save_path': CFG.SAVE_PATH,
    'tolerance': CFG.SOLVER.TOLERANCE,
    'checkpoint_every_epoch': CFG.SOLVER.CHECKPOINT_EVERY_EPOCH,
}