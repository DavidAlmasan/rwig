from copy import deepcopy
import sys

from tqdm import tqdm
sys.path.append('..')
from functools import reduce
from math import factorial
from typing import List
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
from discreteMarkovChain import markovChain
import seaborn as sns

from lib.contact_graph_models import RWIG

class SequenceSimulator:
    def __init__(self, walkers_initial_states, policy, contact_graph_model, verbose=False, **kwargs):
        self.walkers_initial_states = walkers_initial_states.double()
        self.N = walkers_initial_states.shape[0] # Number of states
        self.M = self.walkers_initial_states.shape[-1] # Number of walkers
        if verbose:
            print(f'Number of states: {self.N}, Number of walkers: {self.M}')
        self.walker_indices = list(range(self.M))

        ### Define the step transitions based on the type of policy
        self.walker_transition_function = None  # Placeholder for step function defined below 
        if isinstance(policy, torch.Tensor):
            self.policy = policy.double()
            self.walker_transition_function = self._single_markov_policy_step

        elif isinstance(policy, nn.Module):
            self.policy = policy.double()
            self.walker_transition_function = self._single_general_policy_step

        elif isinstance(policy, List):
            assert len(policy) == self.M
            self.policy = [p.double() for p in policy]

            if isinstance(policy[0], torch.Tensor):
                self.walker_transition_function = self._multiple_markov_policy_step
            elif isinstance(policy[0], nn.Module):
                self.walker_transition_function = self._multiple_general_policy_step
            else:
                raise ValueError('Invalid policy')
        else:
            raise ValueError('Invalid policy')

        self.contact_graph_model = contact_graph_model
        self.walker_agnostic = self.contact_graph_model.walker_agnostic
    
        self.verbose = verbose

        self.steady_state_rwig_distribution = None
        if kwargs.get('compute_steady_state_rwig_distribution', False):
            self.steady_state_rwig_distribution = self.contact_graph_model.generate_contact_network_distribution(self.walkers_initial_states,
                                                                                                                 draw_graphs=False)

    # Step transition functions based on policy types
    def _single_markov_policy_step(self, policy, walker_states_proba_vector):
        return torch.matmul(policy, walker_states_proba_vector)
    
    def _single_general_policy_step(self, policy, walker_states_proba_vector):
        return policy(walker_states_proba_vector.T).T  # transpose to be in torch format (M, N). Transpose again to get it back to (N, M)

    def _multiple_markov_policy_step(self, policy, walker_states_proba_vector):
        walker_states_proba_vector = [torch.matmul(P, walker_states_proba_vector[:, walker_id])
                                        for walker_id, P in enumerate(policy)]
        return torch.stack(walker_states_proba_vector, dim=1)

    def _multiple_general_policy_step(self, policy, walker_states_proba_vector):
        walker_states_proba_vector = [policy[walker_id](walker_states_proba_vector[:, walker_id].unsqueeze(-1).T).squeeze()
                                        for walker_id in range(self.M)]
        return torch.stack(walker_states_proba_vector, dim=1)
    
    def generate_realisation(self, walkers_state_probabilities):
        walker_states = [np.random.choice(range(self.N),
                                        p=walkers_state_probabilities[:, walker_index].detach().clone().numpy()) 
                            for walker_index in range(self.M)]

        walker_partition = self.contact_graph_model.generate_walker_partition(walker_states)
        G_k = self.contact_graph_model.generate_contact_network(walker_partition)

        return G_k

    def generate_data(self, t, no_graphs=False, no_grads=False):
        G_T, S_T = [], []

        # Walker state proba must be in the format (N, M)
        walkers_state_probabilities = self.walkers_initial_states.clone()

        for _ in range(t):
            G_k = self.generate_realisation(walkers_state_probabilities)
            S_T.append(walkers_state_probabilities)
            G_T.append(G_k)
            # Next step
            walkers_state_probabilities = self.walker_transition_function(self.policy, walkers_state_probabilities)
        self.G_T = G_T
        self.S_T = S_T


        if no_grads:
            S_T = [s.detach() for s in S_T]
        if no_graphs:
            return S_T
    
        return G_T, S_T
            
    def compute_sequence_probability(self, G_T, S_T):
        probas = []
        for G_k, S_k in zip(G_T, S_T):
            probas.append(self.contact_graph_model.generate_contact_network_probability(G_k, S_k))
        return probas
    

    def plot_simulated_sequence(self, G_T, k=5, plot_last=False, title='', figpath=None):
        k = min(k, len(G_T))
        print(f'Plotting {k} contact networks using matrix: {title}. Plotting last networks: {plot_last}.')
        fig, axes = plt.subplots(nrows = 1, ncols = k)
        #change figsize of the figure
        fig.set_size_inches(3*k, 4)
        time_index = list(range(len(G_T)))
        if plot_last:
            G_T = G_T[-k:]
            time_index = time_index[-k:]
        else:
            G_T = G_T[:k]
            time_index = time_index[:k]
        # label of y axis
        for index, (k, G_k) in enumerate(zip(time_index, G_T)):
            # Plot vertical lines in between subplots
            if index > 0:
                line = plt.Line2D((index / len(G_T), index / len(G_T)),(.1,.9), color="k", linewidth=3)
                fig.add_artist(line)
            axes[index].set_title(f'$G_{k}$')
            nx.draw(G_k, with_labels=not self.walker_agnostic, 
                    pos=nx.circular_layout(G_k),
                    ax=axes[index],
                    node_size = 200,
                    width=2,
                    margins=0.5)
        fig.tight_layout()
        fig.suptitle(title)
        return fig
    
    def plot_simulated_sequence_with_proba(self, G_T, p_T, k=5, title='', figpath=None):
        k = min(k, len(G_T))
        print(f'Plotting {k} contact graphs for {title}.')
        fig, axes = plt.subplots(nrows = 2, ncols = k)
        #change figsize of the figure
        fig.set_size_inches(2*k, 4.5)
        axes[0, 0].set_ylabel(r'$\Pr[G_k]$')
        time_index = list(range(len(G_T)))
        G_T = G_T[:k]
        p_T = p_T[:k]
        time_index = time_index[:k]
        # label of y axis
        colors = sns.color_palette('pastel') * 1000
        proba_color = colors[0]
        for index, (k, G_k, p_k) in enumerate(zip(time_index, G_T, p_T)):
            axes[0, index].set_xticks([])
            # remove the frame of the chart
            axes[0, index].spines['top'].set_visible(False)
            axes[0, index].spines['right'].set_visible(False)
            if index > 0:
                axes[0, index].set_yticks([])
                axes[0, index].spines['left'].set_visible(False)
            axes[0, index].set_ylim(0, 1)

            axes[0, index].bar(0, p_k, color=proba_color)
            # Draw on top of the bar the actual probability
            axes[0, index].text(0, p_k, f'{p_k:.2f}', ha='center', va='bottom')
            # color the nodes 
            # get cliques 
            cliques = sorted(list(list(x) for x in nx.connected_components(G_k)), key=lambda x: -len(x))
            graph_colors = {node: colors[color_index+1] for color_index, clique in enumerate(cliques) for node in clique}          
            nx.draw(G_k, with_labels=not self.walker_agnostic, 
                    pos=nx.circular_layout(G_k),
                    ax=axes[1, index],
                    node_size = 200,
                    width=2,
                    margins=0.5,
                    node_color=[graph_colors[node] for node in G_k.nodes])
            axes[1, index].set_title(f'$k={k}$')
        fig.tight_layout()
        fig.suptitle(title)
        if figpath:
            fig.savefig(figpath)
        # return fig
 

class SteadyStateDistributionGenerator:
    """
    Generate the steady state distribution of the rwig graphs given the steady states of the walkers
    """
    def __init__(self, steady_stats, contact_graph_model, verbose=False, **kwargs):
        self.steady_stats = steady_stats.double()
        (self.N, self.M) = steady_stats.shape # Number of states, # Number of walkers

        if verbose:
            print(f'Number of states: {self.N}, Number of walkers: {self.M}')
        self.walker_indices = list(range(self.M))

        self.contact_graph_model = contact_graph_model
        self.walker_agnostic = self.contact_graph_model.walker_agnostic
    
        self.verbose = verbose

        self.steady_state_rwig_distribution = self.contact_graph_model.generate_contact_network_distribution(self.walkers_initial_states,
                                                                                                                draw_graphs=False)

   
    
            
 
    def plot_simulated_sequence(self, G_T, k=5, plot_last=False, title='', figpath=None):
        k = min(k, len(G_T))
        print(f'Plotting {k} contact networks using matrix: {title}. Plotting last networks: {plot_last}.')
        fig, axes = plt.subplots(nrows = 1, ncols = k)
        #change figsize of the figure
        fig.set_size_inches(3*k, 4)
        time_index = list(range(len(G_T)))
        if plot_last:
            G_T = G_T[-k:]
            time_index = time_index[-k:]
        else:
            G_T = G_T[:k]
            time_index = time_index[:k]
        # label of y axis
        for index, (k, G_k) in enumerate(zip(time_index, G_T)):
            # Plot vertical lines in between subplots
            if index > 0:
                line = plt.Line2D((index / len(G_T), index / len(G_T)),(.1,.9), color="k", linewidth=3)
                fig.add_artist(line)
            axes[index].set_title(f'$G_{k}$')
            nx.draw(G_k, with_labels=not self.walker_agnostic, 
                    pos=nx.circular_layout(G_k),
                    ax=axes[index],
                    node_size = 200,
                    width=2,
                    margins=0.5)
        fig.tight_layout()
        fig.suptitle(title)
        return fig
 

class CoPresenceReader:
    def __init__(self, fp) -> None:
        self.data = pd.read_csv(fp, 
                                sep=' ', 
                                header=None,
                                names=['timestamp', 'node1', 'node2'])
        print(f'Loaded {len(self.data["timestamp"].unique())} timestamps')
        self.nodes = np.union1d(self.data['node1'].unique(), self.data['node2'].unique())
        self.base_G = nx.empty_graph(self.nodes)
        self.G_T = self.create_contact_graph_sequence(self.data)
        print(f'Loaded {len(self.G_T)} contact graphs')

    def get_records(self):
        return self.data
    
    def get_nodes(self):
        return self.nodes
    
    def filter_graphs_by_nodes(self, nodes, verbose=False):
        if verbose: print(f'Filtering graphs by nodes: {nodes}')
        return [g.subgraph(nodes) for g in self.G_T]
    
    def create_contact_graph_sequence(self, df):
        print('Creating contact graph sequence')
        return df.groupby('timestamp').apply(self._create_contact_graph)

    def _create_contact_graph(self, df):
        G = deepcopy(self.base_G)
        for _, row in df.iterrows():
            G.add_edge(row['node1'], row['node2'])
        return G

    @staticmethod
    def graph_sequence_to_partition_sequence(G_T):
        return pd.Series([str(sorted(sorted(clique) for clique in list(nx.connected_components(G))))     
                for G in G_T]) 
    
    @staticmethod
    def generate_transition_matrix(G_T):
        unique_graphs = G_T.unique().tolist()
        print(f'Unique graphs: {unique_graphs}')
        N = len(unique_graphs)
        P = np.zeros((N, N))
        for i in tqdm(range(len(G_T)-1)):
            P[unique_graphs.index(G_T.iloc[i])][unique_graphs.index(G_T.iloc[i+1])] += 1
        # normalise to be row stochastic
        norm_factor = P.sum(axis=1).repeat(N).reshape(N, N)
        norm_factor = np.where(norm_factor == 0, 1, norm_factor)
        P = P / norm_factor
        return P
        
    
    
def test_dataloader(k=10, walker_agnostic=True):
    N = 4
    M = 8
    # Initial states
    init_state = torch.zeros(N)
    init_state[0] = 1
    walkers_initial_states = init_state.repeat(M).reshape(M, N).t().double()

    # 4 states 8 walkers, complete graph topology
    N = 4
    g = nx.complete_graph(N)
    P = nx.to_numpy_array(g)
    normalising_factor = np.where(P.sum(axis=0) == 0, 1, P.sum(axis=0))
    P = P / normalising_factor
    P = torch.tensor(P, dtype=torch.double)
    # Transition matrix
    P = torch.tensor([[0.1, 0.1, 0.1, 0.1], 
                      [0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1],
                      [0.7, 0.7, 0.7, 0.7]], dtype=torch.double)
    
    policies = [P.clone() for _ in range(M)]
    
    contact_graph_model = RWIG(walker_agnostic)
    data_generator = SequenceSimulator(walkers_initial_states, 
                                       policies,
                                       contact_graph_model=contact_graph_model)
    
    G_T, S_T = data_generator.generate_data(k)

    p_T = data_generator.compute_sequence_probability(G_T, S_T)
    data_generator.plot_simulated_sequence_with_proba(G_T, p_T, k)
    plt.show()

if __name__ == "__main__":
    # test_dataloader(10, False)

    dataset_name = 'LyonSchool'
    fp = f'../datasets/co-presence/tij_pres_{dataset_name}.dat'
    reader = CoPresenceReader(fp)

    nodes = reader.nodes[:2]
    G_T = reader.filter_graphs_by_nodes(nodes)
    G_T_partition = CoPresenceReader.graph_sequence_to_partition_sequence(G_T)
    P = CoPresenceReader.generate_transition_matrix(G_T_partition)
    print(len(P))
    print(P)
    print('----')

    nodes = reader.nodes[1:3]
    G_T = reader.filter_graphs_by_nodes(nodes)
    G_T_partition = CoPresenceReader.graph_sequence_to_partition_sequence(G_T)
    P = CoPresenceReader.generate_transition_matrix(G_T_partition)
    print(len(P))
    print(P)
    print('----')

    nodes = reader.nodes[:3]
    G_T = reader.filter_graphs_by_nodes(nodes)
    G_T_partition = CoPresenceReader.graph_sequence_to_partition_sequence(G_T)
    P = CoPresenceReader.generate_transition_matrix(G_T_partition)
    print(len(P))
    print(P)