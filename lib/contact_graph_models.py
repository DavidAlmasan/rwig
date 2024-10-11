"""
Created on Wed March 20, 2024 

@author: DavidAlmasan
"""
import itertools
import sys
sys.path.append('..')
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from pprint import pprint
from functools import reduce
from math import factorial
from scipy import stats, special
import torch 
import torch.nn as nn
from discreteMarkovChain import markovChain
from more_itertools import set_partitions
import seaborn as sns 
from sympy import symbols, expand, collect

from lib.utils.utils import PartitionManager, EfficientPartitionManager, KPartitionManager


class ContactMatrixGenerator:
    def __init__(self, csv_path, nodes=None, convert_to_rwig=False) -> None:
        if csv_path.endswith('.gz'):
            self.df = pd.read_csv(csv_path, sep=' ',header=None, compression='gzip')
        else:
            self.df = pd.read_csv(csv_path, sep=' ',header=None)
        self.df = self.df

        self.df.columns = ['time', 'id1', 'id2']
        self.nodes = np.union1d(self.df['id1'].unique(), self.df['id2'].unique())
        self.G_T = self.generate_contact_matrix(convert_to_rwig=convert_to_rwig)
        if nodes is None:
            self.subgrap_nodes = self.nodes
        elif isinstance(nodes, int):
            self.subgrap_nodes = np.random.choice(self.nodes, nodes)
        elif isinstance(nodes, list) or isinstance(nodes, np.ndarray):
            self.subgrap_nodes = nodes
        else:
            raise ValueError('Nodes should be None, int or list')
        self.M = len(self.subgrap_nodes)
        self.G_T = self.filter_graphs_by_nodes()
        mapping = {node: i for i, node in enumerate(self.subgrap_nodes)}
        self.G_T = {t: nx.relabel_nodes(G, mapping) for t, G in self.G_T.items()}
        self.subgrap_nodes = list(range(self.M))
    
    def filter_graphs_by_nodes(self):
        timestamps = sorted(self.G_T.keys())
        return {t: self.G_T[t].subgraph(self.subgrap_nodes) for t in timestamps}
    
    def generate_contact_matrix(self, convert_to_rwig):
        G_T = {
            t: nx.empty_graph(self.nodes)
            for t in self.df['time'].unique()
        }   
        
        for row in self.df.itertuples(index=False):
            G_T[row.time].add_edge(row.id1, row.id2, length=1)

        if convert_to_rwig:
            # get connected components
            for t in G_T.keys():
                for clique in nx.connected_components(G_T[t]):
                    if len(clique) > 1:
                        extra_edges = nx.complete_graph(clique).edges()
                        G_T[t].add_edges_from(extra_edges)
                    
        return G_T
    
    def get_contact_matrix_timeseries(self):
        return self.G_T
    
    def plot_simulated_sequence(self, G_T, k=5):
        sorted_keys = sorted(G_T.keys())
        k = min(k, len(G_T))
        print(f'Plotting {k} contact networks')
        fig, axes = plt.subplots(nrows = 1, ncols = k)
        #change figsize of the figure
        fig.set_size_inches(3*k, 4)
        # label of y axis
        for index in range(k):
            G_k = G_T[sorted_keys[index]]
            # Plot vertical lines in between subplots
            if index > 0:
                line = plt.Line2D((index / len(G_T), index / len(G_T)),(.1,.9), color="k", linewidth=3)
                fig.add_artist(line)
            axes[index].set_title(f'$G_{index}$')
            nx.draw(G_k, with_labels=True, 
                    pos=nx.circular_layout(G_k),
                    ax=axes[index],
                    node_size = 200,
                    width=2,
                    margins=0.5)
        fig.tight_layout()
        return fig

    def get_min_Markov_chain(self):
        """
        Return the minimum Markov chain state space (N) that can be constructed from the contact matrix
        """
        N = 0
        for G in self.G_T.values():
            num_connected_components = len(list(nx.connected_components(G)))
            N = max(N, num_connected_components)
        return N
    
    def _assert_is_rwig_graph(self, g):
        return all([nx.is_isomorphic(g.subgraph(clique), nx.complete_graph(len(clique))) for clique in nx.connected_components(g)])
    
    def assert_is_rwig_timeseries(self):
        num_rwig_graphs = 0
        for G in tqdm(self.G_T.values()):
            if self._assert_is_rwig_graph(G):
                num_rwig_graphs += 1
        return num_rwig_graphs / len(self.G_T), (num_rwig_graphs, len(self.G_T))

    def plot_num_cliques_distribution(self, normalize_by_node_count=True, remove_singletons=False):
        """
        Plot the distribution of number of cliques in the graph
        """
        num_cliques = []
        for G in self.G_T.values():
            connected_components = list(nx.connected_components(G))
            num_nodes = G.number_of_nodes()
            if remove_singletons:
                connected_components = [clique for clique in connected_components if len(clique) > 1]
            if normalize_by_node_count:
                num_cliques.append(len(connected_components) / num_nodes)  #num cliques as fraction of nodes
            else:
                num_cliques.append(len(connected_components))
        
        return num_cliques
        
    def plot_clique_size_distribution(self, normalize_by_node_count=True, remove_singletons=False, custom_bins=None, noplot=False):
        """
        Plot the distribution of clique sizes
        """
        all_num_nodes = []
        clique_sizes = []
        for G in self.G_T.values():
            connected_components = list(nx.connected_components(G))
            num_nodes = G.number_of_nodes()
            all_num_nodes.append(num_nodes)
            for clique in connected_components:
                if remove_singletons and len(clique) == 1:
                    continue
                if normalize_by_node_count:
                    clique_sizes.append(len(clique) / num_nodes)  #clique size as fraction of nodes
                else:   
                    clique_sizes.append(len(clique))  
        

        clique_sizes = np.array(clique_sizes)
        max_clique_size = max(clique_sizes)
        if custom_bins is not None:
            # expect 3 bins: small, medium, large
            
            for clique_size_id, bin_edges in enumerate(custom_bins):
                clique_sizes = np.where((clique_sizes >= bin_edges[0]) & (clique_sizes <= bin_edges[1]), clique_size_id, clique_sizes)

        
        if noplot:
            return clique_sizes
        
        color = sns.color_palette('pastel')[0]
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 5))
        # adjust figsize
        fig.set_size_inches(5, 5)
        # ax.hist(clique_sizes, color='orange', density=True)
        sns.histplot(clique_sizes, kde=False, color=color, stat='proportion', ax=ax, discrete=True, alpha = 1)
                    #  binwidth=0.01)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Clique size')
        if custom_bins is not None:
            # different tcks for custom bins
            ax.set_xticks(range(3), [f'[{custom_bins[0][0]}-{custom_bins[0][1]}]', 
                           f'[{custom_bins[1][0]} - {custom_bins[1][1]}]', 
                           f'[{custom_bins[2][0]} - {max_clique_size}]'])
        if normalize_by_node_count:
            ax.set_xlabel('Clique size normalized by node count')
        else:
            ax.set_xlabel('Clique size')

        print(f'Max clique size: {max(clique_sizes)}')
        print(f'Unique clique sizes: {np.unique(clique_sizes)}')
        print(f'Min, max num nodes: {min(all_num_nodes)}, {max(all_num_nodes)}')

        return clique_sizes


    def plot_clique_size_heatmap_timeseries(self, remove_singletons=False, custom_bins=None):
        """
        Plot the distribution of clique sizes
        """
        clique_sizes_timeseries = []
        max_clique_size = 1
        for G in self.G_T.values():
            connected_components = list(nx.connected_components(G))
            clique_size_distribution = [len(clique) for clique in connected_components]
            if remove_singletons:
                clique_size_distribution = [size for size in clique_size_distribution if size > 1]
            
            clique_sizes_timeseries.append(clique_size_distribution)
            max_clique_size = max(max_clique_size, max(clique_size_distribution))

        if custom_bins is None:
            custom_bins = max_clique_size-1
        else:
            custom_bins = [edges[0] for edges in custom_bins] + [max_clique_size]

        histograms = []
        for G_k in clique_sizes_timeseries:
            hist, _ = np.histogram(G_k, bins=custom_bins, density=True)
            histograms.append(hist) 
        histograms = np.array(histograms)
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        #heatmap using seaborn
        sns.heatmap(histograms, cmap='YlGnBu', ax=ax)
        ax.set_ylabel('Time')
        # remove x ticks 
        ax.set_xticks(range(len(custom_bins)-1), [f'[{custom_bins[i]} - {custom_bins[i+1]}]' for i in range(len(custom_bins) - 1)])   
        ax.set_xlabel('Clique size')



class RWIG:
    def __init__(self, walker_agnostic=False) -> None:
        self.walker_agnostic = walker_agnostic
        self.N = None
        self.M = None
        
    def _sigma_product(self, clique, S_k):
        clique = list(reduce(np.union1d, clique))
        sigma_product = torch.prod(S_k[:, clique], dim=1).sum()
        # print(f'[clique: {clique}] Sigma product: {sigma_product}')
        return sigma_product

    def generate_walker_partition(self, walkers_states):
        """
        Simulate a realisation of the random walkers
        """
        inverse_mapping = {}
        for walker, state in enumerate(walkers_states):
            if state not in inverse_mapping.keys():
                inverse_mapping[state] = [walker]
            else:
                inverse_mapping[state].append(walker)

        if self.walker_agnostic:
            # clique sizes only!!!
            walker_partition = sorted([len(clique) for clique in inverse_mapping.values()])
        else:
            walker_partition = sorted([sorted(clique) for clique in inverse_mapping.values()])
        return str(walker_partition)

    def generate_walker_partition_from_graph(self, G_k):
        """
        Generate the walker partition from the graph
        """
        if isinstance(G_k, nx.Graph):
            cliques = list(list(x) for x in nx.connected_components(G_k))
        else:
            cliques = G_k
        walker_partition = sorted([sorted(clique) for clique in cliques])
        return str(walker_partition)
    

    def generate_contact_network(self, walker_partition, return_as_partition=False):
        """
        walker_partition = Partition on all walkers
        """
        if isinstance(walker_partition, str):
            walker_partition = eval(walker_partition)

        G = nx.Graph()
        if self.walker_agnostic:
            cliques = []
            clique_walker_start_index = 0
            for clique_size in walker_partition:
                cliques.append(list(range(clique_walker_start_index, 
                                          clique_walker_start_index + clique_size)))
                clique_walker_start_index += clique_size
        else:
            cliques = walker_partition

        if return_as_partition:
            return cliques
        for clique in cliques:
            G = nx.compose(G, nx.complete_graph(clique))
        return G

    def generate_contact_network_probability(self, G_k, S_k):
        """
        (Theorem 2): RWIG Logic for computing the probability of a contact network 
        """
        num_walkers = S_k.shape[1]
        if isinstance(G_k, nx.Graph):
            cliques = list(list(x) for x in nx.connected_components(G_k))
        else:
            cliques = G_k
        amassed_clique_graphs = PartitionManager(cliques).return_partition(with_union=False)
        G_k_proba = torch.tensor(0, dtype=torch.double)
        for g_pi in amassed_clique_graphs:
            sigma_product = torch.tensor(1., dtype=torch.double)
            num_automorphisms = torch.prod(torch.tensor([factorial(len(clique)-1) for clique in g_pi])).double()
            sign = torch.prod(torch.tensor([(-1)**(len(clique)-1) for clique in g_pi])).double()
            for clique in g_pi:
                sigma_product *= self._sigma_product(clique, S_k)
            G_k_proba += sign * sigma_product * num_automorphisms

        if self.walker_agnostic:
            clique_sizes = [len(clique) for clique in cliques]
            clique_repetitions = list({clique_size: clique_sizes.count(clique_size) for clique_size in set(clique_sizes)}.values())
            scale_factor = factorial(num_walkers) / np.prod([factorial(c_sz) for c_sz in clique_sizes]) / np.prod([factorial(c_rpt) for c_rpt in clique_repetitions])    # permutations without repetition of walkers that are same

            G_k_proba *= torch.tensor(scale_factor, dtype=torch.double)

        if G_k_proba < 0:
            print(f'Negative probability: {G_k_proba}')
            return torch.tensor(0, dtype=torch.double)
        return G_k_proba.squeeze()
    
    def _generate_sample_space(self, S_k):
        (self.N, self.M)= S_k.shape
        walker_indices = list(range(self.M))
        contact_network_sample_space = EfficientPartitionManager(walker_indices, self.N).return_partition()
        if self.walker_agnostic:
            contact_network_sample_space = [partition
                                            for partition in 
                                            {
                                                tuple(sorted([len(clique) for clique in amassed_graph_partition])):None
                                                for amassed_graph_partition in contact_network_sample_space
                                            }.keys()]

        return contact_network_sample_space

    def generate_contact_network_distribution(self, S_k, m_clique_graph=None, draw_graphs=True, contact_network_sample_space=None):
        """
        Generate the distribution of contact networks
        probably move this to utils.utils.PartitionManager
        draw_graphs: if True, compute nx graphs too
        """
        if contact_network_sample_space is None:
            contact_network_sample_space = self._generate_sample_space(S_k)
            print(f'Generated {len(contact_network_sample_space)} contact networks')
        self.contact_networks = {
            str(walker_partition): {
                'partition': walker_partition,
                'probability': self.generate_contact_network_probability(self.generate_contact_network(walker_partition, return_as_partition=True), S_k)
            } for walker_partition in contact_network_sample_space
        }

        if draw_graphs:
            for walker_partition in self.contact_networks.keys():
                self.contact_networks[walker_partition]['network'] = self.generate_contact_network(walker_partition)
        return self.contact_networks
    
    def generate_empirical_contact_network_distribution(self, G_T):
        self.contact_networks = {}
        G_T_partitions = [self.generate_walker_partition_from_graph(G_k) for G_k in G_T.values()]
        unique_graphs, graph_counts = np.unique(G_T_partitions, return_counts=True)
        for walker_partition, count in zip(unique_graphs, graph_counts):
            G_k = self.generate_contact_network(walker_partition)
            if walker_partition not in self.contact_networks.keys():
                self.contact_networks[walker_partition] = {
                    'partition': walker_partition,
                    'probability': count / len(G_T),
                    'network': G_k
                }
        return self.contact_networks
    
    @staticmethod
    def plot_contact_graph_sequence(G_T, k=3):
        """
        Plot the top k contact networks one below the other
        """
        k = min(k, len(G_T))
        G_T = G_T[:k]

        labels = [f'k={i}' for i in range(k)]

        # Plotting the bar chart
        # fig = plt.figure(figsize=(5*k, 3))
        fig, axes = plt.subplots(nrows = 1, ncols = k)
        axes = axes.flatten()
        #change figsize of the figure
        fig.set_size_inches(2*k, 4.5)
        # label of y axis
        y_label = r'$G_k$'
        axes[0].set_ylabel(y_label)
        # colors 
        colors = sns.color_palette('pastel', 1000)
        for index, G, label in zip(range(k), G_T, labels):
            
            # color the nodes 
            # get cliques 
            cliques = sorted(list(list(x) for x in nx.connected_components(G)), key=lambda x: -len(x))
            graph_colors = {node: colors[color_index+1] for color_index, clique in enumerate(cliques) for node in clique}
            # color last graph with grey
            
            nx.draw(G, with_labels=True, 
                    pos=nx.circular_layout(G),
                    ax=axes[index],
                    node_size = 200,
                    width=2,
                    margins=0.5,
                    node_color=[graph_colors[node] for node in G.nodes])
            
            if index < len(G_T) - 1:
                axes[index].spines['right'].set_visible(True)
        #title on figure
        # print(f'[N={self.N}][M={self.M}] ')
        # fig.suptitle(f'Probability distribution of the steady state contact graph ' + r'$\Pr[G_\infty]$')
        return fig
    
    def plot_topk_contact_networks(self, k):
        """
        Plot the top k contact networks one below the other
        """
        sorted_contact_networks = sorted(self.contact_networks.items(), key=lambda x: x[1]['probability'], reverse=True)
        k = min(k, len(sorted_contact_networks))

        labels = [entry[0] for entry in sorted_contact_networks][:k] 
        probabilities = [entry[1]['probability'] for entry in sorted_contact_networks][:k]
        networks = [entry[1]['network'] for entry in sorted_contact_networks][:k]
        others_label_used = False
        unused_graphs = len(self.contact_networks) - k
        if k < len(self.contact_networks):
            labels.append('Other')
            other_probability = 1 - sum(probabilities)
            probabilities.append(other_probability)
            networks.append(nx.empty_graph(self.M))
            k += 1
            others_label_used = True
            


        # Plotting the bar chart
        # fig = plt.figure(figsize=(5*k, 3))
        fig, axes = plt.subplots(nrows = 2, ncols = k)
        #change figsize of the figure
        fig.set_size_inches(2*k, 4.5)
        # label of y axis
        y_label = r'$\Pr[G^{u}_{\infty}]$' if self.walker_agnostic else r'$\Pr[G_\infty]$'
        axes[0, 0].set_ylabel(y_label)
        # colors 
        colors = sns.color_palette('pastel')
        colors = list(colors) * (1000)
        proba_color = colors[0]
        for index, G, proba in zip(range(k), networks, probabilities):
            # proba_color = colors[0] if index < k-1 else colors[1]
            
            # remove the x and y ticks
            axes[0, index].set_xticks([])
            # remove the frame of the chart
            axes[0, index].spines['top'].set_visible(False)
            axes[0, index].spines['right'].set_visible(False)
            if index > 0:
                axes[0, index].set_yticks([])
                axes[0, index].spines['left'].set_visible(False)
            axes[0, index].set_ylim(0, 1)

            axes[0, index].bar(0, proba, color=proba_color)
            # Draw on top of the bar the actual probability
            axes[0, index].text(0, proba, f'{proba:.2f}', ha='center', va='bottom')


            # color the nodes 
            # get cliques 
            cliques = sorted(list(list(x) for x in nx.connected_components(G)), key=lambda x: -len(x))
            graph_colors = {node: colors[color_index+1] for color_index, clique in enumerate(cliques) for node in clique}
            # color last graph with grey
            if index == k-1 and others_label_used:
                graph_colors = ['lightgrey'] * self.M  
            nx.draw(G, with_labels=not self.walker_agnostic, 
                    pos=nx.circular_layout(G),
                    ax=axes[1, index],
                    node_size = 200,
                    width=2,
                    margins=0.5,
                    node_color=[graph_colors[node] for node in G.nodes])
        
        if others_label_used:
            axes[1, -1].set_title(r'$Other$'+ f' ({unused_graphs})')
            # draw a question mark in the middle of the last nx graph
            axes[1, -1].text(0, 0, '?', ha='center', va='center', fontsize=20)
        
        #title on figure
        # print(f'[N={self.N}][M={self.M}] ')
        # fig.suptitle(f'Probability distribution of the steady state contact graph ' + r'$\Pr[G_\infty]$')
        self.N, self.M = None, None  #zero out the values

    def piechart_topk_contact_networks(self, k):
        """
        Piechart  the top k contact networks one below the other
        """
        sorted_contact_networks = sorted(self.contact_networks.items(), key=lambda x: x[1]['probability'], reverse=True)
        k = min(k, len(self.contact_networks))
        labels = [entry[0] for entry in sorted_contact_networks][:k] 
        probabilities = [entry[1]['probability'] for entry in sorted_contact_networks][:k]

        if k < len(self.contact_networks):
            labels.append('Other')
            other_probability = 1 - sum(probabilities)
            probabilities.append(other_probability)

        # Plotting the bar chart
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        #change figsize of the figure
        fig.set_size_inches(5, 5)
        colors = sns.color_palette('pastel')[0:k+1]
        explode = [0.02] * len(probabilities)
        ax.pie(probabilities, labels = labels, colors = colors, 
               autopct='%.0f%%', 
               explode=explode,
               shadow=True,
               radius=1.1)
        # ax.set_ylabel(r'$\Pr[G_\infty]$')  
        
        
        #title on figure
        print(f'[N={self.N}][M={self.M}] ')
        fig.suptitle(f'Probability distribution of the steady state contact graph ' + r'$\Pr[G_\infty]$')
        self.N, self.M = None, None  #zero out the values
    
    def clique_size_distribution(self, contact_networks, noplot=False):
        """
        Compute the distribution of clique sizes
        """
        probas = [sample['probability'] for sample in contact_networks.values()]
        clique_sizes = [
            [len(x) for x in nx.connected_components(contact_network['network'])] for contact_network in contact_networks.values()
        ]
        sampled_clique_sizes_indices = np.random.choice(len(clique_sizes), 100000, p=probas)
        clique_sizes = [clique_sizes[index] for index in sampled_clique_sizes_indices]
        clique_size_distribution = list(itertools.chain(*clique_sizes))
        if noplot:
            return clique_size_distribution
        # Kdeplot
        colors = sns.color_palette('pastel')
        fig, ax = plt.subplots(nrows = 1, ncols = 1,
                               figsize=(5, 5))
        sns.histplot(clique_size_distribution, discrete=True, ax=ax,
                     stat='probability', kde=False, color=colors[0])
        ax.set_xlabel('Clique size')
        ax.set_ylabel('Probability')
        ax.set_title('Clique size distribution')
        return clique_sizes
    
    def clique_count_distribution(self, contact_networks, noplot=False):
        """
        Compute the distribution of clique counts
        """
        probas = [sample['probability'] for sample in contact_networks.values()]
        clique_sizes = [
            len(list(nx.connected_components(contact_network['network']))) for contact_network in contact_networks.values()
        ]
        clique_counts = np.random.choice(clique_sizes, 100000, p=probas)
        if noplot:
            return clique_counts
        # Kdeplot
        colors = sns.color_palette('pastel')
        fig, ax = plt.subplots(nrows = 1, ncols = 1,
                               figsize=(5, 5))
        sns.histplot(clique_counts, discrete=True, ax=ax,
                     stat='probability', kde=False, color=colors[1])
        ax.set_xlabel('Clique count')
        ax.set_ylabel('Probability')
        ax.set_title('Clique count distribution')
        ax.set_xticks(range(min(clique_counts), max(clique_counts)+1))
        return clique_sizes
    