from colorama import Fore, Back, Style
from colour import Color
from typing import Union, List
import os
import sys
sys.path.append('..')
from functools import reduce
from math import factorial
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm

from lib.data import SequenceSimulator
from lib.models import MLPWalkerPolicy, SoftmaxMarkovChain, SteadyStateSoftmaxMarkovChain, MultiplePolicySteadyStateSoftmaxMarkovChain
from lib.contact_graph_models import RWIG, ContactMatrixGenerator
from lib.utils.utils import evaluate_sequence_alignment
import imageio


class SteadyStateSolver():
    def __init__(self, true_rwig_distribution, true_steady_states, policy_model, contact_graph_model, trial_id, **kwargs) -> None:
        self.true_rwig_distribution = true_rwig_distribution
        self.true_steady_states = true_steady_states  # (N, M)

        self.policy_model = policy_model
        assert isinstance(self.policy_model, SteadyStateSoftmaxMarkovChain) or isinstance(self.policy_model, MultiplePolicySteadyStateSoftmaxMarkovChain), \
                'Policy model must be a SteadyStateSoftmaxMarkovChain or MultiplePolicySteadyStateSoftmaxMarkovChain'
        self.contact_graph_model = contact_graph_model
        self.trial_id = trial_id

        optimizer_type = kwargs.get('optimizer', 'SGD')
        self.lr = kwargs.get('lr', 0.01)
        self.epochs = kwargs.get('epochs', 100)
        self.loss_type = kwargs.get('loss_type', 'mse')
        param_list = [{'params': self.policy_model.parameters(),
                        'lr': self.lr}]

        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(param_list)
        else:  # default to SGD
            self.optimizer = torch.optim.SGD(param_list)


        
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'maximum_likelihood':
            self.criterion = lambda out_probs, target_probs: -torch.sum(torch.log(out_probs))
        elif self.loss_type == 'KL_divergence_forward':
            self.criterion = lambda out_probs, target_probs: torch.sum(out_probs * (torch.log(out_probs) -  torch.log(target_probs)))
        elif self.loss_type == 'KL_divergence_backward':
            self.criterion = lambda out_probs, target_probs: torch.sum(target_probs * (torch.log(target_probs) - torch.log(out_probs)))
        else:
            raise ValueError(f'Invalid loss function {self.loss_type}. Choose from mse, maximum_likelihood')
        
        self.steady_state_vector_alignment = []
        self.loss_timeseries = []
        self.eval_metric_timeseries = []

        self.save_path = kwargs.get('save_path', None)
        if self.save_path is None:
            print(Fore.RED + f'Save path not provided. Cannot save model checkpoints.')
            print(Style.RESET_ALL)
            sys.exit()

        self.tolerance = kwargs.get('tolerance', 1e-10)
        self.checkpoint_every_epoch = kwargs.get('checkpoint_every_epoch', 100)

        self.save_path = os.path.join(self.save_path, f'trial_{self.trial_id}')
        os.makedirs(self.save_path, exist_ok=True)

        self.best_policy_model_by_alignment = None
        self.best_policy_model_by_alignment_loss = np.inf
        self.best_policy_model_by_alignment_alignment = np.inf

        # Worst model
        self.best_policy_model_by_loss = None
        self.best_policy_model_by_loss_loss = np.inf
        self.best_policy_model_by_loss_alignment = np.inf
        
    def loss(self, out_probs, true_probs):   
        out_probs, true_probs = torch.stack(out_probs), torch.tensor(true_probs, dtype=torch.double)
        loss = self.criterion(out_probs, true_probs)
        return loss 
    
    @torch.no_grad()
    def evaluate(self, out_probs, true_probs):  # kl div forward
        out_probs, true_probs = torch.stack(out_probs), torch.tensor(true_probs, dtype=torch.double)
        return torch.sum(out_probs * (torch.log(out_probs) -  torch.log(true_probs)))

    def fit(self, verbose=False):
        self.policy_model.train()

        if self.true_steady_states is not None:
            # Generate a contact network sample space to avoid partition generation every time
            # Otherwise we assume the underlying data is true and thus has no steady-state (not a simulation)
            rwig_sample_space = self.contact_graph_model._generate_sample_space(self.true_steady_states)
        else:
            
            rwig_sample_space = self.contact_graph_model._generate_sample_space(self.policy_model().detach().clone())
        true_Gk_probs = [self.true_rwig_distribution[sample]['probability'] for sample in sorted(self.true_rwig_distribution.keys())]

        for epoch in tqdm(range(1, self.epochs+1), leave=False):
            # Forward pass
            steady_states = self.policy_model()  # (N, M)
            rwig_distribution = self.contact_graph_model.generate_contact_network_distribution(steady_states,
                                                                                                            draw_graphs=False,
                                                                                                            contact_network_sample_space=rwig_sample_space)        
            try:
                out_Gk_probs = [rwig_distribution[sample]['probability'] for sample in sorted(self.true_rwig_distribution.keys())]
            except:
                print(f'true_rwig_distribution: {self.true_rwig_distribution.keys()}')
                print(f'rwig_distribution: {rwig_distribution.keys()}')
                sys.exit()
            loss = self.loss(out_Gk_probs, true_Gk_probs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Early stopping
            if loss.item() < self.tolerance and epoch > self.checkpoint_every_epoch:
                break

            # Plot every 500 epochs but save every 100 epochs
            if epoch % (self.checkpoint_every_epoch * 10) == 0 and verbose:
                kl_div = self.evaluate(out_Gk_probs, true_Gk_probs)
                self.save_metrics(loss.item(), kl_div, epoch, plot=True)

            elif epoch % self.checkpoint_every_epoch == 0:
                kl_div = self.evaluate(out_Gk_probs, true_Gk_probs)
                # Checkpointing
                self.save_metrics(loss.item(), kl_div, epoch, plot=False)


        best_policy_by_alignment_steady_states = self._get_steady_states(self.best_policy_model_by_alignment)
        best_policy_by_loss_steady_states = self._get_steady_states(self.best_policy_model_by_loss)

        results_dict = {
            'best_model_by_alignment': {
                'loss': self.best_policy_model_by_alignment_loss,
                'alignment': self.best_policy_model_by_alignment_alignment,
                'steady_states': best_policy_by_alignment_steady_states
            },
            'best_model_by_loss': {
                'loss': self.best_policy_model_by_loss_loss,
                'alignment': self.best_policy_model_by_loss_alignment,
                'steady_states': best_policy_by_loss_steady_states
            }
        }
        return results_dict
    
    def _get_steady_states(self, model = None):
        if model is None:
            model = self.policy_model
        steady_states = model().clone().detach().T  # (M, N)
        ascending_probability_markov_states_indices = torch.argsort(steady_states[0])
        steady_states = [steady_states[i, ascending_probability_markov_states_indices] for i in range(len(steady_states))]
        steady_states = sorted(steady_states, key=lambda x: x[0])
        steady_states = torch.stack(steady_states).T
        return steady_states

    def save_metrics(self, loss, eval_metric, epoch, plot=True):
        self.loss_timeseries.append(loss)
        self.eval_metric_timeseries.append(eval_metric)

        steady_states = self._get_steady_states()
        
        if self.true_steady_states is not None:
            assert steady_states.shape == self.true_steady_states.shape or self.true_steady_states is None, f'Shape of policy model steady states {steady_states.shape} does not match shape of ground truth steady states {self.true_steady_states.shape}'

            alignment = (steady_states.clone().detach() - self.true_steady_states).pow(2).mean(dim=0).sqrt().mean()
            self.steady_state_vector_alignment.append(alignment)

            # Record the model if it's the best so far
            if alignment == min(self.steady_state_vector_alignment):
                self.best_policy_model_by_alignment = self.policy_model.get_internal_model()
                torch.save(self.best_policy_model_by_alignment, os.path.join(self.save_path, 'best_policy_model_by_alignment.pth'))
                self.best_policy_model_by_alignment_loss = loss
                self.best_policy_model_by_alignment_alignment = alignment
        
        else:  # alignment will be eval metric (kl_div)
            if eval_metric == min(self.eval_metric_timeseries):
                self.best_policy_model_by_alignment = self.policy_model.get_internal_model()
                torch.save(self.best_policy_model_by_alignment, os.path.join(self.save_path, 'best_policy_model_by_alignment.pth'))
                self.best_policy_model_by_alignment_loss = loss
                self.best_policy_model_by_alignment_alignment = eval_metric

        if loss == min(self.loss_timeseries):
            self.best_policy_model_by_loss = self.policy_model.get_internal_model()
            torch.save(self.best_policy_model_by_loss, os.path.join(self.save_path, 'best_policy_model_by_loss.pth'))
            self.best_policy_model_by_loss_loss = loss
            self.best_policy_model_by_loss_alignment = eval_metric



        if plot and self.true_steady_states is not None:
            fig, ax = plt.subplots()
            ax.grid(True)
            legend1 = ax.plot(np.linspace(1, epoch, len(self.steady_state_vector_alignment)), self.steady_state_vector_alignment,
                    color='r', marker='x', label='Mean alignment')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Mean RSE')
            ax.set_title('Mean alignment of steady state vectors with groud truth')

            ax2 = ax.twinx()
            legend2 = ax2.plot(np.linspace(1, epoch, len(self.loss_timeseries)), self.loss_timeseries, 
                    color='g', marker='o', label='Loss')
            ax2.set_ylabel('Loss')
            ax.set_title('Loss and Mean alignment of steady state vectors with groud truth')
            lns = legend1 + legend2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc=0)
            plt.savefig(os.path.join(self.save_path, 'loss_and_steady_state_alignment.png'))

            # Close the figure to avoid memory leak
            plt.close(fig)
        elif plot and self.true_steady_states is None:
            fig, ax = plt.subplots()
            ax.grid(True)
            legend1 = ax.plot(np.linspace(1, epoch, len(self.loss_timeseries)), self.loss_timeseries, 
                    color='g', marker='o', label='Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Metrics values')
            ax_eval_loss = ax.twinx()
            legend2 = ax_eval_loss.plot(np.linspace(1, epoch, len(self.eval_metric_timeseries)), self.eval_metric_timeseries,
                    color='r', marker='x', label='KL divergence')
            ax_eval_loss.set_ylabel('KL divergence')
            ax.set_title('Loss and KL divergence')
            plt.savefig(os.path.join(self.save_path, 'loss.png'))
            plt.legend()
            plt.close(fig)

def run_solver_on_distribution(logger, rwig_distribution, true_steady_states, contact_graph_model, orig_cfg, num_samples_if_distribution_empirical=None):
    cfg = deepcopy(orig_cfg)
    # Adjust the save path 
    if num_samples_if_distribution_empirical is None:
        logger.info('Running solver on true distribution')
        suffix = 'true_distribution'
        
    else:
        logger.info(f'Running solver on empirical distribution with {num_samples_if_distribution_empirical} samples')
        suffix = f'empirical_distribution_{num_samples_if_distribution_empirical}_samples'

    cfg.SAVE_PATH = os.path.join(cfg.SAVE_PATH, suffix)

    os.makedirs(cfg.SAVE_PATH, exist_ok=True)
    cfg.SOLVER.PARAMETER_DICT['save_path'] = cfg.SAVE_PATH


    best_by_loss_trial_id = None
    best_by_alignment_trial_id = None

    overall_best_model_by_alignment_alignment = []
    overall_best_model_by_loss_alignment = []

    # Try different learning rates
    learning_rates = [0.1 * cfg.SOLVER.PARAMETER_DICT['lr'], cfg.SOLVER.PARAMETER_DICT['lr'], 10 * cfg.SOLVER.PARAMETER_DICT['lr']]
    for lr_id, lr in enumerate(learning_rates):
        for trial_id in range(cfg.NUM_TRIALS):
            trial_id = trial_id + lr_id * cfg.NUM_TRIALS
            cfg.SOLVER.PARAMETER_DICT.lr = lr
            logger.info(f'Trial [{trial_id+1}/{cfg.NUM_TRIALS * len(learning_rates)}] with learning rate {cfg.SOLVER.PARAMETER_DICT.lr}')
            ### Model
            if cfg.MODEL.MULTIPLE_POLICIES:
                model = MultiplePolicySteadyStateSoftmaxMarkovChain(**cfg.MODEL.PARAMETER_DICT)
            else:
                model = SteadyStateSoftmaxMarkovChain(**cfg.MODEL.PARAMETER_DICT)

            ### Solver
            solver = SteadyStateSolver(rwig_distribution, 
                                        true_steady_states,
                                        model, 
                                        contact_graph_model,
                                        trial_id,
                                        **cfg.SOLVER.PARAMETER_DICT)
            trial_results_dict = solver.fit(verbose=cfg.VERBOSE)  # a bit wonky, but it's fine for now.

            overall_best_model_by_alignment_alignment.append(trial_results_dict['best_model_by_alignment']['alignment'])
            overall_best_model_by_loss_alignment.append(trial_results_dict['best_model_by_loss']['alignment'])

            trial_best_model_by_alignment_steady_states = trial_results_dict['best_model_by_alignment']['steady_states']
            trial_best_model_by_loss_steady_states = trial_results_dict['best_model_by_loss']['steady_states']

            if cfg.VERBOSE:
                if true_steady_states is not None:
                    logger.info(f"Best model by alignment [alignment/loss]: [{trial_results_dict['best_model_by_alignment']['alignment']}/{trial_results_dict['best_model_by_alignment']['loss']}]")
                    for i, pi in enumerate(trial_best_model_by_alignment_steady_states.T):
                        logger.info(f'Stationary distribution for walker {i}: {pi.numpy()}')

                logger.info(f"Best model by loss [alignment/loss]: [{trial_results_dict['best_model_by_loss']['alignment']}/{trial_results_dict['best_model_by_loss']['loss']}]")
                for i, pi in enumerate(trial_best_model_by_loss_steady_states.T):
                    logger.info(f'Stationary distribution for walker {i}: {pi.numpy()}')

            # Save the steady states with best alignment
            if trial_results_dict['best_model_by_alignment']['alignment'] == min(overall_best_model_by_alignment_alignment):
                best_model_by_alignment = trial_results_dict['best_model_by_alignment']['steady_states']
                best_by_alignment_trial_id = trial_id
                best_model_by_alignment_loss = trial_results_dict['best_model_by_alignment']['loss']
                best_model_by_alignment_alignment = trial_results_dict['best_model_by_alignment']['alignment']

            # Save the steady states of the WORST alignment model from the best loss trials
            if trial_results_dict['best_model_by_loss']['alignment'] == max(overall_best_model_by_loss_alignment):
                best_model_by_loss = trial_results_dict['best_model_by_loss']['steady_states']
                best_by_loss_trial_id = trial_id
                best_model_by_loss_loss = trial_results_dict['best_model_by_loss']['loss']
                best_model_by_loss_alignment = trial_results_dict['best_model_by_loss']['alignment']


    # Save the best model
    if true_steady_states is not None:
        logger.info(f'[{suffix}] (overall) Best model by alignment trial id: {best_by_alignment_trial_id}')
        logger.info(f"Best model by alignment [alignment/loss]: [{best_model_by_alignment_alignment}/{best_model_by_alignment_loss}]")
        for i, pi in enumerate(best_model_by_alignment.T):
            logger.info(f'Stationary distribution for walker {i}: {pi.numpy()}')
        torch.save(best_model_by_alignment, os.path.join(cfg.SAVE_PATH, 'best_model_by_alignment_steady_states.pth'))

    else:
        # Alignments are KL divergences
        logger.info(f'[{suffix}] (overall) Best model by kl div trial id: {best_by_alignment_trial_id}')
        logger.info(f"Best model by kl div [kl div/loss]: [{best_model_by_alignment_alignment}/{best_model_by_alignment_loss}]")
        for i, pi in enumerate(best_model_by_alignment.T):
            logger.info(f'Stationary distribution for walker {i}: {pi.numpy()}')
        torch.save(best_model_by_alignment, os.path.join(cfg.SAVE_PATH, 'best_model_by_alignment_steady_states.pth'))
    logger.warning(f'Best model by loss is saved as the model with the highest alignment from the models with the best losses!')

    logger.info(f'[{suffix}] (overall) Best model by loss trial id: {best_by_loss_trial_id}')
    logger.info(f"Best model by loss [alignment/loss]: [{best_model_by_loss_alignment}/{best_model_by_loss_loss}]")
    distribution = contact_graph_model.generate_contact_network_distribution(best_model_by_loss,
                                                                            draw_graphs=False)
    distribution = {k: v['probability'] for k, v in distribution.items()}
    logger.info(f'[{suffix}] (overall) Best model by loss distribution: {distribution}')

    for i, pi in enumerate(best_model_by_loss.T):
        logger.info(f'Stationary distribution for walker {i}: {pi.numpy()}')
    torch.save(best_model_by_loss, os.path.join(cfg.SAVE_PATH, 'best_model_by_loss_steady_states.pth'))

    try:
        # Seaborn histplot of best alignment and worst alignment
        fig, ax = plt.subplots()
        logger.info(f'Saving alignment histogram in {cfg.SAVE_PATH}')
        color_palette = sns.color_palette('pastel')
        sns.histplot(overall_best_model_by_alignment_alignment, ax=ax, stat='probability', color=color_palette[1], bins = 10, label='By alignment', alpha=0.7)
        # ax2 = ax.twinx()
        sns.histplot(overall_best_model_by_loss_alignment, ax=ax, stat='probability', color=color_palette[0], bins = 10, label='By loss', alpha=0.7)
        ax.set_title('Alignment of steady state vectors with ground truth')
        ax.set_xlabel('Mean RSE')
        ax.set_ylabel('Frequency')
        ax.grid()
        ax.legend(loc=0)
        plt.savefig(os.path.join(cfg.SAVE_PATH, 'alignment_histogram.png'))

        plt.close()
    except:  # no alignmet because data 
        pass

def load_and_run_experiment(cfg):
    from loguru import logger
    logger.remove()
    logger.add(os.path.join(cfg.SAVE_PATH, 'experiment_log.txt'), format="{time} {level} {message}", level="INFO")

    logger.info(f'Loading experiment: {cfg.NAME}')

    true_steady_states = cfg.DATA.STEADY_STATES

    for i, pi in enumerate(true_steady_states.T):
        logger.info(f'Stationary distribution for walker {i}: {pi.numpy()}')

    contact_graph_model = RWIG(cfg.DATA.WALKER_AGNOSTIC)
    true_rwig_distribution = contact_graph_model.generate_contact_network_distribution(true_steady_states, draw_graphs=False)

    empirical_distribution_dict = {}
    if 'NUM_SAMPLES_IN_SEQUENCE' in cfg.DATA.keys():
        # Zero out the probabilities of the samples that were not drawn
        empirical_distribution = {sample: {'probability': 0.0} for sample in true_rwig_distribution.keys()}
        sample_space = true_rwig_distribution.keys()
        sample_probabilities = [true_rwig_distribution[sample]['probability'] for sample in sample_space]

        for num_samples in cfg.DATA.NUM_SAMPLES_IN_SEQUENCE:
            logger.info(f'Sampling {num_samples} contact graphs and constructing empirical distribution')

            samples = np.random.choice(list(sample_space), num_samples, p=sample_probabilities)
            unique_samples, sample_counts = np.unique(samples, return_counts=True)

            # empirical_distribution_dict[num_samples] = deepcopy(empirical_distribution)     # THIS COPIES FULL DISTRIBUTION AND ZEROES OUT PROBABILITIES OF SAMPLES NOT DRAWN
            empirical_distribution_dict[num_samples] = {}  # THIS RESETS THE DICT TO EMPTY AND ONLY ADDS THE SAMPLES THAT WERE DRAWN. this should remove the log(proba=0) = -inf error in solver.loss() 
            for sample, count in zip(unique_samples, sample_counts):
                empirical_distribution_dict[num_samples][sample] = {'probability': count / num_samples}

    # Run the solver on the true distribution
    run_solver_on_distribution(logger, true_rwig_distribution, true_steady_states, contact_graph_model, cfg)

    # Run the solver on the empirical distributions
    for num_samples, empirical_distribution in empirical_distribution_dict.items():
        run_solver_on_distribution(logger, empirical_distribution, true_steady_states, contact_graph_model, cfg, num_samples)

def load_and_run_true_data_experiment(cfg):
    from loguru import logger
    logger.remove()
    logger.add(os.path.join(cfg.SAVE_PATH, 'experiment_log.txt'), format="{time} {level} {message}", level="INFO")

    logger.info(f'Loading experiment: {cfg.NAME}')

    true_steady_states = cfg.DATA.STEADY_STATES
    assert true_steady_states is None, 'True steady states cannot be provided for this experiment'

    # Generate the contact graphs
    contact_network_generator = ContactMatrixGenerator(cfg.DATA.CSV_PATH, cfg.DATA.NODES)
    G_T = contact_network_generator.get_contact_matrix_timeseries()

    # Modify the M of the config 
    cfg.MODEL.PARAMETER_DICT['M'] = contact_network_generator.M
    if cfg.MODEL.PARAMETER_DICT['N'] < contact_network_generator.get_min_Markov_chain():
        cfg.MODEL.PARAMETER_DICT['N'] = contact_network_generator.get_min_Markov_chain()

        
    contact_graph_model = RWIG(cfg.DATA.WALKER_AGNOSTIC)
    true_rwig_distribution = contact_graph_model.generate_empirical_contact_network_distribution(G_T)
    print_rwig_distribution = {k: v['probability'] for k, v in true_rwig_distribution.items()}
    logger.info(f'True distribution: {print_rwig_distribution}')

    # Run the solver on the true distribution
    run_solver_on_distribution(logger, true_rwig_distribution, true_steady_states, contact_graph_model, cfg)
