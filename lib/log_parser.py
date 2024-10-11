from datetime import datetime
import sys 
import os
from pprint import pprint
from colorama import Fore, Back, Style
import numpy as np

class LogParser:
    def __init__(self, experiment_name, achive=''):
        self.exp_name = experiment_name
        self.exp_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '../experiments',
                                        achive,
                                        self.exp_name)
        
        self.log_fp = os.path.join(self.exp_folder, 'experiment_log.txt')

        with open(self.log_fp, 'r') as f:
            self.lines = f.readlines()
        
        exp_start, exp_end = None, None
        for line_id, line in enumerate(self.lines):
            if 'Loading experiment' in line:
                if self.exp_name in line:
                    exp_start = line_id
                else:
                    if exp_start is None:
                        continue
                    else:
                        exp_end = line_id
                        break
        self.lines = self.lines[exp_start:exp_end]

    def strip_logger(self, line):
        return ' '.join(line.split(' ')[2:]).strip()

    def extract_steady_state_from_line(self, line):
        return np.array([[float(y) for y in x.replace('[', '').replace(']', '').split(' ') if y not in ['', '\x1b0m']] for x in self.strip_logger(line).split(':')[-1].split(',')]).squeeze()
    
    
    def get_experiment_results(self, markers=['overall', 'true_distribution']):
        results = {
            'true_steady_states': [],
            'best_steady_states_by_loss': [],
            'best_steady_states_by_alignment': []
        }
        M = int(self.lines[0].split('M=')[1].split(']')[0])
        for i in range(1, M+1):
            print(self.strip_logger(self.lines[i]))
            results['true_steady_states'].append(self.extract_steady_state_from_line(self.lines[i]))
        print('\n')
        if markers is None:
            return results
        
        for line_id, line in enumerate(self.lines):
            if all([marker in line for marker in markers]):
                print(Fore.BLUE + self.strip_logger(line))
                print(Fore.RED + self.strip_logger(self.lines[line_id + 1]))

                # check which model it is
                if 'by loss' in self.strip_logger(self.lines[line_id + 1]):
                    model_type = 'best_steady_states_by_loss'
                else:
                    model_type = 'best_steady_states_by_alignment'
                steady_state_line_id = 2
                while 'Stationary distribution for walker' in self.lines[line_id + steady_state_line_id]:
                    print(Fore.GREEN + 
                        self.strip_logger(self.lines[line_id + steady_state_line_id]))
                    results[model_type].append(self.extract_steady_state_from_line(self.lines[line_id + steady_state_line_id]))

                    steady_state_line_id += 1
                    if line_id + steady_state_line_id >= len(self.lines):
                        break

        print(Style.RESET_ALL)
        results['true_steady_states'] = np.stack(results['true_steady_states'])
        results['best_steady_states_by_loss'] = np.stack(results['best_steady_states_by_loss'])
        results['best_steady_states_by_alignment'] = np.stack(results['best_steady_states_by_alignment'])

        return results
    
    def get_experiment_runtime(self):
        format_ = "%Y-%m-%dT%H:%M:%S.%f%z"
        start_time = datetime.strptime(self.lines[0].split(' ')[0], format_)
        assert 'Saving alignment histogram in' in self.lines[-1]
        end_time = datetime.strptime(self.lines[-1].split(' ')[0], format_)

        return end_time - start_time
    
if __name__ == '__main__':
    # exp_name = '[N=5][M=3]MultiPolyicySteadyStateSoftmaxMarkovChain'
    exp_name = '[N=2][M=3][KL_divergence_backward][MultipleSteadyStateSoftmaxMarkovChain]'
    # distribution_marker = ['1000_']
    # distribution_marker = ['10000_']
    distribution_marker = ['true_distribution']


    parser = LogParser(exp_name)
    markers = distribution_marker + ['overall']
    results = parser.get_experiment_results(markers)
    print(results)