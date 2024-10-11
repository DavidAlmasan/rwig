from copy import deepcopy
from functools import reduce
import more_itertools
import numpy as np
import networkx as nx
from more_itertools import set_partitions

class PartitionManager:
    def __init__(self, object_set, max_partitions = None, verbose=False) -> None:
        self.object_set = object_set
        self.max_partitions = max_partitions + 1 if max_partitions is not None else None  # We add 1 to the max_partitions to account range(start, stop+1)
        self.partition = []
        if verbose:
            print(f'Generating walker partitions')
        self._generate_partition(self.object_set, 0, [])

    def _generate_partition(self, object_set, index, ans):
        """
        Generate the sample space of contact networks
        """

        if index == len(object_set):
            # If we have considered all elements in the set, print the partition
            self._record_partition(deepcopy(ans))
            return
    
        # For each subset in the partition, add the current element to it and recall
        for i in range(len(ans)):
            ans[i].append(object_set[index])
            self._generate_partition(object_set, index + 1, ans)
            ans[i].pop()
    
        # Add the current element as a singleton subset and recall
        ans.append([object_set[index]])
        self._generate_partition(object_set, index + 1, ans)
        ans.pop()

    def _record_partition(self, ans):
        """
        Function to print a partition
        """
        self.partition.append(ans)

    def return_partition(self, with_union=False):
        """
        Return the partition
        """
        if with_union:
            return [[list(reduce(np.union1d, realisation)) for realisation in partition] for partition in self.partition]
        
        # sort the partitions 
        self.partition = sorted([sorted(partition) for partition in self.partition])
        return self.partition

class EfficientPartitionManager(PartitionManager):
    def __init__(self, object_set, max_partitions, verbose=False) -> None:
        super().__init__(object_set, max_partitions, verbose)

    def _generate_partition(self, object_set, index, ans):
        """
        Generate M partitions using more_itertools
        """
        max_partitions = self.max_partitions if self.max_partitions is not None else len(object_set)+1
        for m in range(1, max_partitions):
            for partition in set_partitions(object_set, m):
                self._record_partition(partition)

class KPartitionManager:
    def __init__(self, object_set) -> None:
        self.object_set = object_set

    def return_k_partition(self, k):
        """
        Return the k-th partition
        """
        return set_partitions(self.object_set, k)


def evaluate_sequence_alignment(model_G_T, true_G_T, use_isomorphism_alignment=True):
    """
    Evaluate the sequence alignment of two sequences of contact networks
    """
    if use_isomorphism_alignment:
        return [nx.is_isomorphic(out, true) for out, true in zip(model_G_T, true_G_T)]
    else:
        return [nx.utils.graphs_equal(out, true) for out, true in zip(model_G_T, true_G_T)]
        

if __name__ == '__main__':
    s = [[0, 0], [1, 1], [5,6,7]]
    partition_manager = EfficientPartitionManager(s)
    print(partition_manager.return_partition())