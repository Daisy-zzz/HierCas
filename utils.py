import numpy as np

def shuffle_within_group(group):
    n = len(group)
    perm = np.random.permutation(n)
    return group.iloc[perm]

### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
# Define a custom layout function
def custom_layout(G, root, width=1, vert_gap=0.1, vert_loc=0.5, xcenter=0.1, pos=None, depth=0):
    if pos is None:
        pos = {}
    pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if not children or depth > 500:
        return pos
    dx = width
    nextx = xcenter + dx
    for child in children:
        child_vert_loc = vert_loc - vert_gap * random.uniform(-0.3, 0.3)  # Randomly adjust vert_loc for each child
        pos[child] = (nextx, child_vert_loc)
        pos = custom_layout(G, child, width=dx, vert_gap=vert_gap, vert_loc=child_vert_loc, xcenter=nextx, pos=pos, depth=depth+1)
        nextx += dx
    return pos
