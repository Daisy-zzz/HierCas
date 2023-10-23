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


def plot_graph(k, src, dst, values):
    plt.figure(dpi=300)
    plt.clf()
    bwith = 2 #边框宽度设置为2
    ax = plt.gca()#获取边框
    #设置边框
    ax.spines['bottom'].set_linewidth(bwith)#图框下边
    ax.spines['left'].set_linewidth(bwith)#图框左边
    ax.spines['top'].set_linewidth(bwith)#图框上边
    ax.spines['right'].set_linewidth(bwith)#图框右边
    # Create a graph
    G = nx.DiGraph()
    values = values.squeeze(1).tolist()
    src_list = src.tolist()
    dst_list = dst.tolist()
    node_list = set()
    node_list.add(src_list[0])
    for d in dst_list:
        node_list.add(d)
    node_dict = {old_node: new_node for new_node, old_node in enumerate(node_list)}
    src = [node_dict[x] for x in src]
    dst = [node_dict[x] for x in dst]
    att_values = {}
    for idx, node in enumerate(src):
        if att_values.get(node):
            att_values[node] += values[idx]
        else:
            att_values[node] = values[idx]
    for idx, node in enumerate(dst):
        if att_values.get(node):
            att_values[node] += values[idx]
        else:
            att_values[node] = values[idx]
    sum_value = 0
    for node, value in att_values.items():
        sum_value += value
    for node, value in att_values.items():
        G.add_node(node, value=value / sum_value)

    # Add edges
    added_dst = []
    dst_re = dst[::-1]
    src_re = src[::-1]
    for idx, d in enumerate(dst):
        if d not in added_dst:
            G.add_edge(src[idx], dst[idx])
            added_dst.append(d)
    #G.add_edges_from(zip(src, dst))

    # Extract node values
    node_values = nx.get_node_attributes(G, 'value')

    # Draw the graph with node colors based on values
    # Set root node and layout
    root = node_dict[src_list[0]]
    pos = custom_layout(G, root)
    node_colors = list(node_values.values())
    cmap = cm.YlOrRd  # Choose a colormap
    vmin = min(node_colors)
    vmax = max(node_colors)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, cmap=cmap, vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.2, arrowstyle='->', arrowsize=5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # Empty array, required for the colorbar to work
    plt.colorbar(sm)
    plt.savefig("visualize_aps/test" + str(k) + ".png", bbox_inches='tight')
