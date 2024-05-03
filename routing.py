from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
from networkx.algorithms.flow import build_residual_network
from arborescences import *
import numpy as np
import networkx as nx

# global variables in this file
seed = 1
n = 10
rep = 1
k = 8
p = 0.2
f_num = 40
samplesize = 20
name = "experiment-routing"


def set_routing_params(params):
    global seed, n, rep, k, samplesize, name, f_num
    [n, rep, k, samplesize, f_num, seed, name] = params


# build data structure for square one algorithm only with s-d pair
def PrepareSQ1SD(G, s, d):
    global SQ1
    H = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(H, 'capacity')
    SQ1 = {n: {} for n in G}
    k = sorted(list(nx.edge_disjoint_paths(G, s, d, auxiliary=H, residual=R)), key=len)
    SQ1[s][d] = k


# Route with Square One algorithm
# source s
# destination d
# link failure set fails
# arborescence decomposition T
# can survive k-1 link failures by k
# Paper: CASA
def RouteSQ1(s, d, fails, T):
    curRoute = SQ1[s][d][0]
    k = len(SQ1[s][d])
    detour_edges = []
    index = 1
    hops = 0
    switches = 0
    c = s  # current node
    n = len(T[0].nodes())
    while (c != d):
        nxt = curRoute[index]
        if (nxt, c) in fails or (c, nxt) in fails:
            for i in range(2, index + 1):
                detour_edges.append((c, curRoute[index - i]))
                c = curRoute[index - i]
            switches += 1
            c = s
            hops += (index - 1)
            curRoute = SQ1[s][d][switches % k]
            index = 1
        else:
            if switches > 0:
                detour_edges.append((c, nxt))
            c = nxt
            index += 1
            hops += 1
        if hops > 3 * n or switches > k * n:
            print("cycle square one")
            return (True, hops, switches, detour_edges)
    return (False, hops, switches, detour_edges)


# preprocess routing algorithm on graph g (only when using simulate_graph)
def preprocess_simulate_graph(g, f, targeted=False, dest=None):
    edg = list(g.edges())
    fails = g.graph['fails']
    if fails is not None:
        if len(fails) < f:
            fails = fails + edg[:f - len(fails) + 1]
        edg = fails
    if f > len(edg):
        print('more failures than edges')
        print('simulate', len(g.edges()), len(fails), f)
        return -1
    g.graph['k'] = k
    fails = edg[:f]
    if targeted:
        fails = []
    failures = {(u, v): g[u][v]['arb'] for (u, v) in fails}
    failures.update({(v, u): g[u][v]['arb'] for (u, v) in fails})

    g = g.copy(as_view=False)
    g.remove_edges_from(failures.keys())
    dist = nx.shortest_path_length(g, target=dest)
    return g, failures, fails, dist


# postprocess routing algorithm on graph g (only when using simulate_graph)
def postprocess_simulate_graph(g, stat, failures, targeted=False):
    if not targeted:
        for ((u, v), i) in failures.items():
            g.add_edge(u, v)
            g[u][v]['arb'] = i
    stat.finalize()
    sys.stdout.flush()


# run routing algorithm on graph g
# RANDOM: don't use failset associated with g, but construct one at random
# stats: statistics object to fill
# f: number of failed links
# samplesize: number of nodes from which we route towards the root
# dest: nodes to exclude from using in sample
# tree: arborescence decomposition to use
# run routing algorithm on graph g
def simulate_graph(g, stat, failures, fails, dist, precomputation=None, source=None, dest=None, tree=None,
                   targeted=False):
    if precomputation is None:
        precomputation = tree
        if precomputation is None:
            precomputation = GreedyArborescenceDecomposition(g)

    if targeted:
        fails = list(nx.minimum_edge_cut(g, s=source, t=dest))[1:]
        random.shuffle(fails)
        failures1 = {(u, v): g[u][v]['arb'] for (u, v) in fails}
        g.remove_edges_from(failures1.keys())
        x = dist[source]
        dist[source] = nx.shortest_path_length(g, source=source, target=dest)
    if (source == dest) or (not source in dist):
        stat.fails += 1
        return fails, g
    (fail, hops) = stat.update(source, dest, fails, precomputation, dist[source])
    if fail:
        stat.hops = stat.hops[:-1]
        stat.stretch = stat.stretch[:-1]
    elif hops < 0:
        stat.hops = stat.hops[:-1]
        stat.stretch = stat.stretch[:-1]
        stat.succ = stat.succ - 1
    if targeted:
        for ((u, v), i) in failures.items():
            g.add_edge(u, v)
            g[u][v]['arb'] = i

    return fails, g


# class to collect statistics on routing simulation
class Statistic:
    def __init__(self, routeFunction, name, g=None):
        self.funct = routeFunction
        self.name = name
        self.has_graph = g is not None
        if g is not None:
            self.graph = g

    def reset(self, nodes):
        self.totalHops = 0
        self.totalSwitches = 0
        self.fails = 0
        self.succ = 0
        self.stretchNorm = [-2]
        self.stretch = [-2]
        self.hops = [-2]
        self.lastsuc = True
        self.load = {(u, v): 0 for u in nodes for v in nodes}
        self.lat = 0

    # add data for routing simulations from source s to destination
    # despite the failures in fails, using arborescences T and the shortest
    # path length is captured in shortest
    def update(self, s, d, fails, T, shortest):
        if not self.has_graph:
            (fail, hops, switches, detour_edges_used) = self.funct(s, d, fails, T)
        else:
            (fail, hops, switches, detour_edges_used) = self.funct(s, d, fails, T, self.graph)
        # if switches == 0:
        #    fail = False
        if fail:
            self.fails += 1
            self.lastsuc = False
            self.stretchNorm.append(-1)
            self.stretch.append(-1)
            self.hops.append(-1)
            for e in detour_edges_used:
                self.load[e] += 1
        else:
            self.totalHops += hops
            self.succ += 1
            self.totalSwitches += switches
            if shortest == 0:
                shortest = 1
            self.stretchNorm.append(hops / shortest)
            self.stretch.append(hops - shortest)
            self.hops.append(hops)
            for e in detour_edges_used:
                self.load[e] += 1
            self.lastsuc = True
        return (fail, hops)

    # compute statistics when no more data will be added
    def finalize(self):
        self.lat = -1
        self.load = max(self.load.values())
        if len(self.hops) > 1:
            self.hops = self.hops[1:]
            self.stretch = self.stretch[1:]
            self.stretchNorm = self.stretchNorm[1:]
        else:
            self.hops = [0]
            self.stretch = [0]
            self.stretchNorm = [0]
        if len(self.hops) > 0:
            self.lat = np.mean(self.hops)
        return max(self.stretch)

