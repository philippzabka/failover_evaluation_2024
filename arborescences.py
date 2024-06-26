import sys
import networkx as nx
import random
from heapq import heappush, heappop

swappy = []

# reset the arb attribute for all edges to -1, i.e., no arborescence assigned yet
def reset_arb_attribute(g):
    for (u, v) in g.edges():
        g[u][v]['arb'] = -1


# given a graph g and edge (u1, v1) and edge (u2,v2) swap the arborescences
# they belong to (will crash if the edges dont belong to the graph)
def swap(g, u1, v1, u2, v2):
    i1 = g[u1][v1]['arb']
    i2 = g[u2][v2]['arb']
    g[u1][v1]['arb'] = i2
    g[u2][v2]['arb'] = i1


# given a graph g and a minimum degree min, nodes with degree two are contracted into a link
# recursively and then all nodes with a degree < min are removed
# from g recursively until the graph is empty or all remaining nodes have at
# least degree min
def trim_merge(g, min):
    while True:
        while True:
            rem = []
            for v in g.nodes():
                if g.degree(v) == 2:
                    neighbors = list(g.neighbors(v))
                    for i in range(len(neighbors) - 1):
                        g.add_edge(neighbors[0], neighbors[i + 1])
                    rem.append(v)
            if len(rem) == 0:
                break
            g.remove_nodes_from(rem)
        rem = []
        for v in g.nodes():
            if g.degree(v) < min:
                rem.append(v)
        if len(rem) == 0:
            break
        g.remove_nodes_from(rem)
    return g


# given a graph g and a minimum degree min_degree, nodes with a lower degree are removed
# from g and their ex-neighbors are connected recursively until the graph is
# empty or all remaining nodes have at least degree min_degree
def trim2(g, min_degree):
    while True:
        rem = []
        for v in g.nodes():
            if g.degree(v) < min_degree:
                rem.append(v)
                for u in g[v]:
                    for u1 in g[v]:
                        g.add_edge(u, u1)
                break
        g.remove_nodes_from(rem)
        if len(rem) == 0:
            break
    return g


# given a graph return the arborescences in a dictionary with indices as keys
def get_arborescence_dict(g):
    arbs = {}
    for (u, v) in g.edges():
        index = g[u][v]['arb']
        if index not in arbs:
            arbs[index] = nx.DiGraph()
            arbs[index].graph['root'] = g.graph['root']
            arbs[index].graph['index'] = index
        arbs[index].add_edge(u, v)
    return arbs


# given a graph return a list of its arborescences
def get_arborescence_list(g):
    arbs = get_arborescence_dict(g)
    sorted_indices = sorted([i for i in arbs.keys() if i >= 0])
    return [arbs[i] for i in sorted_indices]


# return the stretch of the arborescence with index i on g (how much longer the
# path to the root is in the arborescence than in the original graph)
def stretch_index(g, index):
    arbs = get_arborescence_list(g)
    dist = nx.shortest_path_length(g, target=g.graph['root'])
    distA = {}
    if g.graph['root'] in arbs[index].nodes():
        distA = nx.shortest_path_length(arbs[index], target=g.graph['root'])
    else:
        return float("inf")
    stretch_vector = []
    for v in g.nodes():
        if v != g.graph['root']:
            stretch = -1
            if v in arbs[index].nodes() and v in distA:
                stretch = max(stretch, distA[v] - dist[v])
            else:
                return float("inf")
            stretch_vector.append(stretch)
    return max(stretch_vector)


# return the stretch of the arborence with the largest stretch
def stretch(g):
    stretch_vector = []
    for index in range(g.graph['k']):
        stretch_vector.append(stretch_index(g, index))
    return max(stretch_vector)


# return the longest path to the root in all arborescences
def depth(g):
    arbs = get_arborescence_list(g)
    distA = [{} for index in range(len(arbs))]
    for index in range(len(arbs)):
        if g.graph['root'] in arbs[index].nodes():
            distA[index] = nx.shortest_path_length(
                arbs[index], target=g.graph['root'])
        else:
            return float("inf")
    depth_vector = []
    for v in g.nodes():
        if v != g.graph['root']:
            depth = -1
            for index in range(len(arbs)):
                if v in arbs[index].nodes() and v in distA[index]:
                    depth = max(depth, distA[index][v])
                else:
                    return float("inf")
                depth_vector.append(depth)
    return max(depth_vector)


# return the nodes belonging to arborence i
def nodes_index(g, i):
    return set([u for (u, v, d) in g.edges(data=True) if d['arb'] == i or u == g.graph['root']])


# return length of shortest path between u and v on the indexth arborescence of g
def shortest_path_length(g, index, u, v):
    arbs = get_arborescence_dict(g)
    return nx.shortest_path_length(arbs[index], u, v)


# return nodes in connected component with node d (after failures have been removed)
def connected_component_nodes_with_d_after_failures(g, failures, d):
    G = g.to_undirected()
    G.remove_edges_from(failures)
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    for i in range(len(Gcc)):
        if d in Gcc[i]:
            return list(Gcc[i])


# return the edge connectivity of g between s and t
def TestCut(g, s, t):
    return nx.edge_connectivity(g, s, t)


# compute the k^th arborescence of g greedily
def FindTree(g, k):
    T = nx.DiGraph()
    T.add_node(g.graph['root'])
    R = {g.graph['root']}
    dist = dict()
    dist[g.graph['root']] = 0
    # heap of all border edges in form [(edge metric, (e[0], e[1])),...]
    h = []
    preds = sorted(g.predecessors(
        g.graph['root']), key=lambda k: random.random())
    for x in preds:
        heappush(h, (0, (x, g.graph['root'])))
        if k > 1:
            continue
    while len(h) > 0:
        (d, e) = heappop(h)
        g.remove_edge(*e)
        if e[0] not in R and (k == 1 or TestCut(g, e[0], g.graph['root']) >= k - 1):
            dist[e[0]] = d + 1
            R.add(e[0])
            preds = sorted(g.predecessors(e[0]), key=lambda k: random.random())
            for x in preds:
                if x not in R:
                    heappush(h, (d + 1, (x, e[0])))
            T.add_edge(*e)
        else:
            g.add_edge(*e)
    if len(R) < len(g.nodes()):
        # print(
        #    "Couldn't find next edge for tree with g.graph['root'], ", k, len(R))
        sys.stdout.flush()
    return T


# associate a greedy arborescence decomposition with g
def GreedyArborescenceDecomposition(g):
    reset_arb_attribute(g)
    gg = g.to_directed()
    K = g.graph['k']
    k = K
    while k > 0:
        T = FindTree(gg, k)
        if T is None:
            return None
        for (u, v) in T.edges():
            g[u][v]['arb'] = K - k
        gg.remove_edges_from(T.edges())
        k = k - 1
    return get_arborescence_list(g)


# Helper class (some algorithms work with Network, others without),
# methods as above
class Network:
    # initialize variables
    def __init__(self, g, K, root):
        self.g = g
        self.K = K
        self.root = root
        self.arbs = {}
        self.build_arbs()
        self.dist = nx.shortest_path_length(self.g, target=root)

    # create arbs data structure from edge attributes
    def build_arbs(self):
        self.arbs = {index: nx.DiGraph() for index in range(self.K)}
        for (u, v) in self.g.edges():
            index = self.g[u][v]['arb']
            if index > -1:
                self.arbs[index].add_edge(u, v)

    # create arborescence for index given edge attributes
    def build_arb(self, index):
        self.arbs[index] = nx.DiGraph()
        for (u, v) in self.g.edges():
            if self.g[u][v]['arb'] == index:
                self.arbs[index].add_edge(u, v)

    # return graph of edges not assigned to any arborescence
    def rest_graph(self, index):
        rest = nx.DiGraph()
        for (u, v) in self.g.edges():
            i = self.g[u][v]['arb']
            if i > index or i == -1:
                rest.add_edge(u, v)
        return rest

    # add edge (u,v) to arborescence of given index
    def add_to_index(self, u, v, index):
        old_index = self.g[u][v]['arb']
        self.g[u][v]['arb'] = index
        if index > -1:
            self.arbs[index].add_edge(u, v)
        if old_index > -1:
            self.build_arb(old_index)

    # remove edge (u,v) from the arborescence it belonged to
    def remove_from_arbs(self, u, v):
        old_index = self.g[u][v]['arb']
        self.g[u][v]['arb'] = -1
        if old_index > -1:
            self.build_arb(old_index)

    # swap arborescence assignment for edges (u1,v1) and (u2,v2)
    def swap(self, u1, v1, u2, v2):
        i1 = self.g[u1][v1]['arb']
        i2 = self.g[u2][v2]['arb']
        self.g[u1][v1]['arb'] = i2
        self.g[u2][v2]['arb'] = i1
        self.build_arb(i1)
        self.build_arb(i2)

    # return true if graoh of given index is really an arborescence
    def is_arb(self, index):
        arb = self.arbs[index]
        root = self.root
        if root in arb.nodes():
            distA = nx.shortest_path_length(arb, target=root)
        else:
            return False
        for v in arb.nodes():
            if v == root:
                continue
            if arb.out_degree(v) != 1 or v not in distA:
                return False
            # if self.K - index > 1:
            #   rest = self.rest_graph(index)
            # if not v in rest.nodes() or TestCut(self.rest_graph(index), v, root) < self.K-index-1:
            #    return False
        return True

    # return nodes that are part of arborescence for given index
    def nodes_index(self, index):
        if index > -1:
            arb = self.arbs[index]
            l = list(arb.nodes())
            for u in l:
                if u != self.root and arb.out_degree(u) < 1:
                    arb.remove_node(u)
            return arb.nodes()
        else:
            return self.g.nodes()

    # return number of nodes in all arborescences
    def num_complete_nodes(self):
        return len(self.complete_nodes())

    # return nodes which belong to all arborescences
    def complete_nodes(self):
        c = set(self.g.nodes())
        for arb in self.arbs.values():
            c = c.intersection(set(arb.nodes()))
        return c

    # return number of nodes in all arborescences
    def shortest_path_length(self, index, u, v):
        return nx.shortest_path_length(self.arbs[index], u, v)

    # return true iff node v is in shortest path from node u to root in
    # arborescence of given index
    def in_shortest_path_to_root(self, v, index, u):
        return (v in nx.shortest_path(self.arbs[index], u, self.root))

    # return predecessors of node v in g (as a directed graph)
    def predecessors(self, v):
        return self.g.predecessors(v)


# set up network data structures before using them
def prepareDS(n, h, dist, reset=True):
    if reset:
        reset_arb_attribute(n.g)
    for i in range(n.K):
        dist.append({n.root: 0})
        preds = sorted(n.g.predecessors(n.root), key=lambda k: random.random())
        heapT = []
        for x in preds:
            heappush(heapT, (0, (x, n.root)))
        h.append(heapT)
        n.arbs[i].add_node(n.root)


# try to swap an edge on arborescence index for network with heap h
def trySwap(n, h, index):
    ni = list(n.nodes_index(index))
    for v1 in ni:
        for u in n.g.predecessors(v1):
            index1 = n.g[u][v1]['arb']
            if u == n.root or index1 == -1 or u in ni:
                continue
            for v in n.g.successors(u):
                if n.g[u][v]['arb'] != -1 or v not in n.nodes_index(index1):
                    continue
                if not n.in_shortest_path_to_root(v1, index1, v):
                    n.add_to_index(u, v, index)
                    n.swap(u, v, u, v1)
                    if n.is_arb(index) and n.is_arb(index1):
                        update_heap(n, h, index)
                        update_heap(n, h, index1)
                        add_neighbors_heap(n, h, [u, v, v1])
                        return True
                    # print("undo swap")
                    n.swap(u, v, u, v1)
                    n.remove_from_arbs(u, v)
    return False


# add a new items to the heap
def update_heap(n, h, index):
    new = []
    for (d, e) in list(h[index]):
        if e[1] in n.arbs[index].nodes():
            d = n.shortest_path_length(index, e[1], n.root) + 1
            heappush(new, (d, e))
    h[index] = new


# add neighbors to heap
def add_neighbors_heap(n, h, nodes):
    n.build_arbs()
    for index in range(n.K):
        add_neighbors_heap_index(n, h, index, nodes)


# add neighbors to heap for a given index and nodes
def add_neighbors_heap_index(n, h, index, nodes):
    ni = n.nodes_index(index)
    dist = nx.shortest_path_length(n.g, target=n.root)
    for v in nodes:
        if v not in ni:
            continue
        preds = sorted(n.g.predecessors(v), key=lambda k: random.random())
        d = n.shortest_path_length(index, v, n.root) + 1
        stretch = d
        for x in preds:
            if x not in ni and n.g[x][v]['arb'] == -1:
                heappush(h[index], (stretch, (x, v)))


def round_robin(g, cut=False, swap=False, reset=True):
    global swappy
    if reset:
        reset_arb_attribute(g)
    n = Network(g, g.graph['k'], g.graph['root'])
    K = n.K
    h = []
    dist = []
    prepareDS(n, h, dist, reset)
    index = 0
    swaps = 0
    count = 0
    num = len(g.nodes())
    count = 0
    while n.num_complete_nodes() < num and count < K * num * num:
        count += 1
        if len(h[index]) == 0:
            if swap and trySwap(n, h, index):
                index = (index + 1) % K
                swaps += 1
                continue
            else:
                if swap:
                    print("1 couldn't swap for index ", index)
                # drawArborescences(g, "balanced")
                # sys.stdout.flush()
                # plt.show()
                return -1
        (d, e) = heappop(h[index])
        while e != None and n.g[e[0]][e[1]]['arb'] > -1:  # in used_edges:
            if len(h[index]) == 0:
                if swap and trySwap(n, h, index):
                    index = (index + 1) % K
                    swaps += 1
                    e = None
                    continue
                else:
                    if swap:
                        print("2 couldn't swap for index ", index)
                    g = n.g
                    # print("2uuu", count, index)
                    # drawArborescences(g, "balanced")
                    # sys.stdout.flush()
                    # plt.show()
                    return -1
            else:
                (d, e) = heappop(h[index])
        ni = n.nodes_index(index)
        condition = (e != None and e[0] not in ni and e[1] in ni)
        if cut:
            condition = condition and (
                    K - index == 1 or TestCut(n.rest_graph(index), e[0], n.root) >= K - index - 1)
        if condition:
            n.add_to_index(e[0], e[1], index)
            # print("normal add for index", index, e)
            # print(get_arborescence_dict(g)[index].nodes())
            # print(get_arborescence_dict(g)[index].edges())
            add_neighbors_heap_index(n, h, index, [e[0]])
            index = (index + 1) % K
    swappy.append(swaps)
    g = n.g
    return get_arborescence_list(g)
