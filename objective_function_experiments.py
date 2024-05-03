import math
import glob
import fnss
from routing import *

# global variables
seed = 1
n = 10
rep = 1
k = 8
f_num = 40
samplesize = 20
name = "experiment-objective-function"


# set global parameters in this file and in routing_stats
def set_parameters(params):
    set_objective_parameters(params)


# set global parameters in this file and in routing_stats
def set_objective_parameters(params):
    global seed, n, rep, k, samplesize, name, f_num
    [n, rep, k, samplesize, f_num, seed, name] = params
    set_routing_params(params)


# generate rep random k-regular graphs with connectivity k using seed and
# write them to file
def write_graphs():
    d = []
    ecc = []
    sp = []
    path = "benchmark_graphs/regular/"
    # directory = os.listdir(path)
    # if len(directory) == 0:
    for i in range(rep):
        g = nx.random_regular_graph(k, n).to_directed()
        while nx.edge_connectivity(g) < k:
            g = nx.random_regular_graph(k, n).to_directed()
        prepare_graph(g, k, 0)
        GreedyArborescenceDecomposition(g)
        d.append(depth(g))
        ecc.append(nx.eccentricity(g, 0))
        sp.append(nx.average_shortest_path_length(g))
        s = ''
        for e in g.graph['fails']:
            s = s + str(e[0]) + ' ' + str(e[1]) + '\n'
        f = open(path + name + str(seed) + '_graph_' +
                 str(n) + '_' + str(k) + "_" + str(i) + '.txt', 'w')
        f.write(s[:-1])
        f.close()


def write_erdos_renyi_graphs():
    d = []
    ecc = []
    sp = []
    path = "benchmark_graphs/erdos-renyi/"
    # directory = os.listdir(path)
    # if len(directory) == 0:
    for i in range(rep):
        p = calculate_p()
        g = nx.erdos_renyi_graph(n, p)
        while not nx.is_connected(g):
            g = nx.erdos_renyi_graph(n, p)
        # Use p as seed here
        prepare_graph(g, nx.node_connectivity(g), p)
        GreedyArborescenceDecomposition(g)
        d.append(depth(g))
        ecc.append(nx.eccentricity(g, 0))
        sp.append(nx.average_shortest_path_length(g))
        s = ''
        for e in g.graph['fails']:
            s = s + str(e[0]) + ' ' + str(e[1]) + '\n'
        f = open(path + name + '_graph_' +
                 str(n) + "_" + str(round(p, 2)) + "_" + str(i) + '.txt', 'w')
        f.write(s[:-1])
        f.close()


# Calculate the probability p for an Erdős–Rényi graph G(n, p) to be almost surely connected.
# n (int): Number of nodes in the graph.
# epsilon (float): Small positive constant to ensure the graph is above the connectivity threshold.
# increment: Small increment to ensure that p is sliglthy above threshold
def calculate_p(epsilon=0.1, increment=0.0001):
    if n <= 1:
        return 0  # A graph with 1 or fewer nodes doesn't need edges to be connected.
    return ((1 + epsilon) * math.log(n) / n) + increment


# read generated k-regular graphs from file system
def read_graph(i):
    path = "benchmark_graphs/regular/"
    g = nx.read_edgelist(path + name + str(seed) + '_graph_' +
                         str(n) + '_' + str(k) + "_" + str(i) + '.txt', nodetype=int).to_directed()
    for (u, v) in g.edges():
        g[u][v]['arb'] = -1
    g.graph['seed'] = 0
    g.graph['k'] = k
    g.graph['root'] = 0
    fails = []
    f = open(path + name + str(seed) +
             '_graph_' + str(n) + '_' + str(k) + "_" + str(i) + '.txt', 'r')
    for line in f:
        s = line.replace('\n', '').split(' ')
        fails.append((int(s[0]), int(s[1])))
    f.close()
    g.graph['fails'] = fails
    return g


def read_erdos_renyi_graph(i):
    path = "benchmark_graphs/erdos-renyi/"
    g = nx.read_edgelist(path + name + '_graph_' +
                         str(n) + '_' + str(round(calculate_p(), 2)) + "_" + str(i) + '.txt',
                         nodetype=int).to_directed()
    for (u, v) in g.edges():
        g[u][v]['arb'] = -1
    g.graph['seed'] = 0
    g.graph['k'] = nx.node_connectivity(g)
    g.graph['root'] = 0
    fails = []
    f = open(path + name + '_graph_' + str(n) + '_' + str(round(calculate_p(), 2)) + "_" + str(i) + '.txt', 'r')
    for line in f:
        s = line.replace('\n', '').split(' ')
        fails.append((int(s[0]), int(s[1])))
    f.close()
    g.graph['fails'] = fails
    return g


# generate random ring of clique graphs with n nodes and connectivity k1-1
# in cliques and k2 between neighboring cliques
def create_ring_of_cliques(l, k1, k2, seed):
    # print('l', l, 'k1', k1, 'k2', k2)
    if k2 >= k1 * k1:
        print('k2 must be at most k1*k1 for create_ring_of_cliques')
        sys.exit()
    n = l * (k1)
    m = l * (k1 * (k1 - 1) / 2 + k2)
    random.seed(seed)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(l):
        ## wire inside each clique
        for u in range(i * k1, (i + 1) * k1):
            for v in range(u, (i + 1) * k1):
                g.add_edge(u, v)
        ## wire between cliques
        if i > 0:
            for j in range(k2):
                u = random.choice(range(i * k1, (i + 1) * k1))
                v = random.choice(range((i - 1) * k1, (i) * k1))
                while (u, v) in g.edges():
                    u = random.choice(range(i * k1, (i + 1) * k1))
                    v = random.choice(range((i - 1) * k1, (i) * k1))
                g.add_edge(u, v)
        else:
            for j in range(k2):
                u = random.choice(range(0, k1))
                v = random.choice(range((l - 1) * k1, (l) * k1))
                while (u, v) in g.edges():
                    u = random.choice(range(0, k1))
                    v = random.choice(range((l - 1) * k1, (l) * k1))
                g.add_edge(u, v)
    # n selfloops to be removed
    g.remove_edges_from(nx.selfloop_edges(g))
    if (len(g.edges()) != m):
        print("Bug in ring of clique generation")
        sys.exit()
    g = g.to_directed()
    prepare_graph(g, 2 * k2, seed)
    return g


# set attributes for algorithms
def prepare_graph(g, k, seed):
    g.graph['seed'] = seed
    g.graph['k'] = k
    g.graph['root'] = 0
    g2 = g.to_undirected()
    g2.remove_edges_from(nx.selfloop_edges(g2))
    fails = list(g2.edges())
    random.seed(seed)

    good = False
    count = 0
    while not good:
        count += 1
        random.shuffle(fails)
        G = g.to_undirected()
        n = len(g.nodes())
        G.remove_edges_from(fails[:n])
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        if 0 in Gcc[0]:
            good = True
        elif count > 10:
            g.graph['root'] = list(Gcc[0])[0]
            good = True
    g.graph['fails'] = fails


# return j th zoo graph if it can be trimmed into a graph of connectivity at least 4 and at
# least 10 nodes
def read_zoo(j, min_connectivity):
    path = "./benchmark_graphs/zoo/"
    zoo_list = list(glob.glob(path + "*.graphml"))
    if len(zoo_list) == 0:
        print("Add Internet Topology Zoo graphs (*.graphml files) to directory benchmark_graphs")
        print("Download them from: http://www.topology-zoo.org/dataset.html")
        sys.exit()
    if len(zoo_list) <= j:
        return None
    g1 = nx.Graph(nx.read_graphml(zoo_list[j]))
    g2 = nx.convert_node_labels_to_integers(g1)
    g2.remove_edges_from(nx.selfloop_edges(g2))
    g2 = g2.to_directed()
    n_before = len(g2.nodes)
    degree = min(1, min_connectivity)
    while nx.edge_connectivity(g2) < min_connectivity:
        g2 = trim2(g2, degree)
        if len(g2.nodes) == 0:
            return None
        degree += 1
    g = g2.to_directed()
    print(j, zoo_list[j], 'n_before=', n_before, 'n_after=', len(g.nodes), 'm_after=', len(g.edges), 'connectivity=',
          nx.edge_connectivity(g2), 'degree=', degree)
    for (u, v) in g.edges():
        g[u][v]['arb'] = -1
    prepare_graph(g, nx.edge_connectivity(g), seed)
    g.graph['undirected failures'] = False
    g.graph['pos'] = nx.spring_layout(g)
    return g


# read AS graphs and trims them to be of connectivity at least conn
def generate_trimmed_AS(conn):
    path = "benchmark_graphs/rocket_fuel/"
    files = glob.glob(path + '*.cch')
    if len(files) == 0:
        print("Add Rocketfuel Graphs (*.cch) to directory benchmark_graphs.")
        print("Download them from: https://research.cs.washington.edu/networking/rocketfuel/")
        sys.exit()
    for x in files:
        if 'r0' in x or 'r1' in x or 'pop' in x or 'README' in x:
            continue
        g = nx.Graph()
        print(x)
        g.add_edges_from(fnss.parse_rocketfuel_isp_map(x).edges())
        gt = trim_merge(g, conn)
        # relabelling
        gtL = nx.convert_node_labels_to_integers(gt)
        if (gtL.number_of_nodes() == 0):
            print("AS-Graph %s contains no node after trimming" % x)
            continue
        if (gtL.number_of_nodes() >= 1000):
            print("AS-Graph %s contains too many nodes" % x, gtL.number_of_nodes())
            continue
        if (nx.edge_connectivity(gtL) < conn):
            print("AS-Graph %s is not connected enough for connectivity %i" % (x, conn))
            continue
        else:
            print("AS-Graph %s with %i nodes is good" % (x, gtL.number_of_nodes()))
            nx.write_edgelist(gtL, x[:-4].replace("graphs/rocket_fuel/", "graphs/as/AS") + "-" + str(conn) + ".csv")
