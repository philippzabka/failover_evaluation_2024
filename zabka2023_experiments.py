import time
from copy import deepcopy
from datetime import datetime
from objective_function_experiments import *
from FeigenbaumAlg import DAG, FeigenbaumAlg
from TwoResilientAlg import PreComputeTwoResilient, RouteTwoResilient

DEBUG = False

algos = {
    'Feigenbaum': [DAG, FeigenbaumAlg],
    'TwoResilient': [PreComputeTwoResilient, RouteTwoResilient],
    'SquareOne': [PrepareSQ1SD, RouteSQ1]
}


def one_experiment(g, seed, out, algo, f_list, first_string):
    # If failes > edges
    if f_list[0] >= len(g.edges()):
        out.write(', %f, %f, %f, %f, %f, %f\n' %
                  (float('inf'), float('inf'), float('inf'), 0, 0, 0))
        try:
            print('precomputation didnt work', algo[:2], g.graph['seed'], seed)
        except Exception as e:
            print(e)
        return [0]

    dest = g.graph['root']
    nodes = list(set(g.nodes()) - {dest, dest})
    random.shuffle(nodes)
    success_ratios = []

    [precomputation_algo, routing_algo] = algo[:2]
    reset_arb_attribute(g)
    random.seed(seed)
    if DEBUG:
        print('experiment for ', algo[0])
    if routing_algo == RouteDetCircSkip or routing_algo == KeepForwardingRouting:
        stat = Statistic(routing_algo, str(routing_algo), g.to_undirected())
    else:
        g = g.to_undirected()
        stat = Statistic(routing_algo, str(routing_algo))

    precomputation = None
    pt = 0
    if routing_algo != RouteTwoResilient and routing_algo != RouteSQ1:
        # Pre-precomputation needed for SQ1
        pt_start = time.time()
        if routing_algo == FeigenbaumAlg:
            precomputation = precomputation_algo(g, dest)
        else:
            precomputation = precomputation_algo(g)
        pt = time.time() - pt_start
        if precomputation == -1:  # error...
            out.write(', %f, %f, %f, %f, %f, %f\n' %
                      (float('inf'), float('inf'), float('inf'), 0, 0, pt))
            try:
                print('precomputation didnt work', algo[:2], g.graph['seed'], seed)
            except Exception as e:
                print(e)
            return [0]

    for f_num in f_list:
        out.write(first_string)
        if attack == "CLUSTER":
            g.graph['fails'] = targeted_attacks_against_clusters(g, f_num)

        cc_size = len(
            set(connected_component_nodes_with_d_after_failures(g, g.graph['fails'][:f_num], g.graph['root'])))
        stat.reset(g.nodes())
        random.seed(seed)
        if routing_algo == RouteTwoResilient or routing_algo == RouteSQ1:
            pt = 0
        rt = 0
        targeted = False

        # If failes > edges with CLUSTER attack
        if f_num >= len(g.edges()):
            out.write(', %f, %f, %f, %f, %f, %f\n' %
                      (float('inf'), float('inf'), float('inf'), 0, 0, 0))
            try:
                print('precomputation didnt work', algo[:2], g.graph['seed'], seed)
            except Exception as e:
                print(e)
            return [0]

        graph, failures, fails, dist = preprocess_simulate_graph(g, f_num, targeted, dest)

        for s in nodes[:sample_size]:
            # TwoResilient needs precomputation for every source & destination pair
            if routing_algo == RouteTwoResilient or routing_algo == RouteSQ1:
                # pt = 0
                pt_start = time.time()
                precomputation = precomputation_algo(g, s, dest)
                pt += (time.time() - pt_start)
                if precomputation == -1:  # error...
                    out.write(', %f, %f, %f, %f, %f, %f\n' %
                              (float('inf'), float('inf'), float('inf'), 0, 0, pt))
                    try:
                        print('precomputation didnt work', algo[:2], g.graph['seed'], seed)
                    except Exception as e:
                        print(e)
                    return [0]

            rt_start = time.time()
            result, graph = simulate_graph(g=graph, stat=stat, failures=failures, fails=fails, dist=dist,
                                           precomputation=precomputation, source=s, dest=dest, targeted=targeted)
            if -1 == result:
                print('simulate graph returns -1', algo[1], len(g.edges()), f_num)
            rt_end = time.time()
            rt += (rt_end - rt_start)

        postprocess_simulate_graph(graph, stat, failures, targeted)
        num_experiments = (stat.succ + stat.fails)
        if num_experiments < min(len(g.nodes()) - 1, sample_size):
            print(num_experiments, sample_size, cc_size, algo[1], "sample size doesn't match number of experiments")
        if routing_algo == RouteTwoResilient or routing_algo == RouteSQ1:
            pt_ratio = pt / max(num_experiments, 1)
        else:
            pt_ratio = pt

        rt_ratio = rt / max(num_experiments, 1)
        success_ratio = stat.succ / max(num_experiments, 1)

        # Calc fraction of connected nodes in dest component
        graph_copy = deepcopy(graph)
        graph_copy.remove_edges_from(failures.keys())
        fraction_connected = 0
        if nx.is_directed(graph_copy):
            ccs = nx.strongly_connected_components(graph_copy)
            for cc in ccs:
                if dest in cc:
                    fraction_connected = len(cc) / graph_copy.number_of_nodes()
        else:
            ccs = nx.connected_components(graph_copy)
            for cc in ccs:
                if dest in cc:
                    fraction_connected = len(cc) / graph_copy.number_of_nodes()

        if stat.succ > cc_size - 1:
            print('more success %i, than cc size %i, investigate seed' % (algo[1], (stat.succ, cc_size, seed)))
            sys.exit()
        # write results
        if stat.succ > 0:
            if DEBUG:
                print('success', stat.succ, algo[0])
            # stretch, load, hops, success, cc_size/n, routing time, precomputation time
            out.write(', %i, %i, %f, %i, %i, %f, %i, %f, %f, %f\n' %
                      (f_num, np.max(stat.stretch), np.max(stat.stretchNorm), stat.load, np.max(stat.hops),
                       success_ratio, cc_size, fraction_connected, rt_ratio, pt_ratio))
            out.flush()
        else:
            if DEBUG:
                print('no success_ratio', algo[0], 'seed', g.graph['seed'], 'expected ratio <=', cc_size / n)
            out.write(', %i, %f, %f, %f, %f, %f, %i, %f, %f, %f\n' %
                      (f_num, float('inf'), float('inf'), float('inf'), float('inf'), float(0), cc_size,
                       float(fraction_connected), float(rt_ratio), float(pt_ratio)))
        if cc_size > 1:
            success_ratios.append(success_ratio)
        else:
            success_ratios.append(-1)
    return success_ratios


# run experiments with AS graphs (rocketfuel graphs)
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# rep denotes the number of repetitions in the shuffle for loop
def run_AS(out=None, seed=0, rep=5):
    for i in range(4, 5):
        generate_trimmed_AS(i)
    files = glob.glob('./benchmark_graphs/as/AS*.csv')
    original_params = [n, rep, k, sample_size, f_num, seed, name]
    for x in files:
        random.seed(seed)
        kk = int(x[-5:-4])
        g = nx.read_edgelist(x).to_directed()
        g.graph['k'] = kk
        nn = len(g.nodes())
        mm = len(g.edges())
        ss = min(int(nn / 2), sample_size)
        fn = min(int(mm / 4), f_num)
        max_f_num_index = 2
        f_list = [i * f_num for i in range(1, max_f_num_index)]
        fails = random.sample(list(g.edges()), fn)
        g.graph['fails'] = fails
        set_parameters([nn, rep, kk, ss, fn, seed, name + "AS-"])
        shuffle_and_run(g, out, seed, rep, x, f_list)
        set_parameters(original_params)


# run experiments with zoo graphs
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# rep denotes the number of repetitions in the shuffle for loop
def run_zoo(out=None, seed=0, rep=5, min_connectivity=1):
    original_params = [n, rep, k, sample_size, f_num, seed, name]
    if DEBUG:
        print('n_before, n_after, m_after, connectivity, degree')
    for i in range(261):
        random.seed(seed)
        g = read_zoo(i, min_connectivity)
        if g is None:
            continue
        nn = len(g.nodes())
        # only run it on graphs with more than 20 and less than 50 nodes
        if nn < 20 or nn > 300 or (short and len(g.nodes()) > 2500):
            continue
        kk = nx.edge_connectivity(g)
        mm = len(g.edges())
        ss = min(int(nn / 1.1), sample_size)
        step = int(mm / 4)
        step = f_num
        max_f_num_index = 2
        f_list = [i * step for i in range(1, max_f_num_index)]
        print(i, f_list, 'failure number list', time.asctime(time.localtime(time.time())))
        set_parameters([nn, rep, kk, ss, -1, seed, name + "zoo-"])
        results = shuffle_and_run(g, out, seed, rep, str(i), f_list)
        set_parameters(original_params)
        for fn in f_list:
            print('#connected destination component of size 1:', results[fn]['small_cc'] / len(algos), 'repetitions',
                  rep)
            if results[fn]['small_cc'] / len(algos) == rep:
                continue
            print('(mean(cc_size)-1)/(n-1):', (np.mean(results[fn]['cc_size']) - 1) / (n - 1))
            print('mean success ratio, min, std')
            for (algoname, algo) in algos.items():
                scores = results[fn]['scores'][algoname]
                try:
                    print('%.4E, %.4E, %.4E : %s' % (np.mean(scores), np.max(scores), np.std(scores), algoname))
                    algos[algoname] = algo[:2]
                except ValueError:
                    pass
                finally:
                    sys.stdout.flush()
        if short and i > 5:
            break


# shuffle root nodes and run algorithm
def shuffle_and_run(g, out, seed, rep, x, f_list):
    random.seed(seed)
    results = {f: {'scores': {algoname: [] for algoname in algos.keys()}, 'small_cc': 0, 'cc_size': []} for f in f_list}
    nodes = list(g.nodes())
    for count in range(rep):
        g.graph['root'] = nodes[count % len(nodes)]
        for (algoname, algo) in algos.items():
            # graph, size, connectivity, algorithm, index,
            first_string = '%s, %i, %i, %i, %s, %i' % (x, len(g.nodes()), len(g.edges()), g.graph['k'], algoname, count)
            scores = one_experiment(g, seed + rep, out, algo, f_list, first_string)
            for i in range(len(f_list)):
                f = f_list[i]
                if scores[i] > -1:
                    algos[algoname] += [scores[i]]
                    c = [len(set(
                        connected_component_nodes_with_d_after_failures(g, g.graph['fails'][:f], g.graph['root'])))]
                    results[f]['scores'][algoname] += [scores[i]]
                    results[f]['cc_size'] += c
                else:
                    results[f]['small_cc'] += 1
    return results

    # run experiments with ring of cliques graphs


# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# rep denotes the number of repetitions in the secondary for loop
def run_ring_of_cliques(out=None, seed=0, rep=5, k1_list=[], k2_list=[], l_list=[], f_list=[]):
    results = {}
    for k1 in k1_list:
        for k2 in k2_list:
            for l in l_list:
                for i in range(rep):
                    random.seed(seed + i)
                    g = create_ring_of_cliques(l, k1, k2, seed + i)
                    count = 0
                    random.seed(seed + i)
                    while round_robin(g, cut=True, swap=True) == -1:
                        count += 1
                        g = create_ring_of_cliques(l, k1, k2, count * seed + i)
                        if count > 1:
                            print('bonsai count', count)
                        random.seed(seed + i)
                    n = len(g.nodes())
                    k = 2 * k2
                    if f_list == []:
                        f_list = [f_num]
                    results = {f: {'scores': {algoname: [] for algoname in algos.keys()}, 'small_cc': 0, 'cc_size': []}
                               for f in f_list}
                    set_parameters([len(g.nodes), rep, k, sample_size, f_num, seed, name + "ring-" + str(seed) + "-"])
                    random.seed(seed + i)
                    for (algoname, algo) in algos.items():
                        # graph, size, connectivity, algorithm, index,
                        first_string = '%s, %i, %i, %i, %i, %i, %i, %s, %i' % ("ring", k1, k2, l, n, len(g.edges()),
                                                                               k, algoname, i)
                        scores = one_experiment(g, seed + i, out, algo, f_list, first_string)
                        for i in range(len(f_list)):
                            f = f_list[i]
                            score = scores[i]
                            if score > -1:
                                algos[algoname] += [score]
                                c = [len(set(connected_component_nodes_with_d_after_failures(g, g.graph['fails'][:f],
                                                                                             g.graph['root'])))]
                                results[f]['scores'][algoname] += [score]
                                results[f]['cc_size'] += c
                        else:
                            results[f]['small_cc'] += 1
    return results


# run experiments with erdos-renyi graphs
# out denotes file handle to write results to
# rep denotes the number of repetitions in the secondary for loop
def run_erdos_renyi(out=None, rep=5):
    ss = min(int(n / 2), sample_size)
    fn = min(int(n * k / 4), f_num)
    set_parameters([n, rep, k, ss, fn, seed, name + "erdos-renyi"])
    write_erdos_renyi_graphs()
    f_list = [f_num]
    for i in range(rep):
        g = read_erdos_renyi_graph(i)
        for (algoname, algo) in algos.items():
            # graph, size, connectivity, algorithm, index,
            first_string = '%s, %i, %i, %i, %s, %i' % ("erdos-renyi", n, len(g.edges()), nx.node_connectivity(g),
                                                       algoname, i)
            algos[algoname] += [one_experiment(g, seed + i, out, algo, f_list, first_string)]


# run experiments with d-regular graphs
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# rep denotes the number of repetitions in the secondary for loop
def run_regular(out=None, seed=0, rep=5):
    ss = min(int(n / 2), sample_size)
    fn = min(int(n * k / 4), f_num)
    set_parameters([n, rep, k, ss, fn, seed, name + "regular-"])
    write_graphs()
    f_list = [f_num]
    for i in range(rep):
        random.seed(seed + i)
        g = read_graph(i)
        random.seed(seed + i)
        for (algoname, algo) in algos.items():
            # graph, size, connectivity, algorithm, index,
            first_string = '%s, %i, %i, %i, %s, %i' % ("regular", n, len(g.edges()), k, algoname, i)
            algos[algoname] += [one_experiment(g, seed + i, out, algo, f_list, first_string)]


# Custom targeted link failure model
# Return the selected links incident to nodes with a non-zero value of the
# clustering coefficient
def targeted_attacks_against_clusters(g, f_num):
    candidate_links_to_fail = list()
    links_to_fail = list()
    clustering_coefficients = nx.clustering(g)
    for (v, cc) in clustering_coefficients.items():
        if cc == 0.0:
            continue
        neighbors = nx.neighbors(g, v)
        for w in neighbors:
            if not (v, w) in candidate_links_to_fail and not (w, v) in candidate_links_to_fail:
                candidate_links_to_fail.append((v, w))
    # Select up to f_num bi-directional links that should be disabled
    if len(candidate_links_to_fail) > f_num:
        links_to_fail = random.sample(candidate_links_to_fail, f_num)
    else:
        links_to_fail.extend(candidate_links_to_fail)
    # Append the opposite arcs to the list (we assume failures affect links in both directions)
    for (v, w) in links_to_fail:
        if not (w, v) in candidate_links_to_fail:
            links_to_fail.append((w, v))
    return links_to_fail


# start file to capture results
def start_file(filename):
    out = open(filename + ".csv", 'w')
    if 'ring' in filename:
        out.write(
            "graph, k1, k2, l, size, edge_size, connectivity, algorithm, repetition, failures, " +
            "stretch, stretch_norm, load, hops, success, cc_size, frac_conn_nodes, " +
            "routing_time, precomputation_time,\n")
    else:
        out.write(
            "graph, size, edge_size, connectivity, algorithm, repetition, failures, " +
            "stretch, stretch_norm, load, hops, success, cc_size, frac_conn_nodes, " +
            "routing_time, precomputation_time\n")
    # times are in seconds
    return out


# run experiments
# seed is used for pseudorandom number generation in this run
# switch determines which experiments are run
def experiments(switch="all", seed=0, rep=100):
    global n
    date_now = datetime.now()
    formatted_date = date_now.strftime("%Y-%m-%d-%H:%M:%S")

    if switch in ["erdos-renyi", "all"]:
        out_str = ("results_erdos_renyi/" + name + "erdos-renyi-" + "-" + str(n) + "-" +
                   str(rep) + "-" + str(sample_size) + "-" + str(f_num) + "-" + attack + "-" + formatted_date)
        out = start_file(out_str)
        run_erdos_renyi(out=out, rep=rep)
        out.close()

    if switch in ["regular", "all"]:
        out_str = ("results_regular/" + name + "regular-seed-" + str(seed) + "-" + str(n) + "-" + str(k) + "-" +
                   str(rep) + "-" + str(sample_size) + "-" + str(f_num) + "-" + attack + "-" + formatted_date)
        out = start_file(out_str)
        run_regular(out=out, seed=seed, rep=rep)
        out.close()

    if switch in ["zoo", "all"]:
        min_connectivity = 1
        out_str = ("results_zoo/" + name + "zoo-min-connectivity-" + str(min_connectivity) + "-seed-" + str(seed)
                   + "-" + str(rep) + "-" + str(sample_size) + "-" + str(f_num) + "-" + attack +
                   "-" + formatted_date)
        out = start_file(out_str)
        run_zoo(out=out, seed=seed, rep=rep, min_connectivity=min_connectivity)
        out.close()

    if switch in ["AS"]:
        out_str = ("results_as/" + name + "AS-seed-" + str(seed) + "-" + str(rep) + "-" + str(sample_size) +
                   "-" + str(f_num) + "-" + attack + "-" + formatted_date)
        out = start_file(out_str)
        run_AS(out=out, seed=seed, rep=rep)
        out.close()

    if switch in ["ring", "all"]:
        k1 = 10
        out_str = ("results_ring/" + name + "ring-seed" + str(seed) + "-k1-k2-ratio-" + str(k) + "-" +
                   str(rep) + "-" + str(sample_size) + "-" + str(f_num) + "-" + attack + "-" + formatted_date)
        out = start_file(out_str)
        for k2 in [2]:
            for l in [10]:
                n = l * k1
                m = l * k1 * (k1 - 1) / 2 + l * k2
                step = 10
                max_f_num_index = int(np.ceil((m - n) / step))
                print('k1=', k1, 'k2=', k2, 'l=', l, 'n=', n, 'm=', m)
                f_list = [i * step for i in range(1, max_f_num_index)]
                results = run_ring_of_cliques(out=out, seed=seed, rep=rep, k1_list=[k1], k2_list=[k2], l_list=[l],
                                              f_list=f_list)
                for fn in f_list:
                    print('#connected destination component of size 1:', results[fn]['small_cc'] / len(algos))
                    print('(mean(cc_size)-1)/(n-1):', (np.mean(results[fn]['cc_size']) - 1) / (n - 1))
                    print('mean success ratio, min, std')
                    for (algoname, algo) in algos.items():
                        scores = results[fn]['scores'][algoname]
                        print('%.4E, %.4E, %.4E : %s' % (np.mean(scores), np.max(scores), np.std(scores), algoname))
                        algos[algoname] = algo[:2]
                        sys.stdout.flush()
        out.close()
        return

    print(attack)
    for (algoname, algo) in algos.items():
        print('%.5E %s' % (np.mean(algo[2:]), algoname))
    print("\nlower is better")


if __name__ == "__main__":
    # default arguments
    switch = 'all'  # which experiments to run with same parameters [all, regular, zoo, ringi, AS]
    seed = 0  # random seed
    rep = 100  # number of experiments
    n = 100  # number of nodes
    sample_size = 20  # number of sources to route a packet to destination
    f_num = 40  # number of failed links
    attack = 'RANDOM'  # how edge failures are chosen
    short = False  # if true only small zoo graphs < 25 nodes are run

    # default values, not setable
    k = 3  # base connectivity
    name = "zabka-"  # result files start with this name

    start = time.time()
    print(time.asctime(time.localtime(start)))
    if len(sys.argv) > 1:
        switch = sys.argv[1]
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])
    if len(sys.argv) > 3:
        rep = int(sys.argv[3])
    if len(sys.argv) > 4:
        n = int(sys.argv[4])
    if len(sys.argv) > 5:
        sample_size = int(sys.argv[5])
    if len(sys.argv) > 6:
        f_num = int(sys.argv[6])
    if len(sys.argv) > 7:
        attack = sys.argv[7]  # [RANDOM, CLUSTER]
    if len(sys.argv) > 8:
        short = sys.argv[8] == 'True'  # True or False
    print(time.asctime(time.localtime(start)), 'attack', attack)
    random.seed(seed)
    set_parameters([n, rep, k, sample_size, f_num, seed, name])
    experiments(switch=switch, seed=seed, rep=rep)
    end = time.time()
    print("time elapsed", end - start)
    print("start time", time.asctime(time.localtime(start)))
    print("end time", time.asctime(time.localtime(end)))

    # example call: python zabka2023_experiments.py zoo 45 10 100 20 20 RANDOM False
