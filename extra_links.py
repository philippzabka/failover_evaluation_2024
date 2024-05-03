from objective_function_experiments import *
import sys
import networkx as nx

# keep forwarding routing
def KeepForwardingRouting(s, d, fails, precomp, g):
    verbose = 0  # set to 0 for normal, to 1 for verbose output (latter limits hops to 20)
    [label_size, label, edge_weight, node_weight, down_links, A_links, up_links] = precomp
    hops = 0
    switches = 0  # doesn't make sense in this context, keep it so it fits with other data strucutrures
    failure_encountered = False
    detour_edges = []  # add edges taken to this list when the first failure has been encountered...
    n = len(g.nodes())
    incoming_link = (s, s)
    incoming_node = s
    if verbose == 1: print(
        ' ################################################################ start new experiment with source ' + str(
            s) + ' and destination ' + str(d))
    while (s != d):

        if verbose == 1: print(
            'Start new try to find a next link ++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # remove incoming node from all link lists
        curr_dl = list(down_links[s])
        if incoming_node in curr_dl:
            curr_dl.remove(incoming_node)
        curr_al = list(A_links[s])
        if incoming_node in curr_al:
            curr_al.remove(incoming_node)
        curr_ul = list(up_links[s])
        if incoming_node in curr_ul:
            curr_ul.remove(incoming_node)

        # sort up/down according to weights (higher->earlier) and a-list according to labels (lower->earlier)  #maybe refactor to only sort if there is a failure to speed  up...
        curr_dl = sorted(curr_dl, key=lambda x: int(node_weight[x]), reverse=True)
        for t in curr_dl:
            if verbose == 1: print(
                'The weight of down-node ' + str(t) + ' is ' + str(node_weight[t]))
        curr_al = sorted(curr_al, key=lambda x: int(label[(s, x)]),
                         reverse=False)  # KeyError: (670, 670) ...???? Maybe related to preprocessing..?
        for t in curr_al:
            if verbose == 1: print(
                'The label of a-node ' + str(t) + ' is ' + str(label[(s, t)]))
        curr_ul = sorted(curr_ul, key=lambda x: int(node_weight[x]), reverse=True)
        for t in curr_ul:
            if verbose == 1: print(
                'The weight of up-node ' + str(t) + ' is ' + str(node_weight[t]))

        # init for a links
        a_overflow = 0  # counter, to try all a-links only once
        a_count = -1  # init counter for label of link (for safety)
        if incoming_link in list(A_links[incoming_node]):  # if incoming was a-link, get correct counter
            a_count = label[incoming_link]  # get label from incoming link
        elif len(curr_al) > 0:
            a_count = label[(s, curr_al[0])]
        if len(curr_dl) > 0:  # if down list is not empty, set nxt as first element of down list
            curr_list = curr_dl
            nxt = curr_list[0]
            curr_index = 0  # 0 for down, 1 for A, 2 for up
            if verbose == 1: print('I try the down link from ' + str(s) + ' to ' + str(nxt))
        elif len(
                curr_al) > 0:  # if a list is not empty, set nxt as next a link: if incoming is a-link, then next, else, as first element of down list
            curr_list = curr_al
            if incoming_link in list(A_links[incoming_node]):  # if incoming was a link
                a_count = (a_count + 1) % label_size[incoming_link]  # increase counter by 1
                nxt = next(i for i in list(A_links[s]) if label[(s, i)] == a_count)
            else:  # if incoming was not a-link
                nxt = curr_list[0]
                a_count = label[(s, curr_list[0])]
            curr_index = 1  # 0 for down, 1 for A, 2 for up
        elif len(curr_ul) > 0:  # if a list is not empty, set nxt as first element of down list
            curr_list = curr_ul
            nxt = curr_list[0]
            curr_index = 2  # 0 for down, 1 for A, 2 for up
            if verbose == 1: print('I try the up link from ' + str(s) + ' to ' + str(nxt))
        else:  # note: this should not happen, as we did not yet check if the next link is failed, but added for good measure...
            nxt = incoming_node
            curr_index = 3
            if verbose == 1: print('Oh no: Only the incoming edge is left to take from ' + str(s) + ' to ' + str(
                nxt) + ', even though the last hop was from ' + str(
                incoming_node))
        if verbose == 1: print('Currently s is ' + str(s) + ' and nxt is ' + str(nxt) + ' and d is ' + str(
            d) + ' and the incoming node is ' + str(incoming_node))

        while (s, nxt) in fails or (nxt, s) in fails:
            if verbose == 1: print(' ### failure on the link ' + str((s, nxt)))
            if curr_index == 0:  # down_links usage
                if curr_list.index(nxt) < len(curr_list) - 1:  # are there elements left?
                    if incoming_node != curr_list[curr_list.index(nxt) + 1]:
                        nxt = curr_list[curr_list.index(nxt) + 1]  # next item from down_links
                        if verbose == 1: print(
                            '#################### found another down link################# from ' + str(
                                s) + ' to ' + str(nxt))

                elif a_count > -1:
                    if verbose == 1: print('No elements left in down_link: I switch to a_links')
                    curr_index = 1
                    curr_list = curr_al
                    if a_count == label[(s, curr_al[0])]:
                        nxt = next(
                            i for i in list(A_links[s]) if label[(s, i)] == a_count)
                        if verbose == 1: print('in down-loop, the first to try a-link is from ' + str(s) + ' to ' + str(
                            nxt) + ' with an a_count of ' + str(a_count))
                    else:
                        a_count = (a_count + 1) % label_size[incoming_link]  # increase counter by 1
                        nxt = next(
                            i for i in list(A_links[s]) if label[(s, i)] == a_count)
                        if verbose == 1: print(
                            'the a-link is from ' + str(s) + ' to ' + str(nxt) + ' with an a_count of ' + str(a_count))

                else:
                    if verbose == 1: print('No elements left in a_link or a_link empty: I switch to up_links')
                    curr_index = 2
                    curr_list = curr_ul  # list(up_links[s])
                    if len(curr_list) > 0:
                        nxt = curr_list[0]
                        if verbose == 1: print('the up-link is from ' + str(s) + ' to ' + str(nxt))
                    else:
                        nxt = incoming_node
                        curr_index = 3
                        if verbose == 1: print(
                            'Oh no: Only the incoming edge is left to take from ' + str(s) + ' to ' + str(
                                nxt) + ', even though the last hop was from ' + str(incoming_node))

            elif curr_index == 1:
                if a_overflow < label_size[(s, curr_list[0])]:
                    a_overflow = a_overflow + 1
                    if curr_list.index(nxt) < len(curr_list) - 1:
                        if verbose == 1: print('the current index is ' + str(
                            curr_list.index(nxt)) + ' and the length of the current list is ' + str(
                            len(curr_list)) + ' and the label is ' + str(label[(s, nxt)]))
                        if verbose == 1: print('the next element is ' + str(
                            curr_list[curr_list.index(nxt) + 1]) + ' with an index of ' + str(curr_list.index(nxt) + 1))
                        nxt = next(i for i in list(curr_list) if label[(s, i)] > a_count)
                        a_count = label[(s, nxt)]
                        if verbose == 1: print('trigger 1: the a-link is from ' + str(s) + ' to ' + str(
                            nxt) + ' with an a_count of ' + str(a_count))
                    else:
                        nxt = curr_list[0]
                        a_count = label[(s, nxt)]
                    if verbose == 1: print('the a-link is from ' + str(s) + ' to ' + str(nxt))

                else:
                    if verbose == 1: print('No elements left in a_link: I switch to up_links')
                    curr_index = 2
                    curr_list = curr_ul  # list(up_links[s])
                    if len(curr_list) > 0:
                        nxt = curr_list[0]
                        if verbose == 1: print('the up-link is from ' + str(s) + ' to ' + str(nxt))
                    else:
                        nxt = incoming_node
                        curr_index = 3
                        if verbose == 1: print(
                            'Oh no: Only the incoming edge is left to take from ' + str(s) + ' to ' + str(
                                nxt) + ', even though the last hop was from ' + str(incoming_node))

            elif curr_index == 2:  # up_links usage
                if curr_list.index(nxt) < len(curr_list) - 1:  # are there elements left?
                    if incoming_node != curr_list[curr_list.index(nxt) + 1]:
                        nxt = curr_list[curr_list.index(nxt) + 1]  # next item from down_links
                        if verbose == 1: print('#################### found another up link#################')
                    else:
                        if verbose == 1: print('oh no (up) only incoming is alive')
                        curr_index = 3
                        nxt = incoming_node
                else:
                    nxt = incoming_node
                    curr_index = 3
                    if verbose == 1: print(
                        'Oh no: Only the incoming edge is left to take from ' + str(s) + ' to ' + str(
                            nxt) + ', even though the last hop was from ' + str(incoming_node))
            else:
                print('Error: Nxt is ' + str(nxt) + ' current node is ' + str(s))
                sys.exit()

        if failure_encountered:
            detour_edges.append((s, nxt))
        hops += 1
        n_end = n * n + 20
        if verbose == 1: n_end = 20
        if hops > n_end:  # n*n*n:  #to kill early, later set back to n*n*n
            # probably a loop, return
            if verbose == 1: print(
                '********************************************'''''''''''''''''''' I am stuck in a loop with many hops, good bye')
            return (True, -1, switches, detour_edges)
        incoming_link = (s, nxt)
        incoming_node = s
        if verbose == 1: print('Great success: Next hop is alive: I will go from ' + str(s) + ' to ' + str(nxt))
        s = nxt
    if verbose == 1: print('~~~~~~~~~~~~~~~~~~~~ Destination reached!~~~~~~~~~~~~~~~~~~~~~~~~~')
    return (False, hops, switches, detour_edges)


# Route according to deterministic circular routing, skip current arborescence if no neighbors.
# source s
# destination d
# link failure set fails
# arborescence decomposition T
def RouteDetCircSkip(s, d, fails, T, g):
    curT = 0
    detour_edges = []
    hops = 0
    switches = 0
    k = len(T)
    if k == 0:
        return (True, -2, switches, detour_edges)
    n = max([len(T[i].nodes()) for i in range(k)])
    dist = nx.shortest_path_length(g, target=d)
    # print('nodes', g.nodes())
    # print('dist', dist.keys())
    # print('s,d', s, d )
    # drawGraphWithLabels(g,"tst.png")
    while (s != d):
        while (s not in T[curT].nodes()) and switches < k * n:
            curT = (curT + 1) % k
            switches += 1
        if switches >= k * n:
            break
        nxt = list(T[curT].neighbors(s))
        if len(nxt) == 0:
            # print("Warning: no neighbours available --> switching to the next tree")
            curT = (curT + 1) % k
            switches += 1
            continue
        if (d, s) in g.edges() and not ((d, s) in fails or (s, d) in fails):
            nxt = [d] + nxt
        if len(nxt) == 0:
            curT = (curT + 1) % k
            switches += 1
            break
        breaking = False
        # remove bad nodes from list
        len_nxt = len(nxt)
        nxt = [x for x in nxt if x in dist.keys()]
        if len(nxt) < len_nxt:
            print('shortened')
            nx.write_edgelist(g, "somethingwrong.csv")
            drawGraphWithLabels(g, "somethingwrong.png")
            if len(nxt) == 0:
                curT = (curT + 1) % k
                switches += 1
                break
        # sort list of next hops by distance
        nxt = sorted(nxt, key=lambda ele: dist[ele])
        index = 0
        while (nxt[index], s) in fails or (s, nxt[index]) in fails:
            index = index + 1
            if index >= len(nxt):
                curT = (curT + 1) % k
                switches += 1
                breaking = True
                break
        if not breaking:
            if switches > 0 and curT > 0:
                detour_edges.append((s, nxt[index]))
            s = nxt[index]
            hops += 1
        if hops > n * n or switches > k * n:
            return (True, -1, switches, detour_edges)
    return (False, hops, switches, detour_edges)


def drawArborescences(g, pngname="results/weighted_graph.png"):
    plt.clf()
    k = g.graph['k']
    if 'k1' in g.graph.keys():
        k = g.graph['k1']
    edge_labels = {i: {} for i in range(k)}
    edge_labels[-1] = {}
    for e in g.edges():
        arb = g[e[0]][e[1]]['arb']
        edge_labels[arb][(e[0], e[1])] = ""
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'pink', 'olive',
              'brown', 'orange', 'darkgreen', 'navy', 'purple']
    if 'pos' not in g.graph:
        g.graph['pos'] = nx.spring_layout(g)
    pos = g.graph['pos']
    nx.draw_networkx_labels(g, pos)
    nodes = list(g.nodes)
    node_colors = {v: 'gray' for v in nodes}
    for node in nodes:
        if is_complete_node(g, node):
            node_colors[node] = 'black'
    color_list = [node_colors[v] for v in nodes]
    nx.draw_networkx_nodes(g, pos, nodelist=nodes, alpha=0.6,
                           node_color=color_list, node_size=2)
    for j in range(k):
        edge_j = [(u, v) for (u, v, d) in g.edges(data=True) if d['arb'] == j]
        nx.draw_networkx_labels(g, pos)
        nx.draw_networkx_edges(g, pos, edgelist=edge_j,
                               width=1, alpha=0.5, edge_color=colors[j])
    plt.axis('off')
    plt.savefig(pngname)  # save as png
    plt.close()
    for j in range(k):
        edge_j = [(u, v) for (u, v, d) in g.edges(data=True) if d['arb'] == j]
        nx.draw_networkx_labels(g, pos)
        nx.draw_networkx_edges(g, pos, edgelist=edge_j, width=1,
                               alpha=0.5, edge_color=colors[j])  # , arrowsize=20)
        plt.savefig(pngname + str(j) + '.png')  # save as png
        plt.close()
