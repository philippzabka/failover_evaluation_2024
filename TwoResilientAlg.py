import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from typing import Union
from copy import deepcopy

'''
Implementation of the algorithm described in https://dl.acm.org/doi/abs/10.1145/3558481.3591080
Title: A Tight Characterization of Local Fast Failover Routing: Resiliency to Two Link Failures is Possible
To use it in benchmark_template.py, import TwoResilientAlg
and extend algos with 'TwoResilient': [PreComputeTwoResilient, RouteTwoResilient]
The first entry of the list is the precomputation algorithm, the second the actual routing algorithm.
The graph needs to be precomputed for every source & destination pair.
Precomputation returns a list of subgraphs of an undirected graph G for given source & destination pair.
Implemented by Philipp Zabka
'''

def get_edge_connectivity(g: nx.Graph, source: Union[str, int], target: Union[str, int]) -> int:
    return len(get_edge_disjoint_paths(g, source, target))


def get_node_connectivity(g: nx.Graph, source: Union[str, int], target: Union[str, int]) -> int:
    return len(get_node_disjoint_paths(g, source, target))


# Get all edge disjoint paths in graph between source & target
def get_edge_disjoint_paths(g: nx.Graph, source: Union[str, int], target: Union[str, int]) -> []:
    return sorted(nx.edge_disjoint_paths(g, source, target), key=len)


# Get all node disjoint paths in graph between source & target
def get_node_disjoint_paths(g: nx.Graph, source: Union[str, int], target: Union[str, int]) -> []:
    return sorted(nx.node_disjoint_paths(g, source, target), key=len)


# Lemma 4.2
# Find 2-(s,t)-cuts in P1 & P2 in graph
def get_two_s_t_cuts(g: nx.Graph, source: Union[str, int], target: Union[str, int]) -> []:
    node_disjoint_paths = get_node_disjoint_paths(g, source, target)
    path_p1 = get_edge_path(node_disjoint_paths[0])
    path_p2 = get_edge_path(node_disjoint_paths[1])
    two_s_t_cuts = []
    g_copy = g.copy()
    for p1_edge in path_p1:
        g_copy.remove_edges_from([p1_edge])
        for p2_edge in path_p2:
            g_copy.remove_edges_from([p2_edge])
            if not nx.has_path(g_copy, source, target):
                two_s_t_cuts.append([p1_edge, p2_edge])
            g_copy.add_edges_from([p2_edge])
        g_copy.add_edges_from([p1_edge])

    return two_s_t_cuts


# Extract connected components from graph based on 2-(s,t)-cuts
def get_connected_components(g: nx.Graph, two_s_t_cuts: []) -> []:
    g_copy = g.copy()
    cuts = set()
    [(cuts.add(x[0]), cuts.add(x[1])) for x in two_s_t_cuts]
    [g_copy.remove_edge(edge[0], edge[1]) for edge in cuts]
    connected_components = list(nx.biconnected_components(g_copy))
    connected_components = join_components(g_copy, connected_components)
    # copy otherwise reference problems again
    connected_components = [g.subgraph(component).copy() for component in connected_components]
    return connected_components


# Join components on shared nodes
def join_components(g: nx.Graph, components: []) -> []:
    # create a dictionary that maps each node to the component it belongs to
    node_to_component = {}
    for i, component in enumerate(components):
        for node in component:
            node_to_component[node] = i

    # create a list of sets to represent the merged components
    merged_components = [set(component) for component in components]

    # iterate over all nodes and merge the components they belong to
    for node in node_to_component:
        component_i = node_to_component[node]
        for neighbor in g.neighbors(node):
            component_j = node_to_component.get(neighbor)
            if component_j is not None and component_j != component_i:
                merged_components[component_i].update(merged_components[component_j])
                for neighbor_node in merged_components[component_j]:
                    node_to_component[neighbor_node] = component_i
                merged_components[component_j].clear()

    # remove empty components and return the list of merged components
    return [component for component in merged_components if len(component) > 0]


# In Def. 4.3
# Set C includes all components where the amount of nodes intersected
# with the union of paths P & Q (P1 & P2) is >= 2
def get_connected_component_set(g: nx.Graph, connected_components: [], source: Union[str, int],
                                target: Union[str, int]) -> {}:
    paths = get_node_disjoint_paths(g, source, target)
    p_q_joined = paths[0] + paths[1]
    component_set = set()
    for component in connected_components:
        nodes = [node for node in component.nodes if node in p_q_joined]
        if len(nodes) >= 2:
            component_set.add(component)
    return component_set


# Def. 4.3
# Get nodes in component and path that intersect without source and target
def calc_delta(path: [], connected_component: nx.Graph, source: Union[str, int], target: Union[str, int]) -> []:
    nodes = []
    for node in connected_component.nodes:
        if node in path and node != source and node != target:
            nodes.append(node)
    return nodes


# Def 4.3 (Types of connected components)
# Categorize all components in set C based on types 1-6
# Works with reference
def find_connected_component_types(g: nx.Graph, connected_components: {}, source: Union[str, int],
                                   target: Union[str, int]):
    paths = get_node_disjoint_paths(g, source, target)
    p, q = paths[0], paths[1]
    for component in connected_components:
        p_node_sum = len(calc_delta(p, component, source, target))
        q_node_sum = len(calc_delta(q, component, source, target))

        if (source in component and target not in component) \
                and (p_node_sum >= 1 and q_node_sum == 0 or q_node_sum >= 1 and p_node_sum == 0):
            component.graph["type"] = 1
            component.graph["source"] = source
        elif (target in component and source not in component) \
                and (p_node_sum >= 1 and q_node_sum == 0 or q_node_sum >= 1 and p_node_sum == 0):
            component.graph["type"] = 1
            component.graph["target"] = target
        elif (source in component and target not in component) and (p_node_sum >= 1 and q_node_sum >= 1):
            component.graph["type"] = 2
            component.graph["source"] = source
        elif (target in component and source not in component) and (p_node_sum >= 1 and q_node_sum >= 1):
            component.graph["type"] = 2
            component.graph["target"] = target
        elif source not in component and target not in component \
                and p_node_sum == 1 and q_node_sum == 1:
            component.graph["type"] = 3
        elif source not in component and target not in component and \
                p_node_sum >= 2 and q_node_sum == 0 or q_node_sum >= 2 and p_node_sum == 0:
            component.graph["type"] = 4
        elif source not in component and target not in component \
                and p_node_sum >= 2 and q_node_sum == 1 or q_node_sum >= 2 and p_node_sum == 1:
            component.graph["type"] = 5
        elif source not in component and target not in component and\
                p_node_sum >= 2 and q_node_sum >= 2:
            component.graph["type"] = 6


# Def. 4.3
# Find all key nodes in connected components
# Works with reference
def find_key_nodes(g: nx.Graph, connected_component: nx.Graph, source: Union[str, int], target: Union[str, int]):
    paths = get_node_disjoint_paths(g, source, target)
    p, q = paths[0], paths[1]

    nodes_along_path = []
    for node in p:
        if node in connected_component.nodes:
            nodes_along_path.append(node)
    connected_component.graph["key-nodes"] = {}
    connected_component.graph["key-nodes"]["left-P-port"] = None
    connected_component.graph["key-nodes"]["right-P-port"] = None
    connected_component.graph["key-nodes"]["left-Q-port"] = None
    connected_component.graph["key-nodes"]["right-Q-port"] = None

    if nodes_along_path:
        connected_component.graph["key-nodes"]["left-P-port"] = nodes_along_path[0]
        connected_component.graph["key-nodes"]["right-P-port"] = nodes_along_path[-1]

    nodes_along_path = []
    for node in q:
        if node in connected_component.nodes:
            nodes_along_path.append(node)
    if nodes_along_path:
        connected_component.graph["key-nodes"]["left-Q-port"] = nodes_along_path[0]
        connected_component.graph["key-nodes"]["right-Q-port"] = nodes_along_path[-1]


def gadget_factory(connected_components: [], source: Union[str, int], target: Union[str, int]) -> []:
    gadgets = []
    for component in connected_components:
        if component.graph["type"] == 1:
            gadget = compute_type1_4_gadget(component)
            gadget.graph["type"] = 1
        elif component.graph["type"] == 2:
            gadget = compute_type2_gadget(component, source, target)
            gadget.graph["type"] = 2
        elif component.graph["type"] == 3:
            gadget = compute_type3_gadget(component)
            gadget.graph["type"] = 3
        elif component.graph["type"] == 4:
            gadget = compute_type1_4_gadget(component)
            gadget.graph["type"] = 4
        elif component.graph["type"] == 5:
            gadget = compute_type5_gadget(component)
            gadget.graph["type"] = 5
        elif component.graph["type"] == 6:
            gadget = compute_type6_gadget(component, source, target)
            gadget.graph["type"] = 6
        else:
            raise (Exception("No matching component!"))
        gadgets.append(gadget)
    return gadgets


# Computing gadget of component type 3 according to Lemma 5.1
def compute_type3_gadget(component: nx.Graph) -> nx.Graph:
    a = component.graph["key-nodes"]["left-P-port"]
    c = component.graph["key-nodes"]["left-Q-port"]
    shortest_path = nx.shortest_path(component, a, c)
    gadget = nx.Graph()
    gadget.add_edges_from(get_edge_path(shortest_path))
    return gadget


# Computing gadget of component type 1 or 4 according to Lemma 5.2
def compute_type1_4_gadget(component: nx.Graph) -> nx.Graph:
    # Check on which path the key nodes are
    if component.graph["key-nodes"]["left-P-port"] != component.graph["key-nodes"]["right-P-port"]:
        a = component.graph["key-nodes"]["left-P-port"]
        b = component.graph["key-nodes"]["right-P-port"]
    else:
        a = component.graph["key-nodes"]["left-Q-port"]
        b = component.graph["key-nodes"]["right-Q-port"]
    disjoint_paths = sorted(nx.edge_disjoint_paths(component, a, b), key=len)
    gadget = nx.Graph()
    gadget.add_edges_from(get_edge_path(disjoint_paths[0]))
    gadget.add_edges_from(get_edge_path(disjoint_paths[1]))

    return gadget


def compute_type5_gadget(component: nx.Graph) -> nx.Graph:
    # Check if two ports in path P or Q
    a = component.graph["key-nodes"]["left-P-port"]
    b = component.graph["key-nodes"]["right-P-port"]
    c = component.graph["key-nodes"]["left-Q-port"]
    if a == b:
        a = component.graph["key-nodes"]["left-Q-port"]
        b = component.graph["key-nodes"]["right-Q-port"]
        c = component.graph["key-nodes"]["left-P-port"]

    gadget = nx.Graph()
    if len(list(nx.edge_disjoint_paths(component, a, c))) == 1:
        paths = []
        paths_ab = list(nx.edge_disjoint_paths(component, a, b))
        path_ac = list(nx.edge_disjoint_paths(component, a, c))
        path_bc = list(nx.edge_disjoint_paths(component, b, c))
        paths.extend(paths_ab)
        paths.extend(path_bc)
        paths.extend(path_ac)
        [gadget.add_edges_from(get_edge_path(path)) for path in paths]
    elif len(list(nx.edge_disjoint_paths(component, a, c))) >= 2:
        paths = list(nx.edge_disjoint_paths(component, a, c))
        [gadget.add_edges_from(get_edge_path(path)) for path in paths]
    else:
        raise Exception("Component type-5 connectivity error: " +
                        str(len(list(nx.edge_disjoint_paths(component, a, c)))))
    return gadget


def compute_type2_gadget(component: nx.Graph, source: Union[str, int], target: Union[str, int]) -> nx.Graph:
    # Similar logic as in Type-6 component
    augmented_component, dummy_nodes = compute_augmented_component(component, source, target)
    edge_disjoint_paths = sorted(nx.edge_disjoint_paths(augmented_component, source, target, flow_func=edmonds_karp,
                                                      cutoff=4), key=len)
    gadget = nx.Graph()
    edge_disjoint_paths_cleaned = []
    if "source" in component.graph:
        source_or_target = target
    else:
        source_or_target = source
    for path in edge_disjoint_paths:
        # Remove dummy nodes, source and target from paths
        edge_disjoint_paths_cleaned.append(
            [node for node in path if node not in dummy_nodes and node is not source_or_target]
        )
    [gadget.add_edges_from(get_edge_path(path)) for path in edge_disjoint_paths_cleaned]

    component_port_path = component.copy()
    if len(edge_disjoint_paths_cleaned) == 3:
        if "source" in component.graph:
            a = component.graph["key-nodes"]["right-P-port"]
            c = component.graph["key-nodes"]["right-Q-port"]
            # Remove s to route path through u between a,c
            # By removing s we make sure that no other path than the one through u will be taken
            # Otherwise the path through s could be shorter
            component_port_path.remove_node(source)
            shortest_path_ac = nx.shortest_path(component_port_path, a, c)
            gadget.add_edges_from(get_edge_path(shortest_path_ac))
            gadget.graph["pattern"] = "c"
        else:
            b = component.graph["key-nodes"]["left-P-port"]
            d = component.graph["key-nodes"]["left-Q-port"]
            # Remove t to route path through u between b,d
            # Same as above only with t
            component_port_path.remove_node(target)
            shortest_path_bd = nx.shortest_path(component_port_path, b, d)
            gadget.add_edges_from(get_edge_path(shortest_path_bd))
            gadget.graph["pattern"] = "f"
    elif len(edge_disjoint_paths_cleaned) == 4:
        if "source" in component.graph:
            gadget.graph["pattern"] = "b"
        else:
            gadget.graph["pattern"] = "e"
    else:
        raise Exception("Component type-2 connectivity error: " +
                        str(len(list(nx.edge_disjoint_paths(component, source, target)))))
    return gadget


# Lemma 5.5
def compute_type6_gadget(component: nx.Graph, source: Union[str, int], target: Union[str, int]) -> nx.Graph:
    augmented_component, dummy_nodes = compute_augmented_component(component, source, target)
    # Max-flow function which uses Edmonds-Karp, here capacities are needed
    # flow_val, flow_dict = nx.maximum_flow(augmented_component, source, target, flow_func=edmonds_karp)
    # edge_disjoint_paths = get_edge_disjoint_paths_from_flow_dict(flow_dict, source, target)

    # Calculates edge disjoint paths with Edmonds-Karp which is a Ford-Fulkerson Algo implementation
    # Here no edge capacities are needed
    edge_disjoint_paths = sorted(nx.edge_disjoint_paths(augmented_component, source, target, flow_func=edmonds_karp,
                                                      cutoff=4), key=len)
    gadget = nx.Graph()
    # Remove dummy nodes, source and target from paths
    edge_disjoint_paths_cleaned = []
    for path in edge_disjoint_paths:
        edge_disjoint_paths_cleaned.append(
            [node for node in path if node not in dummy_nodes and node is not source and node is not target]
        )

    a = component.graph["key-nodes"]["left-P-port"]
    c = component.graph["key-nodes"]["left-Q-port"]
    b = component.graph["key-nodes"]["right-P-port"]
    d = component.graph["key-nodes"]["right-Q-port"]

    # We need that later when removing path c,b to get a planar graph (reduced kernel graph)
    gadget.graph["key-nodes"] = {}
    gadget.graph["key-nodes"]["left-P-port"] = a
    gadget.graph["key-nodes"]["left-Q-port"] = c
    gadget.graph["key-nodes"]["right-P-port"] = b
    gadget.graph["key-nodes"]["right-Q-port"] = d

    ab_paths = []
    cd_paths = []
    ad_paths = []
    cb_paths = []

    # Assign capacities to the edges in AC based on connectivity
    if len(edge_disjoint_paths_cleaned) == 3:
        # Capacities not needed for nx.edge_disjoint_paths, but needed for normal maxflow
        # for edge in augmented_component.edges.data():
        #     if not edge[2]:
        #         edge[2]["capacity"] = 0.5

        # Problem: We don't get all edges in the connected component, because the middle edge-disjoint path i.e. x-y
        # is used either by ab or cd
        # We compute the alternative edge-disjoint paths for the key-node pair which is not present in
        # the initial computation
        for path in edge_disjoint_paths_cleaned:
            if a == path[0] and d == path[-1]:
                ad_paths.append(path)
            if c == path[0] and b == path[-1]:
                cb_paths.append(path)

        # If none is the case because AD and CB == 0, just leave it be, but it is not a type-6 but rather type-4
        if len(ad_paths) == 0 and len(cb_paths) != 0:
            [augmented_component.remove_edges_from(get_edge_path(path)) for path in ad_paths]
        if len(cb_paths) == 0 and len(ad_paths) != 0:
            [augmented_component.remove_edges_from(get_edge_path(path)) for path in cb_paths]

        edge_disjoint_paths_2 = sorted(
            nx.edge_disjoint_paths(augmented_component, source, target, flow_func=edmonds_karp,
                                   cutoff=4), key=len)
        edge_disjoint_paths_cleaned_2 = []
        for path in edge_disjoint_paths_2:
            edge_disjoint_paths_cleaned_2.append(
                [node for node in path if node not in dummy_nodes and node is not source and node is not target]
            )
        [gadget.add_edges_from(get_edge_path(path)) for path in edge_disjoint_paths_cleaned]
        [gadget.add_edges_from(get_edge_path(path)) for path in edge_disjoint_paths_cleaned_2]
        gadget.graph["pattern"] = "c"
    elif len(edge_disjoint_paths_cleaned) == 4:
        # Capacities not needed for nx.edge_disjoint_paths, but needed for normal maxflow
        # for edge in augmented_component.edges.data():
        #     if not edge[2]:
        #         edge[2]["capacity"] = 1

        # Determine if AC is of Type-a,b or c in Fig. 6 based on 4 edge-disjoint paths in AC
        # Type-a has only one path where nodes a & b respectively c & d are present
        # Type-b has two paths where a & b respectively c & d are present
        # Type-c has two paths where a & d respectively c & b are present
        # Only checking start and end of path if it contains key node pair
        for path in edge_disjoint_paths_cleaned:
            if a == path[0] and b == path[-1]:
                ab_paths.append(path)
            if c == path[0] and d == path[-1]:
                cd_paths.append(path)
            if a == path[0] and d == path[-1]:
                ad_paths.append(path)
            if c == path[0] and b == path[-1]:
                cb_paths.append(path)

        # Lemma 5.5 Proof
        # If there are any shared nodes so that AC of Type-b and Type-c form Type-a, we get them by default, by
        # applying the edge-disjoint-path algorithm with Edmonds karp (Ford-Fulkerson Impl.) and we get pattern 7a.
        # If there are no shared nodes we need to calculate the connecting paths for AC of Type-b and Type-c.
        if len(ab_paths) == 1 and len(cd_paths) == 1:
            # AC Type-a
            [gadget.add_edges_from(get_edge_path(path)) for path in edge_disjoint_paths_cleaned]
            gadget.graph["pattern"] = "a"
        elif len(ab_paths) == 2 and len(cd_paths) == 2:
            # AC Type-b
            # Get connecting path
            # (a,d) or (c,b) it doesn't matter
            shortest_path_ad = nx.shortest_path(component, a, d)
            [gadget.add_edges_from(get_edge_path(path)) for path in edge_disjoint_paths_cleaned]
            gadget.add_edges_from(get_edge_path(shortest_path_ad))
            gadget.graph["pattern"] = "b"
        elif len(ad_paths) == 2 and len(cb_paths) == 2:
            # AC Type-c
            # There are 2x a,d and 2x c,b paths
            # Take the paths from each ad & cb
            gadget.add_edges_from(get_edge_path(ad_paths[0]))
            gadget.add_edges_from(get_edge_path(cb_paths[0]))
            gadget.add_edges_from(get_edge_path(ad_paths[1]))
            gadget.add_edges_from(get_edge_path(cb_paths[1]))
            gadget.graph["pattern"] = "a"
        else:
            raise Exception("Component type-6 AC-Type error: Key-node pairs must be " +
                            "ab & cd == 1 or 2 or ad & cb == 2! But are: ", len(ab_paths), len(cd_paths),
                            len(ad_paths), len(cb_paths))
    else:
        raise Exception("Component type-6 connectivity error: " +
                        str(len(list(nx.edge_disjoint_paths(component, source, target)))))

    return gadget


# Computing augmented components according to Def. 5.3
def compute_augmented_component(component: nx.Graph, source: Union[str, int], target: Union[str, int]) -> []:
    # Networkx does not support Multigraphs when performing maxflow. So we have to bring in "intermediary"
    # nodes, to simulate the parallel edges.
    augmented_component = nx.Graph()
    augmented_component.add_edges_from(component.edges)
    dummy_nodes = ["dummy-1", "dummy-2", "dummy-3", "dummy-4"]

    # Check if dummy nodes already exist and if yes change name of dummy node and try again
    for i in range(len(dummy_nodes)):
        while True:
            if dummy_nodes[i] not in component:
                break
            else:
                dummy_nodes[i] += "1"

    if component.graph["type"] == 2:
        if "source" in component.graph:
            a = component.graph["key-nodes"]["right-P-port"]
            c = component.graph["key-nodes"]["right-Q-port"]
            augmented_component.add_edges_from(
                [
                    (a, target),
                    (a, dummy_nodes[0]),
                    (dummy_nodes[0], target),
                    (c, target),
                    (c, dummy_nodes[1]),
                    (dummy_nodes[1], target)
                ], capacity=1
            )
        else:
            b = component.graph["key-nodes"]["left-P-port"]
            d = component.graph["key-nodes"]["left-Q-port"]
            augmented_component.add_edges_from(
                [
                    (b, source),
                    (b, dummy_nodes[0]),
                    (dummy_nodes[0], source),
                    (d, source),
                    (d, dummy_nodes[1]),
                    (dummy_nodes[1], source)
                ], capacity=1
            )
    else:
        a = component.graph["key-nodes"]["left-P-port"]
        c = component.graph["key-nodes"]["left-Q-port"]
        b = component.graph["key-nodes"]["right-P-port"]
        d = component.graph["key-nodes"]["right-Q-port"]
        augmented_component.add_edges_from(
            [
                (a, source),
                (dummy_nodes[0], a),
                (dummy_nodes[0], source),
                (c, source),
                (dummy_nodes[1], c),
                (dummy_nodes[1], source),
                (b, target),
                (dummy_nodes[2], b),
                (dummy_nodes[2], target),
                (d, target),
                (dummy_nodes[3], d),
                (dummy_nodes[3], target),
            ], capacity=1
        )

    return augmented_component, dummy_nodes


def compute_kernel_graph(gadgets: [], cuts: []) -> nx.Graph:
    kernel_graph = nx.compose_all([gadget for gadget in gadgets])
    [kernel_graph.add_edges_from(cut_pair) for cut_pair in cuts]
    return kernel_graph


def compute_reduced_kernel_graph(gadgets: [], cuts: []) -> nx.Graph:
    # Same as reduced kernel pattern but no path-contracted patterns
    # Remove path c-b in Type-6a gadgets
    reduced_kernel_graph = nx.Graph()
    for gadget in gadgets:
        if gadget.graph["type"] == 6:
            if gadget.graph["pattern"] == "a":
                a = gadget.graph["key-nodes"]["left-P-port"]
                b = gadget.graph["key-nodes"]["right-P-port"]
                c = gadget.graph["key-nodes"]["left-Q-port"]
                d = gadget.graph["key-nodes"]["right-Q-port"]
                # Calc 2-edge disjoint paths between c,b
                edge_disjoint_paths = list(nx.edge_disjoint_paths(gadget, c, b, cutoff=2))
                # Remove the path that doesn't contain the other key-nodes
                for path in edge_disjoint_paths:
                    if a not in path or d not in path:
                        gadget.remove_edges_from(get_edge_path(path))
        reduced_kernel_graph = nx.compose(reduced_kernel_graph, gadget)
    [reduced_kernel_graph.add_edges_from(cut_pair) for cut_pair in cuts]
    return reduced_kernel_graph


def get_q1(kernel_graph: nx.Graph, source: Union[str, int], target: Union[str, int]) -> []:
    paths = sorted(nx.edge_disjoint_paths(kernel_graph, source, target), key=len)
    return paths[0]


# Start routing on outer face (primary_path_Q1) of planar graph
def two_resilient_routing(planar_kernel_graph: nx.PlanarEmbedding, source: Union[str, int], target: Union[str, int],
                          failed_links: []) -> []:
    # Number of hops passed
    hops = 0
    # Number of faces switched to
    switches = 0
    # Number of edges passed on a face
    detour_edges = []
    q1 = get_q1(planar_kernel_graph, source, target)
    edge_path_q1 = get_edge_path(q1)
    # Traverse primary path Q1
    for link in edge_path_q1:
        # If we encounter a link failure during face routing
        if link in failed_links:
            # Get outer face of subgraph where link failed
            face_hops = planar_kernel_graph.traverse_face(link[0], link[1])
            # Reverse list of hops on the face and prepend last element at index 0
            # First element is always the starting node of failed link i.e. link[0]
            face_hops.reverse()
            face_hops.insert(0, face_hops.pop())
            edge_face_hops = get_edge_path(face_hops)
            recursive_depth = 0
            # Start recursive routing
            result = recursive_routing(planar_kernel_graph, edge_face_hops, failed_links, target,
                                       link[1], link[1], recursive_depth)
            success = result[0]
            target_found = result[2]
            hops += result[3]
            # End of recursion, add up switches for every recursion entry
            switches += result[4]
            detour_edges.extend(result[5])
            if target_found:
                return True, hops, switches, detour_edges
            elif success:
                continue
            else:
                return False, hops, switches, detour_edges
        hops += 1
    return True, hops, switches, detour_edges


# Recursive helper function for resilient routing
# prev_intermediary_target: Other side of failed link, we need to route to from previous run (to shorten routing)
# intermediary_target: Other side of failed link, we need to route to
def recursive_routing(planar_kernel_graph: nx.PlanarEmbedding, edge_face_hops: [],failed_links: [],
                      target: Union[str, int], prev_intermediary_target: Union[str, int],
                      intermediary_target: Union[str, int], recursive_depth: int) -> []:
    hops = 0
    switches = recursive_depth
    detour_edges = []
    for face_link in edge_face_hops:
        if face_link[0] == target or face_link[1] == target:
            # 3rd bool: target found
            return True, True, True, hops, switches, detour_edges
        # If we encounter a link failure during face routing
        if face_link in failed_links:
            # Get outer face of subgraph where link failed
            face_hops = planar_kernel_graph.traverse_face(face_link[0], face_link[1])
            # Reverse list of hops on the face and prepend last element at index 0
            # First element is always the starting node of failed link i.e. link[0]
            face_hops.reverse()
            face_hops.insert(0, face_hops.pop())
            edge_face_hops_new = get_edge_path(face_hops)
            # Stop if max recursion depth is exceeded
            # Max recursion depth can be set at beginning of file
            try:
                result = recursive_routing(planar_kernel_graph, edge_face_hops_new, failed_links, target,
                                           intermediary_target, face_link[1], recursive_depth)
            except RecursionError:
                return False, False, False, hops, switches, detour_edges
            target_found = result[2]
            hops += result[3]
            # We already add up every recursion we make so only assign the newest recursion depth value
            switches = result[4]
            detour_edges.extend(result[5])
            # Target found, stop recursion
            if target_found:
                return True, True, False, hops, switches, detour_edges
            if result[0] and result[1]:
                continue
            elif result[0] and not result[1]:
                return True, False, False, hops, switches, detour_edges
            else:
                return False, False, False, hops, switches, detour_edges
        hops += 1
        detour_edges.append(face_link)
        # if we encounter the intermediary target hop i.e. failed node from previous run (link[1]) we can stop
        # routing on this face and resume routing on the previous face since we reached it will
        # be shorter
        if face_link[0] == prev_intermediary_target or face_link[1] == prev_intermediary_target:
            return True, False, False, hops, switches, detour_edges
    # Whole face was routed on without fails, but target not found
    # 1st bool: Was face routing overall successful?
    # 2nd bool: Helper bool to check if we need to route current face or go back and route on the previous if
    # there was a fail -> because we encountered the failed node on the current face
    # 3rd bool: Did routing encounter the original target node already?
    return True, True, False, hops, switches, detour_edges


# Walks just along a simple path
def simple_routing(path: [], failed_links: []) -> []:
    hops = 0
    detour_edges = []
    edge_path = get_edge_path(path)
    for link in edge_path:
        if link in failed_links:
            # Detour_edges: Simulate going back to start when encountering failed link
            # Hops*2 to account for detour hops
            return False, hops*2, detour_edges
        hops += 1
        detour_edges.append(link)
    # Empty detour_edges, because didn't need to go back to start
    return True, hops, []


def get_edge_path(path: []) -> []:
    # Create edge path from simple node path
    edge_path = []
    for i in range(len(path)):
        if i + 1 < len(path):
            edge_path.append((path[i], path[i + 1]))
    return edge_path

# Precomputation for TwoResilientAlg
# graph
# source of routing
# target of routing
# Returns a list of subgraphs
def PreComputeTwoResilient(graph: nx.Graph, source: Union[str, int], target: Union[str, int]) -> []:
    biconnected_components = list(nx.biconnected_components(graph))
    try:
        shortest_path = list(nx.shortest_path(graph, source, target))
    except nx.NetworkXNoPath as e:
        print(e)
        return -1

    filtered_biconnected_components = list()
    for component in biconnected_components:
        nodes_on_path = [node for node in shortest_path if node in component]
        if len(nodes_on_path) >= 2:
            filtered_biconnected_components.append([component, [nodes_on_path[0], nodes_on_path[-1]]])

    kernel_graphs = []
    for component in filtered_biconnected_components:
        # Notable: Maybe create new graphs instead of deep copy -> may boost performance?
        subgraph = deepcopy(graph.subgraph(component[0]))
        sub_source = component[1][0]
        sub_target = component[1][1]

        # Sub-graph is 2-edge connected and at least 2-node connected
        if get_edge_connectivity(subgraph, sub_source, sub_target) == 2 \
                and get_node_connectivity(subgraph, sub_source, sub_target) >= 2:
            cuts = get_two_s_t_cuts(subgraph, sub_source, sub_target)
            connected_components = get_connected_components(subgraph, cuts)

            '''
            If there are no connected components or source and target are in the connected component
            then don't calculate kernel graph
            '''
            if len(connected_components) == 0:
                kernel_graphs.append(subgraph)
            elif source in connected_components[0] and target in connected_components[0]:
                kernel_graphs.append(subgraph)
            elif sub_source in connected_components[0] and sub_target in connected_components[0]:
                kernel_graphs.append(subgraph)

            else:
                connected_components = get_connected_component_set(subgraph, connected_components, sub_source, sub_target)
                find_connected_component_types(subgraph, connected_components, sub_source, sub_target)
                [find_key_nodes(subgraph, conn_component, sub_source, sub_target) for conn_component in
                 connected_components]
                gadgets = gadget_factory(connected_components, sub_source, sub_target)
                kernel_graph = compute_kernel_graph(gadgets, cuts)
                is_planar, subgraph = nx.check_planarity(kernel_graph)
                if not is_planar:
                    print(nx.NetworkXException("Subgraph is not planar!"))
                    return -1

        subgraph.graph["source"] = sub_source
        subgraph.graph["target"] = sub_target
        kernel_graphs.append(subgraph)

    return kernel_graphs


# Route according to two resilient routing
# Original source s
# Original destination d
# Link failure set fails
# Precomputed subgraphs of g
def RouteTwoResilient(source: Union[str, int], target: Union[str, int], failed_links: [],
                      kernel_graphs: [Union[nx.Graph, nx.PlanarEmbedding]]) -> []:
    hops = 0
    switches = 0
    detour_edges = []
    for graph in kernel_graphs:
        sub_source = graph.graph["source"]
        sub_target = graph.graph["target"]
        if type(graph) is nx.PlanarEmbedding:
            result = two_resilient_routing(graph, sub_source, sub_target, failed_links)
            success = result[0]
            hops += result[1]
            switches += result[2]
            detour_edges.extend(result[3])
            if not success:
                return True, -1, switches, detour_edges
        else:
            paths = sorted(nx.node_disjoint_paths(graph, sub_source, sub_target), key=len)
            success = False
            # Try all shortest paths between s & t as long as routing is unsuccessful
            for path in paths:
                result = simple_routing(path, failed_links)
                success = result[0]
                hops += result[1]
                detour_edges.extend(result[2])
                if success:
                    break
            # If simple routing in all node disjoint paths fails, stop process here
            if not success:
                return True, -1, switches, detour_edges

    # If routing has not stopped until here the routing process was successful
    return False, hops, switches, detour_edges