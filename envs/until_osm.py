import osmnx as ox
import networkx as nx
from networkx.algorithms.components import strongly_connected_components
import random

def load_graph(place_name='District 1, Ho Chi Minh City, Vietnam'):
    G = ox.graph_from_place(place_name, network_type='drive')
    largest_cc = max(strongly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    return G

def sample_start_goal(G):
    nodes = list(G.nodes)
    start, goal = random.sample(nodes, 2)  # Select two distinct nodes
    return start, goal
