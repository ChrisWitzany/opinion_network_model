import time
import enum
import math
from collections import defaultdict
from typing import Tuple, Any, DefaultDict, Callable

import numpy as np
import networkx as nx
import matplotlib as plt


class GraphType(enum.IntEnum):
  CAVEMAN = 0
  GAUSSIAN_RANDOM_PARTITION = 1
  WINDMILL = 2
  SMALLWORLD = 3


# --- DEFAULT ---
def gen_default() -> nx.Graph:
  """
  Generates default graph (should be empty)

  :return: empty graph
  """

  return nx.empty_graph()


def visualize_default(graph: nx.Graph):
  """
  Default method for drawing graph

  :param graph: input graph
  """

  nx.draw(graph)


# --- CAVEMAN ---
def gen_caveman(l: int, k: int) -> nx.Graph:
  """
  Generates Connected Caveman Graph as specified in:
  https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.connected_caveman_graph.html

  :param l: number of cliques
  :param k: size of cliques

  :return: Connected Caveman Graph of l cliques each of size k
  """

  return nx.connected_caveman_graph(l, k)


def visualize_caveman(graph: nx.Graph):
  """
  Visualize Caveman Graph, uses nx.draw_kamada_kawai(G)

  :param graph: input graph
  """

  nx.draw_kamada_kawai(graph)


# --- GAUSSIAN_RANDOM_PARTITION
def gen_gaussian_random_partition(n: int, s: float, v: float, p_in: float, p_out: float) -> nx.Graph:
  """
  Generates Gaussian Random Partition Graph as specified in:
  https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.gaussian_random_partition_graph.html#networkx.generators.community.gaussian_random_partition_graph

  :param n: number of nodes
  :param s: mean cluster size
  :param v: shape parameter, the variance of cluster size distribution is s/v
  :param p_in: probabilty of intra cluster connection
  :param p_out: probabilty of inter cluster connection

  :return: Gaussian Random Partition Graph of n nodes with specified parameters
  """

  return nx.gaussian_random_partition_graph(n, s, v, p_in, p_out)


def visualize_gaussian_random_partition(graph: nx.Graph):
  """
  Visualize Gaussian Random Partition Graph, uses nx.draw_circular(G)

  :param graph: input graph
  """

  nx.draw_circular(graph)


# --- WINDMILL ---
def gen_windmill(n: int, k: int) -> nx.Graph:
  """
  Generates Windmill Graph as specified in:
  https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.windmill_graph.html#networkx.generators.community.windmill_graph

  :param n: number of cliques
  :param k: size of cliques

  :return: Windmill Graph of n cliques each of size k
  """

  return nx.windmill_graph(n, k)


def visualize_windmill(graph: nx.Graph):
  """
  Visualize Windmill Graph, uses nx.draw_spring(G)

  :param graph: input graph
  """

  nx.draw_spring(graph)


# --- SMALLWORLD ---
def gen_smallworld(n: int, k: int, p: float) -> nx.Graph:
  """
  Generates Newman–Watts–Strogatz small-world Graph as specified in:
  https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.newman_watts_strogatz_graph.html#networkx.generators.random_graphs.newman_watts_strogatz_graph

  :param n: number of nodes
  :param k: each node is joined with its k nearest neighbors in a ring topology
  :param p: probability of adding a new edge for each edge

  :return: Newman–Watts–Strogatz small-world Graph of n nodes with specified parameters
  """

  return nx.newman_watts_strogatz_graph(n, k, p)


def visualize_smallworld(graph: nx.Graph):
  """
  Visualize Newman–Watts–Strogatz small-world Graph, uses nx.draw_spring(G)

  :param graph: input graph
  """

  nx.draw_circular(graph)


# --- Wrapper Methods ---
generators = defaultdict(gen_default)
generators[GraphType.CAVEMAN] = gen_caveman
generators[GraphType.GAUSSIAN_RANDOM_PARTITION] = gen_gaussian_random_partition
generators[GraphType.WINDMILL] = gen_windmill
generators[GraphType.SMALLWORLD] = gen_smallworld

visualizers = defaultdict(visualize_default)
visualizers[GraphType.CAVEMAN] = visualize_caveman
visualizers[GraphType.GAUSSIAN_RANDOM_PARTITION] = visualize_gaussian_random_partition
visualizers[GraphType.WINDMILL] = visualize_windmill
visualizers[GraphType.SMALLWORLD] = visualize_smallworld


def generate(graph_type: GraphType, params: Tuple) -> nx.Graph:
  """
  Allows for more flexibility when generating random graphs

  :param graph_type: type of graph to generate
  :param params: arguments will be passed to specific generator

  :return: Graph according to specific generator
  """

  return generators[graph_type](*params)


def visualize(graph_type: GraphType, graph: nx.Graph):
  """
  Selects best method for visualizing the graph type

  :param graph_type: type of graph to visualize
  :param graph: input graph
  """

  visualizers[graph_type](graph)
