import time
import enum
import math
from collections import defaultdict
from typing import Tuple, Any, DefaultDict, Callable, Union

import numpy as np
import networkx as nx
import matplotlib as plt


# --- DEFAULT ---
class DefaultNetwork:
  @staticmethod
  def generate(n: int, p: float) -> nx.Graph:
    """
    Generates default graph, G_{n, p}

    :param n: number of nodes
    :param p: probability of edge creation
    :return: random graph
    """

    return nx.fast_gnp_random_graph(n, p)

  @staticmethod
  def visualize(graph: nx.Graph):
    """
    Default method for drawing graph

    :param graph: input graph
    """

    nx.draw(graph)


# --- CAVEMAN ---
class CavemanNetwork(DefaultNetwork):
  @staticmethod
  def generate(l: int, k: int) -> nx.Graph:
    """
    Generates Connected Caveman Graph as specified in:
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.connected_caveman_graph.html

    :param l: number of cliques
    :param k: size of cliques

    :return: Connected Caveman Graph of l cliques each of size k
    """

    return nx.connected_caveman_graph(l, k)

  @staticmethod
  def visualize(graph: nx.Graph):
    """
    Visualize Caveman Graph, uses nx.draw_kamada_kawai(G)

    :param graph: input graph
    """

    nx.draw_kamada_kawai(graph)


# --- GAUSSIAN_RANDOM_PARTITION
class GaussianRandomPartitionNetwork(DefaultNetwork):
  @staticmethod
  def generate(n: int, s: float, v: float, p_in: float, p_out: float) -> nx.Graph:
    """
    Generates Gaussian Random Partition Graph as specified in:
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.gaussian_random_partition_graph.html#networkx.generators.community.gaussian_random_partition_graph

    :param n: number of nodes
    :param s: mean cluster size
    :param v: shape parameter, the variance of cluster size distribution is s/v
    :param p_in: probability of intra cluster connection
    :param p_out: probability of inter cluster connection

    :return: Gaussian Random Partition Graph of n nodes with specified parameters
    """

    return nx.gaussian_random_partition_graph(n, s, v, p_in, p_out)

  @staticmethod
  def visualize(graph: nx.Graph):
    """
    Visualize Gaussian Random Partition Graph, uses nx.draw_circular(G)

    :param graph: input graph
    """

    nx.draw_circular(graph)


# --- WINDMILL ---
class WindmillNetwork(DefaultNetwork):
  @staticmethod
  def generate(n: int, k: int) -> nx.Graph:
    """
    Generates Windmill Graph as specified in:
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.windmill_graph.html#networkx.generators.community.windmill_graph

    :param n: number of cliques
    :param k: size of cliques

    :return: Windmill Graph of n cliques each of size k
    """

    return nx.windmill_graph(n, k)

  @staticmethod
  def visualize(graph: nx.Graph):
    """
    Visualize Windmill Graph, uses nx.draw_spring(G)

    :param graph: input graph
    """

    nx.draw_spring(graph)


# --- SMALLWORLD ---
class SmallWorldNetwork(DefaultNetwork):
  @staticmethod
  def generate(n: int, k: int, p: float) -> nx.Graph:
    """
    Generates Newman–Watts–Strogatz small-world Graph as specified in:
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.newman_watts_strogatz_graph.html#networkx.generators.random_graphs.newman_watts_strogatz_graph

    :param n: number of nodes
    :param k: each node is joined with its k nearest neighbors in a ring topology
    :param p: probability of adding a new edge for each edge

    :return: Newman–Watts–Strogatz small-world Graph of n nodes with specified parameters
    """

    return nx.newman_watts_strogatz_graph(n, k, p)

  @staticmethod
  def visualize(graph: nx.Graph):
    """
    Visualize Newman–Watts–Strogatz small-world Graph, uses nx.draw_spring(G)

    :param graph: input graph
    """

    nx.draw_circular(graph)


class NetworkType(enum.IntEnum):
  CAVEMAN = 0
  GAUSSIAN_RANDOM_PARTITION = 1
  WINDMILL = 2
  SMALLWORLD = 3


class Network:
  @staticmethod
  def network_by_type(network_type: NetworkType) -> DefaultNetwork:
    networks = defaultdict(DefaultNetwork)
    networks[NetworkType.CAVEMAN] = CavemanNetwork
    networks[NetworkType.GAUSSIAN_RANDOM_PARTITION] = GaussianRandomPartitionNetwork
    networks[NetworkType.WINDMILL] = WindmillNetwork
    networks[NetworkType.SMALLWORLD] = SmallWorldNetwork

    return networks[network_type]

  @staticmethod
  def generate(network_type: NetworkType, params: Tuple) -> nx.Graph:
    """
    Allows for more flexibility when generating random graphs

    :param network_type: type of network to generate
    :param params: arguments will be passed to specific generator

    :return: Graph according to specific generator
    """

    return Network.network_by_type(network_type).generate(*params)

  @staticmethod
  def visualize(network_type: NetworkType, graph: nx.Graph):
    """
    Selects best method for visualizing the graph type

    :param network_type: type of graph to visualize
    :param graph: input graph
    """

    Network.network_by_type(network_type).visualize(graph)
