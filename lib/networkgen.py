import time
import enum
import math
import random
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
    Visualize Newman–Watts–Strogatz small-world Graph, uses nx.draw_circular(G)

    :param graph: input graph
    """

    nx.draw_circular(graph)


# --- ERDOS_RENYI ---
class ErdosRenyiNetwork(DefaultNetwork):
  @staticmethod
  def generate(n: int, p: float) -> nx.Graph:
    """
    Generates Erdos-Renyi Graph as specified in:
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html#networkx.generators.random_graphs.erdos_renyi_graph

    :param n: number of nodes
    :param p: probability of edge creation

    :return: Erdos-Renyi Graph of n nodes with specified parameters
    """

    return nx.erdos_renyi_graph(n, p)

  @staticmethod
  def visualize(graph: nx.Graph):
    """
    Visualize Erdos-Renyi Graph, uses nx.draw_spring(G)

    :param graph: input graph
    """

    nx.draw_spring(graph)


# --- RELAXED_CAVEMAN ---
class RelaxedCavemanNetwork(DefaultNetwork):
  @staticmethod
  def generate(l: int, k: int, p: float, f: float) -> nx.Graph:
    """
    Generates Relaxed Caveman Graph as specified in:
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.relaxed_caveman_graph.html

    While randomly adding edges

    :param l: number of cliques
    :param k: size of cliques
    :param p: probability of rewiring each edge
    :param f: fraction of edges that are not in the original graph to be sampled

    :return: Connected Caveman Graph of l cliques each of size k
    """

    G = nx.relaxed_caveman_graph(l, k, p)
    non_edges = list(nx.non_edges(G))

    new_edges = random.sample(non_edges, int(f * len(non_edges)))
    G.add_edges_from(new_edges)

    return G

  @staticmethod
  def visualize(graph: nx.Graph):
    """
    Visualize Relaxed Caveman Graph, uses nx.draw_kamada_kawai(G)

    :param graph: input graph
    """

    nx.draw_kamada_kawai(graph)

# --- STAR_CLIQUES ---
class StarClique(DefaultNetwork):
  @staticmethod
  def generate(l: int, n: float, m: float) -> nx.Graph:
    """
    Generates a graph similar to the Relaxed Caveman Graph as specified in:
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.relaxed_caveman_graph.html

    where the cliques are Windmill graphs as specified in: 


    :param l: number of windmill cliques
    :param n: number of cliques in the windmill subgraphs
    :param m: size of cliques in the windmill subgraphs 

    :return: Connected Star Clique Graph of l cliques each of size k
    """
    #size of cliques in the main graph
    k = n * m + 1

    clique_list = []
    for i in range(l):
      clique_list.append(nx.windmill_graph(n, m))

    #now need to connect the cliques of clique_list in a caveman fashion
    G = nx.compose_all(clique_list)

    caveman_nodes = []

    for component in nx.connected_components(G):
      caveman_nodes.append(component.nodes[random.randint(0, k)])
    
    for i in range(len(caveman_nodes)):
      if i==len(caveman_nodes)-1:
        G.add_edge(caveman_nodes[i], caveman_nodes[0])
      else:
        G.add_edge(caveman_nodes[i], caveman_nodes[i+1])

    return G

  @staticmethod
  def visualize(graph: nx.Graph):
    """
    Visualize Star Clique Graph, uses nx.draw_kamada_kawai(G)

    :param graph: input graph
    """

    nx.draw_kamada_kawai(graph)


# --- BARABASI_ALBERT ---
class BarabasiAlbertNetwork(DefaultNetwork):
  @staticmethod
  def generate(n: int, m1: int, m2: int, p: float) -> nx.Graph:
    """
    Generates Dual Barabasi-Albert Graph as specified in:
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.dual_barabasi_albert_graph.html

    :param n: number of nodes
    :param m1: number of edges to link each new node to existing nodes with probability p
    :param m2: number of edges to link each new node to existing nodes with probability 1-p
    :param p: probability of attaching m1 edges (as opposed to m2 edges)


    :return: Random graph using dual Barabási–Albert preferential attachment
    """

    return nx.dual_barabasi_albert_graph(n, m1, m2, p)

  @staticmethod
  def visualize(graph: nx.Graph):
    """
    Visualize Barabasi-Albert Graph, uses nx.draw_kamada_kawai(G)

    :param graph: input graph
    """

    nx.draw_kamada_kawai(graph)


class NetworkType(enum.IntEnum):
  DEFAULT = -1
  CAVEMAN = 0
  GAUSSIAN_RANDOM_PARTITION = 1
  WINDMILL = 2
  SMALLWORLD = 3
  ERDOS_RENYI = 4
  RELAXED_CAVEMAN = 5
  STAR_CLIQUE = 6
  BARABASI_ALBERT = 7


class Network:
  @staticmethod
  def network_by_type(network_type: NetworkType) -> type(DefaultNetwork):
    networks = defaultdict(DefaultNetwork)
    networks[NetworkType.CAVEMAN] = CavemanNetwork
    networks[NetworkType.GAUSSIAN_RANDOM_PARTITION] = GaussianRandomPartitionNetwork
    networks[NetworkType.WINDMILL] = WindmillNetwork
    networks[NetworkType.SMALLWORLD] = SmallWorldNetwork
    networks[NetworkType.ERDOS_RENYI] = ErdosRenyiNetwork
    networks[NetworkType.RELAXED_CAVEMAN] = RelaxedCavemanNetwork
    networks[NetworkType.STAR_CLIQUE] = StarClique
    networks[NetworkType.BARABASI_ALBERT] = BarabasiAlbertNetwork

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
