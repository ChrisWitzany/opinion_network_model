import random
import time
import enum
import math
from collections import namedtuple
from typing import Tuple, NamedTuple, List

import numpy as np
import pandas as pd
import pylab as plt
import networkx as nx
from scipy.stats import bernoulli
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

from networkgen import Network, NetworkType


class Opinion(enum.IntEnum):
  DISAGREE = 0
  BELIEVE = 1
  UNSURE = 2


class OpinionAgent(Agent):
  """
  This class defines the behaviour of our agents in the Model
  """

  class Params(NamedTuple):
    initial_opinion: float  # TODO: docs
    weight_decay: float  # TODO: docs

  def __init__(self,
      unique_id: int,
      model: Model,
      params: Params
    ) -> None:
    """
    Create a new Agent that defines opinion behaviour

    :param unique_id: unique numerical identifier for the agent
    :param model: instance of a model that contains the agent
    """

    super().__init__(unique_id, model)
    self.params = params

    #if self.params.initial_opinion is None:
    self.state = np.clip(np.random.normal(self.params.initial_opinion, 0.1), 0, 1)  # TODO: parameterize this
    #else:
    #  self.state = self.params.initial_opinion

  def compute_similarity(self, b):
    return self._compute_similarity(self.state, b)

  def _compute_similarity(self, a, b):
    vecA = np.array([np.cos(a * np.pi / 2), np.sin(a * np.pi / 2)])
    vecB = np.array([np.cos(b * np.pi / 2), np.sin(b * np.pi / 2)])
    return np.square(np.dot(vecA, vecB))

  def update_neighbor_weights(self):
    neighbors = self.model.grid.get_neighbors(self.unique_id, include_center=False)
    for neighbor_id in neighbors:
      neighbor_agent: 'OpinionAgent' = self.model.schedule.agents[neighbor_id]
      self.model.network.edges[neighbor_id, self.unique_id]['weight'] = 0.2 + 0.8 * self.state * self.compute_similarity(neighbor_agent.state)

  def step(self) -> None:
    """@override(Agent)
    Defines behaviour within a single step
    """

    neighbors = self.model.grid.get_neighbors(self.unique_id, include_center=False)

    # Add a bit of noise
    self.state = np.clip(np.random.normal(self.state, 0.1), 0, 1)  # TODO: make hyperparameter
    self.update_neighbor_weights()

    sampled_neighbors = random.sample(neighbors, 3)
    accumulated_step = 0.0
    global_opinion = np.round(np.mean([agent.state for agent in self.model.grid.get_all_cell_contents()]))
    for neighbor_id in sampled_neighbors:
      neighbor_agent: 'OpinionAgent' = self.model.schedule.agents[neighbor_id]

      effect = self.model.network.edges[neighbor_id, self.unique_id]['weight']
      similarity = self.compute_similarity(neighbor_agent.state)
      neighbor_opinion = np.round(neighbor_agent.state)

      if similarity > 0.25:
        self.state = (self.state + neighbor_agent.state + neighbor_opinion * effect) / 3

      #global_effect = 1 - self._compute_similarity(global_opinion, 0.5)
      #similarity = self.compute_similarity(neighbor_agent.state)

      #step = (neighbor_agent.state - self.state) * importance * 0.1  # TODO: make hyperparameter
      #self.state += step
      #neighbor_agent.state += step * similarity

      #neighbor_opinion = np.round(neighbor_agent.state)
      #step = (0.2 + 0.8 * self.compute_similarity(neighbor_opinion)) * (neighbor_opinion - self.state)
      #accumulated_step += step * effect

      #global_step = (0.05 + 0.05 * self.compute_similarity(global_opinion)) * (global_opinion - self.state)
      #accumulated_step += global_step * global_effect

      #diff = neighbor_agent.state - self.state

      #scaling = neighbor_agent.state + self.state - abs(diff)

      #diff *= self.model.network.edges[neighbor_id, self.unique_id]['weight']
      #diff *= random.uniform(0, 1)

      #self.state = min(max(self.state + diff, 0), 1)  # clip inside [0, 1]
      #neighbor_agent.state = min(max(neighbor_agent.state + diff*0.25, 0), 1)  # clip inside [0, 1]

      #scaling = self.model.network.edges[neighbor_id, self.unique_id]['weight']

      #if self.state != neighbor_agent.state:
        #if (self.state, neighbor_agent.state) in [(Opinion.DISAGREE, Opinion.BELIEVE), (Opinion.BELIEVE, Opinion.DISAGREE)]:
        #  if bernoulli(scaling):
        #    self.state = neighbor_agent.state
        #elif self.state == Opinion.UNSURE:
        #  if bernoulli(scaling):
        #    self.state = neighbor_agent.state


class OpinionModel(Model):
  """
  This class defines the network model
  """

  def __init__(self,
      fraction_believers: float,
      agent_params: OpinionAgent.Params,
      network_type: NetworkType,
      network_params: Tuple
    ) -> None:
    """
    Create a new model, initialize network and setup data collection

    :param fraction_believers: fraction of population that are initially believers
    :param agent_params: parameters of agents
    :param network_type: type of network, see networkgen.py for more information
    :param network_params: parameters of network, depends on network type
    """

    super().__init__()

    self.fraction_believers = fraction_believers
    self.network = Network.generate(network_type, network_params).to_directed()
    self.population_size = len(self.network.nodes)

    # Initialize Weights
    for u, v in self.network.edges:
      self.network.edges[u, v]['weight'] = 1.0

    self.grid = NetworkGrid(self.network)

    # Scheduling
    self.schedule = RandomActivation(self)

    # Populate Model with Agents
    for node in self.network.nodes:
      agent = OpinionAgent(node, self, agent_params)

      self.schedule.add(agent)
      self.grid.place_agent(agent, node)

    for agent in self.grid.get_all_cell_contents():
      agent.update_neighbor_weights()

    # Initialize initial believers
    n_initial_believers = math.floor(self.population_size * fraction_believers)
    initial_believers: List[OpinionAgent] = random.sample(self.schedule.agents, n_initial_believers)

    for agent in initial_believers:
      agent.state = np.random.normal(0.75, 0.05) #Opinion.BELIEVE

    # Data Collector keeps track of agents' states
    self.data_collector = DataCollector(agent_reporters={"State": "state"})

  def step(self) -> None:
    """@override(Model)
    Executes a step in model simulation and updates data collector
    """

    self.data_collector.collect(self)
    self.schedule.step()

  def run(self, steps: int) -> None:
    """
    Runs simulation on model for specified number of steps

    :param steps: number of steps to simulate
    """

    for _ in range(steps):
      self.step()



