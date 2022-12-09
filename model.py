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
  SUSCEPTIBLE = 0
  EXPOSED = 1
  SKEPTIC = 2
  INFECTED = 3


class OpinionAgent(Agent):
  """
  This class defines the behaviour of our agents in the Model
  """

  class Params(NamedTuple):
    initial_opinion: Opinion  # TODO: docs
    prob_S_with_I: float  # TODO: docs
    prob_S_with_Z: float  # TODO: docs
    neighbor_threshold: float = None  # TODO: docs

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
    self.state = self.params.initial_opinion

  def step(self) -> None:
    """@override(Agent)
    Defines behaviour within a single step
    """

    # TODO: add docs explaining this
    neighbors = self.model.grid.get_neighbors(self.unique_id, include_center=False)

    if len(neighbors) == 0:
      return

    if self.params.neighbor_threshold is not None:
      if self.state == Opinion.SKEPTIC:
        n_infected = 0
        for neighbor_id in neighbors:
          neighbor = self.model.schedule.agents[neighbor_id]
          if neighbor.state == Opinion.INFECTED:
            n_infected += 1
        if n_infected > self.params.neighbor_threshold * len(neighbors):
          self.state = Opinion.INFECTED
          return

      if self.state == Opinion.INFECTED:
        n_skeptic = 0
        for neighbor_id in neighbors:
          neighbor = self.model.schedule.agents[neighbor_id]
          if neighbor.state == Opinion.SKEPTIC:
            n_skeptic += 1
        if n_skeptic > self.params.neighbor_threshold * len(neighbors):
          self.state = Opinion.SKEPTIC
          return

    # Normal transitions

    for neighbor_id in random.sample(neighbors, min(1, len(neighbors))):
      neighbor = self.model.schedule.agents[neighbor_id]

      if self.state == Opinion.EXPOSED:
        transition_I = bernoulli.rvs(self.params.prob_S_with_I)
        transition_Z = bernoulli.rvs(self.params.prob_S_with_Z)

        neighbor_not_skeptic = neighbor.state != Opinion.SKEPTIC
        neighbor_not_infected = neighbor.state != Opinion.INFECTED

        if transition_I and not transition_Z and neighbor_not_skeptic:
          self.state = Opinion.INFECTED

        if transition_Z and not transition_I and neighbor_not_infected:
          self.state = Opinion.SKEPTIC

      if (self.state, neighbor.state) == (Opinion.SUSCEPTIBLE, Opinion.INFECTED):
        if bernoulli.rvs(self.params.prob_S_with_I):
          self.state = Opinion.INFECTED
        else:
          self.state = Opinion.EXPOSED

      if (self.state, neighbor.state) == (Opinion.SUSCEPTIBLE, Opinion.SKEPTIC):
        if bernoulli.rvs(self.params.prob_S_with_Z):
          self.state = Opinion.SKEPTIC
        else:
          self.state = Opinion.EXPOSED


class OpinionModel(Model):
  """
  This class defines the network model
  """

  def __init__(self,
      fraction_believers: float,
      fraction_skeptics: float,
      agent_params: OpinionAgent.Params,
      network_type: NetworkType,
      network_params: Tuple
    ) -> None:
    """
    Create a new model, initialize network and setup data collection

    :param fraction_believers: fraction of population that are initially believers
    :param fraction_skeptics: fraction of population that are initially skeptics
    :param agent_params: parameters of agents
    :param network_type: type of network, see networkgen.py for more information
    :param network_params: parameters of network, depends on network type
    """

    super().__init__()

    # Setup Network
    self.fraction_believers = fraction_believers
    self.fraction_skeptics = fraction_skeptics
    self.network = Network.generate(network_type, network_params)
    self.population_size = len(self.network.nodes)

    self.grid = NetworkGrid(self.network)

    # Scheduling
    self.schedule = RandomActivation(self)

    # Populate Model with Agents
    for node in self.network.nodes:
      agent = OpinionAgent(node, self, agent_params)

      self.schedule.add(agent)
      self.grid.place_agent(agent, node)

    # Initialize initial believers/skeptics
    n_initial_believers = math.floor(self.population_size * fraction_believers)
    n_initial_skeptics = math.floor(self.population_size * fraction_skeptics)

    initial_biased: List[OpinionAgent] = random.sample(self.schedule.agents, n_initial_believers + n_initial_skeptics)

    for i in range(n_initial_believers):
      initial_biased[i].state = Opinion.INFECTED

    for i in range(n_initial_believers, len(initial_biased)):
      initial_biased[i].state = Opinion.SKEPTIC

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



