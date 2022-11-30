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
    initial_opinion: Opinion = Opinion.DISAGREE  # By default the agents disagree
    p_opinion_change: float = 0.5  # Probability of changing opinion

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

    neighbors = self.model.grid.get_neighbors(self.unique_id, include_center=False)

    for neighbor_id in neighbors:
      neighbor_agent: 'OpinionAgent' = self.model.schedule.agents[neighbor_id]

      if self.state != neighbor_agent.state:
        if (self.state, neighbor_agent.state) in [(Opinion.DISAGREE, Opinion.BELIEVE), (Opinion.BELIEVE, Opinion.DISAGREE)]:
          if bernoulli(self.params.p_opinion_change):
            self.state = Opinion.UNSURE
        elif self.state == Opinion.UNSURE:
          if bernoulli(self.params.p_opinion_change):
            self.state = neighbor_agent.state


class OpinionModel(Model):
  """
  This class defines the network model
  """

  def __init__(self,
      population_size: int,
      fraction_believers: float,
      p_opinion_change: float,
      network_type: NetworkType,
      network_params: Tuple
    ) -> None:
    """
    Create a new model, initialize network and setup data collection

    :param population_size: number of agents in model (note: this needs to agree with number of nodes specified in network_params)
    :param fraction_believers: fraction of population that are initially believers
    :param p_opinion_change: probability for an agent to switch opinion
    :param network_type: type of network, see networkgen.py for more information
    :param network_params: parameters of network, depends on network type
    """

    super().__init__()

    self.population_size = population_size
    self.fraction_believers = fraction_believers
    self.network = Network.generate(network_type, network_params)

    self.grid = NetworkGrid(self.network)

    # Scheduling
    self.schedule = RandomActivation(self)

    # Populate Model with Agents
    for node in self.network.nodes:
      agent = OpinionAgent(node, self, OpinionAgent.Params(p_opinion_change=p_opinion_change))

      self.schedule.add(agent)
      self.grid.place_agent(agent, node)

    # Initialize initial believers
    n_initial_believers = math.floor(self.population_size * fraction_believers)
    initial_believers: List[OpinionAgent] = random.sample(self.schedule.agents, n_initial_believers)

    for agent in initial_believers:
      agent.state = Opinion.BELIEVE

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



