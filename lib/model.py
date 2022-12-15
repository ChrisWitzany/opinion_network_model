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
from mesa import Agent
from mesa.time import RandomActivation, BaseScheduler
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

from networkgen import Network, NetworkType
from modelgen import Model


class ModelWrapper:
  """
  This class defines the network model
  """

  def __init__(self,
      model_type: Model.ModelType,
      model_params: Tuple,
      agent_params: Tuple,
      network_type: NetworkType,
      network_params: Tuple
    ) -> None:
    """
    Create a new model, initialize network and setup data collection

    :param model_type: type of model, see modelgen.py for more information
    :param model_params: parameters of environment
    :param agent_params: parameters of agents
    :param network_type: type of network, see networkgen.py for more information
    :param network_params: parameters of network, depends on network type
    """

    super().__init__()

    # Setup Model and Network
    self.model = Model.get(model_type, model_params)
    self.model.network = Network.generate(network_type, network_params)

    self.model.population_size = len(self.model.network.nodes)

    self.model.grid = NetworkGrid(self.model.network)

    # Scheduling
    self.model.schedule = RandomActivation(self.model)

    # Populate Model with Agents
    for node in self.model.network.nodes:
      agent = self.model.create_agent(node, agent_params)

      self.model.schedule.add(agent)
      self.model.grid.place_agent(agent, node)

    # Data Collector keeps track of agents' states
    self.model.data_collector = DataCollector(agent_reporters={"State": "state"})

    # Run additional setup if needed
    self.model.setup()

  def schedule(self) -> BaseScheduler:
    return self.model.schedule

  def data_collector(self) -> DataCollector:
    return self.model.data_collector

  def step(self) -> None:
    """@override(Model)
    Executes a step in model simulation and updates data collector
    """

    self.model.data_collector.collect(self.model)
    self.model.schedule.step()

  def run(self, steps: int) -> None:
    """
    Runs simulation on model for specified number of steps

    :param steps: number of steps to simulate
    """

    for _ in range(steps):
      self.step()



