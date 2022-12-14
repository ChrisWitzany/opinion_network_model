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

import time
import enum
import math
import random
from collections import defaultdict
from typing import Tuple, Any, DefaultDict, Callable, Union

import numpy as np
import networkx as nx
import matplotlib as plt


# --- Base ---
class BaseAgentState(enum.IntEnum):
  DEFAULT = 0  # Default state is always 0


class BaseAgent(Agent):
  class Params(NamedTuple):
    pass

  def __init__(self, unique_id: int, model: 'DefaultModel', initial_state, params: Params):
    super().__init__(unique_id, model)

    self.model = model
    self.state = initial_state
    self.params = params

  def step(self):
    self.model.resolve(self)


# --- SEIZplus ---
class SEIZplusStates(enum.IntEnum):
  DEFAULT = 0
  SUSCEPTIBLE = 0
  EXPOSED = 1
  SKEPTIC = 2
  INFECTED = 3


# TODO: docs
class SEIZplusModel(Model):
  class Params(NamedTuple):
    initial_infected: float
    initial_skeptics: float
    prob_S_with_I: float  # TODO: docs
    prob_S_with_Z: float  # TODO: docs
    neighbor_threshold: float = None  # TODO: docs

  def __init__(self, params: Params):
    """
    Implements the SEIZ+ Model
    """

    super().__init__()

    self.params = params

  def setup(self):
    # Initialize initial believers/skeptics
    n_initial_believers = math.floor(self.population_size * self.params.initial_infected)
    n_initial_skeptics = math.floor(self.population_size * self.params.initial_skeptics)

    initial_biased: List[BaseAgent] = random.sample(self.schedule.agents, n_initial_believers + n_initial_skeptics)

    for i in range(n_initial_believers):
      initial_biased[i].state = SEIZplusStates.INFECTED

    for i in range(n_initial_believers, len(initial_biased)):
      initial_biased[i].state = SEIZplusStates.SKEPTIC

  def create_agent(self, unique_id: int, agent_params: BaseAgent.Params):
    return BaseAgent(unique_id, self, SEIZplusStates.DEFAULT, agent_params)

  def resolve(self, agent: BaseAgent) -> None:
    """
    Resolves interaction between agents

    :param agent: agent that is to update
    """

    # TODO: explain this
    neighbors = self.grid.get_neighbors(agent.unique_id, include_center=False)

    if len(neighbors) == 0:
      return

    if self.params.neighbor_threshold is not None:
      if agent.state == SEIZplusStates.SKEPTIC:
        n_infected = 0
        for neighbor_id in neighbors:
          neighbor = self.schedule.agents[neighbor_id]
          if neighbor.state == SEIZplusStates.INFECTED:
            n_infected += 1
        if n_infected > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZplusStates.INFECTED
          return

      if agent.state == SEIZplusStates.INFECTED:
        n_skeptic = 0
        for neighbor_id in neighbors:
          neighbor = self.schedule.agents[neighbor_id]
          if neighbor.state == SEIZplusStates.SKEPTIC:
            n_skeptic += 1
        if n_skeptic > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZplusStates.SKEPTIC
          return

    # Normal transitions

    for neighbor_id in random.sample(neighbors, min(1, len(neighbors))):
      neighbor = self.schedule.agents[neighbor_id]

      if agent.state == SEIZplusStates.EXPOSED:
        transition_I = bernoulli.rvs(self.params.prob_S_with_I)
        transition_Z = bernoulli.rvs(self.params.prob_S_with_Z)

        neighbor_not_skeptic = neighbor.state != SEIZplusStates.SKEPTIC
        neighbor_not_infected = neighbor.state != SEIZplusStates.INFECTED

        if transition_I and not transition_Z and neighbor_not_skeptic:
          agent.state = SEIZplusStates.INFECTED

        if transition_Z and not transition_I and neighbor_not_infected:
          agent.state = SEIZplusStates.SKEPTIC

      if (agent.state, neighbor.state) == (SEIZplusStates.SUSCEPTIBLE, SEIZplusStates.INFECTED):
        if bernoulli.rvs(self.params.prob_S_with_I):
          agent.state = SEIZplusStates.INFECTED
        else:
          agent.state = SEIZplusStates.EXPOSED

      if (agent.state, neighbor.state) == (SEIZplusStates.SUSCEPTIBLE, SEIZplusStates.SKEPTIC):
        if bernoulli.rvs(self.params.prob_S_with_Z):
          agent.state = SEIZplusStates.SKEPTIC
        else:
          agent.state = SEIZplusStates.EXPOSED


class ModelType(enum.IntEnum):
  SIR = 0
  SEIZ = 1
  SEIZplus = 2
  SEIZM = 3


class Model:
  @staticmethod
  def model_by_type(model_type: ModelType) -> type(Model):
    models = defaultdict()
    models[ModelType.SEIZplus] = SEIZplusModel

    return models[model_type]

  @staticmethod
  def get(model_type: ModelType, params: Tuple):
    """
    Allows for more flexibility when trying different model dynamics

    :param model_type: type of model to generate
    :param params: arguments will be passed to specific model constructor

    :return: Model according to specific params
    """

    return Model.model_by_type(model_type)(params)
