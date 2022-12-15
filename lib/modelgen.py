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

# --- Money Agent ---

class MoneyAgent(Agent):
  class Params(NamedTuple):

    class Tendency(NamedTuple):
      history: list
      sentiment: float

    certainty: float
    influence: float
    money: int
    #tendency: namedtuple('Tendency', ['history', 'sentiment'])
    tendency: Tendency

  def __init__(self, unique_id: int, model: Model, initial_state, params: Params) -> None:
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

# --- SEIZM ---

class SEIZMstates(enum.IntEnum):
  DEFAULT = 0
  SUSCEPTIBLE = 0
  EXPOSED = 1
  SKEPTIC = 2
  INFECTED = 3


# TODO: docs
class SEIZMModel(Model):
  class Params(NamedTuple):
    #Maybe some of these will be useless
    initial_infected: float
    initial_skeptics: float
    prob_S_with_I: float  # TODO: docs
    prob_S_with_Z: float  # TODO: docs
    certainty_threshold: float
    influence_threshold: float
    money_theshold: float
    influence_increase: float
    certainty_increase: float
    neighbor_threshold: float = None  # TODO: docs


  def __init__(self, params: Params):
    """
    Implements the SEIZM Model
    """

    super().__init__()

    self.params = params

  def setup(self):
    # Initialize initial believers/skeptics
    n_initial_believers = math.floor(self.population_size * self.params.initial_infected)
    n_initial_skeptics = math.floor(self.population_size * self.params.initial_skeptics)

    initial_biased: List[MoneyAgent] = random.sample(self.schedule.agents, n_initial_believers + n_initial_skeptics)

    for i in range(n_initial_believers):
      initial_biased[i].state = SEIZMstates.INFECTED

    for i in range(n_initial_believers, len(initial_biased)):
      initial_biased[i].state = SEIZMstates.SKEPTIC

  def create_agent(self, unique_id: int, agent_params: MoneyAgent.Params):
    return MoneyAgent(unique_id, self, SEIZMstates.DEFAULT, agent_params)

  def resolve(self, agent: MoneyAgent) -> None:
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

      #agent is one interactor and neighbor is the other

      #CASE 1 - agent is exposed
      if agent.state == SEIZMstates.EXPOSED:

        transition_I = bernoulli.rvs(self.params.prob_S_with_I)
        transition_Z = bernoulli.rvs(self.params.prob_S_with_Z)

        if transition_I and not transition_Z and neighbor_not_skeptic:
          agent.state = SEIZplusStates.INFECTED

        if transition_Z and not transition_I and neighbor_not_infected:
          agent.state = SEIZplusStates.SKEPTIC

        #CASE 1.1 - DONE
        if neighbor.state == SEIZMstates.EXPOSED:

          agent_sentiment = agent.params.tendency.sentiment
          neighbor_sentiment = neighbor.params.tendency.sentiment
          balanced_sentiment = agent_sentiment + neighbor_sentiment / 2

          agent.params.tendency._replace(sentiment=balanced_sentiment)
          neighbor.params.tendency._replace(sentiment=balanced_sentiment)

        #CASE 1.2 - neighor is skeptic 
        if neighbor.state == SEIZMstates.SKEPTIC:

          #SUCCESSFULLY CONVINCING 
          if agent.params.tendency.sentiment <= 0.5 and agent.params.certainty <= neighbor.params.certainty:
            agent.state = SEIZMstates.SKEPTIC
            agent.params.certainty += neighbor.params.certainty * 0.1
            neighbor.params.influence = max(neighbor.params.influence + 0.1 * (0.5 - agent.params.tendency.sentiment), 1)
            #increase represents how good he was at convinicing him 

          #TOO HARSH
          if agent.params.tendency.sentiment > 0.5 and agent.params.certainty > neighbor.params.certainty:
            pass

          #CONVINCING BUT NOT ENOUGH SO ONLY UPDATE TENDENCY AND DECREASE CERTAINTY
          if agent.params.tendency.sentiment > 0.5 and agent.params.certainty <= neighbor.params.certainty:
            agent.params.certainty = agent.params.certainty - 0.05 * agent.params.certainty
            agent.params.tendency.sentiment = agent.params.tendency.sentiment
            neighbor.params.influence = 0 #TODO
            pass


        #CASE 1.3 - neighbor is infected
        if neighbor.state == SEIZMstates.INFECTED:

          if neighbor.params.money >= 0 and neighbor.params.certainty > self.params.certainty_threshold and neighbor.params.influence > self.params.influence_threshold:
            #qualifies to ask for money 
            p = random.uniform(0,1)
            #TODO make this condition more complex
            if p > self.params.money_theshold:
              #ask to spend money 
              if agent.params.tendency.sentiment > 0.5 and :
                #agree
                agent.params.money -= 1
                neighbor.params.money += 1
                agent.state = SEIZMstates.INFECTED
                #increase certainty by a factor of neighbors certainty and influence 
                agent.params.certainty += 0.1 * ((neighbor.params.certainty - self.params.certainty_threshold) + (neighbor.params.influence - self.params.influence_threshold))
                neighbor.params.influence += 0.1 * (1 - agent.params.tendency.sentiment)
              else:
                #disagree
                neighbor.params.influence -= 0.1 * min(0, agent.params.tendency.sentiment - 0.5) #because should have been easier to convince them if they tended towards you 
                agent.params.tendency.sentiment += 0
            else:
              pass
              #do not ask to spend money 


          #possible money proposition

      #CASE 2 - agent is susceptible
      if agent.state == SEIZMstates.SUSCEPTIBLE:

        #CASE 2.1 - neighor is skeptic 
        if neighbor.state == SEIZMstates.SKEPTIC:

          if bernoulli.rvs(self.params.prob_S_with_Z):
            agent.state = SEIZMstates.SKEPTIC
            neighbor.params.influence = max(1, self.params.influence_increase * (1 - abs(neighbor.params.influence - 0.5)) + neighbor.params.influence)
          else:
            agent.state = SEIZMstates.EXPOSED
            sentiment_after_interaction = 0 #has to be 0 or 1 #TODO
            agent.params.tendency.history.insert(0, sentiment_after_interaction)
            del agent.params.tendency.history[5]
            agent.params.tendency.sentiment = np.average(agent.params.tendency.history)
            #DO SMTH TO CERTAINTY AND INFLUENCE BASED ON TENDENCY

        #CASE 2.2 - neighbor is infected
        if neighbor.state == SEIZMstates.INFECTED:
          #transition to EXPOSED
          if bernoulli.rvs(self.params.prob_S_with_I):
            agent.state = SEIZMstates.INFECTED
            neighbor.params.influence = max(1, self.params.influence_increase * (1 - abs(neighbor.params.influence - 0.5)) + neighbor.params.influence)
          else:
            agent.state = SEIZMstates.EXPOSED
            sentiment_after_interaction = 0 #has to be 0 or 1 #TODO
            agent.params.tendency.history.insert(0, sentiment_after_interaction)
            del agent.params.tendency.history[5]
            agent.params.tendency.sentiment = np.average(agent.params.tendency.history)
            #DO SMTH TO CERTAINTY AND INFLUENCE BASED ON TENDENCY

      #CASE 3 - agent is skeptic
      if agent.state == SEIZMstates.SKEPTIC:

        #CASE 3.1 - neighbor is exposed
        if neighbor.state == SEIZMstates.EXPOSED:
          pass
          #symmetric to CASE 1.2
          #check if agent certainty and influence is too high wrt to sentiment

        #CASE 3.2 - neighbor is susceptible
        if neighbor.state == SEIZMstates.SUSCEPTIBLE:
          #symmetric to CASE 2.1
          if bernoulli.rvs(self.params.prob_S_with_I):
            neighbor.state = SEIZMstates.SKEPTIC
            #DO SMTH TO CERTAINTY AND INFLUENCE 
          else:
            neighbor.state = SEIZMstates.EXPOSED
            sentiment_after_interaction = 0 #has to be 0 or 1 
            neighbor.params.tendency.history.insert(0, sentiment_after_interaction)
            del neighbor.params.tendency.history[5]
            neighbor.params.tendency.sentiment = np.average(neighbor.params.tendency.history)
            #DO SMTH TO CERTAINTY AND INFLUENCE BASED ON TENDENCY

        #CASE 3.3 - neighor is skeptic 
        if neighbor.state == SEIZMstates.SKEPTIC:
          pass

        #CASE 3.4 - neighbor is infected
        if neighbor.state == SEIZMstates.INFECTED:
          #symmetric to CASE 4.3

          if neighbor.params.influence > agent.params.influence and neighbor.params.balance == 0: #times a ceratin epsilon (i.e. close to 0)
            #infected is stronger
            if agent.params.certainty < self.params.certainty_threshold:
              agent.state == SEIZMstates.EXPOSED #or infected
            else:
              agent.params.certainty = agent.params.certainty - 0.05
          else:
            #skeptic is stronger
            if neighbor.params.certainty < self.params.certainty_threshold:
              neighbor.state == SEIZMstates.EXPOSED #or infected
            else:
              neighbor.params.certainty = neighbor.params.certainty - 0.05

      #CASE 4 - agent is infected
      if agent.state == SEIZMstates.INFECTED:

        #CASE 4.1 - neighbor is exposed
        if neighbor.state == SEIZMstates.EXPOSED:
          #symmetric to CASE 1.3
          pass
          #check the tendency of neighbor
          #possible money proposition

        #CASE 4.2 - neighbor is susceptible
        if neighbor.state == SEIZMstates.SUSCEPTIBLE:
          #symmetric to CASE 2.2

          if bernoulli.rvs(self.params.prob_S_with_I):
            neighbor.state = SEIZMstates.INFECTED
            #DO SMTH TO CERTAINTY AND INFLUENCE 
          else:
            neighbor.state = SEIZMstates.EXPOSED
            sentiment_after_interaction = 0 #has to be 0 or 1 
            neighbor.params.tendency.history.insert(0, sentiment_after_interaction)
            del neighbor.params.tendency.history[5]
            neighbor.params.tendency.sentiment = np.average(neighbor.params.tendency.history)
            #DO SMTH TO CERTAINTY AND INFLUENCE BASED ON TENDENCY

        #CASE 4.3 - neighor is skeptic 
        if neighbor.state == SEIZMstates.SKEPTIC:
          #symmetric to CASE 3.4
          pass

        #CASE 4.4 - neighbor is infected
        if neighbor.state == SEIZMstates.INFECTED:
          #possible money proposition
          pass


    #----------- PAST --------------

      #Agent exposed
      if agent.state == SEIZplusStates.EXPOSED:
        transition_I = bernoulli.rvs(self.params.prob_S_with_I)
        transition_Z = bernoulli.rvs(self.params.prob_S_with_Z)

        neighbor_not_skeptic = neighbor.state != SEIZplusStates.SKEPTIC
        neighbor_not_infected = neighbor.state != SEIZplusStates.INFECTED

        if transition_I and not transition_Z and neighbor_not_skeptic:
          agent.state = SEIZplusStates.INFECTED

        if transition_Z and not transition_I and neighbor_not_infected:
          agent.state = SEIZplusStates.SKEPTIC

      #Agent susceptible and neighbor infected
      if (agent.state, neighbor.state) == (SEIZplusStates.SUSCEPTIBLE, SEIZplusStates.INFECTED):
        if bernoulli.rvs(self.params.prob_S_with_I):
          agent.state = SEIZplusStates.INFECTED
        else:
          agent.state = SEIZplusStates.EXPOSED

      #Agent susceptible and neighbor skeptic 
      if (agent.state, neighbor.state) == (SEIZplusStates.SUSCEPTIBLE, SEIZplusStates.SKEPTIC):
        if bernoulli.rvs(self.params.prob_S_with_Z):
          agent.state = SEIZplusStates.SKEPTIC
        else:
          agent.state = SEIZplusStates.EXPOSED



class Model:
  
  class ModelType(enum.IntEnum):
    SIR = 0
    SEIZ = 1
    SEIZplus = 2
    SEIZM = 3

  @staticmethod
  def model_by_type(model_type: ModelType) -> 'type(Model)':
    models = defaultdict()
    models[Model.ModelType.SEIZplus] = SEIZplusModel
    models[Model.ModelType.SEIZM] = SEIZMModel

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
