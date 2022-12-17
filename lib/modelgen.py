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
  class Params:
    certainty: float
    influence: float
    money: int
    sentiment: float

    def __init__(self, certainty: float, influence: float, money: int, sentiment: float) -> None:
      self.certainty = certainty
      self.influence = influence
      self.money = money
      self.sentiment = sentiment

  def __init__(self, unique_id: int, model: Model, initial_state, params: Params) -> None:
    super().__init__(unique_id, model)

    self.model = model
    self.state = initial_state
    self.params = MoneyAgent.Params(
      np.random.normal(params.certainty / 2, params.influence / 4),
      np.random.normal(params.influence / 2, params.influence / 4),
      params.money,
      params.sentiment + math.pow(-1, random.randint(0, 1)) * random.uniform(0, 0.2)
    )

  
  def step(self):
    self.model.resolve(self)


# --- SIR ---
class SIRStates(enum.IntEnum):
  DEFAULT = 0
  UNSURE = 0
  DISAGREE = 1
  BELIEVE = 2


# TODO: docs
class SIRModel(Model):
  class Params(NamedTuple):
    initial_infected: float
    initial_disagree: float
    p_opinion_change: float

  def __init__(self, params: Params):
    """
    Implements the SIR Model
    """

    super().__init__()

    self.params = params

  def setup(self):
    # Initialize initial believers/skeptics
    n_initial_believers = math.floor(self.population_size * self.params.initial_infected)
    n_initial_disagree = math.floor(self.population_size * self.params.initial_disagree)

    initial_biased: List[BaseAgent] = random.sample(self.schedule.agents, n_initial_believers + n_initial_disagree)

    for i in range(n_initial_believers):
      initial_biased[i].state = SIRStates.BELIEVE

    for i in range(n_initial_believers, len(initial_biased)):
      initial_biased[i].state = SIRStates.DISAGREE

  def create_agent(self, unique_id: int, agent_params: BaseAgent.Params):
    return BaseAgent(unique_id, self, SIRStates.DEFAULT, agent_params)

  def resolve(self, agent: BaseAgent) -> None:
    """
    Resolves interaction between agents

    :param agent: agent that is to update
    """

    neighbors = self.grid.get_neighbors(agent.unique_id, include_center=False)

    for neighbor_id in neighbors:
      neighbor_agent = self.schedule.agents[neighbor_id]

      if agent.state != neighbor_agent.state:
        if (agent.state, neighbor_agent.state) in [(SIRStates.DISAGREE, SIRStates.BELIEVE), (SIRStates.BELIEVE, SIRStates.DISAGREE)]:
          if bernoulli.rvs(self.params.p_opinion_change):
            agent.state = SIRStates.UNSURE
        elif agent.state == SIRStates.UNSURE:
          if bernoulli.rvs(self.params.p_opinion_change):
            agent.state = neighbor_agent.state


# --- SEIZ ---
class SEIZStates(enum.IntEnum):
  DEFAULT = 0
  SUSCEPTIBLE = 0
  EXPOSED = 1
  SKEPTIC = 2
  INFECTED = 3


# TODO: docs
class SEIZModel(Model):
  class Params(NamedTuple):
    initial_infected: float
    initial_skeptics: float
    prob_S_with_I: float  # TODO: docs
    prob_S_with_Z: float  # TODO: docs
    prob_E_to_I: float  # TODO: docs

  def __init__(self, params: Params):
    """
    Implements the SEIZ Model
    """

    super().__init__()

    self.params = params

  def setup(self):
    # Initialize initial believers/skeptics
    n_initial_believers = math.floor(self.population_size * self.params.initial_infected)
    n_initial_skeptics = math.floor(self.population_size * self.params.initial_skeptics)

    initial_biased: List[BaseAgent] = random.sample(self.schedule.agents, n_initial_believers + n_initial_skeptics)

    for i in range(n_initial_believers):
      initial_biased[i].state = SEIZStates.INFECTED

    for i in range(n_initial_believers, len(initial_biased)):
      initial_biased[i].state = SEIZStates.SKEPTIC

  def create_agent(self, unique_id: int, agent_params: BaseAgent.Params):
    return BaseAgent(unique_id, self, SEIZStates.DEFAULT, agent_params)

  def resolve(self, agent: BaseAgent) -> None:
    """
    Resolves interaction between agents

    :param agent: agent that is to update
    """

    # TODO: explain this
    neighbors = self.grid.get_neighbors(agent.unique_id, include_center=False)

    if len(neighbors) == 0:
      return

    # Normal transitions
    for neighbor_id in random.sample(neighbors, min(1, len(neighbors))):
      neighbor = self.schedule.agents[neighbor_id]

      if agent.state == SEIZStates.EXPOSED:
        if bernoulli.rvs(self.params.prob_E_to_I):
          agent.state = SEIZStates.INFECTED
          return

      if (agent.state, neighbor.state) == (SEIZStates.EXPOSED, SEIZStates.INFECTED):
        if bernoulli.rvs(self.params.prob_S_with_I):
          agent.state = SEIZStates.INFECTED

      if (agent.state, neighbor.state) == (SEIZStates.SUSCEPTIBLE, SEIZStates.INFECTED):
        if bernoulli.rvs(self.params.prob_S_with_I):
          agent.state = SEIZStates.INFECTED
        else:
          agent.state = SEIZStates.EXPOSED

      if (agent.state, neighbor.state) == (SEIZStates.SUSCEPTIBLE, SEIZStates.SKEPTIC):
        if bernoulli.rvs(self.params.prob_S_with_Z):
          agent.state = SEIZStates.SKEPTIC
        else:
          agent.state = SEIZStates.EXPOSED


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
      if agent.state == SEIZMstates.SKEPTIC:
        n_infected = 0
        for neighbor_id in neighbors:
          neighbor = self.schedule.agents[neighbor_id]
          if neighbor.state == SEIZMstates.INFECTED:
            n_infected += 1
        if n_infected > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZMstates.INFECTED
          return

      if agent.state == SEIZMstates.INFECTED:
        n_skeptic = 0
        for neighbor_id in neighbors:
          neighbor = self.schedule.agents[neighbor_id]
          if neighbor.state == SEIZMstates.SKEPTIC:
            n_skeptic += 1
        if n_skeptic > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZMstates.SKEPTIC
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
      if agent.state == SEIZMstates.SKEPTIC:
        n_infected = 0
        for neighbor_id in neighbors:
          neighbor = self.schedule.agents[neighbor_id]
          if neighbor.state == SEIZMstates.INFECTED:
            n_infected += 1
        if n_infected > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZMstates.INFECTED
          return

      if agent.state == SEIZMstates.INFECTED:
        n_skeptic = 0
        for neighbor_id in neighbors:
          neighbor = self.schedule.agents[neighbor_id]
          if neighbor.state == SEIZMstates.SKEPTIC:
            n_skeptic += 1
        if n_skeptic > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZMstates.SKEPTIC
          return

    # Normal transitions

    for neighbor_id in random.sample(neighbors, min(1, len(neighbors))):
      neighbor = self.schedule.agents[neighbor_id]

      #CASE 1 - agent is exposed
      if agent.state == SEIZMstates.EXPOSED:

        #CASE 1.1 - neighbor is exposed - DONE
        if neighbor.state == SEIZMstates.EXPOSED:

          #balance out the sentiments of both agents

          agent_sentiment = agent.params.sentiment
          neighbor_sentiment = neighbor.params.sentiment
          balanced_sentiment = agent_sentiment + neighbor_sentiment / 2

          agent.params.sentiment = balanced_sentiment
          neighbor.params.sentiment = balanced_sentiment

        #CASE 1.2 - DONE
        if neighbor.state == SEIZMstates.SKEPTIC:

          #Good setiment towards skeptics
          if agent.params.sentiment <= 0.5:
            
            if agent.params.certainty <= neighbor.params.certainty and neighbor.params.influence > self.params.influence_threshold:
              #skeptic is more certain and influent enough - move to skeptic
              agent.state = SEIZMstates.SKEPTIC
              agent.params.certainty += neighbor.params.certainty * self.params.certainty_increase
              neighbor.params.influence = max(neighbor.params.influence + self.params.influence_increase * (0.5 - agent.params.sentiment), 1)
              #increase represents how good he was at convinicing him 
            else:
              #exposed is more certain or skeptic not convincing enough - move towards skeptic
              agent.params.sentiment -= random.uniform(0,1) * agent.params.certainty
          
          else:
            #bad sentiment towards skeptics

            if neighbor.params.influence > self.params.influence_threshold: #still tries
              p = random.uniform(0,1)
              if (p > self.params.influence_threshold):
                #suceeds
                agent.state = SEIZMstates.SKEPTIC
                neighbor.params.influence += random.uniform(0,1) * (agent.params.sentiment) # 1 - for influence 
              else:
                #fails - sees the counterpart as too bold
                neighbor.params.influence -= random.uniform(0,1) * (1 - agent.params.sentiment) # not 1 - for inlfuence
                agent.params.sentiment += random.uniform(0,1) * agent.params.certainty
                agent.params.certainty += neighbor.params.influence * random.uniform(0,1)

            else:
              #small step in random direction
              agent.params.sentiment += math.pow(-1, random.randint(0,1)) * random.uniform(0,1) * agent.params.certainty



        #CASE 1.3 - DONE
        if neighbor.state == SEIZMstates.INFECTED:

          if neighbor.params.money >= 0 and neighbor.params.certainty > self.params.certainty_threshold and neighbor.params.influence > self.params.influence_threshold and random.uniform(0,1) > self.params.money_theshold:
            #ask to spend money 

            if agent.params.sentiment > 0.5 and agent.params.influence < self.params.influence_threshold and agent.params.certainty > self.params.certainty_threshold:
              #agree
              agent.params.money -= 1
              neighbor.params.money += 1
              agent.state = SEIZMstates.INFECTED
              #increase certainty by a factor of neighbors certainty and influence 
              agent.params.certainty += 0.1 * ((neighbor.params.certainty - self.params.certainty_threshold) + (neighbor.params.influence - self.params.influence_threshold))
              neighbor.params.influence += 0.1 * (1 - agent.params.sentiment)
            else:
              #disagree
              neighbor.params.influence -= 0.1 * min(0, agent.params.sentiment - 0.5) #because should have been easier to convince them if they tended towards you 
              agent.params.sentiment -= 0.1 * neighbor.params.influence

          else:
            #do not ask to spend money 
            if agent.params.sentiment > 0.5:
              #good sentiment towards infected
            
              if agent.params.certainty <= neighbor.params.certainty and neighbor.params.influence > self.params.influence_threshold:
                #infected is more certain and influent enough - move to infected
                agent.state = SEIZMstates.INFECTED
                agent.params.certainty = max(agent.params.certainty + neighbor.params.certainty * self.params.certainty_increase, 1)
                neighbor.params.influence = max(neighbor.params.influence + self.params.influence_increase * abs(0.5 - agent.params.sentiment), 1)
                #increase represents how good he was at convinicing him 
              else:
                #exposed is more certain or infected not convincing enough - move towards infected
                agent.params.sentiment += random.uniform(0,1) * agent.params.certainty
            
            else:
              #bad sentiment towards skeptics

              if neighbor.params.influence > self.params.influence_threshold: #still tries
                p = random.uniform(0,1)
                if (p > self.params.influence_threshold):
                  #suceeds
                  agent.state = SEIZMstates.INFECTED
                  neighbor.params.influence += random.uniform(0,1) * (1 - agent.params.sentiment) # 1 - for influence 
                else:
                  #fails - sees the counterpart as too bold
                  neighbor.params.influence -= random.uniform(0,1) * (agent.params.sentiment) # not 1 - for inlfuence
                  agent.params.sentiment -= random.uniform(0,1) * agent.params.certainty
                  agent.params.certainty += neighbor.params.influence * random.uniform(0,1) #higher increase if neighbor had high influence

              else:
                #small step in random direction
                agent.params.sentiment += math.pow(-1, random.randint(0,1)) * random.uniform(0,1) * agent.params.certainty



      #CASE 2 - agent is susceptible
      if agent.state == SEIZMstates.SUSCEPTIBLE:

        #CASE 2.1 - DONE - next step make more difficult to reach high levels and low levels of attributes
        if neighbor.state == SEIZMstates.SKEPTIC:

          if bernoulli.rvs(self.params.prob_S_with_Z):
            agent.state = SEIZMstates.SKEPTIC
            neighbor.params.influence = max(1, self.params.influence_increase * (1 - abs(neighbor.params.influence - 0.5)) + neighbor.params.influence)
          else:
            agent.state = SEIZMstates.EXPOSED
            #initialize all suscptible with a 0.5 sentiment
            step = neighbor.params.influence * random.uniform(0,1)
            agent.params.sentiment -= step
            if step > random.uniform(0,1) * self.params.certainty_threshold:
              agent.params.certainty += random.uniform(0,1) * step
            if step > random.uniform(0,1) * self.params.influence_threshold:
              neighbor.params.influence += random.uniform(0,1) * step

        #CASE 2.2 - DONE
        if neighbor.state == SEIZMstates.INFECTED:
          #transition to EXPOSED
          if bernoulli.rvs(self.params.prob_S_with_I):
            agent.state = SEIZMstates.INFECTED
            neighbor.params.influence = max(1, self.params.influence_increase * (1 - abs(neighbor.params.influence - 0.5)) + neighbor.params.influence)
          else:
            agent.state = SEIZMstates.EXPOSED
            step = neighbor.params.influence * random.uniform(0,1)
            agent.params.sentiment += step
            if step > random.uniform(0,1) * self.params.certainty_threshold:
              agent.params.certainty += random.uniform(0,1) * step
            if step > random.uniform(0,1) * self.params.influence_threshold:
              neighbor.params.influence += random.uniform(0,1) * step

      #CASE 3 - agent is skeptic
      if agent.state == SEIZMstates.SKEPTIC:

        #CASE 3.1 - DONE
        if neighbor.state == SEIZMstates.EXPOSED:

          #Good setiment towards skeptics
          if neighbor.params.sentiment <= 0.5:
            
            if neighbor.params.certainty <= agent.params.certainty and agent.params.influence > self.params.influence_threshold:
              #skeptic is more certain and influent enough - move to skeptic
              neighbor.state = SEIZMstates.SKEPTIC
              neighbor.params.certainty += agent.params.certainty * self.params.certainty_increase
              agent.params.influence = max(agent.params.influence + self.params.influence_increase * (0.5 - neighbor.params.sentiment), 1)
              #increase represents how good he was at convinicing him 
            else:
              #exposed is more certain or skeptic not convincing enough - move towards skeptic
              neighbor.params.sentiment -= random.uniform(0,1) * neighbor.params.certainty
          
          else:
            #bad sentiment towards skeptics

            if agent.params.influence > self.params.influence_threshold: #still tries
              p = random.uniform(0,1)
              if (p > self.params.influence_threshold):
                #suceeds
                neighbor.state = SEIZMstates.SKEPTIC
                agent.params.influence += random.uniform(0,1) * (neighbor.params.sentiment) # 1 - for influence 
              else:
                #fails - sees the counterpart as too bold
                agent.params.influence -= random.uniform(0,1) * (1 - neighbor.params.sentiment) # not 1 - for inlfuence
                neighbor.params.sentiment += random.uniform(0,1) * neighbor.params.certainty
                neighbor.params.certainty += agent.params.influence * random.uniform(0,1)

            else:
              #small step in random direction
              neighbor.params.sentiment += math.pow(-1, random.randint(0,1)) * random.uniform(0,1) * neighbor.params.certainty

        #CASE 3.2 - DONE
        if neighbor.state == SEIZMstates.SUSCEPTIBLE:

          if bernoulli.rvs(self.params.prob_S_with_Z):
            neighbor.state = SEIZMstates.SKEPTIC
            agent.params.influence = max(1, self.params.influence_increase * (1 - abs(agent.params.influence - 0.5)) + agent.params.influence)
          else:
            neighbor.state = SEIZMstates.EXPOSED
            #initialize all suscptible with a 0.5 sentiment
            step = agent.params.influence * random.uniform(0,1)
            neighbor.params.sentiment -= step
            if step > random.uniform(0,1) * self.params.certainty_threshold:
              neighbor.params.certainty += random.uniform(0,1) * step
            if step > random.uniform(0,1) * self.params.influence_threshold:
              agent.params.influence += random.uniform(0,1) * step

        #CASE 3.3 - DONE
        if neighbor.state == SEIZMstates.SKEPTIC:

          certainty_reinforcement = neighbor.params.certainty + agent.params.certainty + 1 / 3
          neighbor.params.certainty += certainty_reinforcement
          agent.params.certainty += certainty_reinforcement

        #CASE 3.4 - DONE
        if neighbor.state == SEIZMstates.INFECTED:

          certainty_condition_neighbor_wins = agent.params.certainty <= self.params.certainty_threshold and neighbor.params.certainty >= self.params.certainty_threshold
          certainty_condition_agent_wins = neighbor.params.certainty < self.params.certainty_threshold and agent.params.certainty > self.params.certainty_threshold
          influence_condition_neighbor_wins = agent.params.influence <= self.params.influence_threshold and neighbor.params.influence >= self.params.influence_threshold
          influence_condition_agent_wins = neighbor.params.influence < self.params.influence_threshold and agent.params.influence > self.params.influence_threshold

          if influence_condition_neighbor_wins: #times a ceratin epsilon (i.e. close to 0)
            if certainty_condition_neighbor_wins:
              agent.state = SEIZMstates.EXPOSED
            else:
              agent.params.certainty -= 0.01 * neighbor.params.influence

          if influence_condition_agent_wins and neighbor.params.money == 0:
            if certainty_condition_agent_wins:
              neighbor.state = SEIZMstates.EXPOSED
            else:
              neighbor.params.certainty -= 0.01 * agent.params.influence

      #CASE 4 - agent is infected
      if agent.state == SEIZMstates.INFECTED:

        #CASE 4.1 - DONE
        if neighbor.state == SEIZMstates.EXPOSED:
          
          if agent.params.money >= 0 and agent.params.certainty > self.params.certainty_threshold and agent.params.influence > self.params.influence_threshold and random.uniform(0,1) > self.params.money_theshold:
            #ask to spend money 

            if neighbor.params.sentiment > 0.5 and neighbor.params.influence < self.params.influence_threshold and neighbor.params.certainty > self.params.certainty_threshold:
              #agree
              neighbor.params.money -= 1
              agent.params.money += 1
              neighbor.state = SEIZMstates.INFECTED
              #increase certainty by a factor of neighbors certainty and influence 
              neighbor.params.certainty += 0.1 * ((agent.params.certainty - self.params.certainty_threshold) + (agent.params.influence - self.params.influence_threshold))
              agent.params.influence += 0.1 * (1 - neighbor.params.sentiment)
            else:
              #disagree
              agent.params.influence -= 0.1 * min(0, neighbor.params.sentiment - 0.5) #because should have been easier to convince them if they tended towards you 
              neighbor.params.sentiment -= 0.1 * agent.params.influence

          else:
            #do not ask to spend money 
            if neighbor.params.sentiment > 0.5:
              #good sentiment towards infected
            
              if neighbor.params.certainty <= agent.params.certainty and agent.params.influence > self.params.influence_threshold:
                #infected is more certain and influent enough - move to infected
                neighbor.state = SEIZMstates.INFECTED
                neighbor.params.certainty = max(neighbor.params.certainty + agent.params.certainty * self.params.certainty_increase, 1)
                agent.params.influence = max(agent.params.influence + self.params.influence_increase * abs(0.5 - neighbor.params.sentiment), 1)
                #increase represents how good he was at convinicing him 
              else:
                #exposed is more certain or infected not convincing enough - move towards infected
                neighbor.params.sentiment += random.uniform(0,1) * neighbor.params.certainty
            
            else:
              #bad sentiment towards skeptics

              if agent.params.influence > self.params.influence_threshold: #still tries
                p = random.uniform(0,1)
                if (p > self.params.influence_threshold):
                  #suceeds
                  neighbor.state = SEIZMstates.INFECTED
                  agent.params.influence += random.uniform(0,1) * (1 - neighbor.params.sentiment) # 1 - for influence 
                else:
                  #fails - sees the counterpart as too bold
                  agent.params.influence -= random.uniform(0,1) * (neighbor.params.sentiment) # not 1 - for inlfuence
                  neighbor.params.sentiment -= random.uniform(0,1) * neighbor.params.certainty
                  neighbor.params.certainty += agent.params.influence * random.uniform(0,1) #higher increase if neighbor had high influence

              else:
                #small step in random direction
                neighbor.params.sentiment += math.pow(-1, random.randint(0,1)) * random.uniform(0,1) * neighbor.params.certainty



        #CASE 4.2 - DONE
        if neighbor.state == SEIZMstates.SUSCEPTIBLE:

          if bernoulli.rvs(self.params.prob_S_with_I):
            neighbor.state = SEIZMstates.INFECTED
            agent.params.influence = max(1, self.params.influence_increase * (1 - abs(agent.params.influence - 0.5)) + agent.params.influence)
          else:
            neighbor.state = SEIZMstates.EXPOSED
            step = agent.params.influence * random.uniform(0,1)
            neighbor.params.sentiment += step
            if step > random.uniform(0,1) * self.params.certainty_threshold:
              neighbor.params.certainty += random.uniform(0,1) * step
            if step > random.uniform(0,1) * self.params.influence_threshold:
              agent.params.influence += random.uniform(0,1) * step

        #CASE 4.3 - DONE
        if neighbor.state == SEIZMstates.SKEPTIC:

          certainty_condition_neighbor_wins = agent.params.certainty <= self.params.certainty_threshold and neighbor.params.certainty >= self.params.certainty_threshold
          certainty_condition_agent_wins = neighbor.params.certainty < self.params.certainty_threshold and agent.params.certainty > self.params.certainty_threshold
          influence_condition_neighbor_wins = agent.params.influence <= self.params.influence_threshold and neighbor.params.influence >= self.params.influence_threshold
          influence_condition_agent_wins = neighbor.params.influence < self.params.influence_threshold and agent.params.influence > self.params.influence_threshold

          if influence_condition_neighbor_wins and agent.params.money == 0: #times a ceratin epsilon (i.e. close to 0)
            if certainty_condition_neighbor_wins:
              agent.state = SEIZMstates.EXPOSED
            else:
              agent.params.certainty -= 0.01 * neighbor.params.influence

          if influence_condition_agent_wins:
            if certainty_condition_agent_wins:
              neighbor.state = SEIZMstates.EXPOSED
            else:
              neighbor.params.certainty -= 0.01 * agent.params.influence

        #CASE 4.4 - DONE
        if neighbor.state == SEIZMstates.INFECTED:

          positive_balance_agent = agent.params.money >= 0
          certainty_condition_agent = agent.params.certainty > self.params.certainty_threshold
          influence_condition_agent = agent.params.influence > self.params.influence_threshold
          random_param = random.uniform(0,1) > self.params.money_theshold

          if positive_balance_agent and certainty_condition_agent and influence_condition_agent and random_param:
            #aqgent asks neighbor to spend money 

            if neighbor.params.influence < self.params.influence_threshold and neighbor.params.money <= 0:
              #agree
              neighbor.params.money -= 1
              agent.params.money += 1
              #increase certainty by a factor of neighbors certainty and influence 
              neighbor.params.certainty += 0.1 * ((agent.params.certainty - self.params.certainty_threshold) + (agent.params.influence - self.params.influence_threshold))
              agent.params.influence += 0.1 * (1 - (agent.params.influence - neighbor.params.influence))
            else:
              #disagree
              agent.params.influence -= 0.1 * (min(self.params.influence_threshold - neighbor.params.influence, 0)) #because should have been easier to convince them if they tended towards you 
              neighbor.params.influence += 0.1 * agent.params.influence

          positive_balance = neighbor.params.money >= 0
          certainty_condition = neighbor.params.certainty > self.params.certainty_threshold
          influence_condition = neighbor.params.influence > self.params.influence_threshold
          random_param = random.uniform(0,1) > self.params.money_theshold

          if positive_balance and certainty_condition and influence_condition and random_param:
            #neighbor asks agent to spend money 

            if agent.params.influence < self.params.influence_threshold and agent.params.money <= 0:
              #agree
              agent.params.money -= 1
              neighbor.params.money += 1
              #increase certainty by a factor of neighbors certainty and influence 
              agent.params.certainty += 0.1 * ((neighbor.params.certainty - self.params.certainty_threshold) + (neighbor.params.influence - self.params.influence_threshold))
              neighbor.params.influence += 0.1 * (1 - (neighbor.params.influence - agent.params.influence))
            else:
              #disagree
              neighbor.params.influence -= 0.1 * (min(self.params.influence_threshold - agent.params.influence, 0)) #because should have been easier to convince them if they tended towards you 
              agent.params.influence += 0.1 * neighbor.params.influence

          #add reinforcement otherwise 


class Model:
  
  class ModelType(enum.IntEnum):
    SIR = 0
    SEIZ = 1
    SEIZplus = 2
    SEIZM = 3

  @staticmethod
  def model_by_type(model_type: ModelType) -> 'type(Model)':
    models = defaultdict()
    models[Model.ModelType.SIR] = SIRModel
    models[Model.ModelType.SEIZ] = SEIZModel
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
