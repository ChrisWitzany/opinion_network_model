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
  """
  Base implementation of an Agent, it only has a single state variable and no specialized parameters
  It also depicts the interface for specialized agents
  """

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

  """
    Implements the agent instances for the SEIZM model as extension of the BaseAgent wiht more advanced agent parameters

    :param certainty: float between 0 and 1 indicating how certain an agent is about its belief
    :param influence: float between 0 and 1 indicating how influent an agent is 
    :param money: money balance of an agent, sign indicated the balance (spent or earned money) and the absolute value indicates nr. of transactions
    :param sentiment: float between 0 and 1 indicating towards which opinion - Skeptic or Infected - an agent tends

  """

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


class SIRModel(Model):
  """
  Implements the SIR Model
  """

  class Params(NamedTuple):
    initial_infected: float
    initial_disagree: float
    p_opinion_change: float

  def __init__(self, params: Params):
    super().__init__()

    self.params = params

  def setup(self):
    # Setup initial believers/skeptics
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
    Resolves interaction between agents.

    :param agent: agent that is to update
    """

    neighbors = self.grid.get_neighbors(agent.unique_id, include_center=False)

    # Go over all neighbors
    for neighbor_id in neighbors:
      neighbor_agent = self.schedule.agents[neighbor_id]

      if agent.state != neighbor_agent.state:
        # If neighbor has different opinion, the agent becomes unsure with some probability
        if (agent.state, neighbor_agent.state) in [(SIRStates.DISAGREE, SIRStates.BELIEVE), (SIRStates.BELIEVE, SIRStates.DISAGREE)]:
          if bernoulli.rvs(self.params.p_opinion_change):
            agent.state = SIRStates.UNSURE
        # If unsure, the agent adapts the neighbors' opinion with some probability
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


class SEIZModel(Model):
  """
  Implements the SEIZ Model
  """

  class Params(NamedTuple):
    initial_infected: float
    initial_skeptics: float
    prob_S_with_I: float
    prob_S_with_Z: float
    prob_E_to_I: float

  def __init__(self, params: Params):
    super().__init__()

    self.params = params

  def setup(self):
    # Setup initial believers/skeptics
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

    neighbors = self.grid.get_neighbors(agent.unique_id, include_center=False)

    if len(neighbors) == 0:
      return

    # Normal transitions

    # Select random neighbor
    for neighbor_id in random.sample(neighbors, min(1, len(neighbors))):
      neighbor = self.schedule.agents[neighbor_id]

      # Exposed agents can transition to being infected with some probability
      if agent.state == SEIZStates.EXPOSED:
        if bernoulli.rvs(self.params.prob_E_to_I):
          agent.state = SEIZStates.INFECTED
          return

      # Exposed agents can transition to being infected with some probability
      # when in contact with infected agent
      if (agent.state, neighbor.state) == (SEIZStates.EXPOSED, SEIZStates.INFECTED):
        if bernoulli.rvs(self.params.prob_S_with_I):
          agent.state = SEIZStates.INFECTED

      # Susceptible agents can transition to being infected with some probability
      # when in contact with infected agent, otherwise they become exposed
      if (agent.state, neighbor.state) == (SEIZStates.SUSCEPTIBLE, SEIZStates.INFECTED):
        if bernoulli.rvs(self.params.prob_S_with_I):
          agent.state = SEIZStates.INFECTED
        else:
          agent.state = SEIZStates.EXPOSED

      # Susceptible agents can transition to being skeptic with some probability
      # when in contact with skeptic agent, otherwise they become exposed
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


class SEIZplusModel(Model):
  """
  Implements the SEIZ+ Model
  """

  class Params(NamedTuple):
    initial_infected: float
    initial_skeptics: float
    prob_S_with_I: float
    prob_S_with_Z: float
    neighbor_threshold: float = None

  def __init__(self, params: Params):
    super().__init__()

    self.params = params

  def setup(self):
    # Setup initial believers/skeptics
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

    neighbors = self.grid.get_neighbors(agent.unique_id, include_center=False)

    if len(neighbors) == 0:
      return

    # Group pressure effect, an agent might change his state when a certain threshold of neighboring
    # agents that have a counter-opinion is reached.
    if self.params.neighbor_threshold is not None:
      if agent.state == SEIZplusStates.SKEPTIC:
        n_infected = sum([
          1 if self.schedule.agents[neighbor_id].state == SEIZplusStates.INFECTED
          else 0
          for neighbor_id in neighbors
        ])

        if n_infected > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZplusStates.INFECTED
          return

      if agent.state == SEIZplusStates.INFECTED:
        n_skeptic = sum([
          1 if self.schedule.agents[neighbor_id].state == SEIZplusStates.SKEPTIC
          else 0
          for neighbor_id in neighbors
        ])

        if n_skeptic > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZplusStates.SKEPTIC
          return

    # Normal transitions

    # Select random neighbor
    for neighbor_id in random.sample(neighbors, min(1, len(neighbors))):
      neighbor = self.schedule.agents[neighbor_id]

      # If agent is exposed they can decide on either skeptic or infected if exclusively
      # the corresponding random variable is true and they are currently not in contact
      # with an agent of opposing opinion
      if agent.state == SEIZplusStates.EXPOSED:
        transition_I = bernoulli.rvs(self.params.prob_S_with_I)
        transition_Z = bernoulli.rvs(self.params.prob_S_with_Z)

        neighbor_not_skeptic = neighbor.state != SEIZplusStates.SKEPTIC
        neighbor_not_infected = neighbor.state != SEIZplusStates.INFECTED

        if transition_I and not transition_Z and neighbor_not_skeptic:
          agent.state = SEIZplusStates.INFECTED

        if transition_Z and not transition_I and neighbor_not_infected:
          agent.state = SEIZplusStates.SKEPTIC

      # If agent is susceptible and in contact with an infected agent, they also become
      # infected with some probability, otherwise they become exposed
      if (agent.state, neighbor.state) == (SEIZplusStates.SUSCEPTIBLE, SEIZplusStates.INFECTED):
        if bernoulli.rvs(self.params.prob_S_with_I):
          agent.state = SEIZplusStates.INFECTED
        else:
          agent.state = SEIZplusStates.EXPOSED

      # If agent is susceptible and in contact with a skeptic agent, they also become
      # skeptic with some probability, otherwise they become exposed
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


class SEIZMModel(Model):
  """
  Implements the SEIZM Model
  """

  class Params(NamedTuple):
    initial_infected: float
    initial_skeptics: float
    prob_S_with_I: float
    prob_S_with_Z: float
    certainty_threshold: float
    influence_threshold: float
    money_threshold: float
    influence_increase: float
    certainty_increase: float
    neighbor_threshold: float = None

  def __init__(self, params: Params):
    super().__init__()

    self.params = params

  def setup(self):
    # Setup initial believers/skeptics
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

    neighbors = self.grid.get_neighbors(agent.unique_id, include_center=False)

    if len(neighbors) == 0:
      return

    # Group pressure effect, an agent might change his state when a certain threshold of neighboring
    # agents that have a counter-opinion is reached.
    if self.params.neighbor_threshold is not None:
      if agent.state == SEIZMstates.SKEPTIC:
        n_infected = sum([
          1 if self.schedule.agents[neighbor_id].state == SEIZMstates.INFECTED
          else 0
          for neighbor_id in neighbors
        ])

        if n_infected > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZMstates.INFECTED
          return

      if agent.state == SEIZMstates.INFECTED:
        n_skeptic = sum([
          1 if self.schedule.agents[neighbor_id].state == SEIZMstates.SKEPTIC
          else 0
          for neighbor_id in neighbors
        ])

        if n_skeptic > self.params.neighbor_threshold * len(neighbors):
          agent.state = SEIZMstates.SKEPTIC
          return

    # Normal transitions

    for neighbor_id in random.sample(neighbors, min(1, len(neighbors))):
      neighbor = self.schedule.agents[neighbor_id]

      #CASE 1 - agent is exposed
      if agent.state == SEIZMstates.EXPOSED:

        #CASE 1.1 - neighbor is exposed - DONE
        if neighbor.state == SEIZMstates.SUSCEPTIBLE:
          neighbor.state = SEIZMstates.EXPOSED

        #CASE 1.2 - neighbor is skeptic
        if neighbor.state == SEIZMstates.SKEPTIC:

          #Good setiment towards skeptics
          if agent.params.sentiment <= 0.5:

            certainty_test = agent.params.certainty <= neighbor.params.certainty
            influence_test = neighbor.params.influence > self.params.influence_threshold
            
            #skeptic is sufficiently influent and certain to trigger a transition of the agent to skeptic
            if certainty_test and influence_test:

              #skeptic is more certain and influent enough - move to skeptic
              agent.state = SEIZMstates.SKEPTIC

              #the more certain the neighbor is, the more certain the agent becomes
              agent.params.certainty = max(1, agent.params.certainty + neighbor.params.certainty * self.params.certainty_increase)

              #increase in influence depends on how good he was at convinicing him - if sentiment was already close to 0 influence goes up less
              neighbor.params.influence = max(neighbor.params.influence + self.params.influence_increase * agent.params.sentiment, 1)
              
            #exposed is more certain or skeptic not convincing enough - semtiments moves towards skeptic
            else:

              #move more drastically towards skeptic the more the agent is certain
              #agent.params.sentiment = min(0, agent.params.sentiment - random.uniform(0,1) * agent.params.certainty)
              agent.params.sentiment = min(0, agent.params.sentiment - random.uniform(0,1) * agent.params.certainty)
          
          #bad sentiment towards skeptics
          else:

            #if skeptic is influent enough, he still tries
            if neighbor.params.influence > self.params.influence_threshold:

              p = random.uniform(0,1)

              #agent accepts with random probability - more likely if skeptic is more influent 
              if (p < neighbor.params.influence):

                #neighbor suceeded in convincing 
                agent.state = SEIZMstates.SKEPTIC

                #influence increase depends on how certain the agent was in the opposite opinion
                neighbor.params.influence = max(1, neighbor.params.influence + agent.params.sentiment)
                #neighbor.params.influence = max(1, neighbor.params.influence + random.uniform(0,1) * agent.params.sentiment)

              #agent rejects and sees the counterpart as too bold
              else:

                #influence decrease depends on how easy it should've been to convince the agent (i.e small decrease if close to 1)
                #neighbor.params.influence = min(0, neighbor.params.influence - random.uniform(0,1) * (1 - agent.params.sentiment))
                neighbor.params.influence = min(0, neighbor.params.influence - (1 - agent.params.sentiment))

                #sentiment is pushed even further in the opposite direction 
                #agent.params.sentiment = max(1, agent.params.sentiment + random.uniform(0,1) * agent.params.certainty)
                agent.params.sentiment = max(1, agent.params.sentiment + agent.params.certainty)


                #the more influent (bolder) the skeptic was, the more the agent becomes more certain
                #agent.params.certainty = max(1, agent.params.certainty + neighbor.params.influence * random.uniform(0,1))
                agent.params.certainty = max(1, agent.params.certainty + neighbor.params.influence)

            #does not try to interact with agent 
            else:

              #small step in random direction
              agent.params.sentiment = max(1, agent.params.sentiment + math.pow(-1, random.randint(0,1)) * random.uniform(0,1) * agent.params.certainty)



        #CASE 1.3 - neighbor is infected
        if neighbor.state == SEIZMstates.INFECTED:

          money_condition = neighbor.params.money >= 0
          certainty_condition = neighbor.params.certainty > self.params.certainty_threshold
          influence_condition = neighbor.params.influence > self.params.influence_threshold
          random_condition = random.uniform(0,1) > self.params.money_threshold

          if money_condition and certainty_condition and influence_condition and random_condition:
            #ask to spend money 

            matching_sentiment = agent.params.sentiment > 0.5
            influence_condition = agent.params.influence < self.params.influence_threshold
            certainty_condition = agent.params.certainty > self.params.certainty_threshold

            #agrees if has a matching sentiment, a low influence (easily influenced), and a high certainty
            if matching_sentiment and influence_condition and certainty_condition:

              agent.params.money -= 1
              neighbor.params.money += 1
              agent.state = SEIZMstates.INFECTED

              #increase certainty by a factor of neighbors certainty and influence
              convincing_factor = (neighbor.params.certainty - self.params.certainty_threshold) + (neighbor.params.influence - self.params.influence_threshold)
              agent.params.certainty = max(1, agent.params.certainty + self.params.certainty_increase * convincing_factor)
              
              #rate of influence increase depends on the agent's sentiment
              neighbor.params.influence = max(1, neighbor.params.influence + self.params.influence_increase * (1 - agent.params.sentiment))
            
            #disagree to spend money
            else:

              #decrease the influence of the infected more if agent's sentiment was already close to 1
              #should have been easier to convince them if they tended towards you 
              neighbor.params.influence = min(0, neighbor.params.influence - self.params.influence_increase * min(0, agent.params.sentiment - 0.5))
              
              #decrease the agent's sentiment towards skeptic since neighbor viewed as too bold
              agent.params.sentiment = min(0, agent.params.sentiment - neighbor.params.influence)

          #do not ask to spend money 
          else:

            #good sentiment towards infected
            if agent.params.sentiment > 0.5:
            
              certainty_condition = agent.params.certainty <= neighbor.params.certainty
              influence_condition = neighbor.params.influence > self.params.influence_threshold
              
              #infected is more certain and influent enough - move to infected
              if certainty_condition and influence_condition:

                agent.state = SEIZMstates.INFECTED

                #increase agent certainty the more the infected is certain
                agent.params.certainty = max(agent.params.certainty + neighbor.params.certainty * self.params.certainty_increase, 1)
                
                #increase represents how good he was at convinicing him 
                neighbor.params.influence = max(neighbor.params.influence + self.params.influence_increase * (1 - agent.params.sentiment), 1)
                
              #exposed is more certain or infected not convincing enough - move towards infected
              else:
            
                agent.params.sentiment = max(1, agent.params.sentiment + agent.params.certainty)
                #agent.params.sentiment = max(1, agent.params.sentiment + random.uniform(0,1) * agent.params.certainty)
            
            #bad sentiment towards infected
            else:

              #if infected is influent enough, he still tries
              if neighbor.params.influence > self.params.influence_threshold: 
                p = random.uniform(0,1)
                if (p > self.params.influence_threshold):
                  #suceeds
                  agent.state = SEIZMstates.INFECTED
                  neighbor.params.influence = max(1, neighbor.params.influence + self.params.influence_increase * (1 - agent.params.sentiment))
                else:
                  #fails - sees the counterpart as too bold
                  neighbor.params.influence = min(0, neighbor.params.influence - random.uniform(0,1) * agent.params.sentiment)
                  #agent.params.sentiment = min(0, agent.params.sentiment - random.uniform(0,1) * agent.params.certainty)
                  agent.params.sentiment = min(0, agent.params.sentiment - agent.params.certainty)
                  #agent.params.certainty = max(1, agent.params.certainty + neighbor.params.influence * random.uniform(0,1)) higher increase if neighbor had high influence
                  agent.params.certainty = max(1, agent.params.certainty + neighbor.params.influence)

              else:
                #small step in random direction
                agent.params.sentiment = max(1, agent.params.sentiment + math.pow(-1, random.randint(0,1)) * random.uniform(0,1) * agent.params.certainty)



      #CASE 2 - agent is susceptible
      if agent.state == SEIZMstates.SUSCEPTIBLE:

        #CASE 2.1 - neighbor is skeptic 
        if neighbor.state == SEIZMstates.SKEPTIC:


          #move directly to skeptic without going to exposed 
          if bernoulli.rvs(self.params.prob_S_with_Z):

            agent.state = SEIZMstates.SKEPTIC

            #increase influence of neighbor
            neighbor.params.influence = max(1, self.params.influence_increase + neighbor.params.influence)
          
          #move to exposed
          else:

            agent.state = SEIZMstates.EXPOSED

            #gain sentiment for the skeptics by a factor of the neighbor's influence 
            step = neighbor.params.influence * random.uniform(0,1)
            agent.params.sentiment = min(0, agent.params.sentiment - step)

            #increase certainty of agent by a factor of neighbor's influence 
            if step > random.uniform(0,1) * self.params.certainty_threshold:
              agent.params.certainty = max(agent.params.certainty + random.uniform(0,1) * step, 1)

            #increase influence of neighbor
            if step > random.uniform(0,1) * self.params.influence_threshold:
              neighbor.params.influence = max(1, neighbor.params.influence + random.uniform(0,1) * step)

        #CASE 2.2 - neighbor is infected
        if neighbor.state == SEIZMstates.INFECTED:

          #directly transition to infected
          if bernoulli.rvs(self.params.prob_S_with_I):

            agent.state = SEIZMstates.INFECTED

            #increase neighbor influence 
            neighbor.params.influence = max(1, self.params.influence_increase + neighbor.params.influence)

          #transition to exposed 
          else:

            agent.state = SEIZMstates.EXPOSED

            #gain sentiment for the infected by a factor of the neighbor's influence 
            step = neighbor.params.influence * random.uniform(0,1)
            agent.params.sentiment += step

            #increase certainty of agent by a factor of neighbor's influence 
            if step > random.uniform(0,1) * self.params.certainty_threshold:
              agent.params.certainty = max(agent.params.certainty + random.uniform(0,1) * step, 1)

            #increase influence of neighbor
            if step > random.uniform(0,1) * self.params.influence_threshold:
              neighbor.params.influence = max(1, neighbor.params.influence + random.uniform(0,1) * step)
        
        #CASE 2.3 - neighbor is Exposed
        if neighbor.state == SEIZMstates.EXPOSED:
          agent.state = SEIZMstates.EXPOSED

      #CASE 3 - agent is skeptic
      if agent.state == SEIZMstates.SKEPTIC:

        #CASE 3.1 - neighbor is exposed
        if neighbor.state == SEIZMstates.EXPOSED:

          #Good setiment towards skeptics
          if neighbor.params.sentiment <= 0.5:

            certainty_test = neighbor.params.certainty <= agent.params.certainty
            influence_test = agent.params.influence > self.params.influence_threshold
            
            #skeptic is sufficiently influent and certain to trigger a transition of the agent to skeptic
            if certainty_test and influence_test:

              #skeptic is more certain and influent enough - move to skeptic
              neighbor.state = SEIZMstates.SKEPTIC

              #the more certain the neighbor is, the more certain the agent becomes
              neighbor.params.certainty = max(1, neighbor.params.certainty + agent.params.certainty * self.params.certainty_increase)

              #increase in influence depends on how good he was at convinicing him - if sentiment was already close to 0 influence goes up less
              agent.params.influence = max(agent.params.influence + self.params.influence_increase * neighbor.params.sentiment, 1)
              
            #exposed is more certain or skeptic not convincing enough - semtiments moves towards skeptic
            else:

              #move more drastically towards skeptic the more the agent is certain
              #agent.params.sentiment = min(0, agent.params.sentiment - random.uniform(0,1) * agent.params.certainty)
              neighbor.params.sentiment = min(0, neighbor.params.sentiment - random.uniform(0,1) * neighbor.params.certainty)
          
          #bad sentiment towards skeptics
          else:

            #if skeptic is influent enough, he still tries
            if agent.params.influence > self.params.influence_threshold:

              p = random.uniform(0,1)

              #neighbor accepts with random probability - more likely if skeptic is more influent 
              if (p < agent.params.influence):

                #agent suceeded in convincing 
                neighbor.state = SEIZMstates.SKEPTIC

                #influence increase depends on how certain the agent was in the opposite opinion
                agent.params.influence = max(1, agent.params.influence + neighbor.params.sentiment)
                #neighbor.params.influence = max(1, neighbor.params.influence + random.uniform(0,1) * agent.params.sentiment)

              #agent rejects and sees the counterpart as too bold
              else:

                #influence decrease depends on how easy it should've been to convince the neighbor (i.e small decrease if close to 1)
                #agent.params.influence = min(0, neighbor.params.influence - random.uniform(0,1) * (1 - agent.params.sentiment))
                agent.params.influence = min(0, agent.params.influence - (1 - neighbor.params.sentiment))

                #sentiment is pushed even further in the opposite direction 
                #neighbor.params.sentiment = max(1, neighbor.params.sentiment + random.uniform(0,1) * neighbor.params.certainty)
                neighbor.params.sentiment = max(1, neighbor.params.sentiment + neighbor.params.certainty)


                #the more influent (bolder) the skeptic was, the more the agent becomes more certain
                #neighbor.params.certainty = max(1, neighbor.params.certainty + agent.params.influence * random.uniform(0,1))
                neighbor.params.certainty = max(1, neighbor.params.certainty + agent.params.influence)

            #does not try to interact with neighbor 
            else:

              #small step in random direction
              neighbor.params.sentiment = max(1, neighbor.params.sentiment + math.pow(-1, random.randint(0,1)) * random.uniform(0,1) * neighbor.params.certainty)


        #CASE 3.2 - neighbor is susceptible 
        if neighbor.state == SEIZMstates.SUSCEPTIBLE:

          #move directly to skeptic without going to exposed 
          if bernoulli.rvs(self.params.prob_S_with_Z):

            neighbor.state = SEIZMstates.SKEPTIC

            #increase influence of agents
            agent.params.influence = max(1, self.params.influence_increase + agent.params.influence)
          
          #move to exposed
          else:

            neighbor.state = SEIZMstates.EXPOSED

            #gain sentiment for the skeptics by a factor of the neighbor's influence 
            step = agent.params.influence * random.uniform(0,1)
            neighbor.params.sentiment = min(0, neighbor.params.sentiment - step)

            #increase certainty of agent by a factor of neighbor's influence 
            if step > random.uniform(0,1) * self.params.certainty_threshold:
              neighbor.params.certainty = max(neighbor.params.certainty + random.uniform(0,1) * step, 1)

            #increase influence of neighbor
            if step > random.uniform(0,1) * self.params.influence_threshold:
              agent.params.influence = max(1, agent.params.influence + random.uniform(0,1) * step)

        #CASE 3.3 - neighbor is skeptic
        if neighbor.state == SEIZMstates.SKEPTIC:
          
          #both skeptics reinforce their beliefs
          certainty_reinforcement = neighbor.params.certainty + agent.params.certainty + 1 / 3
          neighbor.params.certainty = certainty_reinforcement
          agent.params.certainty = certainty_reinforcement

        #CASE 3.4 - neighbor is infected
        if neighbor.state == SEIZMstates.INFECTED:

          certainty_condition_neighbor_wins = agent.params.certainty <= self.params.certainty_threshold and neighbor.params.certainty >= self.params.certainty_threshold
          certainty_condition_agent_wins = neighbor.params.certainty < self.params.certainty_threshold and agent.params.certainty > self.params.certainty_threshold
          influence_condition_neighbor_wins = agent.params.influence <= self.params.influence_threshold and neighbor.params.influence >= self.params.influence_threshold
          influence_condition_agent_wins = neighbor.params.influence < self.params.influence_threshold and agent.params.influence > self.params.influence_threshold

          #infected has a stronger influence 
          if influence_condition_neighbor_wins: 

            #infected has a higher certainty in his belief - skeptic transitions back to exposed 
            if certainty_condition_neighbor_wins:
              agent.state = SEIZMstates.EXPOSED
            #skeptic has a higher certainty in his belief - decrease certainty since infected has strong influence
            else:
              agent.params.certainty = min(0, agent.params.certainty - self.params.certainty_increase * neighbor.params.influence)

          #skeptic has a stronger influence
          if influence_condition_agent_wins:

            #infected has no monetary incentive towards his belief
            if neighbor.params.money == 0:

              #skeptic has higher certainty in his belief - skeptic transitions back to exposed 
              if certainty_condition_agent_wins:
                neighbor.state = SEIZMstates.EXPOSED
              #infected has a higher certainty in his belief - decrease certainty since infected has strong influence
              else:
                neighbor.params.certainty = min(0, neighbor.params.certainty - self.params.certainty_increase * agent.params.influence)
            
            #infected has monetary incentive towards his belief
            else:

              #skeptic has higher certainty in his belief + random factor
              if certainty_condition_agent_wins and random.uniform(0,1) < self.params.money_threshold:
                neighbor.state = SEIZMstates.EXPOSED
              #infected has a higher certainty in his belief - decrease certainty since infected has strong influence
              else:
                neighbor.params.certainty = min(0, neighbor.params.certainty - self.params.certainty_increase * agent.params.influence)


      #CASE 4 - agent is infected
      if agent.state == SEIZMstates.INFECTED:

        #CASE 4.1 - neighbor is exposed
        if neighbor.state == SEIZMstates.EXPOSED:
          
          money_condition = agent.params.money >= 0
          certainty_condition = agent.params.certainty > self.params.certainty_threshold
          influence_condition = agent.params.influence > self.params.influence_threshold
          random_condition = random.uniform(0,1) > self.params.money_threshold

          if money_condition and certainty_condition and influence_condition and random_condition:
            #ask to spend money 

            matching_sentiment = neighbor.params.sentiment > 0.5
            influence_condition = neighbor.params.influence < self.params.influence_threshold
            certainty_condition = neighbor.params.certainty > self.params.certainty_threshold

            #agrees if has a matching sentiment, a low influence (easily influenced), and a high certainty
            if matching_sentiment and influence_condition and certainty_condition:

              neighbor.params.money -= 1
              agent.params.money += 1
              neighbor.state = SEIZMstates.INFECTED

              #increase certainty by a factor of agent's certainty and influence
              convincing_factor = (agent.params.certainty - self.params.certainty_threshold) + (agent.params.influence - self.params.influence_threshold)
              neighbor.params.certainty = max(1, neighbor.params.certainty + self.params.certainty_increase * convincing_factor)
              
              #rate of influence increase depends on the neighbor's sentiment
              neighbor.params.influence = max(1, neighbor.params.influence + self.params.influence_increase * (1 - agent.params.sentiment))
            
            #disagree to spend money
            else:

              #decrease the influence of the infected more if neighbor's sentiment was already close to 1
              #should have been easier to convince them if they tended towards you 
              agent.params.influence = min(0, agent.params.influence - self.params.influence_increase * min(0, neighbor.params.sentiment - 0.5))
              
              #decrease the agent's sentiment towards skeptic since neighbor viewed as too bold
              neighbor.params.sentiment = min(0, neighbor.params.sentiment - agent.params.influence)

          #do not ask to spend money 
          else:

            #good sentiment towards infected
            if neighbor.params.sentiment > 0.5:
            
              certainty_condition = neighbor.params.certainty <= agent.params.certainty
              influence_condition = agent.params.influence > self.params.influence_threshold
              
              #infected is more certain and influent enough - move to infected
              if certainty_condition and influence_condition:

                neighbor.state = SEIZMstates.INFECTED

                #increase agent certainty the more the infected is certain
                neighbor.params.certainty = max(neighbor.params.certainty + agent.params.certainty * self.params.certainty_increase, 1)
                
                #increase represents how good he was at convinicing him 
                agent.params.influence = max(agent.params.influence + self.params.influence_increase * (1 - neighbor.params.sentiment), 1)
                
              #exposed is more certain or infected not convincing enough - move towards infected
              else:
            
                neighbor.params.sentiment = max(1, neighbor.params.sentiment + random.uniform(0,1) * neighbor.params.certainty)
            
            #bad sentiment towards infected
            else:

              #if infected is influent enough, he still tries
              if agent.params.influence > self.params.influence_threshold: 
                p = random.uniform(0,1)
                if (p > self.params.influence_threshold):
                  #suceeds
                  neighbor.state = SEIZMstates.INFECTED
                  agent.params.influence = max(1, agent.params.influence + self.params.influence_increase * (1 - neighbor.params.sentiment))
                else:
                  #fails - sees the counterpart as too bold
                  agent.params.influence = min(0, agent.params.influence - random.uniform(0,1) * neighbor.params.sentiment)
                  
                  neighbor.params.sentiment = min(0, neighbor.params.sentiment - random.uniform(0,1) * neighbor.params.certainty)

                  #higher increase if agent had high influence
                  neighbor.params.certainty = max(1, neighbor.params.certainty + agent.params.influence * random.uniform(0,1))

              else:
                #small step in random direction
                neighbor.params.sentiment = max(1, neighbor.params.sentiment + math.pow(-1, random.randint(0,1)) * random.uniform(0,1) * neighbor.params.certainty)

        #CASE 4.2 - neighbor is susceptible
        if neighbor.state == SEIZMstates.SUSCEPTIBLE:

          #directly transition to infected
          if bernoulli.rvs(self.params.prob_S_with_I):

            neighbor.state = SEIZMstates.INFECTED

            #increase agent influence 
            agent.params.influence = max(1, self.params.influence_increase + agent.params.influence)

          #transition to exposed 
          else:

            neighbor.state = SEIZMstates.EXPOSED

            #gain sentiment for the infected by a factor of the agent's influence 
            step = agent.params.influence * random.uniform(0,1)
            neighbor.params.sentiment += step

            #increase certainty of neighbor by a factor of agent's influence 
            if step > random.uniform(0,1) * self.params.certainty_threshold:
              neighbor.params.certainty = max(neighbor.params.certainty + random.uniform(0,1) * step, 1)

            #increase influence of agent
            if step > random.uniform(0,1) * self.params.influence_threshold:
              agent.params.influence = max(1, agent.params.influence + random.uniform(0,1) * step)

        #CASE 4.3 - neighbor is skeptic
        if neighbor.state == SEIZMstates.SKEPTIC:

          certainty_condition_neighbor_wins = agent.params.certainty <= self.params.certainty_threshold and neighbor.params.certainty >= self.params.certainty_threshold
          certainty_condition_agent_wins = neighbor.params.certainty < self.params.certainty_threshold and agent.params.certainty > self.params.certainty_threshold
          influence_condition_neighbor_wins = agent.params.influence <= self.params.influence_threshold and neighbor.params.influence >= self.params.influence_threshold
          influence_condition_agent_wins = neighbor.params.influence < self.params.influence_threshold and agent.params.influence > self.params.influence_threshold

          #infected has a stronger influence -----
          if influence_condition_agent_wins: 

            #infected has a higher certainty in his belief - skeptic transitions back to exposed 
            if certainty_condition_agent_wins:
              neighbor.state = SEIZMstates.EXPOSED
            #skeptic has a higher certainty in his belief - decrease certainty since infected has strong influence
            else:
              neighbor.params.certainty = min(0, neighbor.params.certainty - self.params.certainty_increase * agent.params.influence)

          #skeptic has a stronger influence
          if influence_condition_neighbor_wins:

            #infected has no monetary incentive towards his belief
            if agent.params.money == 0:

              #skeptic has higher certainty in his belief - skeptic transitions back to exposed 
              if certainty_condition_neighbor_wins:
                agent.state = SEIZMstates.EXPOSED
              #infected has a higher certainty in his belief - decrease certainty since infected has strong influence
              else:
                agent.params.certainty = min(0, agent.params.certainty - self.params.certainty_increase * neighbor.params.influence)
            
            #infected has monetary incentive towards his belief
            else:

              #skeptic has higher certainty in his belief + random factor
              if certainty_condition_neighbor_wins and random.uniform(0,1) < self.params.money_threshold:
                agent.state = SEIZMstates.EXPOSED
              #infected has a higher certainty in his belief - decrease certainty since infected has strong influence
              else:
                agent.params.certainty = min(0, agent.params.certainty - self.params.certainty_increase * neighbor.params.influence)

        #CASE 4.4 - neighbor is infected 
        if neighbor.state == SEIZMstates.INFECTED:

          positive_balance_agent = agent.params.money >= 0
          certainty_condition_agent = agent.params.certainty > self.params.certainty_threshold
          influence_condition_agent = agent.params.influence > self.params.influence_threshold
          random_param = random.uniform(0,1) > self.params.money_threshold

          if positive_balance_agent and certainty_condition_agent and influence_condition_agent and random_param:
            #aqgent asks neighbor to spend money 

            if neighbor.params.influence < self.params.influence_threshold and neighbor.params.money <= 0:
              #agree
              neighbor.params.money -= 1
              agent.params.money += 1
              #increase certainty by a factor of neighbor's certainty and influence 
              neighbor.params.certainty = max(1, neighbor.params.certainty + self.params.certainty_increase * ((agent.params.certainty - self.params.certainty_threshold) + (agent.params.influence - self.params.influence_threshold)))
              agent.params.influence = max(1, agent.params.influence + self.params.influence_increase * (1 - (agent.params.influence - neighbor.params.influence)))
            else:
              #disagree
              #decrease influence by a factor of neighbor's influence because should have been easier to convince them if they tended towards you 
              agent.params.influence = min(0, agent.params.influence - self.params.influence_increase * (min(self.params.influence_threshold - neighbor.params.influence, 0)))
              neighbor.params.influence = max(1, neighbor.params.influence + self.params.influence_increase * agent.params.influence)

          positive_balance = neighbor.params.money >= 0
          certainty_condition = neighbor.params.certainty > self.params.certainty_threshold
          influence_condition = neighbor.params.influence > self.params.influence_threshold
          random_param = random.uniform(0,1) > self.params.money_threshold

          if positive_balance and certainty_condition and influence_condition and random_param:
            #neighbor asks agent to spend money 

            if agent.params.influence < self.params.influence_threshold and agent.params.money <= 0:
              #agree
              agent.params.money -= 1
              neighbor.params.money += 1
              #increase certainty by a factor of neighbors certainty and influence
              agent.params.certainty = max(1, agent.params.certainty + self.params.certainty_increase * ((neighbor.params.certainty - self.params.certainty_threshold) + (neighbor.params.influence - self.params.influence_threshold)))
              neighbor.params.influence = max(1, neighbor.params.influence + self.params.influence_increase * (1 - (neighbor.params.influence - agent.params.influence)))
            else:
              #disagree
              neighbor.params.influence = min(0, neighbor.params.influence - self.params.influence_increase * (min(self.params.influence_threshold - agent.params.influence, 0)))
              agent.params.influence = max(1, agent.params.influence + self.params.influence_increase * neighbor.params.influence)


class Model:
  
  class ModelType(enum.IntEnum):
    SIR = 0
    SEIZ = 1
    SEIZplus = 2
    SEIZM = 3

  @staticmethod
  def model_by_type(model_type: ModelType) -> 'type(Model)':
    """
    Resolves model dynamics by network type

    :param model_type:
    :return: Model dynamics corresponding to model type
    """
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
