"""
Intro to Machine Learning Final
Encompasses the solution to Task 5.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
from tqdm import tqdm
import numpy as np

from task2 import GridEnvironment, Point
from task3 import Agent
from util import accuracy_path

# Q-Learning algorithm, adopted from the slides' pseudocode
def q_learning(
    env: GridEnvironment,
    agent: Agent,
    episodes: int=5000,
    alpha: float=0.1,
    gamma: float=0.9,
    start_state: Point=(0, 0),
    max_steps: int=1000
  ):
  MIN = 0.25 # Minimum accuracy before considering runs
  LO = -0.2 # How much drop to restart a run
  HI = 0.01 # How much rise to count towards a run
  ALPHA = 0.618 # How much new information to incorporate
  RUN = 5 # How long a run should be

  accs = [0.]
  ewma = 0
  run = 0
  below_min = True

  for ep in tqdm(range(episodes), desc="Q-Learning Training"):
    state = start_state

    for _ in range(max_steps):
      action = agent.choose_action(state)
      next_state, reward = env.step(state, action)

      # Same as SARSA except this section, want max
      q = agent.get_q(state, action)
      # max_a Q(S',a)
      q_next_max = np.max(agent.Q[next_state])
      new_q = q + alpha * (reward + gamma * q_next_max - q)
      agent.set_q(state, action, new_q)

      state = next_state

      if state == env.target:
        break
    
    if (ep + 1) % 100 == 0:
      acc, _, _ = accuracy_path(env, agent)
      diff = acc - ewma
      ewma = ewma*(1 - ALPHA) + acc*ALPHA
      accs.append(ewma)

      if below_min:
        if acc < MIN:
          continue
        below_min = False
      
      if diff > 0:
        if diff < HI:
          run += 1
          if run > RUN:
              break
        else:
          run = 0
      elif diff < LO:
        run = 0
    
  return accs