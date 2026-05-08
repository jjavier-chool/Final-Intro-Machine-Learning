"""
Intro to Machine Learning Final
Encompasses the solution to Task 4.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
from tqdm import tqdm

from task2 import GridEnvironment, Point
from task3 import Agent
from util import accuracy_path

# SARSA algorithm, adopted from the slides' pseudocode
def sarsa(
    env: GridEnvironment,
    agent: Agent,
    episodes: int=5000,
    alpha: float=0.1,
    gamma: float=0.9,
    start_state: Point=(0, 0),
    max_steps: int=1000
  ):
  accs = [0]

  # "Loop for each episode"
  for ep in tqdm(range(episodes), desc="SARSA Training"):
    # Init S, choose A from S using policy derived from Q
    state = start_state
    action = agent.choose_action(state)
    # Loop for each step of episode
    for _ in range(max_steps):
      # Take action A, observe R, S'
      next_state, reward = env.step(state, action)
      # Choose A' from S' using policy derived from Q
      next_action = agent.choose_action(next_state)

      # Q(S,A)
      q = agent.get_q(state, action)
      # Q(S',A')
      q_next = agent.get_q(next_state, next_action)
      # Q(S,A)<-Q(S,A)+ alpha[R+gammaQ(S',A')-Q(S,A)]
      new_q = q + alpha * (reward + gamma * q_next - q)
      agent.set_q(state, action, new_q)

      # S<-S', A<-A'
      state, action = next_state, next_action

      # Until S is terminal
      if state == env.target:
        break
    
    if (ep + 1) % 100 == 0:
      acc, _, _ = accuracy_path(env, agent)
      diff = acc - accs[-1]
      accs.append(acc)
      if 0 < diff < 0.01:
        break
  
  return accs