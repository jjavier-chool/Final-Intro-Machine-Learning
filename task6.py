"""
Intro to Machine Learning Final
Encompasses the solution to Task 6.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
from task1 import get_abstract_map
from task2 import GridEnvironment, Point
from task3 import Agent
from task4 import sarsa
from task5 import q_learning
import numpy as np
import torch
import time
import sys

import matplotlib.pyplot as plt

# Helper: Train
#TODO: tune episodes for each test
def train(env: GridEnvironment, method: str, epsilon: float, gamma: float, episodes: int=10000):
  agent = Agent(env.rows, env.cols, ["up", "down", "left", "right"], epsilon=epsilon)

  start = time.time()

  if method == "SARSA":
    sarsa(env, agent, episodes=episodes, gamma=gamma)
  else:
    q_learning(env, agent, episodes=episodes, gamma=gamma)

  elapsed = time.time() - start
  return agent, elapsed

# Helper: accuracy + extra path metrics
#TODO: length kinda pointless? remove?
def accuracy_path(env: GridEnvironment, agent: Agent, max_steps: int=500):
  rows, cols = env.rows, env.cols

  valid = 0
  total = 0
  path_lengths = []

  for r in range(rows):
    for c in range(cols):
      if env.map[r, c] == 0:
        continue

      state = (r, c)
      total += 1
      # Need this in order to prevent loops right?
      visited = set()
      steps = 0
      success = False

      while steps < max_steps:
        if state == env.target:
          success = True
          break

        if state in visited:
          break  # Loop

        visited.add(state)

        # Pretty sure it's correct to do the best action here
        action = agent.best_action(state)
        next_state, _ = env.step(state, action)

        if next_state == state:
          break  # Invalid to just stand in place

        state = next_state
        steps += 1

      if success:
        valid += 1
        path_lengths.append(steps)

  accuracy = valid / total if total else 0
  avg_len = np.mean(path_lengths) if path_lengths else 0
  longest = np.max(path_lengths) if path_lengths else 0

  return accuracy, avg_len, longest

# 1. Complexity Test
def complexity_test():
  print("\n=== Complexity Test ===")

  # TODO: all need to be 40, but first is 20x20. Add enlargening to task1.py
  target_rows = [40, 40, 40, 40]
  target_cols = [40, 40, 40, 40]
  map_episodes = [10000, 10000, 10000, 10000]

  for i in range(1,5):
    grid = get_abstract_map(i, target_rows[i-1], target_cols[i-1])
    '''
    plt.imshow(grid, cmap='gray')
    plt.scatter(20, 20)
    plt.scatter(0, 0)
    plt.show()
    '''
    env = GridEnvironment(i, grid, target=(39, 39), reward_strategy="S2")

    for method in ["SARSA", "Q"]:
      agent, t = train(env, method, epsilon=0.5, gamma=0.5, episodes=map_episodes[i-1])

      acc, avg, longest = accuracy_path(env, agent)

      print(" | ".join([
        f"map {i}",
        method,
        f"{t=:.2f}s",
        f"episodes={map_episodes[i-1]}",
        f"{acc=:.3f}",
        f"{avg=:.2f}",
        f"{longest=!s}"
      ]))
      print()

      plot_policy(agent, env)

# 2. Exploration Test
def exploration_test(grid: np.ndarray):
  print("\n=== Exploration Test ===")

  env = GridEnvironment(4, grid, target=(39, 39), reward_strategy="S2")
  tune_episodes = [10000, 10000, 10000, 10000, 10000, 10000]
  i = 0

  for eps in [0, 0.5, 1]:
    for method in ["SARSA", "Q"]:
      agent, t = train(env, method, epsilon=eps, gamma=0.5, episodes=tune_episodes[i])

      acc, avg, longest = accuracy_path(env, agent)

      print(" | ".join([
        f"{eps=}",
        method,
        f"{t=:.2f}s",
        f"episodes={tune_episodes[i]}",
        f"{acc=:.3f}",
        f"{avg=:.2f}",
        f"{longest=!s}"
      ]))
      print()
      i = i + 1

      plot_policy(agent, env)

# 3. Discount Test
def discount_test(grid: np.ndarray):
  print("\n=== Discount Test ===")

  env = GridEnvironment(4, grid, target=(39, 39), reward_strategy="S2")
  tune_episodes = [10000, 10000, 10000, 10000, 10000, 10000]
  i = 0

  for gamma in [0.1, 0.5, 1]:
    for method in ["SARSA", "Q"]:
      agent, t = train(env, method, epsilon=0.5, gamma=gamma, episodes=tune_episodes[i])

      acc, avg, longest = accuracy_path(env, agent)

      print(" | ".join([
        f"{gamma=}",
        method,
        f"{t=:.2f}s",
        f"episodes={tune_episodes[i]}",
        f"{acc=:.3f}",
        f"{avg=:.2f}",
        f"{longest=!s}"
      ]))
      print()
      i = i + 1

      plot_policy(agent, env)

# 4. Reward Strategy Test
def reward_test(grid: np.ndarray):
  print("\n=== Reward Strategy Test ===")

  # TODO: Find from previous 2 tests
  best_eps = 0.5
  best_gamma = 0.5
  tune_episodes = [10000, 10000, 10000, 10000]
  i = 0

  for strategy in ["S1", "S2"]:
    env = GridEnvironment(4, grid, target=(39, 39), reward_strategy=strategy)

    for method in ["SARSA", "Q"]:
      agent, t = train(env, method, best_eps, best_gamma, tune_episodes[i])

      acc, avg_len, longest = accuracy_path(env, agent)

      print(' | '.join([
        strategy,
        method,
        f"time={t:.2f}s",
        f"episodes={tune_episodes[i]}",
        f"{acc=:.3f}",
        f"avg={avg_len:.2f}",
        f"{longest=!s}"
      ]))
      print()
      i = i + 1

      plot_policy(agent, env)

def softmax(x: np.ndarray):
  a = np.exp(x - x.max(axis=-1, keepdims=True))
  return a/a.sum(axis=-1, keepdims=True)

def plot_policy(agent: Agent, env: GridEnvironment):
  '''
  Plot an image of the map overlaid with arrows indicating the agent's policy.
  '''
  # Plot the map itself
  plt.imshow(env.map, cmap='gray')

  # For every position, convert the action to its delta to plot an arrow
  h, w = env.map.shape
  action_delta = np.array([(-1, 0), (1, 0), (0, -1), (0,1)])
  flow = softmax(agent.Q) @ action_delta
  #flow = np.eye(4)[agent.Q.argmax(axis=-1)] @ action_delta
  dy = flow[..., 0]
  dx = flow[..., 1]
  y, x = np.mgrid[0:h, 0:w]
  plt.quiver(x, y, dx, -dy, color='red')

  # Plot the best path, stopping if it ever loops
  seen = set()
  state = (0, 0)
  while state != (39, 39):
    seen.add(state)
    
    r, c = state
    state, _ = env.step(state, agent.best_action(state, random=False)
    
    if state in seen:
      plt.scatter((c + state[1])/2, (r + state[0])/2, s=100, c='red')
      break
    else:
      plt.plot([c, state[1]], [r, state[0]] , c='blue')

  # Plot the start and goal
  plt.scatter([0, 39], [0, 39])

  plt.axis("equal")
  plt.show()

# Main
def main(todo: str = 'all'):
  torch.manual_seed(42)
  grid = get_abstract_map(4, 50, 50)

  if todo == 'all':
    todo = '1,2,3,4'
  
  for task in todo.split(','):
    match task:
      case '1': complexity_test()
      case '2': exploration_test(grid)
      case '3': discount_test(grid)
      case '4': reward_test(grid)
  
  # Extra custom map test here if we have time

if __name__ == "__main__":
  main(*sys.argv[1:])
