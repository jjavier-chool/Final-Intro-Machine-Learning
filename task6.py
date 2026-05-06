"""
Intro to Machine Learning Final
Encompasses the solution to Task 6.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
from typing import Literal
import numpy as np
import torch
import time
import sys

import matplotlib.pyplot as plt

from task1 import MAPS, Map
from task2 import GridEnvironment, Point
from task3 import Agent
from task4 import sarsa
from task5 import q_learning
from util import softmax

# Helper: Train
#TODO: tune episodes for each test
def train(env: GridEnvironment, method: str, epsilon: float, gamma: float, episodes: int=10000, policy="argmax"):
  agent = Agent(
    env.rows, env.cols,
    ["up", "down", "left", "right"],
    epsilon=epsilon,
    policy=policy
  )

  start = time.time()

  match method:
    case "SARSA": sarsa(env, agent, episodes=episodes, gamma=gamma)
    case "Q": q_learning(env, agent, episodes=episodes, gamma=gamma)
    case _:
      raise NotImplementedError(method)

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
  map_episodes = [10000]*len(MAPS)

  for i, _ in enumerate(MAPS, 1):
    grid = Map(i)
    env = GridEnvironment(i, grid, target=grid.target, reward_strategy="S2")

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

      plot_policy(f"test1/map{i}-{method}", agent, env)

# 2. Exploration Test
def exploration_test(grid: Map):
  print("\n=== Exploration Test ===")

  env = GridEnvironment(4, grid, target=grid.target, reward_strategy="S2")
  tune_episodes = [10000]*len(MAPS)
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

      plot_policy(f"test2/{i}-eps={eps:.1f}-{method}", agent, env)

# 3. Discount Test
def discount_test(grid: Map):
  print("\n=== Discount Test ===")

  env = GridEnvironment(4, grid, target=grid.target, reward_strategy="S2")
  tune_episodes = [10000]*len(MAPS)
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

      plot_policy(f"test3/{i}-gamma={gamma:.1f}-{method}", agent, env)

# 4. Reward Strategy Test
def reward_test(grid: Map):
  print("\n=== Reward Strategy Test ===")

  # TODO: Find from previous 2 tests
  best_eps = 0.5
  best_gamma = 0.5
  tune_episodes = [10000]*len(MAPS)
  i = 0

  for strategy in ["S1", "S2"]:
    env = GridEnvironment(4, grid, target=grid.target, reward_strategy=strategy)

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

      plot_policy(f"test4/{i}-{strategy}-{method}", agent, env)

# 5. (extra) Softmax vs Argmax
def policy_test(grid: Map):
  print("\n=== Policy Sampling Test ===")

  # TODO: Find from previous tests
  best_eps = 0.5
  best_gamma = 0.5
  tune_episodes = 10000
  i = 0

  for policy in ["argmax", "softmax"]:
    env = GridEnvironment(4, grid, target=grid.target, reward_strategy="S2")

    for method in ["SARSA", "Q"]:
      if policy == "softmax":
        schedule = [0.1, 0.5, 1, 1.5, 2]
      else:
        schedule = [best_eps]
      for T in schedule:
        agent, t = train(env, method, T, best_gamma, tune_episodes, policy=policy)

        acc, avg_len, longest = accuracy_path(env, agent)

        print(' | '.join([
          policy,
          method,
          f"{T=:.2f}",
          f"time={t:.2f}s",
          f"episodes={tune_episodes}",
          f"{acc=:.3f}",
          f"avg={avg_len:.2f}",
          f"{longest=!s}"
        ]))
        print()
        i = i + 1

        plot_policy(f"test5/{i}-{policy}-{method}-T={T:.1f}", agent, env)

def plot_policy(name: str, agent: Agent, env: GridEnvironment, plot: Literal['argmax', 'softmax']="softmax"):
  '''
  Plot an image of the map overlaid with arrows indicating the agent's policy.
  '''
  #return
  # Plot the map itself
  plt.imshow(env.map.data, cmap='gray')

  # Plot the best path, backtracking if it ever loops
  path: list[tuple[int, Point]] = []
  seen = set()
  pick = 0
  state = (0, 0)
  loops = 0
  pathlen = 0
  for pathlen in range(1000): # Escape hatch
    if state == env.map.target:
      break
    seen.add(state)
    
    r, c = state
    state, _ = env.step(state, agent.actions[np.argsort(-agent.Q[state])[pick]])
    
    if state in seen:
      loops += 1
      plt.scatter((c + state[1])/2, (r + state[0])/2, s=100, c='blue')
      while path:
        pick, state = path.pop()
        pick += 1
        if pick < len(agent.actions):
          break
      else:
        # Just give up
        pathlen = 999
        break
    else:
      plt.plot([c, state[1]], [r, state[0]] , c='blue')
      path.append((0, state))
      pick = 0
  
  # For every position, convert the action to its delta to plot an arrow
  action_delta = np.array([(-1, 0), (1, 0), (0, -1), (0,1)])
  
  match plot:
    case "argmax":
      # Argmax plot, more boring
      flow = np.eye(4)[agent.Q.argmax(axis=-1)] @ action_delta
    case "softmax":
      # Softmax plot gives a better sense of what direction a state "prefers"
      flow = softmax(agent.Q) @ action_delta
      norm = np.linalg.norm(flow, axis=-1, keepdims=True)
      out = np.where(
        env.map[..., np.newaxis] == 0,
        np.full_like(flow, np.nan),
        np.zeros_like(flow)
      )
      # Avoid a warning with normal division
      np.divide(flow, norm, out=out, where=norm != 0)
      flow = out
    case _:
      raise NotImplementedError(plot)
  
  dy = flow[..., 0]
  dx = flow[..., 1]
  h, w = env.map.shape
  y, x = np.mgrid[0:h, 0:w]
  plt.quiver(x, y, dx, -dy, color='red', scale=1.41, units='xy')

  # Plot the start and goal
  plt.scatter([0, 39], [0, 39])

  if pathlen < 999:
    plt.text(0, -2, f"path={pathlen}")
    plt.text(0, -1, f"{loops=}")
  else:
    plt.text(0, -2, "path=∞")
    plt.text(0, -1, f"loops=∞")

  plt.axis("equal")
  plt.savefig(f"output/{name}.png")
  #plt.show()

# Main
def main(todo: str = 'all'):
  torch.manual_seed(42)
  grid = Map(4)

  if todo == 'all':
    todo = '1,2,3,4,5'
  
  for task in todo.split(','):
    match task:
      case '1': complexity_test()
      case '2': exploration_test(grid)
      case '3': discount_test(grid)
      case '4': reward_test(grid)
      case '5': policy_test(grid)
  
  # Extra custom map test here if we have time

if __name__ == "__main__":
  main(*sys.argv[1:])
