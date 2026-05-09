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
from util import softmax, accuracy_path

# Helper: Train
#TODO: tune episodes for each test
def train(name: str, env: GridEnvironment, method: str, epsilon: float, gamma: float, episodes: int=10000, policy="argmax"):
  agent = Agent(
    env.rows, env.cols,
    ["up", "down", "left", "right"],
    epsilon=epsilon,
    policy=policy
  )

  start = time.time()

  match method:
    case "SARSA": acc = sarsa(env, agent, episodes=episodes, gamma=gamma)
    case "Q": acc = q_learning(env, agent, episodes=episodes, gamma=gamma)
    case _:
      raise NotImplementedError(method)

  elapsed = time.time() - start
  plt.clf()
  plt.plot(acc)
  #plt.show()
  plt.savefig(f"output/{name}-accuracy.png")
  return agent, elapsed, acc

# 1. Complexity Test
def complexity_test():
  print("\n=== 1. Complexity Test ===")
  map_episodes = [10000]*len(MAPS)
  i = 0
  strategy = "S2"
  for m, _ in enumerate(MAPS, 1):
    grid = Map(m)
    env = GridEnvironment(m, grid, target=grid.target, reward_strategy=strategy)

    for method in ["SARSA", "Q"]:
      name = f"test1/map{m}-{method}-{strategy}"
      agent, t, accs = train(name, env, method, epsilon=0.5, gamma=0.5, episodes=map_episodes[m-1])

      acc, avg_len, longest = accuracy_path(env, agent)

      print(" | ".join([
        f"map {m}",
        f"{method:^5}",
        strategy,
        f"{t=:.2f}s",
        f"{acc=:.3f}",
        f"episodes={(len(accs) - 1)*100}",
        f"{avg_len=:.1f}"
      ]))
      print()

      i += 1
      plot_policy(name, agent, env)

# 2. Exploration Test
def exploration_test(grid: Map):
  print("\n=== 2. Exploration Test ===")

  strategy = "S2"
  env = GridEnvironment(4, grid, target=grid.target, reward_strategy=strategy)
  tune_episodes = [10000]*len(MAPS)
  i = 0

  for method in ["SARSA", "Q"]:
    for eps in [0, 0.5, 1]:
      name = f"test2/{method}-{strategy}-eps={eps:.1f}"
      agent, t, accs = train(name, env, method, epsilon=eps, gamma=0.5, episodes=tune_episodes[i])

      acc, avg_len, longest = accuracy_path(env, agent)

      print(" | ".join([
        f"{eps=}",
        f"{method:^5}",
        strategy,
        f"{t=:.2f}s",
        f"{acc=:.3f}",
        f"episodes={(len(accs) - 1)*100}",
        f"{avg_len=:.1f}"
      ]))
      print()
      i = i + 1

      plot_policy(name, agent, env)

# 3. Discount Test
def discount_test(grid: Map):
  print("\n=== 3. Discount Test ===")

  strategy = "S2"
  env = GridEnvironment(4, grid, target=grid.target, reward_strategy=strategy)
  tune_episodes = [10000]*len(MAPS)
  i = 0

  for method in ["SARSA", "Q"]:
    for gamma in [0.1, 0.5, 1]:
      name = f"test3/{method}-{strategy}-gamma={gamma:.1f}"
      agent, t, accs = train(name, env, method, epsilon=0.5, gamma=gamma, episodes=tune_episodes[i])

      acc, avg_len, longest = accuracy_path(env, agent)

      print(" | ".join([
        f"{gamma=}",
        f"{method:^5}",
        strategy,
        f"{t=:.2f}s",
        f"{acc=:.3f}",
        f"episodes={(len(accs) - 1)*100}",
        f"{avg_len=:.1f}"
      ]))
      print()
      i = i + 1

      plot_policy(name, agent, env)

# 4. Reward Strategy Test
def reward_test(grid: Map):
  print("\n=== 4. Reward Strategy Test ===")

  tune_episodes = [10000]*len(MAPS)
  i = 0

  for strategy in ["S1", "S2"]:
    env = GridEnvironment(4, grid, target=grid.target, reward_strategy=strategy)

    for method in ["SARSA", "Q"]:
      # From previous 2 tests
      if method == "SARSA":
        best_eps = 0.5
        best_gamma = 1
      else:
        best_eps = 1
        best_gamma = 0.5

      name = f"test4/{strategy}-{method}"
      agent, t, accs = train(name, env, method, best_eps, best_gamma, tune_episodes[i])

      acc, avg_len, longest = accuracy_path(env, agent)

      print(' | '.join([
        strategy,
        f"{method:^5}",
        f"time={t:.2f}s",
        f"{acc=:.3f}",
        f"episodes={(len(accs) - 1)*100}",
        f"{avg_len=:.1f}"
      ]))
      print()
      i = i + 1

      plot_policy(name, agent, env)

# 5. (extra) Softmax vs Argmax
def policy_test(grid: Map):
  print("\n=== 5. Policy Sampling Test ===")

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
        name = f"test5/{policy}-{method}-T={T:.1f}"
        agent, t, accs = train(name, env, method, T, best_gamma, tune_episodes, policy=policy)

        acc, avg_len, longest = accuracy_path(env, agent)

        print(' | '.join([
          f"{policy:^5}",
          method,
          f"{T=:.2f}",
          f"time={t:.2f}s",
          f"{acc=:.3f}",
          f"episodes={(len(accs) - 1)*100}",
          f"{avg_len=:.1f}"
        ]))
        print()
        i = i + 1

        plot_policy(name, agent, env)

def plot_policy(name: str, agent: Agent, env: GridEnvironment, plot: Literal['argmax', 'softmax']="softmax"):
  '''
  Plot an image of the map overlaid with arrows indicating the agent's policy.
  '''
  #return
  plt.clf()
  # Plot the map itself
  plt.imshow(env.map.data, cmap='gray', zorder=0)

  # Plot the start and goal
  gy, gx = env.target
  plt.scatter([0, gx], [0, gy], zorder=1)

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
      plt.plot([c, state[1]], [r, state[0]] , c='blue', zorder=2)
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
  plt.quiver(x, y, dx, -dy, color='red', scale=2, scale_units='xy', units='xy', zorder=3)

  if pathlen < 999:
    plt.text(0, -3, f"path={pathlen}")
    plt.text(0, -1, f"{loops=}")
  else:
    plt.text(0, -3, "path=∞")
    plt.text(0, -1, f"loops=∞")

  plt.axis("equal")
  plt.savefig(f"output/{name}.png", dpi=300)
  #plt.show()

# Main
def main(todo: str = 'all'):
  torch.manual_seed(42)
  np.random.seed(42)
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
