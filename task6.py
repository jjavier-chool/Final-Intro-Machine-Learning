"""
Intro to Machine Learning Final
Encompasses the solution to Task 6.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
from task1 import get_abstract_map
from task2 import GridEnvironment
from task3 import Agent
from task4 import sarsa
from task5 import q_learning
import numpy as np
import torch
import time
import sys

# Helper: Train
#TODO: tune episodes for each test
def train(env, method, epsilon, gamma, episodes=10000):
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
def accuracy_path(env, agent, max_steps=500):
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
  target_rows = [20, 40, 40, 40]
  target_cols = [20, 40, 40, 40]
  map_episodes = [10000, 10000, 10000, 10000]

  for i in range(1,5):
    grid = get_abstract_map(i, target_rows[i-1], target_cols[i-1])
    env = GridEnvironment(i, grid, target=(39, 39), reward_strategy="S2")

    for method in ["SARSA", "Q"]:
      agent, t = train(env, method, epsilon=0.5, gamma=0.5, map_episodes[i-1])

      acc, avg_len, longest = accuracy_path(env, agent)

      print(f"map{i} | {method} | time={t:.2f}s | episodes={map_episodes[i-1]} | acc={acc:.3f} | avg={avg_len:.2f} | longest={longest}")

# 2. Exploration Test
def exploration_test(grid):
  print("\n=== Exploration Test ===")

  env = GridEnvironment(4, grid, target=(39, 39), reward_strategy="S2")
  tune_episodes = [10000, 10000, 10000, 10000, 10000, 10000]
  i = 0

  for eps in [0, 0.5, 1]:
    for method in ["SARSA", "Q"]:
      agent, t = train(env, method, epsilon=eps, gamma=0.5, tune_episodes[i])

      acc, avg_len, longest = accuracy_path(env, agent)

      print(f"eps={eps} | {method} | time={t:.2f}s | episodes={tune_episodes[i]} | acc={acc:.3f} | avg={avg_len:.2f} | longest={longest}")
      i = i + 1

# 3. Discount Test
def discount_test(grid):
  print("\n=== Discount Test ===")

  env = GridEnvironment(4, grid, target=(39, 39), reward_strategy="S2")
  tune_episodes = [10000, 10000, 10000, 10000, 10000, 10000]
  i = 0

  for gamma in [0.1, 0.5, 1]:
    for method in ["SARSA", "Q"]:
      agent, t = train(env, method, epsilon=0.5, gamma=gamma, tune_episodes[i])

      acc, avg_len, longest = accuracy_path(env, agent)

      print(f"gamma={gamma} | {method} | time={t:.2f}s | episodes={tune_episodes[i]} | acc={acc:.3f} | avg={avg_len:.2f} | longest={longest}")
      i = i + 1

# 4. Reward Strategy Test
def reward_test(grid):
  print("\n=== Reward Strategy Test ===")

  # TODO: Find from previous 2 tests
  best_eps = 0.5
  best_gamma = 0.5
  tune_episodes = [10000, 10000, 10000, 10000]
  i = 0

  for strategy in ["S1", "S2"]:
    env = GridEnvironment(4, grid, target=(39, 39), reward_strategy=strategy)

    for method in ["SARSA", "Q"]:
      agent, t = train(env, method, epsilon=best_eps, gamma=best_gamma, tune_episodes[i])

      acc, avg_len, longest = accuracy_path(env, agent)

      print(f"{strategy} | {method} | time={t:.2f}s | episodes={tune_episodes[i]} | acc={acc:.3f} | avg={avg_len:.2f} | longest={longest}")
      i = i + 1

# Main
def main(todo: str = 'all'):
  torch.manual_seed(42)
  grid = get_abstract_map(4, 50, 50)

  if todo == '1' or todo == 'all':
    complexity_test()

  if todo == '2' or todo == 'all':
    exploration_test(grid)

  if todo == '3' or todo == 'all':
    discount_test(grid)

  if todo == '4' or todo == 'all':
    reward_test(grid)

  # Extra custom map test here if we have time


if __name__ == "__main__":
  main(*sys.argv[1:])
