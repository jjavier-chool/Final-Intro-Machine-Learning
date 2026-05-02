"""
Intro to Machine Learning Final
Encompasses the solution to Task 3.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
import numpy as np
import random

from task2 import Action, Point

# Agent generalized for both SARSA and Q-Learning
class Agent:
  # Initialize with dimensions for building the Q-table + epsilon
  def __init__(self, rows: int, cols: int, actions: list[Action], epsilon: float=0.1):
    self.rows = rows
    self.cols = cols
    self.actions = actions
    self.epsilon = epsilon

    self.num_actions = len(actions)

    # 3D Q-table: Q[row][col][action]
    self.Q = np.zeros((rows, cols, self.num_actions))

    self.action_to_index = {a: i for i, a in enumerate(actions)}

  # Exploration vs exploitation choice
  def choose_action(self, state: Point):
    if random.random() < self.epsilon:
      return random.choice(self.actions)

    return self.best_action(state)

  # Used in task6.py when finding accuracy
  def best_action(self, state: Point, random: bool=True):
    # Can't just use argmax because we need to break ties randomly. This is
    # equivalent to argmax without ties.
    if random:
      A = self.Q[state]
      a = np.random.choice(np.where(A == A.max())[0])
      return self.actions[a]
    else:
      return self.actions[self.Q[state].argmax()]
    
  # Getter
  def get_q(self, state: Point, action: Action):
    return self.Q[*state, self.action_to_index[action]]

  # Setter
  def set_q(self, state: Point, action: Action, value: int):
    self.Q[*state, self.action_to_index[action]] = value
