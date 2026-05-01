"""
Intro to Machine Learning Final
Encompasses the solution to Task 3.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
import numpy as np
import random

# Agent generalized for both SARSA and Q-Learning
class Agent:
  # Initialize with dimensions for building the Q-table + epsilon
  def __init__(self, rows, cols, actions, epsilon=0.1):
    self.rows = rows
    self.cols = cols
    self.actions = actions
    self.epsilon = epsilon

    self.num_actions = len(actions)

    # 3D Q-table: Q[row][col][action]
    self.Q = np.zeros((rows, cols, self.num_actions))

    self.action_to_index = {a: i for i, a in enumerate(actions)}

  # Exploration vs exploitation choice
  def choose_action(self, state):
    if random.random() < self.epsilon:
      return random.choice(self.actions)

    r, c = state
    return self.actions[np.argmax(self.Q[r, c])]

  # Used in task6.py when finding accuracy
  def best_action(self, state):
    r, c = state
    return self.actions[np.argmax(self.Q[r, c])]

  # Getter
  def get_q(self, state, action):
    r, c = state
    return self.Q[r, c, self.action_to_index[action]]

  # Setter
  def set_q(self, state, action, value):
    r, c = state
    self.Q[r, c, self.action_to_index[action]] = value
