"""
Intro to Machine Learning Final
Encompasses the solution to Task 3.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
import numpy as np
from task2 import Action, Point
from util import softmax

# Agent generalized for both SARSA and Q-Learning
class Agent:
  # Initialize with dimensions for building the Q-table + epsilon
  def __init__(self, rows: int, cols: int, actions: list[Action], epsilon: float=0.1, policy="argmax"):
    self.rows = rows
    self.cols = cols
    self.actions = actions
    self.epsilon = epsilon
    self.policy = policy

    self.num_actions = len(actions)

    # 3D Q-table: Q[row][col][action]
    self.Q = np.zeros((rows, cols, self.num_actions))

    self.action_to_index = {a: i for i, a in enumerate(actions)}

  # Exploration vs exploitation choice for the next action
  def choose_action(self, state: Point):
    match self.policy:
      case "argmax":
        if np.random.random() < self.epsilon:
          return np.random.choice(self.actions)
        return self.best_action(state)
      # For the extra policy test in task6.py
      case "softmax":
        P = softmax(self.Q[state]/self.epsilon)
        return self.actions[np.argmax(np.random.multinomial(1, P))]

      case _:
        raise NotImplementedError(self.policy)

  # Used in task6.py when finding accuracy
  def best_action(self, state: Point, random: bool=True):
    # Can't just use argmax because we need to break ties randomly. This is
    # equivalent to argmax without ties.
    if random:
      A = self.Q[state]
      return self.actions[np.random.choice(np.where(A == A.max())[0])]
    else:
      return self.actions[self.Q[state].argmax()]

  # Getter
  def get_q(self, state: Point, action: Action):
    return self.Q[*state, self.action_to_index[action]]

  # Setter
  def set_q(self, state: Point, action: Action, value: int):
    self.Q[*state, self.action_to_index[action]] = value
