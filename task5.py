"""
Intro to Machine Learning Final
Encompasses the solution to Task 5.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
from tqdm import tqdm
import numpy as np

# Q-Learning algorithm, adopted from the slides' pseudocode
def q_learning(env, agent, episodes=5000, alpha=0.1, gamma=0.9, start_state=(0, 0), max_steps=1000):
  for ep in tqdm(range(episodes), desc="Q-Learning Training"):
    state = start_state

    for _ in range(max_steps):
      action = agent.choose_action(state)
      next_state, reward = env.step(state, action)

      # Same as SARSA except this section, want max
      q = agent.get_q(state, action)
      r, c = next_state
      # max_a Q(S',a)
      q_next_max = np.max(agent.Q[r, c])
      new_q = q + alpha * (reward + gamma * q_next_max - q)
      agent.set_q(state, action, new_q)

      state = next_state

      if state == env.target:
        break
