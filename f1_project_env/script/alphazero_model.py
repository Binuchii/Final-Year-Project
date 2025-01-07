import random
import numpy as np


class AlphaZeroModel:
    def __init__(self):
        pass

    def predict(self, state):
        """
        Predict policy and value for the given state.
        """
        # Random policy and value
        policy = np.random.dirichlet(np.ones(len(state["actions"])))
        value = random.uniform(-1, 1)
        return policy, value
