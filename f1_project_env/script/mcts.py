import random


class MCTS:
    def __init__(self, environment, n_simulations=10):
        """
        Initialize MCTS with the environment.
        """
        self.env = environment
        self.n_simulations = n_simulations

    def simulate(self, state):
        """
        Run a single simulation from the given state.
        """
        env_copy = self._copy_env()
        current_state = state
        total_reward = 0
        done = False

        while not done:
            valid_actions = env_copy.get_valid_actions()
            action = random.choice(valid_actions)  # Random action (replace with policy later)
            _, reward, done = env_copy.step(action)
            total_reward += reward

        return total_reward

    def search(self, state):
        """
        Perform MCTS for the given state.
        """
        action_rewards = {}
        valid_actions = self.env.get_valid_actions()

        for action in valid_actions:
            total_reward = 0
            for _ in range(self.n_simulations):
                total_reward += self.simulate(state)
            action_rewards[action] = total_reward / self.n_simulations

        # Choose the action with the highest average reward
        best_action = max(action_rewards, key=action_rewards.get)
        return best_action

    def _copy_env(self):
        """
        Create a copy of the environment for simulation.
        """
        import copy
        return copy.deepcopy(self.env)
