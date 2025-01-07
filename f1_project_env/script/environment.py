import numpy as np
import random

class QualifyingEnvironment:
    def __init__(self, data):
        """
        Initialize the environment with cleaned qualifying data.
        """
        self.data = data
        self.state = None  # Current state (e.g., raceId + driver standings)
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        self.state = self._initialize_state()
        return self.state

    def _initialize_state(self):
        """
        Set up the initial state based on the data.
        """
        # Use raceId and drivers for the qualifying session
        race = random.choice(self.data["raceId"].unique())
        drivers = self.data[self.data["raceId"] == race]
        return {"raceId": race, "drivers": drivers, "actions": drivers["driverId"].tolist()}

    def step(self, action):
        """
        Apply the chosen action and update the state.
        """
        # Simulate the results of an action
        driver = action  # Assume action is driverId
        performance = self._simulate_performance(driver)

        # Update the state based on the action
        self.state["drivers"].loc[self.state["drivers"]["driverId"] == driver, "performance"] = performance
        reward = self._calculate_reward()

        done = len(self.state["drivers"]) == 0  # Check if all drivers are processed
        return self.state, reward, done

    def _simulate_performance(self, driver):
        """
        Simulate qualifying performance for the given driver.
        """
        # Simple placeholder logic for performance
        return random.uniform(0.9, 1.1)

    def _calculate_reward(self):
        """
        Calculate a reward based on the state.
        """
        # Placeholder: Reward better performances
        return 1.0

    def get_valid_actions(self):
        """
        Return valid actions for the current state.
        """
        return self.state["actions"]
