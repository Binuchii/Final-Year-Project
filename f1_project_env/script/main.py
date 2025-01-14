from script.mcts import MCTS


def run_alpha_zero(env, model, n_iterations=10):
    """
    Train AlphaZero with MCTS for qualifying predictions.
    """
    for iteration in range(n_iterations):
        state = env.reset()
        done = False

        while not done:
            policy, value = model.predict(state)
            action = MCTS(env).search(state)
            state, reward, done = env.step(action)

        print(f"Iteration {iteration + 1}: Final reward = {reward}")
