import gym
import numpy as np
import random

def main():

    # Create Taxi environment
    env = gym.make('Taxi-v3')

    # Initialize Q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # Hyperparameters
    learning_rate = 0.9
    discount_rate = 0.9
    epsilon=1.0
    decay_rate = 0.005

    # Training variables
    num_episodes = 1000
    max_steps = 99  # per episode

    # Training
    for episode in range(num_episodes):

        # Reset the environment
        state = env.reset()
        done = False

        for s in range(max_steps):

            # Exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit
                action = np.argmax(qtable[state, :])

            # Take action and observe reward
            new_state, reward, done, info = env.step(action)

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update to our new state
            state = new_state

            # If done, finish episode
            if done:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # Watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):
        print("TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)
        rewards += reward
        env.render()  # Commented out for more efficient training
        print(f"Score: {rewards}")
        state = new_state

        if done==True:
            break

    env.close()

if __name__ == "__main__":
    main()
