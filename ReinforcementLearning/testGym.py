import gym

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Reset the environment to the initial state
observation = env.reset()

# Run the environment for a few steps
for step in range(200):
    # Render the current state of the environment
    env.render()

    # Perform the action and print the returned value
    step_result = env.step(action)
    print(step_result)

    # Unpack the values
    observation, reward, done, info = step_result

    # If the episode is done, reset the environment
    if done:
        observation = env.reset()

# Close the environment
env.close()

		