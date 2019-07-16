# Import Dependencies
import gym
import numpy as np
import matplotlib.pyplot as plt

# Define Environement
env = gym.make("MountainCar-v0")
# Reset Environment
env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

# Observation Space Size; 20 discrete values
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# Discrete Observation Space Window Size
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Create Q-Table
# Q-Table contains all the combinations corresponding to actions 0, 1 and 2.
# The number of combinations depend on the DISCRETE_OS_SIZE
# Each time the car moves, the agent comes to this table, looks at it's new state value and fetches the action to take.
# The action is defined by the action corresponding to the largest state value
# The agent takes this action and keeps updating the table over time
# Initially, these values are random
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Rewards
ep_rewards = []
# Aggregate Rewards
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


# Learning Rate
lr = 0.1

# Discount: Measure of how important are the future actions over current actions
discount = 0.95

# Episodes
episodes = 25000

# Show Every
show_every = 2000

# Epsilon: Higher the value, more the exploration
epsilon = 0.5
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(episodes):
    episode_reward = 0
    if episode % show_every == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())

    # Flag
    done = False

    # Actions in Environment
    # Action 0: Push Car Left
    # Action 1: Do Nothing
    # Action 2: Push Car Right
    while not done:
        if np.random.random() > epsilon:
            # Define action based on q_table discrete_state
            action = np.argmax(q_table[discrete_state])
        else:
            # Define action based on q_table discrete_state
            action = np.random.randint(0, env.action_space.n)
        # We get Position & Velocity of this car in environment
        new_state, reward, done, _ = env.step(action=action)
        episode_reward += reward
        # Get New Discrete State
        new_discrete_state = get_discrete_state(new_state)
        if render:
            # Show the Environment
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q Value
            current_q = q_table[discrete_state + (action, )]
            # New Q Value Calculation Formula
            new_q = (1 - lr) * current_q + lr * (reward + discount * max_future_q)
            # Update Q Table with New Q Value
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print("We made it on episode {}", episode)
            # Reward
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % show_every:
        avg_reward = sum(ep_rewards[-show_every:])/len(ep_rewards[-show_every:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-show_every:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-show_every:]))
        print(f"Episode: {episode} avg: {avg_reward} min: {min(ep_rewards[-show_every:])} max: {max(ep_rewards[-show_every:])}")

# Close the Environment
env.close()

plt.plot( aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot( aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot( aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()