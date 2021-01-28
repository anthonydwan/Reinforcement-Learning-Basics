import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")

"""
mountain car is game env where you try to train the car
(agent) to get over the hill to get to the destination 
(goal).


mountain car has 3 actions you can take:
action 0: push car left
action 1: do nothing
action 2: push car right
"""

# environment provide the space but it is possible
# that env does not show and the agent may need to
# explore.
print(env.observation_space.high)
print(env.observation_space.low)

# how many actions we can take (3)
print(env.action_space.n)

LEARNING_RATE = 0.1
# time discount
DISCOUNT = 0.95
EPISODES = 10_000

VERBOSE = 500

# randomness for exploratory action
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# hardcoding for now, since every env may have
# different dimensions for obs space
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# separate into 20 discrete numbers
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# table of reward per ep
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


# here we reduce the state numbers into discrete numbers
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0

    if episode % VERBOSE == 0:
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    # print(discrete_state)

    # starting q values,
    # q_table[discrete_state]

    # argmax to get the action with max value
    # np.argmax(q_table[discrete_state])

    done = False
    while not done:
        # action should be the one with highest q value
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # new_state is the game environment (position and velocity)
        # agent does not really need to know what the values mean
        # reward is in continuous variable
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        # rendering the game
        if render:
            env.render()

        # updating when the game is not done, ie goal not reached
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            '''
            Q-Learning Formula:
            
            Q_new(s_t, a_t) := (1- alpha)* old_value + LR * learned value
                            := (1- alpha)* old_value + LR * (reward + discount_factor * estimate_optimal_future_val)
                            := (1- alpha)* Q(s_t, a_t)+ alpha(r_t + gamma* max(a) Q(s_(t+1), a) ) 
            '''

            # it is only when we get a future_q that is positive when it would backprop
            # to the correct actions

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # based on the new_state and the reward, we update what action
            # we should have taken
            q_table[discrete_state + (action,)] = new_q

        # when the car reaches the destination
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            # reward for completing the game (non-neg which is max value)
            q_table[discrete_state + (action,)] = 0

        # update
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


    ep_rewards.append(episode_reward)

    # for every episode
    if not episode % VERBOSE:
        average_reward = sum(ep_rewards[-VERBOSE:])/len(ep_rewards[-VERBOSE:])

        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-VERBOSE:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-VERBOSE:]))


        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-VERBOSE:])} max: {max(ep_rewards[-VERBOSE:])}")



    # closing the environment afterwards

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
plt.legend(loc = 4)
plt.show()