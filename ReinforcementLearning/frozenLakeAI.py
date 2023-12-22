import gym
import random
from collections import namedtuple
import collections
import numpy as np
import matplotlib.pyplot as plt

#Define Epsilon, Total number of possible actions
epsilon = 1.0
n_actions = 64

#ε-greedy policy (add a randomness ε for the choice of the action)
def selectEpsilonAction(table, obs, n_actions):
    value, action = bestActionValue(table, obs)

    if random.random() < epsilon:
        return random.randint(0, n_actions-1)
    else:
        return action

#greedy policy (take the best action according to the policy)
def selectGreedyAction(table, obs, n_actions):
    value, action = bestActionValue(table, obs)
    return action

#Explore the table, To find the best action that maximises Q(s,a)
# Corrected bestActionValue function
def bestActionValue(table, state):
    bestAction = 0
    maxValue = 0
    state_key = tuple(state.items())  # Convert state dictionary to a tuple
    for action in range(n_actions):
        if table[(state_key, action)] > maxValue:
            bestAction = action
            maxValue = table[(state_key, action)]

    return maxValue, bestAction


#Define Gamma and Learning Rate
GAMMA = 0.95
LEARNING_RATE = 0.8


#To Update Q(obs0,action) according to Q(obs1,*) and the reward obtained.
# Corrected QLearning function
def QLearning(table, obs0, obs1, reward, action):
    bestValue, _ = bestActionValue(table, obs1)
    state_key = tuple(obs0.items())  # Convert obs0 dictionary to a tuple
    QTarget = reward + GAMMA * bestValue
    QError = QTarget - table[(state_key, action)]
    table[(state_key, action)] += LEARNING_RATE * QError



#Define Test Epidosodes
TEST_EPISODES = 100
#Test Game Loop
def testGame(env, table):
    n_actions = env.actionSpace.n
    rewardGames = []
    for _ in range(TEST_EPISODES):
        obs = env.reset()
        rewards = 0
        while True:
            #Act Greedly
            next_obs, reward, done, _ = env.step(selectGreedyAction(table, obs, n_actions))
            obs = next_obs
            rewards += reward
            if done:
                rewardGames.append(rewards)
                break
    return np.mean(rewardGames)


#Define Max Games and Epsilon Decay Rate
MAX_GAMES = 15000
EPS_DECAY_RATE = 0.9993

#Create Frozen Lake Terrain
env = gym.make("FrozenLake-v1")
obs =  env.reset()

obsLength = env.observation_space.n
n_actions =  env.action_space.n

rewardCount = 0
gamesCount = 0

#Initialize the table with Nil values
table = collections.defaultdict(float)
testRewardList = []

#Reinitialize epsilon after each session
epsilon = 1.0

while gamesCount < MAX_GAMES:

    #Select the action followign an ε-greedy policy
    action = selectEpsilonAction(table, obs, n_actions)
    next_obs, reward, done, _ = env.step(action)

    #Update the Q-Table
    QLearning(table, obs, next_obs, reward, action) 

    rewardCount += reward
    obs = next_obs

    if done:
        epsilon *= EPS_DECAY_RATE

        #Test the new Q-Table every 1000 Games
        if (gamesCount +1) % 1000 == 0:
            testReward = testGame(env, table)
            print('\tGame Count:', gamesCount, "\tTest Reward:", testReward, "Epsilon Value:", np,round(epsilon,2))

            testRewardList.append(testReward)
        
        obs = env.reset()
        rewardCount = 0
        gamesCount +=1

#Plot the Accuracy over the Number of Steps
plt.figure(figsize=(18,9))
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.plot(testRewardList)
plt.show()        

