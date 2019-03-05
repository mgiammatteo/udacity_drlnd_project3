# Udacity DRL nanodegree - Project 3: Collaboration and Competition
In this project, we train two agents in parallel to control rackets to bounce a ball over a net for a game of Tennis.
The environment is provided by a Unity machine learning agent called Tennis. More information on the Unity ml-agents can be found [here](https://github.com/Unity-Technologies/ml-agents).

## Project Details
The two agents interface to an environment which is characterised as follows:

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Repository structure
The code is structured as follows: 
* **Tennis.ipynb**: this is where the deep rl agents are concurrently trained. Both agents are structured according to the DDPG actor critic algorithm, details of which can be found [here](https://arxiv.org/pdf/1509.02971.pdf).
* **ddpg_agent.py**: this module implements a class to represent a vanilla ddpg agent.
* **model.py**: this module contains the implementation of the actor and critic neural networks. The actor is the policy approximator which provides the critic with the best action vector to take at each time step. The critic is the action value function approximator and its aim it to approximate the optimal action value function.
* **checkpoint_actor_1.pth**: this is the binary containing the first agent's trained actor neural network weights.
* * **checkpoint_actor_2.pth**: this is the binary containing the second agent's trained actor neural network weights.
* **checkpoint_critic_1.pth**: this is the binary containing the first agent's critic neural network weights.
* * **checkpoint_critic_2.pth**: this is the binary containing the second agent's trained critic neural network weights.
* **Tennis_Windows_x86_64**: this directory contains the binary for the Tennis environment utilised in this project. It's for running on Windows 10, 64-bit.

### Dependencies
* python 3.6
* numpy: install with 'pip install numpy'.
* PyTorch: install by following the instructions [here](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installing-Unity-ml-agents-on-Windows).
* ml-agents: install by following instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md).

## Getting Started
In cell 1 of Tennis.ipynb we import the binary for the Unity environment 'Tennis.exe'. For a local installation of the Unity ml-agents, please refer to the following two sources:
* [Linux, Mac](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
* [Windows 10](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md)

## Instructions
This is a jupyter notebook project. To run the code and train the deep reinforcement learning agent, you simply execute each of the cells in **Tennis.ipynb**. After training, the average score per hundred episodes will be displayed and a plot of the score per episode will also be shown.