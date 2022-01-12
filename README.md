# Snake and Deep Reinforcement Learning

This project contains the following files:
1. **snake_env.py**
   
   This is the environment that represents the snake game. If you run `python snake_env.py`, you'll be able to play snake manually with the arrow keys.

____
2. **agent_1.py**
   This script trains an agent via a deep Q network to play the game of snake, learning over time. Running `python agent_1.py` causes the agent to train over a specified number of episodes, playing the game autonomously.

____
3. **plotting.py**
   This is a supporting script that graphs some statistics about how the network was trained and the performance of the agent.

____
4. **requirements.txt**
   
   After cloning this repository, run `pip install -r requirements.txt` to set up your environment. This project runs on **Python 3.8.x**, and some of the main modules utilized in this demo are `tensorflow`, `keras`, `turtle`, and `gym`.
____
## To do:
- Allow several user-definable agents via config.json files.
- Clean up plotting.py and add the ability to save figures locally.
- Expand this readme to include a high-level overview of how the network is trained via a Bellman equation.
- Allow for a model to be trained over N episodes and then saved.
- Allow for previously trained models to play without additional training.
- Expand plotting.py to store a collection of training episodes as a gif.
