import random
from collections import deque
from typing import Any, Dict, List

import numpy as np
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from src.environment import Snake

"""
In deep reinforcement learning, we need to create two things:
    - an environment (the snake game universe, including the snake itself)
    - an agent (the algorithm which pilots the snake in the environment)

Our environment is the `Snake` class and our agent is the neural network, `DQN`.

A `Sequential` deep learning model is appropriate for a plain stack of neural network
layers where each layer has exactly one input tensor and one output tensor.

Deques are a generalization of stacks and queues (the name is pronounced "deck" and is
short for "double-ended queue"). Deques support thread-safe, memory-efficient appends
and pops from either side of the deque with approximately the same O(1) performance in
either direction.

A dense layer, also known as a fully connected layer, is a layer in an artificial
neural network where each neuron in the current layer is connected to every neuron in
the previous layer. This complete linkage is what gives the layer its name, and it's
the most commonly used type of layer in artificial neural network networks.

Adam (short for Adaptive Moment Estimation) optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second order moments. Whereas momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction, which thus prefers flat minima in the error surface.
"""


class DQN:
    """A deep-Q neural network that can train itself to play Snake."""

    def __init__(self, env: Snake, params: Dict[str, Any]) -> None:
        """Instantiate a deep-Q network meant to learn the game of Snake.

        Args:
            env (Snake): The `gym.Env` subclass representing the game of Snake.
            params (Dict[str, Any]): Dictionary of relevant hyperparameters.
        """
        # There are only four possible directions to move in, so the action space is 4D.
        self.action_space = env.action_space
        self.state_space = (
            env.state_space
        )  # The dimension of the state space (e.g. 12 binary elements is 12D).
        self.epsilon = params["epsilon"]
        self.gamma = params["gamma"]
        self.batch_size = params["batch_size"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.learning_rate = params["learning_rate"]
        self.layer_sizes = params["layer_sizes"]
        # The deque acts as a rolling window of the state of the game over time.
        self.memory = deque(maxlen=2500)
        self.model = self.build_model()

    def build_model(self) -> Sequential:
        """Build a neural network of fully connected dense layers.

        This model consists of one input layer, `len(self.layer_sizes - 1)` hidden
        layers, and one output layer.

        Returns:
            Sequential: An ordered stack of neural network layers compiled into a model.
        """
        model = Sequential()
        for i, layer_size in enumerate(self.layer_sizes):
            if i == 0:  # The number of input nodes is the dimension of the state space.
                model.add(
                    Dense(
                        layer_size, input_shape=(self.state_space,), activation="relu"
                    )
                )
            else:
                # Recall Rectified Linear Unit, ReLU(x), returns x if x > 0, else 0.
                model.add(Dense(layer_size, activation="relu"))
        # The number of nodes in the output layer is the dimension of the action space.
        model.add(Dense(self.action_space, activation="softmax"))
        # Softmax is used for multi-class classification. For example, maybe the snake
        # needs to travel in more than one direction to get the apple.
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(
        self,
        state: List[int],
        action: int,
        reward: int,
        next_state: List[int],
        done: bool,
    ) -> None:
        """Store an experience tuple in the agent's memory buffer.

        The experience tuple is (state, action, reward, next_state, done). This method
        appends the experience tuple to the agent's memory buffer for later use in
        learning.

        Args:
            state (List[int]): The current state of the environment.
            action (int): The action taken in the current state.
            reward (int): The reward received after taking the action.
            next_state (List[int]): The state of the environment after taking the
                action.
            done (bool): A boolean indicating if the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: List[int]):
        """Move a randomly or in the best policy-predicted direction.

        `self.epsilon` is the explore threshold parameter. If under the threshold, move
        in a random direction, otherwise move in the direction which maximizes the
        probability of a larger total reward.

        Args:
            state (List[int]): The state of the environment.

        Returns:
            int: The direction to move for the Snake's next step.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state, verbose=0)
        # e.g. [[0.08789534, 0.8699538 , 0.03103394, 0.01111698]]
        #           0:up      1:down      2:left      3:right
        return np.argmax(act_values[0])

    def replay(self) -> None:
        """Retrain and update the DQN based on its working memory of past experiences.

        The method operates in five steps:
            1. Sample from working memory.
            2. Process the batch.
            3. Compute the targets.
            4. Adjust the model.
            5. Lessen random exploration.

        """
        # Collect more samples if we don't have enough for a training batch.
        if len(self.memory) < self.batch_size:
            return

        # Get a batch-sized random sample from the working memory buffer.
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([memory[0] for memory in minibatch])
        actions = np.array([memory[1] for memory in minibatch])
        rewards = np.array([memory[2] for memory in minibatch])
        next_states = np.array([memory[3] for memory in minibatch])
        dones = np.array([memory[4] for memory in minibatch])
        # Convert the state vectors from 1x12 matrices to 12-element arrays.
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # The core of this algorithm is a Bellman equation as a simple value iteration
        # update, using the weighted average of the old value and the new information.
        targets = rewards + self.gamma * (
            np.amax(self.model.predict_on_batch(next_states), axis=1)
        ) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.arange(self.batch_size)
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)  # Update the model.
        if self.epsilon > self.epsilon_min:  # Lessen the random exploration.
            self.epsilon *= self.epsilon_decay


def train_dqn(env: Snake, params: Dict[str, Any]) -> List[int]:
    """Train a Deep Q-Network (DQN) agent on a given environment.

    This function runs a series of episodes where the DQN agent interacts with the
    environment, collects experiences, and updates its policy based on those
    experiences. It records the total reward for each episode and returns a list of
    rewards for all episodes.

    Args:
        env (Snake): The environment on which the agent will be trained. It must
            implement methods like `reset` and `step`.
        params (Dict[str, Any]): A dictionary of parameters for training, including:
            - "num_episodes" (int): The number of episodes to train the agent.
            - "max_steps" (int): The maximum number of steps per episode.

    Returns:
        List[int]: A list of total rewards for each episode.

    Example:
        >>> env = Snake()
        >>> params = {"num_episodes": 100, "max_steps": 200}
        >>> history = train_dqn(env, params)
    """
    history = []
    agent = DQN(env, params)
    for episode_num in range(params["num_episodes"]):
        state = env.reset()
        # Convert the initial state to a 1x12 matrix.
        state = np.reshape(state, (1, env.state_space))
        total_reward = 0
        for step_num in range(params["max_steps"]):
            action = agent.act(state)
            prev_state = state
            # The step method allows the agent to move the snake.
            next_state, reward, done, _ = env.step(action, episode_num, step_num)
            total_reward += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # [TO DEV] Include online gradient descent (i.e. batch_size=1).
            if agent.batch_size > 1:
                agent.replay()
            if done:
                prefix = f"{str(prev_state)} {total_reward:<5}"
                suffix = f'({episode_num+1:>3}/{params["num_episodes"]:<3})'
                print(f"{prefix} {suffix}")
                break
        history.append(total_reward)
    return history
