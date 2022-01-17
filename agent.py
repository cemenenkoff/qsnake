import random
import numpy as np
'''
In deep reinforcement learning, we need to create two things:
    - an environment (the snake game universe)
    - an agent (the algorithm which pilots the snake the environment)
Our environment is the Snake class and our agent is the neural network.
'''
from keras import Sequential
'''
A Sequential deep learning model is appropriate for a plain stack of layers
where each layer has exactly one input tensor and one output tensor.
'''
from collections import deque
'''
Deques are a generalization of stacks and queues (the name is pronounced "deck"
and is short for "double-ended queue"). Deques support thread-safe, memory
efficient appends and pops from either side of the deque with approximately the
same O(1) performance in either direction.
'''
from keras.layers import Dense
'''
In any neural network, a dense layer is a layer that is deeply connected with
its preceding layer which means the neurons of the layer are connected to every
neuron of its preceding layer. This layer is the most commonly used layer in
artificial neural network networks.
'''
from tensorflow.keras.optimizers import Adam
'''
Adam optimization is a stochastic gradient descent method that is based on
adaptive estimation of first-order and second order moments. Whereas momentum
can be seen as a ball running down a slope, Adam behaves like a heavy ball with
friction, which thus prefers flat minima in the error surface.
'''

class DQN:
    '''
    a deep-Q neural network that can train itself to play snake
    '''
    def __init__(self, env, params):

        self.action_space = env.action_space # the dimension of the action space (4 here because the snake's only options are up, down, left, right)
        self.state_space = env.state_space # the dimension of the state space (e.g. 12 binary elements)
        self.epsilon = params['epsilon'] # the initial ratio of steps taken to randomly explore vs move in a predicted direction
        self.gamma = params['gamma'] # the discount factor for future rewards (0 is short-sighted; 1 is long-sighted)
        '''
        An important note on batch_size:
        The number of training examples used in the estimate of the error gradient
        is a hyperparameter for the learning algorithm called the "batch size" or simply the "batch".

        The error gradient is a statistical estimate. The more training examples used in the estimate,
        the more accurate this estimate will be and the more likely that the weights of the network
        will be adjusted in a way that will improve the performance of the model.

        The improved estimate of the error gradient comes at the cost of having to use the model
        to make many more predictions before the estimate can be calculated, and in turn, the weights updated.
        '''
        self.batch_size = params['batch_size']
        self.epsilon_min = params['epsilon_min'] # the minimum ratio of time steps we'd like the agent to move randomly vs in a predicted direction
        self.epsilon_decay = params['epsilon_decay'] # how much of the ratio of random moving we want to take into the next iteration of gathering a batch of states
        self.learning_rate = params['learning_rate'] # to what extent newly acquired info overrides old info (0 learn nothing and exploit prior knowledge exclusively; 1 only consider the most recent information)
        self.layer_sizes = params['layer_sizes'] # the number of nodes for the hidden layers of our Q network
        self.memory = deque(maxlen=2500) # our defined working memory array of the state of the agent and the environment over time
        self.model = self.build_model()

    def build_model(self):
        '''
        builds a neural network of dense layers consisting of an input layer, 3 hidden layers, and an output layer
        '''
        model = Sequential()
        for i, layer_size in enumerate(self.layer_sizes):
            if i == 0: # The input layer's shape (i.e. number of nodes) is defined by the dimension of the state space.
                model.add(Dense(layer_size, input_shape=(self.state_space,), activation='relu'))
            else: # The three hidden layers will have an integer number of nodes.
                model.add(Dense(layer_size, activation='relu'))
                # Recall that the Rectified Linear Unit (ReLU) activation
                # function that outputs the input directly if the input is
                # positive, otherwise it outputs zero.
        # The number of nodes in the output layer is the dimension of the action space.
        model.add(Dense(self.action_space, activation='softmax'))
        # The Softmax function is good here because it's best applied to
        # multi-class classification problems where class membership is required
        # on more than two class labels. In this instance, maybe the snake needs
        # to travel in 3 or more directions to get to the apple.
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        '''
        adds the current state, next state, proposed action, total reward, and whether we are done in
        the agent's running memory buffer (deque) of states
        '''
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        '''
        moves in a random direction or the direction predicted to give the best reward outcome
        '''
        if np.random.rand() <= self.epsilon: # If we are under the explore threshold parameter,
            return random.randrange(self.action_space) # move in a random direction.
        # Otherwise, move in the direction which maximizes the probability of a large reward.
        act_values = self.model.predict(state) # e.g. [[0.08789534, 0.8699538 , 0.03103394, 0.01111698]]
                                               #           0:up      1:down      2:left      3:right
        return np.argmax(act_values[0])

    '''
    A note on batch size from Deep Learning by Ian Goodfellow:

    "Optimization algorithms that use the entire training set are called batch
    or deterministic gradient methods, because they process all of the training
    examples simultaneously in a large batch."

    "Optimization algorithms that use only a single example at a time are
    sometimes called stochastic or sometimes online methods. The term online is
    usually reserved for the case where the examples are drawn from a stream of
    continually created examples rather than from a fixed-size training set over
    which several passes are made."

    A note on batch sizing from Jason Brownlee:

    "The number of training examples used in the estimate of the error gradient
    is a hyperparameter for the learning algorithm called the 'batch size,' or
    simply the 'batch.'

    A batch size of 32 means that 32 samples from the training dataset will be
    used to estimate the error gradient before the model weights are updated.
    One training epoch means that the learning algorithm has made one pass
    through the training dataset, where examples were separated into randomly
    selected 'batch size' groups.

    Historically, a training algorithm where the batch size is set to the total
    number of training examples is called 'batch gradient descent' and a
    training algorithm where the batch size is set to 1 training example is
    called 'stochastic gradient descent' or 'online gradient descent.'"

    Batch Gradient Descent:
        batch_size = total number of examples in the training dataset
    Stochastic Gradient Descent:
        batch_size = 1
    Minibatch Gradient Descent:
        1 < batch_size < total number of examples in the training dataset

    Put another way, the batch size defines the number of samples that must be
    propagated through the network before the weights can be updated.
    '''
    def replay(self):
        if len(self.memory) < self.batch_size: # If we haven't conducted enough samples for a training batch,
            return # go collect more samples.

        # If we have enough samples for a learning batch...
        minibatch = random.sample(self.memory, self.batch_size) # Get a batch_size'd random sample from the working memory buffer.
        states = np.array([memory[0] for memory in minibatch])
        actions = np.array([memory[1] for memory in minibatch])
        rewards = np.array([memory[2] for memory in minibatch])
        next_states = np.array([memory[3] for memory in minibatch])
        dones = np.array([memory[4] for memory in minibatch])
        states = np.squeeze(states) # Convert the state vectors from 1x12 matrices to 12-element arrays.
        next_states = np.squeeze(next_states)

        # The core of this algorithm is a Bellman equation as a simple value iteration update,
        # using the weighted average of the old value and the new information.
        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.arange(self.batch_size)
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min: # If our random exploration parameter is greater than the minimum
            self.epsilon *= self.epsilon_decay # attenuate it just a bit.

def train_dqn(env, params):
    history = []
    agent = DQN(env, params)
    for episode_num in range(params['num_episodes']):
        state = env.reset()
        state = np.reshape(state, (1, env.state_space)) # Convert the initial state to a 1x12 matrix.
        total_reward = 0
        max_steps = 10000
        for step_num in range(max_steps):
            action = agent.act(state)
            prev_state = state
            next_state, reward, done, info = env.step(action, episode_num, step_num) # This is where the agent moves.
            total_reward += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if agent.batch_size > 1: # This agent doesn't perform online/stochastic gradient descent (i.e. batch_size=1) for now.
                agent.replay()
            if done:
                print(f'{str(prev_state)} {total_reward:<5} ({episode_num+1:>3}/{params["num_episodes"]:<3})')
                break
        history.append(total_reward)
    return history
