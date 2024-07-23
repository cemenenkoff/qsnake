![banner](/img/readme/qsnake-banner.png)
# [Deep Q Learning](https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm) for Top-Down 2D Games
- In the game of [Snake](https://en.wikipedia.org/wiki/Snake_(video_game_genre)), each time the snake eats an apple, it grows by one chunk.
- The game ends if the snake head hits a wall or its own body.

<p align="center">
  <img src="/img/readme/training-montage.gif"/>
</p>

![learning-curve](/img/readme/learning-curve.png)

[TOC]

# 1. General Overview
- [**Reinforcement learning**](https://en.wikipedia.org/wiki/Reinforcement_learning) is a style of machine learning that relies on past experience to develop a **policy** on what do to next.
- [**Deep neural networks**](https://en.wikipedia.org/wiki/Deep_learning) are used in machine learning to set up decision pathways that maximize reward (i.e. **minimize loss** or error).
- [**Q-learning**](https://en.wikipedia.org/wiki/Q-learning) is a subclass of reinforcement learning that is **model-free**, meaning that it does not require a model of the environment in which it operates and can learn directly from raw experiences.

## 1.1 Winning the Game
- In this exploration, a **policy** is a strategy to win the game of snake, and we want to find the optimal one.
- The **agent** executes the policy.
- The policy begins as *"move randomly and hope for the best"*.
- As the agent gains experience, the policy becomes less random and moves closer to an optimal solution.

***The more the agent plays, the more its random movements are replaced by policy-predicted movements.***

# 2. Setup
If you are fairly new to Python programming, I'd recommend the following steps:

1. Download and install [VS Code](https://code.visualstudio.com/download).

2. Install [Python 3.12.4](https://www.python.org/downloads/release/python-3124/) (☑️ **Add python.exe to PATH** if you have no other Python versions installed).

3. Install [Git bash](https://git-scm.com/downloads).

4. Open VS Code.

5. Press `F1`, and in the command palette, search for `Terminal: Select Default Profile` and set Git bash as the default terminal.

6. Start a new terminal with `Ctrl` + `` ` ``.

7. Clone this repository to a directory where you like to store your coding projects.

8. Open this repository (i.e. the `qsnake` folder) as the current workspace folder with `Ctrl` + `K` `Ctrl` + `O`.

9.  Make sure the terminal path points to the `qsnake` folder, and if it doesn't, navigate there via `cd <path_to_qsnake_folder>`. You can confirm you're in the right spot with quick `ls -la` command.

10. From the terminal, run `pip install virtualenv` to install the `virtualenv` module.

11. Run `python -m virtualenv <myenvname> --python=python3.12.4` to create a virtual environment that runs on Python 3.12.4.

12. Activate the virtual environment with `source <myenvname>/bin/activate`.

13. You should see `(<myenvname>)` two lines above the terminal input line when the environment is active.

14. Press `F1` to open VS Code's command palette, then search for `Python: Select Interpreter` and select `Python 3.12.4 64-bit ('<myenvname>':venv)`.

15. Run `pip install -r requirements.txt` to install all dependencies on your activated virtual environment.

16. Next, ensure the `"human"` option in `configs/config.json` is set to `true`.

17. Once everything is installed and configured, run `python src/q_snake.py` to test if you can play the game manually.

18. Next, specify `"human": false` in `config.json`, and then run  `python src/q_snake.py` again, this time to see if the *agent* is able to play the game.

19. Let the agent run to the end and check that `plotting.py` is able to produce a graph of the learning curve.

20. Play with a few settings in `config.json` and re-run `python src/q_snake.py` to see how the changes affect the agent's behavior. Feel free to do this until you get bored or it sparks a question you want to explore.

21. *OPTIONAL* - If you want to utilize the GIF-making supporting scripts, install [Ghostscript](https://ghostscript.com/releases/index.html) and note the path of the binary. You will need to change a line in the preamble of **gif_creator.py** to specify where the binary is located.

22. *OPTIONAL* - If you were bold enough to install Ghostscript, try saving game frames as EPS files. You can then run `python src/make_gif_from_images.py` to convert the EPS files into PNG files and then into an animated GIF.

23. *OPTIONAL* - Try converting the saved EPS files into PNG and then into a GIF all at once by specifying both `"save_for_gif": true` and `"make_gif": true` in `config.json`. Please note that this process can take 30 to 60 minutes for 50 training episodes.

# 3. Core Files
## 3.1 `q_snake.py`
Run `python src/q_snake.py` to interface with this project.

## 3.2 `config.json`
Define how the script should run. All of the keys in the default configuration included below are required along with their example data types.
```json
{
    "human": true,
    "name": "default",
    "save_for_gif": false,
    "make_gif": false,
    "params":{
        "epsilon": 1.0,
        "gamma": 0.95,
        "batch_size": 256,
        "epsilon_min": 0.001,
        "epsilon_decay": 0.98,
        "learning_rate": 0.00025,
        "layer_sizes": [128, 128, 128],
        "num_episodes": 20,
        "max_steps": 3000,
        "state_definition_type": "default"
    }
}
```
### 3.2.1 Config Core Definitions
#### 3.2.1.1 `human`
Whether to run so a human can play with the arrow keys, or to let the agent play by itself.
#### 3.2.1.2 `name`
The name of the configuration.
#### 3.2.1.3 `save_for_gif`
Whether to save EPS files of the agent as it trains in preparation for making a training montage GIF.
#### 3.2.1.4 `make_gif`
Whether to make a training montage GIF at the end of the run.

### 3.2.2 Config Param Definitions
#### 3.2.2.1 `epsilon`
Initial ratio of random versus policy-predicted steps.
#### 3.2.2.2 `gamma`
Discount factor for future rewards (0 is short-sighted, 1 is long-sighted).
#### 3.2.2.3 `batch_size`
The number of training samples processed before the model's internal parameters are updated during one iteration of training.
#### 3.2.2.4 `epsilon_min`
The minimum ratio of time steps we'd like the agent to move randomly versus in a policy-predicted direction.
#### 3.2.2.5 `epsilon_decay`
How much randomness we want to take into the next iteration of gathering a batch of states. For example, if we move randomly 50% of the time for the current batch and `epsilon_decay` is set to 0.95, we will move randomly 47% of the time on the next batch.
#### 3.2.2.6 `learning_rate`
To what extent newly acquired info overrides old info (0 learn nothing and exploit prior knowledge exclusively; 1 only consider the most recent information).
#### 3.2.2.7 `layer_sizes`
The number of nodes for the hidden layers of our deeply-connected Q network.
#### 3.2.2.8 `num_episodes`
The number of games to play.
#### 3.2.2.9 `max_steps`
The maximum number of steps allowable in a single game.

### 3.2.3 An Important Note on `batch_size`
- The number of training examples used in the *estimate* of the **error gradient** is a *hyperparameter* for the learning algorithm called the **"batch size"** (or simply **"the batch"**).
- A **hyperparameter** is a variable set before learning begins.
- Note that the error gradient is a *statistical* estimate.
- The more training examples used in the estimate
  1. The more accurate the estimate will be.
  2. The more likely the network will adjust such that the model's overall performance improves.
- **An improved estimate of the error gradient comes at a cost**.
  - It requires the model to make many more predictions before the estimate can be calculated and the weights updated.

#### 3.2.3.1 Notes Batch Size from Deep Learning by Ian Goodfellow:
>Optimization algorithms that use the entire training set are called batch or deterministic gradient methods, because they process all of the training examples simultaneously in a large batch.
>
>Optimization algorithms that use only a single example at a time are sometimes called stochastic or sometimes online methods. The term online is usually reserved for the case where the examples are drawn from a stream of continually created examples rather than from a fixed-size training set over which several passes are made.

#### 3.2.3.2 Notes Batch Sizing from Jason Brownlee:
> A batch size of 32 means that 32 samples from the training dataset will be used to estimate the error gradient before the model weights are updated. One training epoch means that the learning algorithm has made one pass through the training dataset, where examples were separated into randomly selected "batch size" groups.
>
> Historically, a training algorithm where the batch size is set to the total number of training examples is called "batch gradient descent" and a training algorithm where the batch size is set to 1 training example is called "stochastic gradient descent" or "online gradient descent".
>
> Put another way, the batch size defines the number of samples that must be propagated through the network before the weights can be updated.

#### 3.2.3.3 Summary of Batch Sizing Styles
| Gradient Descent Type           | Batch Size                            |
|---------------------------------|---------------------------------------|
| **Batch Gradient Descent**      | all training samples                  |
| **Stochastic Gradient Descent** | 1                                     |
| **Minibatch Gradient Descent**  | 1 < batch size < all training samples |

## 3.3 `requirements.txt`
The modules required for running the project. After creating a virtual environment with [Python 3.12.4](https://www.python.org/downloads/release/python-3124/), run `pip install -r requirements.txt` to install all necessary dependencies.

## 3.4 `environment.py`
This subclass of `gymnasium.Env` represents the Snake game environment (which includes the snake itself).

## 3.5 `agent.py`
Train an agent to play Snake via a deep Q-learning network.

## 3.6 `plotting.py`
Graph some statistics about how the Q-network was trained and the performance of the agent.

## 3.7 `gif_builder.py`
Convert saved EPS files into PNG files and then PNG files into an animated GIF. Using this class requires a separate installation of [Ghostscript](https://ghostscript.com/releases/index.html), and `EpsImagePlugin.gs_windows_binary = <path_to_binary>` must be appropriately edited within `gif_builder.py`.

## 3.8 `make_gif_from_images.py`
Convert already-saved EPS or PNG image files into a GIF without having to re-run the game. Again, [Ghostscript](https://ghostscript.com/releases/index.html) is required.

# 4. Q-Learning Overview
## 4.1 What is the [Bellman Equation](https://en.wikipedia.org/wiki/Bellman_equation)?
The Bellman equation is a fundamental recursive formula in reinforcement learning that expresses the value (i.e. Q-value) of a **state** in terms of the expected reward and the values of subsequent states. It has an abstract general definition, but in Q-learning, it's more easily understood when rearranged into the [**Q-value update rule**](https://en.wikipedia.org/wiki/Q-learning#Algorithm).

![banner](/img/readme/q-update-rule.png)

- $Q(s_t,a_t)$ is the Q-value of taking action $a$ in state $s$ at time $t$.
- $\alpha$ is the learning rate, which determines the extent to which new experience overrides past experience.
- $r_t$ is the immediate reward received after taking action $a$ in state $s$ at time $t$.
- $\gamma$ is the discount factor, which balances the importance of immediate versus future rewards.
- $s_{t+1}$ is the next state resulting from taking action $a$ in state $s$ at time $t$.
- $\underset{a}{\max}\,Q(s_{t+1},a)$ is the maximum Q-value for the next state $s_{t+1}$ over all possible actions $a$.

An episode of the algorithm ends when state $s_{t+1}$ is a **final** state (referred to as a `done` state in `agent.py`). For all final states $s_f$, $Q(s_f,a)$ is never updated, but is instead set to the reward value $r$ observed for state $s_f$.

### 4.1.1 Adjustable Hyperparameters
#### 4.1.1.1 The Learning Rate
   - $\alpha$ determines to what extent newly acquired information overrides old information.
     - $\alpha\in(0,1]$
     - As $\alpha$ approaches 0, the agent more exclusively uses prior knowledge.
     - As $\alpha$ approaches 1, the agent more exclusively considers the most recent data.
     - In deterministic environments, a value of 1 is optimal.
     - In environments with some randomness, $\alpha$ needs to eventually decrease to zero (meaning an optimal strategy is found).
       - In practice, however, using a constant learning rate works just fine.
#### 4.1.1.2 The Discount Factor
   - $\gamma$ determines the importance of future rewards.
      - $\gamma\in(0,1]$
      - As $\gamma$ approaches 0, the agent further prioritizes short-term rewards.
     - As $\gamma$ approaches 1, the agent more exclusively acts to achieve long-term rewards.
      - Starting with a small discount factor and increasing it toward a final value over time tends to accelerate learning.

## 4.2 Applying the Bellman Equation to Snake
In the Snake game, the agent (the brain controlling the snake) navigates the game grid, eats food to grow, and avoids collisions with itself and the walls.
### 4.2.1 General Definitions in Terms of Snake
- State $s$
  - A configuration of the game (i.e. the position of the snake, the walls, and the food).
- Action $a$
  - Possible moves (i.e. up, down, left, or right).
- Reward $r$
  - The reward received for taking action $a$ in state $s$.
    | **Reward Type** | **Description**                              |
    |-----------------|----------------------------------------------|
    | $+$             | Reward for eating food.                       |
    | $-$             | Penalty for colliding with walls or itself.   |
    | $0$             | No reward for moving without eating food.     |

### 4.2.2 Bellman Snake Algorithm
1. ***Initialize DQN***
   - Start with an empty deeply-connected neural net where all Q-values are zero.
2. ***Observe Current State***
   - Record the current position of the snake and the apple on the grid.
3. ***Choose Action***
   - Choose to act based more on past experience or for random exploration (based on $\epsilon$, another hyperparameter set in `config.json` under `epsilon_decay`).
4. ***Execute Action and Observe Reward***
   - Take the action $a$, move the snake, then receive the immediate reward $r$.
5. ***Observe Next State***
   - Record the new position of the snake and the apple on the grid.
6. ***Update Q-Value***
   - Apply the Bellman equation to update the Q-value for the state-action pair $(s, a)$
7. ***Iterate***
   - Repeat the process, updating the weights in the DQN as the agent continues to play.

# 5. Feature Roadmap
Here are some ideas for future development.
- Enable model saving functionality.
- Allow previously trained models to play with saved settings.
- Allow previously trained models to play and resume training.

# 6. References
Here are the key references I used to develop this project.
1. [Create a Snake-Game using Turtle in Python](https://www.geeksforgeeks.org/create-a-snake-game-using-turtle-in-python/)
2. [Q-learning](https://en.wikipedia.org/wiki/Q-learning)
3. [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation)
4. [Gymnasium's `Env` Class](https://gymnasium.farama.org/api/env/)
5. [Gymnasium's Classic Control Examples](https://gymnasium.farama.org/environments/classic_control/)
6. [Snake Played by a Deep Reinforcement Learning Agent](https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36)
7. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
8. [Create and Save Animated GIF with Python – Pillow](https://www.geeksforgeeks.org/create-and-save-animated-gif-with-python-pillow/)