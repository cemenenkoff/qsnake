# Snake and Deep Q Reinforcement Learning
### Each time the snake eats the apple, it grows by one chunk. The game ends if the snake head hits a wall or its own body.
![training montage](/example_output/training-montage.gif)
![learning curve](/example_output/learning-curve.png)
## Overview
____
- _**REINFORCEMENT LEARNING**_ is a style of machine learning that relies on past experience to develop a policy on what do to next.
- _**DEEP NEURAL NETWORKS**_ are used in machine learning to set up decision pathways that maximize reward (i.e. minimize loss or error).
- _**Q-LEARNING**_ is a subclass of reinforcement learning that is *model-free*, meaning that it does not require a model of the environment in which it operates. Given a reward structure (e.g. 10 points for eating an apple, but -100 for eating a body chunk), Q-learning can handle problems with inherent randomness (e.g. the apple respawning).

In this exploration, a _**policy**_ is a strategy to win the game of snake. Q-learning finds an *optimal* policy in the sense of maximizing the expected value of the total reward over any and all steps in the game, starting from an initial state.

The agent's policy begins as *"move randomly and hope for the best"* but changes as it completes more games in efforts to find an _**optimal action-selection policy**_ (where the possible actions are up, down, left, and right).

### **The more the agent plays, the more its random movements are replaced by policy-predicted movements.**
____
This project contains the following files:
____
**explore.py**

>After everything is set up, running `python explore.py -c config.json` is the main way to interface with this project. User options like where to store output, whether to save images, build a gif, and also hyperparameters for the learning agent are all specified in **config.json**.
____
**config.json**

>This configuration file defines how the script should run. All of the keys in the default configuration included below are required along with their example data types.

```yaml
{
    "project_root_dir": ".",
    "human": false,
    "name": "test",
    "save_for_gif": false,
    "make_gif": false,
    "params":{
        "epsilon": 1,
        "gamma": 0.95,
        "batch_size": 256,
        "epsilon_min": 0.001,
        "epsilon_decay": 0.98,
        "learning_rate": 0.00025,
        "layer_sizes": [128, 128, 128],
        "num_episodes": 15,
        "state_definition_type": "default"
    }
}
```
____
**requirements.txt**

>After cloning this repository (and hopefully setting up a virtual environment), run `pip install -r requirements.txt` to install all necessary dependencies. This project runs on **Python 3.9.7**, and some of the main modules utilized in this demo are **tensorflow**, **keras**, **turtle**, and **gym**.
____
**environment.py**

>This environment class inherits gym.Env and represents our snake game.
____
**agent.py**

>This script trains an agent to play snake via a deep Q-learning network.
____
**plotting.py**

>This is a supporting script that graphs some statistics about how the network was trained and the performance of the agent.
____
**gif_creator.py**

>This supporting class can convert saved eps files into png files and then png files into an animated gif. *Using this class requires a separate installation of Ghostscript!*
____
**make_gif_from_images.py**

>This small script allows for converting already-saved eps or png image files into a gif without having to re-run the game. Again, *using this script requires a separate installation of Ghostscript!*
____

## I recommend setting up this project in a **virtual environment**.

#### If you are new to python programming, I'd reccommend the following steps:

1. Download and install VS Code.

2. Install Python 3.9.7 (add it to PATH if you have no other Python versions installed).

3. Install Git bash.

4. *OPTIONAL:* If you want to utilize the gif-making supporting scripts, install Ghostscript and note the path of the binary. You will need to change a line in the preamble of **gif_creator.py** to specify where the binary is located.

5. From the Git bash shell, run `pip install virtualenv` to install the `virtualenv` module.

6. Run `python -m virtualenv <myenvname> --python=python3.9.7` to create a virtual environment that runs on Python 3.9.7.

7. In your shell, navigate to the main project folder that contains `<myenvname>` via `cd <projdir>`. You can confirm the environment folder exists with a quick `ls -la` command.

8. On Windows, run `./<myenvname>/Scripts/activate` to activate the virtual environment.

9.  On Linux or Mac (or if prompted by Git bash), run `source <myenvname>/bin/activate` to do the same thing.

10. Once your virtual environment is activated, close and restart your VS Code terminal.

11. You should see a `(<myenvname>)` string next to the terminal input when the environment is active.

12. Press `Ctrl+Shift+P` (on Windows) to open VS Code's command palette.

13. From the dropdown menu, click `Python: Select Interpreter`.

14. Select `Python 3.9.7 64-bit ('<myenvname>':venv)` (it may already be selected automatically).

15. Run `pip list` to see a list of installed packages. It should only have two modules.

16. Run `pip install -r requirements.txt` to install all dependencies on your activated virtual environment.

17. Once everything is installed, run `python explore.py -c config.json` to test if you can play the game manually.

18. Next, specify `"human": false` in config.json, save it, and then run  `python explore.py -c config.json` again, this time to see if the *agent* is able to play the game.

19. Let the agent run to the end and check that **plotting.py** is able to produce a graph of the learning curve.

20. Play with a few settings in config.json and re-run `python explore.py` to see how the changes affect the agent's behavior. Feel free to do this until you get bored or it sparks a questions you want to explore.

21. *OPTIONAL:* If you were bold enough to install Ghostscript, try saving game frames as eps files. You can then run `make_gif_from_images.py` (with some manual adjustments) to convert the eps files into png files and then into an animated gif.

22. *OPTIONAL:* Try converting the saved eps files into png and then into a gif all at once by specifying `"save_for_gif": true` and `"make_gif": true` in config.json. Please note that this process can take ~30 minutes for 50 training episodes.

23. To get a feel for the design of project, I would recommend reading the algorithm overview below, then **explore.py**, and then **environment.py** before **agent.py**.
____
## [What is a Bellman equation?](https://en.wikipedia.org/wiki/Bellman_equation)
![equation](/example_output/bellman.png)

What's shown above is an [algorithmic solution to a Bellman equation](https://en.wikipedia.org/wiki/Q-learning#Algorithm) using value iteration (also known as backward induction). The linked article is very well-written, but here's a summary:

>### **Q is a function that computes expected rewards for an action taken in a given state.**
It has a few hyperparameters, but two are especially important:
1. The _**learning rate**_, α, where α ∈ (0,1]. The learning rate determines _**to what extent newly acquired information overrides old information**_. A factor of 0 makes the agent exclusively use prior knowledge, whereas 1 makes it consider only the most recent data. In deterministic environments, a value of 1 is optimal, but when there is randomness involved, it needs to eventually decrease to zero. In practice however, using a constant learning rate works just fine.
2. The _**discount factor**_, γ, where γ ∈ [0,1]. The discount factor determines the _**importance of future rewards**_. 0 makes the agent short-sighted, whereas 1 makes it strive for a long-term, larger reward. Starting with a small discount factor and increasing it toward a final value over time tends to accelerate learning.

### The reward value generated by **Q is the sum of three factors**:
1. The _**current value**_ weighted by the learning rate. Changes in Q become more rapid as the learning rate approaches 1.
2. The _**reward to earn via the proposed action**_ in the current state, weighted by the learning rate.
3. The _**max possible reward**_ from the next state, weighted by the learning rate and discount factor.
____
## To do:

- ~~Allow for several user-definable agents via **config.json** files.~~

- ~~Clean up **plotting.py** and add the ability to save figures locally.~~

- ~~Add hyperparameter settings to the learning curve figure generated by **plotting.py**.~~

- ~~Expand **README.md** to include a high-level overview of how the network is trained via a **Bellman equation**.~~

- **Allow for models to be saved.**

- **Allow previously trained models to play with saved settings.**

- **Allow previously trained models to play and resume training.**

- ~~Expand **plotting.py** to store a hyperparameter configuration's collection of training episodes as a gif.~~
