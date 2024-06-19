# code_thesis
The cobel directory was taken over from https://github.com/sencheng/CoBeL-RL but only the neccessary scripts have been adopted (with slight changes).\\
The simple_FW.py script is based on the myfastweights.py from https://github.com/ischlag/Fast-Weight-Memory-public/tree/main/metaRL.

## Used setup
Python version 3.10.13

numpy 1.26.1

PyTorch 2.1.0

gym 0.26.2

matplotlib 3.7.1

scikit-learn 1.3.0

pandas 2.1.1

## Getting started
gridworld_demo_ppo_fwm.py is the script that runs the training (or testing) on a grid world for the memory-augmented agent (FWM) or the agent without a memory module (LSTM).

ppo_fwm.py is the script that contains the PPO algorithm.

simple_FW.py is the script that contains the FWM module.

The cobel directory contains all necessaryfunctions to create the grid world.

The more_gridworlds directory contains grid world 2 and 3 mentioned in the thesis.

All scripts that were used to create the figures of the thesis can be found in the "plots" directory.
