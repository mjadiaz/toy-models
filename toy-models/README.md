# Toy models environments
Searching for points in toy functions with RL.

Install the environments in developer mode with:

`pip install --editable .`

# The reward function

To change the reward function, edit the `reward_functions.py` file in `toy_models/envs` by adding a new reward function as a plugin to the `REWARD_FUNCTIONS` dictionary. Then edit the '.get_reward()' method in the `/toy_models/envs/toy_functions.py`.


