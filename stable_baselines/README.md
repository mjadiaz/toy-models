# Training TD3 

The folder [scripts](./scripts) contains all the scripts that we are using for trainig the TD3 agent, the metrics and the utility functions for Monte Carlo Integration of the Q-Network.

- The function `new_run` was defined.
  - The idea is that every run that we made should have a different `run_name` to have the checkpoints in different folders. 
  - load_agent was defined to load a saved agent and continue the training. It should be the path for the saved agent.  
  
# Wandb integration
To use the Wandb callback install Wandb and login following this guide: [Weights and Biases quickstart](https://docs.wandb.ai/quickstart).

To running on the cluster we need to set offline mode, then following  the [track experiments guide](https://docs.wandb.ai/guides/track/launch), in the `new_run` function set the argument `cluster_mode=True`. Then, add your wandb key in your .zshrc file as:

`export WANDB_KEY = KEY`

and import in the `train.py` script with,
```python
import os
wandb_key = os.environ.get('WANDB_KEY')
```
then pass it to the `new_run` function as the `wandb_key = wandb_key`. This is already done in the `train.py` script.

While running the training, run the 'bash wandb_sync.sh' to constantly upload the training logs to Weights and Biases!

# Analizer 

The `analizer.py` script is an streamlit app to visualise the Q-network analisys. To run it install [Streamlit](https://streamlit.io/) and execute,

`streamlit run analizer.py`
