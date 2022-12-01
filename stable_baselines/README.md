# Training TD3 

The folder [scripts](./scripts) contains all the scripts that we are using for trainig the TD3 agent, the metrics and the utility functions for Monte Carlo Integration of the Q-Network.

- The function `new_run` was defined.
  - The idea is that every run that we made should have a different `run_name` to have the checkpoints in different folders. 
  - load_agent was defined to load a saved agent and continue the training. It should be the path for the saved agent.  
  
# Wandb integration
To use the Wandb callback install Wandb and login following this guide: [Weights and Biases quickstart](https://docs.wandb.ai/quickstart).

# Analizer 

The `analizer.py` script is an streamlit app to visualise the Q-network analisys. To run it install [Streamlit](https://streamlit.io/) and execute,

`streamlit run analizer.py`
