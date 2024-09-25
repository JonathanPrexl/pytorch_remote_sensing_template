# PyTorch Template for Deep Learning in Remote Sensing

A flexible template for setting up deep learning experiments with pytorch and hydra (in the context of satellite based Earth observation data).

This repo was prepared and presented in the context of the Munich autumn school for remote sensing (MARS) 2023 and 2024.



# Requirements

Requires a Python environment with (at least) `PyTorch`, `torchvision`, `torchinfo`, `hydra`, `rasterio`, `numpy`, 
`pandas`, `tqdm`, `matplotlib`



# Modules and Usage

* `python3 model.py`: run an empty batch through the model and get the output shape. Useful for model testing and debugging.
* `python3 dataset.py`: load dataset with default config and get shape of first dataset element. Useful for debugging.
* `python3 main.py`: Main entry point for training.
  * Show help with `--help`
  * Use another configuration file: `python3 main.py --config-name other_configuration_file`
  * Overwrite configuration parameters using the command line: `python3 main.py config_name=new_value gpu_idx=cpu`
* `configs` directory contains all configuration files, there can exist multiple ones
* `notebooks` implement the dataset split, check the dataset class and analyze the model output during inference

(If Docker is installed: a container can be build and run with bash script `docker/startcontainer.sh`. Set mounted 
volumes to have you data available inside the container.)



# Dataset for testing

You can get the EuroSat dataset [here](https://github.com/phelber/EuroSAT).



# Updates

### Sep. 2024

* Change the data-set to work with a pre-defined split file for more control over the data split
* Implement Vision Transformer for the 2024 Workshop "Transformers from Scratch" as a minimal viable example



# Contributer 

Jonathan Prexl & Thomas Roßberg
