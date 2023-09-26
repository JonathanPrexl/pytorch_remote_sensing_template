# PyTorch Template for Deep Learning in Remote Sensing

A flexible template for setting up deep learning experiments with pytorch and hydra.

Requires a Python environment with (at least) PyTorch, torchvision, torchinfo, hydra, rasterio, numpy, 
pandas, tqdm, matplotlib

* `python3 model.py`: run an empty batch through the model and get the output shape. Useful for model testing and debugging.
* `python3 dataset.py`: load dataset with default config and get shape of first dataset element. Useful for debugging.
* `python3 main.py`: Main entry point for training.
  * Show help with `--help`
  * Use another configuration file: `python3 main.py --config-name other_configuration_file`
  * Overwrite configuration parameters using the command line: `python3 main.py config_name=new_value gpu_idx=cpu`
* `configs` directory contains all configuration files, there can exist multiple ones


(If Docker is installed: a container can be build and run with bash script `docker/startcontainer.sh`. Set mounted 
volumes to have you data available inside the container.)
