#!/bin/bash
# Build and run a docker container for development.

# Safer bash scripts, which fail early
# See: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

# Set some variables
image_name="l91bjopr/pytorchtutorial"
container_name="pytorchtutorial"

docker build -t ${container_name} \
  --build-arg USER_ID=$(id -u) \
  --build-arg HOST_UID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .

# Change, which volumes are mounted!
# Add '--gpus all' to use GPUs in the container
# This might also require '--privileged --ipc=host --shm-size=128G'
docker run -it --gpus all --name ${container_name} \
--shm-size=8G \
--volume /home/jprexl//Code/pytorch_remote_sensing_template/src:/home/user/src/ \
--volume /home/jprexl/Data/EuroSATallBands/:/home/user/data/ \
--volume /home/jprexl//Code/pytorch_remote_sensing_template/data:/home/user/datainference/ \
--volume /home/jprexl/Results/MARS24_workshop/:/home/user/results/ \
${container_name}