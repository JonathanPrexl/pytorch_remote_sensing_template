docker build -t l91bjopr/pytorchtutorial \
  --build-arg USER_ID=$(id -u) \
  --build-arg HOST_UID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .


docker run -it --gpus all --privileged --shm-size=128g --name pytorchtutorial --ipc=host \
-v /home/jprexl/Code/pytorch_remote_sensing_template/src:/home/user/src/ \
l91bjopr/pytorchtutorial
