### pyDBoW3

This repository updates the pyDBoW3 project: https://github.com/foxis/pyDBoW3 For usage examples see the original repository.

We performed slight modifications for compatibility with OpenCV 4.0.1 and provide a Docker image to easily create a Python wheel for use within another project on a Ubuntu 18.04 host.


### Building the Python wheel

To create a Python wheel of pyDBoW3 for use within another project, follow the steps below.

1) Clone the repo to your machine
```
git clone https://github.com/LukasBommes/pyDBoW3-builder.git
```

2) Build the Docker image which contains the full build environment
```
sudo docker build . -t pydbow3
```
This will automatically install OpenCV 4.0.1, DBoW3 and compile the pyDBoW3 source against OpenCV and DBoW. Additionally, a Python wheel of pyDBoW3 is created.

3) Startup the container
```
sudo docker run -it --name pydbow3_container pydbow3 /bin/bash
```

4) Copy the Python wheel to your local machine (home directory) by running the command below in a new terminal
```
sudo docker cp pydbow3_container:/home/pydbow3/build/dist/dist/pyDBoW3-0.3-cp36-cp36m-linux_x86_64.whl ~

```
Note, that depending on the python version which is being installed in the Docker image the name of the wheel can vary. You can navigate to `/home/pydbow3/build/dist/dist` inside the container to determine the name of the wheel.

### Using the Python wheel in another project

1) Install dependencies (Boost and OpenCV)
```
apt-get update && apt-get install -y libboost-all-dev libopencv-dev
```

2) Move the previously generated wheel anywhere on your host and install it
```
python -m pip install pyDBoW3-0.3-cp36-cp36m-linux_x86_64.whl

```
Note, that the Python version of your project has to match the Python version for which the wheel was build. In the above example, the wheel is only compatible with Python 3.6. To build for another Python version you have to modify the Python version inside the Docker image and rebuild the wheel.
