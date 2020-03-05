### Building the Docker image

Change into base directory of project and run
```
sudo docker build -t dronemapper .
```

### Running the Docker image

On the host disable X server access control
```
xhost +
```
and start the docker container
```
sudo docker run -it \
    --runtime=nvidia \
    --ipc=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/dronemapper \
    -p 8888:8888 \
    dronemapper /bin/bash
```

### Using Jupyter Lab

To startup jupyter lab use the following command
```
jupyter lab --ip='0.0.0.0' --port=8888 --no-browser --allow-root
```
Open the URL which is displayed to you in the web browser on the host.
