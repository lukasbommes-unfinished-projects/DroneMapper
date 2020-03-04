### Building the Docker image

Change into base directory of project and run
```
sudo docker build -t dronemapper .
```

### Running the Docker image

```
sudo docker run -it --runtime=nvidia --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $(pwd):/dronemapper dronemapper /bin/bash
```
