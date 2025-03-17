# depth_model
Creating an efficient depth map generation network for a quadcopter using NPU TH1520

### Test on Linux
```
cd ./md_data
python -m venv depth
source ./depth/bin


### Install Docker
https://docs.docker.com/desktop/setup/install/linux/

### To build and start the container
```bash
docker compose up -d --build --remove-orphans
```

### To run built container
```bash
docker compose up -d
```

### To go into container
```bash
docker exec -ti mobile_depth /bin/bash
```

### To run tests in container
```bash
cd /md_data
./run_tests.sh
```

