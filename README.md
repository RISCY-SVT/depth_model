# depth_model
Создание эффективной сети построения карты глубин для квадрокоптера с использованием NPU TH1520

### Install Docker
https://docs.docker.com/desktop/setup/install/linux/


### To build and start the container
```bash
docker compose up -d --build --remove-orphans
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
