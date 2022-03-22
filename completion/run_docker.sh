#!/bin/bash

# github.com.cnpmjs.org 
docker run -it --shm-size="10g" --gpus all -v /disk2/yanyj/MVP_Benchmark/MVP_Benchmark:/home/mvp  --net host mvp bash 

docker exec -it id bash 

git fetch
git branch -a
git branch 
git checkout 
git pull