#!/bin/bash

CUDA_VISIBLE_DEVICES='0' python train_mult_gpu.py -c ./cfgs/vrcnet.yaml

#CUDA_VISIBLE_DEVICES='1' python train_HOUV.py -c ./cfgs/houv.yaml