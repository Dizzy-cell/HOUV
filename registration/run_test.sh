#!/bin/bash

# CUDA_VISIBLE_DEVICES='0' python test_mult.py -c ./cfgs/vrcnet.yaml -l 0 -r 4 & CUDA_VISIBLE_DEVICES='1' python test_mult.py -c ./cfgs/vrcnet.yaml -l 4 -r 8 & CUDA_VISIBLE_DEVICES='2' python test_mult.py -c ./cfgs/vrcnet.yaml -l 8 -r 12 & CUDA_VISIBLE_DEVICES='3' python test_mult.py -c ./cfgs/vrcnet.yaml -l 12 -r 16


CUDA_VISIBLE_DEVICES='0' python test_mult.py -c ./cfgs/vrcnet.yaml -l 0 -r 500 & CUDA_VISIBLE_DEVICES='1' python test_mult.py -c ./cfgs/vrcnet.yaml -l 500 -r 1000 & CUDA_VISIBLE_DEVICES='2' python test_mult.py -c ./cfgs/vrcnet.yaml -l 1000 -r 1500 & CUDA_VISIBLE_DEVICES='3' python test_mult.py -c ./cfgs/vrcnet.yaml -l 1500 -r 2000

# num_size=2000 
# step=500
# l=0
# r=500
# for i in in {1..4}
# do
#     echo $l $r
#     #CUDA_VISIBLE_DEVICES='0' python test_mult.py -c ./cfgs/vrcnet.yaml -l $l -r $r 
#     let l+=$step
#     let r+=$step
# done

# Wait for the above run to end
sleep 600
echo “Combine Dataset”
python test_mult.py -c ./cfgs/vrcnet.yaml --combine True

cd ./log/vrcnet
zip -R submission_last.zip results.h5