# The champion method HOUV about Regitstration on MVP Benchmark :* Multi-View Partial Point Clouds for Completion and Registration
<p align="center"> 
<img src="images/logo.png">
</p>

## [MVP Benchmark]

### Overview
This repository mainly about champion method on the Partial-to-Partial Point Cloud Registration Completion.
We propose a novel method called the Hybrid Optimization with Unconstrained Variables (HOUV) on 3D Registration.
To the best of our knowledge, HOUV is the first work to directly optimize unconstrained variables in sophisticated registration, surpassing the advanced optimization and learning-based methods. 
This repository also introduces the MVP Benchmark for partial point cloud **[COMPLETION](https://github.com/paul007pl/MVP_Benchmark/tree/main/completion)** and **[REGISTRATION](https://github.com/paul007pl/MVP_Benchmark/tree/main/registration)**, and it also includes following recent methods:

+ **Completetion:**
    &nbsp;&nbsp;[1] [PCN](https://github.com/wentaoyuan/pcn); &nbsp;&nbsp;[2] [ECG](https://github.com/paul007pl/ECG); &nbsp;&nbsp;[3] [VRCNet](https://github.com/paul007pl/VRCNet)

+ **Registration:**
    &nbsp;&nbsp;[1] [DCP](https://github.com/WangYueFt/dcp); &nbsp;&nbsp;[2] [DeepGMR](https://github.com/wentaoyuan/deepgmr); &nbsp;&nbsp;[3] [IDAM](https://github.com/jiahaowork/idam)

This repository is implemented in Python 3.7, PyTorch 1.5.0, CUDA 10.1 and gcc > 5. 


### Installation
Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), and then use the following command:
```
git clone --depth=1 https://github.com/paul007pl/MVP_Benchmark.git
cd MVP_Benchmark; source setup.sh;
```
If your connection to conda and pip is unstable, it is recommended to manually follow the setup steps in `setup.sh`.


### MVP Dataset
Download corresponding dataset:
  + **Completion** :&nbsp;&nbsp;&nbsp;&nbsp; [Google Drive](https://drive.google.com/drive/folders/1XxZ4M_dOB3_OG1J6PnpNvrGTie5X9Vk_) &nbsp;&nbsp; or &nbsp;&nbsp; [百度网盘](https://pan.baidu.com/s/18pli79KSGGsWQ8FPiSW9qg)&nbsp;&nbsp;(code: p364)
  + **Registration** :&nbsp;&nbsp;&nbsp;&nbsp; [Google Drive](https://drive.google.com/drive/folders/1RlUW0vmmyqxkBTM_ITVguAjxzIS1MFz4) &nbsp;&nbsp; or &nbsp;&nbsp; [百度网盘](https://pan.baidu.com/s/18pli79KSGGsWQ8FPiSW9qg)&nbsp;&nbsp;(code: p364)


### Usage
For registration
  + `cd registration` into the workspace.
  + `mkdir log` and `mkdir log/houv` to build the savedir.
  + `touch log/houv/train.log` to build the log file.
  + To test `MVP_ExtraTest_RG.h5` dataset: run `bash run_test.sh`. 
    - This script needs four gpus.
    - It maybe take one or two hours.
    - If run error, please step by step commands in `run_test.sh`.
  + Config for those methods can be found in `cfgs/`.
  + The results are saved in `./log/houv/results.h5`
  + The submission in `submission_last.zip`
More information please refer to `registraion/read.me`.


+ Different partial point clouds for the same CAD Model:
<p align="center"> 
<img src="images/partial_pcds.gif", style="zoom: 75%;">
</p>

+ High-quality complete point clouds:
<p align="center"> 
<img src="images/complete_pcds.gif", style="zoom: 75%;">
</p>

<br>


## [Citation]
If you find our code useful, please cite our paper:
```bibtex
@article{pan2021variational,
  title={Variational Relational Point Completion Network},
  author={Pan, Liang and Chen, Xinyi and Cai, Zhongang and Zhang, Junzhe and Zhao, Haiyu and Yi, Shuai and Liu, Ziwei},
  journal={arXiv preprint arXiv:2104.10154},
  year={2021}
}
```

<br>


## [License]
Our code is released under Apache-2.0 License.

<br>


## [Acknowledgement]
We include the following PyTorch 3rd-party libraries:  
[1] [CD](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)  
[2] [EMD](https://github.com/Colin97/MSN-Point-Cloud-Completion)  
[3] [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)  

We include the following algorithms:  
[1] [PCN](https://github.com/wentaoyuan/pcn)  
[2] [ECG](https://github.com/paul007pl/ECG)  
[3] [VRCNet](https://github.com/paul007pl/VRCNet)  
[4] [DCP](https://github.com/WangYueFt/dcp)  
[5] [DeepGMR](https://github.com/wentaoyuan/deepgmr)  
[6] [IDAM](https://github.com/jiahaowork/idam)  
