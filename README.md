# DashGaussian with TC-GS

This repo applies [TC-GS](https://arxiv.org/pdf/2505.24796v2) to [DashGaussian](https://dashgaussian.github.io/) , achieving 1.2x speedup on average during the training process.

Only the forward pass utilizes the Tensor Cores.

If you want to use TC-GS module, try to install  ```submodules/tcgs_rasterizer``` **instead of** ```submodules/diff-gaussian-rasterization``` 

## Usage

1. clone this repo
```shell
git clone https://github.com/TensorCore3DGS/TCGS_Dev/tree/master
```

2. Setup the environment as [DashGaussian](https://dashgaussian.github.io/) 

3. Replace the renderer by TC-GS module.
```shell
pip install submodules/tcgs_rasterizer
```


# DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds
### [Webpage](https://dashgaussian.github.io/) | [Paper](https://arxiv.org/pdf/2503.18402) | [arXiv](https://arxiv.org/abs/2503.18402) | [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

The implementation of **DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds**, a powerful 3DGS training acceleration method. Accepted by CVPR 2025 (highlight).

In this repository, we show how to plug DashGaussian into [the up-to-date 3DGS implementation](https://github.com/graphdeco-inria/gaussian-splatting). 

**To notice**, the official implementation of 3DGS has been updating since the paper of DashGaussian is published, so the reproduced results from this repository can be different from that reported in the paper.
We plan to provide more examples for DashGaussian beyond vanilla 3DGS in the near future. 

## Environment Setup
To prepare the environment, 

1. Clone this repository. 
	```
	git clone https://github.com/YouyuChen0207/DashGaussian.git
	```
2. Follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install dependencies. 

	Please notice, that the ```diff-gaussian-rasterization``` module contained in this repository has already been switched to the ```3dgs-accel``` branch for efficient backward computation.
3. Install our Lanczos-resampling implementation for anti-aliased image downsampling. 
	```
	pip install submodules/lanczos-resampling
	```

## Run DashGaussian

### Running Comand
Set the data paths in ```scripts/full_eval.sh``` to your local data folder, and run.
```
bash scripts/full_eval.sh
```

### Running Options
In ```full_eval.py```, you can set, 
* ```--dash``` Enable DashGaussian.
* ```--fast``` Use the Sparse Adam optimizer. 
* ```--preset_upperbound``` Set the primitive number upperbound manually for DashGaussian and disable the momentum-based primitive upperbound budgeting method. This option is disabled by default.

### Plug DashGaussian into Other 3DGS Backbones
This repository is an example to plug DashGaussian into 3DGS backbones. 
Search keyword ```DashGaussian``` within the project, you can find all code pieces integrating DashGaussian into the backbone. 

## Results
The following experiment results are produced with a personal NVIDIA RTX 4090 GPU.
The average of rendering quality metrics, number of Gaussian primitives in the optimized 3DGS model, and training time, are reported. 
### [Mipnerf-360 Dataset](https://jonbarron.info/mipnerf360/)
|  Method | Optimizer | PSNR | SSIM | LPIPS | N_GS | Time (min) |
|-----|-----|-----|-----|-----|-----|-----|
| 3DGS | Adam | 27.51 | 0.8159 | 0.2149 | 2.73M | 12.70 |
| 3DGS-Dash | Adam | 27.38 | 0.8080 | 0.2316 | 2.26M | 6.96 | 
| 3DGS-fast | Sparse Adam | 27.33 | 0.8102 | 0.2240 | 2.46M | 7.91 | 
| 3DGS-fast-Dash | Sparse Adam | 27.37 | 0.8041 | 0.2391 | 2.06M | 4.54 |

### [Deep-Blending Dataset](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)
|  Method | Optimizer | PSNR | SSIM | LPIPS | N_GS | Time (min) |
|-----|-----|-----|-----|-----|-----|-----|
| 3DGS | Adam | 29.83 | 0.9069 | 0.2377 | 2.48M | 10.74 |
| 3DGS-Dash | Adam | 29.71 | 0.9061 | 0.2482 | 1.80M | 4.10 | 
| 3DGS-fast | Sparse Adam | 29.48 | 0.9068 | 0.2461 | 2.31M | 6.71 | 
| 3DGS-fast-Dash | Sparse Adam | 29.76 | 0.9046 | 0.2539 | 1.64M | 2.67 |

### [Tanks&Temple Dataset](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)
|  Method | Optimizer | PSNR | SSIM | LPIPS | N_GS | Time (min) |
|-----|-----|-----|-----|-----|-----|-----|
| 3DGS | Adam | 23.73 | 0.8526 | 0.1694 | 1.57M | 8.04 |
| 3DGS-Dash | Adam | 24.01 | 0.8502 | 0.1838 | 1.16M | 4.29 | 
| 3DGS-fast | Sparse Adam | 23.78 | 0.8502 | 0.1741 | 1.53M | 6.11 | 
| 3DGS-fast-Dash | Sparse Adam | 24.00 | 0.8499 | 0.1856 | 1.14M | 3.15 |

## Citation
```
@inproceedings{chen2025dashgaussian,
  title     = {DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds},
  author    = {Chen, Youyu and Jiang, Junjun and Jiang, Kui and Tang, Xiao and Li, Zhihao and Liu, Xianming and Nie, Yinyu},
  booktitle = {CVPR},
  year      = {2025}
}
```
