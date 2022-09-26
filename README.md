# INS-Conv: Incremental Sparse Convolution for Online 3D segmentation

This is the incremental sparse convolution library implemented based on [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) and [Live Semantic 3D Perception for Immersive Augmented Reality](https://ieeexplore.ieee.org/abstract/document/8998140). The later describes a more efficient GPU implementation of the original submanifold sparse convolution. Our method supports incremental computing of sparse convolution, including SSC, convolution/deconvolution, BN, IO, and residual structure, etc.
## Environment setup

### Preliminary Requirements:
* Ubuntu 16.04
* CUDA 9.0
<!-- 
### Conda environment
Create the conda environment using:
```bash
conda env create -f p1.yml
```
and activate it. -->

### Install
```conda
conda env create -f p1.yml
```

```bash
sh all_build.sh
```

### Demo
For training, you could train an arbitary model using the original sparseconvnet.

For incremental inference, demo.py gives an example of the INS-Conv library.

The whole online segmentation system is complicated and large. If you are interested, contact liu-ly20@mails.tsinghua.edu.cn for more details.