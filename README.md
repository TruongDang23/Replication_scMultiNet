# scMultiNet  
A deep adversarial network model for multi-task analysis of single-cell omics data  

## Table of Contents  
- [Overview](#overview)  
- [Tutorial](#tutorial)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Citation](#citation)  
- [Contact](#contact)  

## <a name="overview"></a>Overview  
scMultiNet is a deep adversarial network designed for **multi-task analysis of single-cell multi-omics data**, supporting tasks such as data denoising, clustering, integration, and cross-modal prediction.  

## <a name="tutorial"></a>Tutorial  
A step-by-step tutorial for scMultiNet is available at:  
👉 [scMultiNet Tutorial Documentation](https://scmultinet-tutorial.readthedocs.io/en/latest/)  

## <a name="installation"></a>Installation  

**Environment requirements**  
- Python 3.8.x  
- PyTorch (tested with 1.9.0+cu111)  
- CUDA 11.1 (tested on NVIDIA RTX 2080 Ti)  

We recommend installing dependencies in a **conda environment**:  

```bash
conda create -n scMultiNet python=3.8
conda activate scMultiNet
```

Then install the required packages:  

```bash
pip install h5py==3.9.0 torch==1.9.0+cu111 anndata==0.9.2 scanpy==1.9.3 scikit-learn==0.22.2
```  

## <a name="usage"></a>Usage  

1. Prepare the input data in **.h5** format (see README in the `data` folder).  
2. Run scMultiNet following the [tutorial](https://scmultinet-tutorial.readthedocs.io/en/latest/), or directly from the command line:  

```bash
python train.py --dataset=BMNC
```  

## <a name="citation"></a>Citation  
If you use scMultiNet in your research, please cite our work (citation details coming soon).  

## <a name="contact"></a>Contact  
Junlin Xu  
📧 Email: xjl@hnu.edu.cn  
