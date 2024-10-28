**[ECCV 2024] Online Video Quality Enhancement with Spatial-Temporal Look-up Tables.** [Paper](https://arxiv.org/abs/2311.13616)

This is the offical implementation of the STLVQE.

## 🔖 Abstract

Low latency rates are crucial for online video-based applications, such as video conferencing and cloud gaming, which make improving video quality in online scenarios increasingly important. However, existing quality enhancement methods are limited by slow inference speed and the requirement for temporal information contained in future frames, making it challenging to deploy them directly in online tasks. In this paper, we propose a novel method, STLVQE, specifically designed to address the rarely studied online video quality enhancement (Online-VQE) problem.
Our STLVQE designs a new VQE framework which contains a Module-Agnostic Feature Extractor that greatly reduces the redundant computations and redesign the propagation, alignment, and enhancement module of the network. A Spatial-Temporal Look-up Tables (STL) is proposed, which extracts spatial-temporal information in videos while saving substantial inference time. To the best of our knowledge, we are the first to exploit the LUT structure to extract temporal information in video tasks. Extensive experiments on the MFQE 2.0 dataset demonstrate our STLVQE achieves a satisfactory performance-speed trade-off.

## ⚙️ Setup

### 0. Install Environment via Anaconda (Recommended)
```bash
conda create -n STLVQE
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
git clone https://github.com/quzefan/STLVQE
cd STLVQE
cd (mmediting/modulated_deform/quadrilinear4d_cpp/quadrilinear6d_cpp)
python setup.py install
cd ..
cd mmediting
```

### 1. Prepare the data
You can get the MFQE 2.0 dataset at [here](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset) and compress the video by following its instruction.

Put the compressed video in the mmediting/data folder.

### 2. Train
Please modify the address to your compressed video and GT in the config file first.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh --config ./config/STLVQE_Train.py
```

### 3. Finetune
Please modify the address to your weight in the config file (load_from).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh --config ./config/STLVQE_Finetune.py
```

### 4. Transfer to LUT
```bash
python Transfer_to_lut.py
python Transfer_to_lut_6d.py
```

### 5. Inference
```bash
python ./tools/test.py --config ./config/STLVQE_Inference.py
```

## 😉 Citation
If you find our work is helpful to your research, please cite the papers as follows:
```
@article{qu2023online,
  title={Online Video Quality Enhancement with Spatial-Temporal Look-up Tables},
  author={Qu, Zefan and Jiang, Xinyang and Yang, Yifan and Li, Dongsheng and Zhao, Cairong},
  journal={arXiv preprint arXiv:2311.13616},
  year={2023}
}

@article{yin2024online,
  title={Online streaming video super-resolution with convolutional look-up table},
  author={Yin, Guanghao and Qu, Zefan and Jiang, Xinyang and Jiang, Shan and Han, Zhenhua and Zheng, Ningxin and Yang, Huan and Liu, Xiaohong and Yang, Yuqing and Li, Dongsheng and others},
  journal={IEEE Transactions on Image Processing},
  volume={33},
  pages={2305--2317},
  year={2024},
  publisher={IEEE}
}
```


## 🤗 Acknowledgements
Our codebase builds on [mmagic](https://github.com/open-mmlab/mmagic) (Previously called MMEditing), [modulated-deform-conv](https://github.com/CHONSPQX/modulated-deform-conv) and [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus). 
Thanks the authors for sharing their awesome codebases! 


## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****
