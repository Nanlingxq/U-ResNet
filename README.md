# U-ResNet: A Parallel Fusion Network for Image Classification and Segmentation

This repository contains the official implementation of U-ResNet, a novel parallel fusion architecture combining ResNet and U-Net for joint image classification and segmentation, as presented in our paper:</br>
## 🧠 Overview
### U-ResNet integrates the residual learning of ResNet and the encoder-decoder structure of U-Net in a parallel manner to achieve:</br>
✅ High accuracy in both image classification and semantic segmentation

✅ Rapid convergence and mitigation of the vanishing gradient problem

✅ Enhanced pixel-level feature extraction via EUCB* and Selected Upsampling (SU) modules

✅ State-of-the-art performance on multiple public and private datasets
## 🏗️ Architecture
### U-ResNet consists of three main components:

<li>ResBlock: Residual learning path for classification and gradient stability.</li>

<li>UBlock: U-Net inspired path for pixel-level segmentation.</li>

<li>Feature Merge: Fusion module combining features from both paths.</li>

### Additional modules:

<li>Selected Upsampling (SU): Enhances low-resolution image features.</li>

<li>EUCB*: Improved upsampling block with channel shuffle for faster convergence.</li>

## 📦 Installation
### Requirements
<li>Python 3.8+</li>

<li>PyTorch 2.5.1</li>

<li>CUDA 12.1 + cuDNN 9</li>

### Other dependencies:
<li>torchvision, numpy, opencv-python, tqdm, scikit-image</li>

### To install the package from python, please run the code:
```
  pip install -r requirements.txt
```

### Datasets
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)</br>
[MiniImageNet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)</br>
[Cityscapes](https://www.cityscapes-dataset.com)</br>

## 🔎 Train and Evaluate
If you have prepared the datasets,  run such code to train the model:
```
  python Segmentation/Cityscapes/TrainCode.py
```

## 🎼 Citation
If you think this project is helpful, please leave a ⭐️ and cite our paper.
