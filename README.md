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
![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](/Assets/images/Arc.png)
