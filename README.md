# FSPI-DM
**Paper:** High-resolution iterative reconstruction at extremely low sampling rate for Fourier single-pixel imaging via diffusion model

**Authors:** Xianlin Song, Xuan Liu,  Zhouxu Luo, Huilin Zhou, Jiaqing Dong, Wenhua Zhong, Guijun Wang, Binzhong He, Qiegen Liu, Senior Member, IEEE

Date : Jan-9-2024  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2024, Department of Electronic Information Engineering, Nanchang University.  

<div align="justify">
The trade-off between imaging efficiency and imaging quality has always been encountered by Fourier single-pixel imaging (FSPI). To achieve high-resolution imaging, the increase in the number of measurements is necessitated, resulting in a reduction of imaging efficiency. Here, a novel high-quality reconstruction method for FSPI imaging via diffusion model was proposed. A score-based diffusion model is designed to learn prior information of the data distribution. The real-sampled low-frequency Fourier spectrum of the target is employed as a consistency term to iteratively constrain the model in conjunction with the learned prior information, achieving high-resolution reconstruction at extremely low sampling rates. The performance of the proposed method is evaluated by simulations and experiments. The results show that the proposed method has achieved superior quality compared with the traditional FSPI method and the U-Net method. Especially at the extremely low sampling rate (e.g., 1%), an approximately 241% improvement in edge intensity-based score was achieved by the proposed method for the coin experiment, compared with the traditional FSPI method. The method has the potential to achieve high-resolution imaging without compromising imaging speed, which will further expanding the application scope of FSPI in practical scenarios.
</div>

# Scheme of the system and the photographs of practical system.
![图片描述](Figures/1.png)
# Flow chart of high-resolution iterative reconstruction based on diffusion model.

# The reconstruction results obtained by different methods for animal and coin under various sampling rates, as well as the corresponding ground truth and Fourier spectra.

# Requirements and Dependencies
python==3.7.11  
Pytorch==1.7.0  
tensorflow==2.4.0  
torchvision==0.8.0  
tensorboard==2.7.0  
scipy==1.7.3  
numpy==1.19.5  
ninja==1.10.2  
matplotlib==3.5.1  
jax==0.2.26  

# Checkpoints
We provide pretrained checkpoints of the dog. You can download pretrained models from [Baidu cloud] (https://pan.baidu.com/s/1IYIG5fQ_Ju_iRAbX455dSg) Extract the code (FSPI)

# Dataset
