# A Cascaded Refinement GAN for Phase Contrast Microscopy Image Super Resolution

This is a tensorflow implementation of cascaded refinement generative adversarial network (GAN) to super-resolve phase contrast microscopy images.



# Setup

Prerequisites:

Windows or Linux

Required python libraries: Tensorflow + Scipy + Numpy + Pillow.

NVIDIA GPU + CUDA(>=8.0) + CuDNN(>=5.0) (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

# Training and Testing

To train a model at 256p resolution, please set "is_training=True" and change the file paths for training and test sets accordingly in "demo_256p.py". Then run "demo_256p.py".

To train a model at 256p resolution, please set "is_training=True" and change the file paths for training and test sets accordingly in "demo_256p.py". Then run "demo_256p.py".

To train a model at 256p resolution, please set "is_training=True" and change the file paths for training and test sets accordingly in "demo_256p.py". Then run "demo_256p.py".

To train a model at 256p resolution, please set "is_training=True" and change the file paths for training and test sets accordingly in "demo_256p.py". Then run "demo_256p.py".

# Citation

If you use our code for research, please cite our paper:

Liang Han and Zhaozheng Yin. A Cascaded Refinement GAN for Phase Contrast Microscopy Image Super Resolution. In MICCAI 2018.

# Acknowledegments

Part of the code borrows from 'Photographic Image Synthesis with Cascaded Refinement Networks'.
