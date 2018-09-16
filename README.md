# A Cascaded Refinement GAN for Phase Contrast Microscopy Image Super Resolution

This is a tensorflow implementation of cascaded refinement generative adversarial network (GAN) to super-resolve phase contrast microscopy images.



# Setup

Prerequisites:

Windows or Linux

Required python libraries: Tensorflow + Scipy + Numpy + Pillow.

NVIDIA GPU + CUDA(>=8.0) + CuDNN(>=5.0) (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

# Training and Testing

To train a model that super-resolves the phase contrast image only using the adversarial loss, please run "adversarial_loss_only.py".

To train a model that super-resolves the phase contrast image only using the content loss, please run "contnet_loss_only.py".

To train a model that super-resolves the phase contrast image using our proposed cascaded refinement GAN, please run "cascaded_refinement_GAN.py".

To train a model that super-resolves the phase contrast image only using a single image input, i.e., no differential pattern filter (DPF) is used, please run "single_image_input.py".

# Citation

If you use our code for research, please cite our paper:

Liang Han and Zhaozheng Yin. A Cascaded Refinement GAN for Phase Contrast Microscopy Image Super Resolution. In MICCAI 2018.

# Acknowledegments

Part of the code in this project borrows from 'Photographic Image Synthesis with Cascaded Refinement Networks'.
