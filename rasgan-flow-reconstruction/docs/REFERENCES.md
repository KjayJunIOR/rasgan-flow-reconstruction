# Method references

The implementation combines ideas from several established model families and
training techniques. These references are scholarly attribution; they do not
imply that third-party source code is vendored in the repository.

1. C. Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a
   Generative Adversarial Network*, CVPR 2017.
   https://doi.org/10.48550/arXiv.1609.04802
2. X. Wang et al., *ESRGAN: Enhanced Super-Resolution Generative Adversarial
   Networks*, ECCV Workshops 2018.
   https://doi.org/10.48550/arXiv.1809.00219
3. A. Jolicoeur-Martineau, *The Relativistic Discriminator: a Key Element
   Missing from Standard GAN*, ICLR 2019.
   https://doi.org/10.48550/arXiv.1807.00734
4. M. Mirza and S. Osindero, *Conditional Generative Adversarial Nets*.
   https://doi.org/10.48550/arXiv.1411.1784
5. P. Isola et al., *Image-to-Image Translation with Conditional Adversarial
   Networks*, CVPR 2017.
   https://doi.org/10.48550/arXiv.1611.07004
6. Q. Wang et al., *ECA-Net: Efficient Channel Attention for Deep Convolutional
   Neural Networks*, CVPR 2020.
   https://doi.org/10.48550/arXiv.1910.03151
7. J. Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018.
   https://doi.org/10.48550/arXiv.1709.01507
8. T. Miyato et al., *Spectral Normalization for Generative Adversarial
   Networks*, ICLR 2018.
   https://doi.org/10.48550/arXiv.1802.05957
9. T. Salimans et al., *Improved Techniques for Training GANs*, NeurIPS 2016.
   https://doi.org/10.48550/arXiv.1606.03498
10. L. Mescheder et al., *Which Training Methods for GANs Do Actually
    Converge?*, ICML 2018.
    https://doi.org/10.48550/arXiv.1801.04406
11. M. Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to
    a Local Nash Equilibrium*, NeurIPS 2017.
    https://doi.org/10.48550/arXiv.1706.08500
12. W.-S. Lai et al., *Deep Laplacian Pyramid Networks for Fast and Accurate
    Super-Resolution*, CVPR 2017.
    https://doi.org/10.48550/arXiv.1704.03915
13. K. Fukami, Y. Nabae, K. Kawai, and K. Fukagata, *Super-resolution
    reconstruction of turbulent velocity fields using a GAN-based AI
    framework*.
    https://doi.org/10.1063/1.5127031
14. M. Bode et al., *Using physics-informed enhanced super-resolution
    generative adversarial networks for subfilter modeling in turbulent
    reactive flows*.
    https://doi.org/10.1016/j.proci.2020.06.022
15. M. Bode et al., *Influence of adversarial training on super-resolution
    turbulence reconstruction*.
    https://doi.org/10.1103/PhysRevFluids.9.064601
16. L. Jiang et al., *Focal Frequency Loss for Image Reconstruction and
    Synthesis*, ICCV 2021.
    https://doi.org/10.48550/arXiv.2012.12821
17. G. Morales-Brotons et al., *Exponential Moving Average of Weights in Deep
    Learning: Dynamics and Benefits*.
    https://doi.org/10.48550/arXiv.2411.18704

## Software lineage

Early development used the following implementation as a baseline:

- TensorLayer/SRGAN: https://github.com/tensorlayer/SRGAN/tree/master

The linked upstream repository states that its code is for academic and
non-commercial use only. The present source-provenance comparison is documented
in `UPSTREAM_COMPARISON.md`.
