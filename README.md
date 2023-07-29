# Fish Vocoder

This repo is designed as an uniform interface for developing various vocoders.

Configs:
- [x] hifigan (baseline): HiFiGAN generator with UnivNet discriminators.
- [x] bigvgan: BigVGAN generator.
- [x] vocos: Vocos (ConvNext) generator.
- [ ] hifigan-mms: HiFiGAN generator with UnivNet discriminators and replaced mel-spectrogram with MMS model.
- [ ] hifigan-vae: HiFiGAN generator with UnivNet discriminators and replaced mel-spectrogram with VAE model.

## References
- TIMM: https://github.com/huggingface/pytorch-image-models
- BigVGAN: https://github.com/NVIDIA/BigVGAN
- Vocos: https://github.com/charactr-platform/vocos
- UnivNet: https://github.com/mindslab-ai/univnet
- ConvNext: https://github.com/facebookresearch/ConvNeXt
- HiFiGAN: https://github.com/jik876/hifi-gan
- Fish Diffusion: https://github.com/fishaudio/fish-diffusion
- RefineGAN: https://arxiv.org/abs/2111.00962
- Encodec: https://github.com/facebookresearch/encodec
