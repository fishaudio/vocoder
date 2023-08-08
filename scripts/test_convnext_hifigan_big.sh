python fish_vocoder/test.py task_name=convnext-hifigan-more-supervised \
    model/generator=convnext-hifigan-big \
    model.num_mels=128 \
    model.mel_transforms.modules.input.f_min=40 \
    model.mel_transforms.modules.input.f_max=16000 \
    ckpt_path=other/convnext_hifigan_more_supervised/step_001560000.ckpt \
    'input_path="other"' \
    'output_path="results/test-audios"'
