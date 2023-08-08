python fish_vocoder/test.py task_name=convnext-hifigan-more-supervised \
    model/generator=convnext-hifigan \
    ckpt_path=other/convnext_hifigan_more_supervised/step_001560000.ckpt \
    'input_path="/mnt/nvme2/audio-data/DSD100"' \
    'output_path="results/DSD100"'
