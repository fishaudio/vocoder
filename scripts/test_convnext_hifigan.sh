python fish_vocoder/test.py task_name=convnext-hifigan \
    model/generator=convnext-hifigan \
    ckpt_path=other/convnext_hifigan/step_000300000.ckpt \
    'input_path="dataset/LibriTTS/test-other"' \
    'output_path="results/LibriTTS/test-other"'
