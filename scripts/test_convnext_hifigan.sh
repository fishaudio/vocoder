python fish_vocoder/test.py task_name=convnext-hifigan \
    model/generator=convnext-hifigan \
    ckpt_path=other/convnext_hifigan/step_000600000.ckpt \
    'input_path="other"' \
    'output_path="results/test-audios"'
