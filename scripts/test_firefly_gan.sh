export CUDA_VISIBLE_DEVICES=0

python fish_vocoder/test.py task_name=firefly-gan-test \
    model/generator=firefly-gan-base \
    ckpt_path=checkpoints/firefly-gan-base.ckpt \
    'input_path="other"' \
    'output_path="results/test-audios"'
