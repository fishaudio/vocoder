python fish_vocoder/train.py task_name=convnext-hifigan-base \
    model/generator=convnext-hifigan-base \
    model.num_frames=16 \
    data.batch_size=32 \
    logger=tensorboard
