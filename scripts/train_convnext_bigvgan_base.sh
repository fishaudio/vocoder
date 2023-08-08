python fish_vocoder/train.py task_name=convnext-bigvgan-base \
    model/generator=convnext-bigvgan-base \
    model.num_frames=16 \
    data.batch_size=32 \
    logger=tensorboard
