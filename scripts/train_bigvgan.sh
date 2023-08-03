python fish_vocoder/train.py task_name=convnext-bigvgan \
    model/generator=convnext-bigvgan \
    data.datasets.train.datasets.hifi-8000h.dataset.root=filelist.hifi-8000h.train \
    model.num_frames=16 \
    logger=tensorboard
