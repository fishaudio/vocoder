python fish_vocoder/train.py task_name=convnext-hifigan \
    model/generator=convnext-hifigan \
    data.datasets.train.datasets.hifi-8000h.dataset.root=filelist.train \
    logger=tensorboard
