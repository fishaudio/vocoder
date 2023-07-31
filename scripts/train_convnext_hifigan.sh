python fish_vocoder/train.py task_name=convnext-hifigan \
    model/generator=convnext-hifigan \
    data.datasets.train.root=dataset/hifi-8000h \
    data.datasets.train.root=filelist.train \
    logger=tensorboard
