python fish_vocoder/train.py task_name=vocos \
    model/generator=vocos \
    data.datasets.train.root=dataset/hifi-8000h \
    data.datasets.train.root=filelist.train \
    logger=tensorboard
