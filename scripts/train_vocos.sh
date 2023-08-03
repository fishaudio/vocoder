python fish_vocoder/train.py task_name=vocos \
    model/generator=vocos \
    data.datasets.train.datasets.hifi-8000h.dataset.root=filelist.hifi-8000h.train \
    logger=tensorboard
