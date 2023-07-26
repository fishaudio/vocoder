python fish_vocoder/train.py task_name=bigvgan-base \
    model=bigvgan-base \
    data.datasets.train.root=dataset/hifi-8000h \
    data.datasets.train.transform._args_.3.crop_length=8192
