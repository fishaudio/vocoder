python fish_vocoder/train.py task_name=vocos-huge \
    model/generator=vocos-huge \
    data.batch_size=4 \
    data.datasets.train.root=dataset/hifi-8000h \
    data.datasets.train.root=filelist.train \
    logger=tensorboard
