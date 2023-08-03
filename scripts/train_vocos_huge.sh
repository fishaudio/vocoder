python fish_vocoder/train.py task_name=vocos-huge \
    model/generator=vocos-huge \
    data.batch_size=4 \
    data.datasets.train.datasets.hifi-8000h.dataset.root=filelist.hifi-8000h.train \
    logger=tensorboard
