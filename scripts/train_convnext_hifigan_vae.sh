python fish_vocoder/train.py task_name=convnext-hifigan-vae \
    model=vae \
    model/generator=convnext-hifigan-vae \
    data.datasets.train.root=dataset/hifi-8000h \
    data.datasets.train.root=filelist.train
