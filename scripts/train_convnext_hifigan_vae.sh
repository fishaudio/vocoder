python fish_vocoder/train.py task_name=convnext-hifigan-vae \
    model=vae \
    model/generator=convnext-hifigan-vae \
    data.datasets.train.datasets.hifi-8000h.dataset.root=filelist.hifi-8000h.train
