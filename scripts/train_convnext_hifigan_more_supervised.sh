python fish_vocoder/train.py task_name=convnext-hifigan \
    model/generator=convnext-hifigan \
    data.datasets.train.datasets.hifi-8000h.dataset.root=filelist.hifi-8000h.train \
    logger=tensorboard \
    model.feature_matching_loss=true \
    model.mel_loss=true \
    data.batch_size=4
