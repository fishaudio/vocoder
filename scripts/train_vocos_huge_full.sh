python fish_vocoder/train.py task_name=vocos-huge-full \
    model/generator=vocos-huge \
    model.num_mels=160 \
    model.mel_transforms.modules.input.f_min=0 \
    model.mel_transforms.modules.input.f_max=22050 \
    data.batch_size=4 \
    trainer.precision=16-mixed \
    logger=tensorboard
