CUDA_VISIBLE_DEVICES= python fish_vocoder/train.py task_name=convnext-hifigan-snake \
    model/generator=convnext-hifigan-snake \
    logger=tensorboard \
    model.feature_matching_loss=true \
    model.mel_loss=true \
    data.batch_size=4 \
    ckpt_path=other/convnext_hifigan_more_supervised/step_001180000.ckpt \
    resume_weights_only=true
