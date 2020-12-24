loss = dict(
    type='CrossEntropyLoss')

lr = 1e-4
optimizer = dict(
    type='Adam',
    lr=lr,
    weight_decay=1e-6)

lr_config = dict(
    type='CosineAnnealingLR',
    T_max=10,
    eta_min=0.01 * lr)

output_transforms = dict(
    accuracy=dict(
        type='ActivatedTransform'))

trainer = dict(
    amp_backend='native',
    auto_select_gpus=True,
    gradient_clip_val=0.5,
    log_every_n_steps=50,
    precision=32,
    max_epochs=10,
    accelerator=None,
    move_metrics_to_cpu=False)

callbacks = dict(
    patience=4)