model = dict(
    type='EfficientNet',
    in_channels=3,
    n_classes=5,
    scale=0,
    se_rate=0.25,
    drop_connect_rate=0.2,
    frozen_stages=-1)