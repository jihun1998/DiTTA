"""DiTTA on VSPW with a SegFormer-B5 ISS model, 10% warm-up (W2F protocol).

This is the setting reported as "DiTTA (Ours), 10%" in the paper.
"""

cfg = dict(
    # ---- data -------------------------------------------------------------
    data_root='data/vspw/VSPW_480p',
    split='val',
    # W2F: adapt on the first `warmup_ratio` percent of each video, then freeze
    # and evaluate on the rest.
    warmup_ratio=10,
    # clip fed to the model: frame offsets relative to the current frame.
    # (-1,) means the add-on attends from frame t to frame t-1.
    dilation=(-1,),

    # ---- ISS model --------------------------------------------------------
    model=dict(
        num_classes=124,
        embed_dim=256,
        dropout_ratio=0.1,
        window=5,          # radius of the add-on's local attention window
    ),

    # ---- SAM2 distillation targets ---------------------------------------
    sam2=dict(
        model_cfg='sam2_hiera_l.yaml',
        max_track=100,           # max objects prompted per video
        frame0_conf_thre=0.8,    # ISS confidence needed to prompt on frame 0
        frameN_conf_thre=0.8,    # ... and on later warm-up frames
        b=0.3,                   # lambda_area   (in-mask area weighting)
        c=0.8,                   # lambda_freq   (video-level frequency weighting)
    ),

    # ---- test-time adaptation --------------------------------------------
    tta=dict(
        lr=1e-3,
        iters=5,                 # passes over the warm-up frames
        opt_params='decoder',    # decoder | cls_head | addon
        momentum=0.99,           # EMA rate of the momentum encoder
        temperature=1.0,         # T in the contrastive loss
        soft_contra=True,        # weight prototypes by the reliability map
    ),

    # ---- evaluation -------------------------------------------------------
    evaluation=dict(
        tau=0.8,                 # reliability threshold for logit fusion
        vc_windows=(8, 16),      # mVC_8 / mVC_16
    ),
)
