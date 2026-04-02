from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml")

# 训练参数配置 - 针对类别不平衡优化
model.train(
    data="M4-SAR-sampled_0.1.yaml",  # 使用 10% 子集进行快速验证
    epochs=200,  # 增加训练轮数，让小样本类别充分学习
    batch=16,
    imgsz=640,
    device=0,  # 使用 GPU
    workers=8,  # 数据加载线程数
    optimizer='AdamW',  # 优化器选择
    lr0=0.001,  # 提高初始学习率，帮助模型更快收敛
    lrf=0.01,  # 最终学习率 (lr0 * lrf)
    momentum=0.937,  # SGD momentum/Adam beta1
    weight_decay=0.0005,  # 权重衰减
    warmup_epochs=5.0,  # 增加预热轮数，稳定训练
    warmup_momentum=0.8,  # 预热动量
    box=7.5,  # box loss gain
    cls=3.0,  # cls loss gain (从 1.5 提高到 3.0，大幅增强分类损失权重)
    dfl=1.5,  # dfl loss gain
    # 数据增强策略 - 增强小样本可见性
    hsv_h=0.02,  # HSV-Hue augmentation (略微增加)
    hsv_s=0.8,  # HSV-Saturation augmentation (增加)
    hsv_v=0.5,  # HSV-Value augmentation (增加)
    degrees=15.0,  # rotation augmentation (遥感图像旋转不变性)
    translate=0.2,  # translation augmentation (增加)
    scale=0.8,  # scale augmentation (增强多尺度)
    shear=5.0,  # shear augmentation (轻微剪切)
    perspective=0.0,  # perspective augmentation
    flipud=0.1,  # probability of flip up-down (增加上下翻转)
    fliplr=0.5,  # probability of flip left-right
    mosaic=1.0,  # probability of mosaic augmentation (保持)
    mixup=0.15,  # probability of mixup augmentation (略微增加)
    copy_paste=0.5,  # probability of copy-paste augmentation (增加到 0.5，复制小样本目标)
    # 训练策略
    patience=100,  # 增加 patience，允许更长时间收敛
    save_period=10,  # Save checkpoint every x epochs
    seed=42,  # random seed
    exist_ok=False,  # overwrite existing experiment
    pretrained=True,  # use pretrained weights
    amp=True,  # automatic mixed precision training
    project='runs/detect',  # 保存项目的目录
    name='OptiSAR-Net-M4-SAR-10pct-balanced',  # 实验名称
)