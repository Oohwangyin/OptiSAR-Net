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
    hsv_h=0.0,  # SAR 没有色调信息，禁用
    hsv_s=0.0,  # 饱和度不变
    hsv_v=0.0,  # 亮度变化会改变后向散射特性，谨慎使用
    degrees=0.0,  # SAR 几何校正后不应旋转
    translate=0.05,  # 微小平移增强泛化
    scale=0.3,  # 尺度变化（SAR 目标尺度变化较大）
    shear=0.0,  # 剪切会破坏几何精度
    perspective=0.0,  # 透视变换不适用
    flipud=0.0,  # SAR 垂直翻转会改变目标方向（一般不使用）
    fliplr=0.3,  # 水平翻转（如果目标方向无关）
    mosaic=0.5,  # mosaic 仍有效，但降低概率避免引入过多噪声
    mixup=0.0,  # mixup 可能混合 SAR 目标与背景，效果不确定
    copy_paste=0.0,  # 复制粘贴会破坏 SAR 背景一致性
    patience=50,  # 小数据集可以适当减少 patience
    save_period=10,  # Save checkpoint every x epochs
    seed=42,  # random seed
    exist_ok=False,  # overwrite existing experiment
    pretrained=True,  # use pretrained weights
    amp=True,  # automatic mixed precision training
    project='runs/detect',  # 保存项目的目录
    name='OptiSAR-Net-M4-SAR-10pct-balanced',  # 实验名称
)