from ultralytics import YOLO

model = YOLO('runs/detect/OptiSAR-Net-HRSC2016-4class/weights/best.pt')

# 验证参数配置
results = model.val(
    data='HRSC2016_YOLO.yaml',
    batch=16,
    imgsz=640,
    device=0,
    split='test',  # 使用测试集进行评估
    plots=True,  # 生成可视化图表
    verbose=True,  # 打印每类的详细结果
    save_json=False,  # 不保存 JSON 结果
    save_txt=False,  # 不保存 TXT 预测结果
    save_conf=False,  # 不保存置信度
    save_crop=False,  # 不保存裁剪图像
    show_labels=True,  # 显示标签
    show_conf=True,  # 显示置信度
)

# 打印关键指标
print("\n" + "="*60)
print("验证结果汇总")
print("="*60)
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
print("="*60)
