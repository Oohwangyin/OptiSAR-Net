from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt')

# 关键参数说明：
# plots=True: 生成所有可视化图表（包括混淆矩阵）
# verbose=True: 打印每类的详细结果
# save_json=False: 不保存 JSON（避免跳过 print_results）
results = model.val(data='CDHD.yaml', batch=1, plots=True, verbose=True, save_json=False)