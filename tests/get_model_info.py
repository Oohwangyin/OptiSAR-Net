#!/usr/bin/env python3
"""
通用模型信息查看脚本
支持：
- 官方 Ultralytics 模型（yolov8n.pt 等）
- 自定义模型（需要自定义模块在 ultralytics/nn/modules/ 下）
用法：
    python load_model_info.py <模型路径>
示例：
    python load_model_info.py yolov8n.pt
    python load_model_info.py ../runs/.../best.pt
"""

import sys
import torch
from ultralytics import YOLO


# ------------------------------------------------------------
# 1. 如果需要加载自定义模型，请取消下面两行的注释，并修改路径
# project_root = "/root/autodl-tmp/OptiSAR-Net"
# sys.path.insert(0, project_root)
# 然后导入自定义模块（若有）
# try:
#     from ultralytics.nn.modules import OptiSAR_Net_Module
# except ImportError:
#     pass
# ------------------------------------------------------------

def count_params_from_state_dict(state_dict):
    """从 state_dict 统计参数量（不依赖模型定义）"""
    total = 0
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and "num_batches_tracked" not in key:
            total += value.numel()
    return total


def get_model_info_from_pt(pt_path):
    """
    直接从 .pt 文件读取参数量，无需完整模型定义。
    返回参数量（int），GFLOPs（float或None）
    """
    ckpt = torch.load(pt_path, map_location="cpu")

    # 提取 state_dict
    if "model" in ckpt:
        model_obj = ckpt["model"]
        if hasattr(model_obj, "state_dict"):
            sd = model_obj.state_dict()
        else:
            sd = model_obj
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    total_params = count_params_from_state_dict(sd)

    # 尝试获取 GFLOPs（如果 ckpt 中保存了模型配置）
    gflops = None
    if "model" in ckpt and hasattr(ckpt["model"], "gflops"):
        gflops = ckpt["model"].gflops
    elif "train_args" in ckpt and "gflops" in ckpt["train_args"]:
        gflops = ckpt["train_args"]["gflops"]
    else:
        # 对于官方模型，可以尝试用 YOLO 加载获取
        try:
            tmp_model = YOLO(pt_path)
            if hasattr(tmp_model.model, "gflops"):
                gflops = tmp_model.model.gflops
        except:
            pass

    return total_params, gflops


def main():
    if len(sys.argv) < 2:
        print("用法: python load_model_info.py <模型路径>")
        sys.exit(1)

    model_path = sys.argv[1]

    # 方法一：尝试用 YOLO 加载（最准确，包含 GFLOPs）
    try:
        model = YOLO(model_path)
        # 获取参数量（单位：百万）
        params_m = model.model.params / 1e6
        gflops = model.model.gflops
        print(f"✅ YOLO 加载成功")
        print(f"   参数量: {params_m:.2f} M ({model.model.params:,})")
        print(f"   GFLOPs: {gflops:.1f}")
        # 可选：打印详细信息
        # model.info(verbose=False)
    except ModuleNotFoundError as e:
        print(f"⚠️ YOLO 加载失败（缺少自定义模块）: {e}")
        print("   改用 torch 直接统计参数量...")
        total_params, gflops = get_model_info_from_pt(model_path)
        print(f"   参数量: {total_params:,} ({total_params / 1e6:.2f} M)")
        if gflops is not None:
            print(f"   GFLOPs: {gflops:.1f}")
        else:
            print("   GFLOPs: 无法获取（需要完整模型定义）")
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        print("   尝试使用 torch 直接统计参数量...")
        total_params, gflops = get_model_info_from_pt(model_path)
        print(f"   参数量: {total_params:,} ({total_params / 1e6:.2f} M)")
        if gflops is not None:
            print(f"   GFLOPs: {gflops:.1f}")


if __name__ == "__main__":
    main()