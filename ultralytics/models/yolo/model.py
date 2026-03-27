# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import yaml_load, ROOT


class YOLO(Model):
    # task：任务类型（detect/segment/classify/pose）
    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """如果文件名包含-world，会切换到YOLOWorld模型"""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """
        返回任务类型到具体类的映射关系
        不同任务（检测/分隔/分类等）对应不同的模型、训练器、验证器、预测器
        """
        return {
            # 图像分类任务映射      用于单标签或多标签图像分类
            "classify": {
                "model": ClassificationModel,                           #分类模型类
                "trainer": yolo.classify.ClassificationTrainer,         #分类训练器
                "validator": yolo.classify.ClassificationValidator,     #分类验证器
                "predictor": yolo.classify.ClassificationPredictor,     #分类预测器
            },
            # 目标检测任务映射（OptiSAR-Net 使用）      用于检测并定位图像中的多个目标
            "detect": {
                "model": DetectionModel,                                 #分类模型类
                "trainer": yolo.detect.DetectionTrainer,                 #检测训练器（管理训练过程）
                "validator": yolo.detect.DetectionValidator,             #检测验证器（评估 mAP 等指标）
                "predictor": yolo.detect.DetectionPredictor,             #检测预测器（执行推理）
            },
            # 目标分割任务映射      不仅要检测目标，还要生成像素级掩码
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            # 姿态估计任务映射      用于检测人体关键点（如关节位置）
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            # 旋转框检测任务映射     用于检测有方向的目标（如船舶、文字等）
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt") -> None:
        """
        Initializes the YOLOv8-World model with the given pre-trained model file. Supports *.pt and *.yaml formats.

        Args:
            model (str | Path): Path to the pre-trained model. Defaults to 'yolov8s-world.pt'.
        """
        super().__init__(model=model, task="detect")

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes
