# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, ap_per_class, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.is_coco = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt")  # is COCO
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
        self.area_ranges = {"S": (0.0, 32.0**2), "M": (32.0**2, 96.0**2), "L": (96.0**2, float("inf"))}
        self.size_stats = {k: dict(tp=[], conf=[], pred_cls=[], target_cls=[]) for k in self.area_ranges}

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 7) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP75", "mAP50-95")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                self._update_size_stats(None, bbox, cls)
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            self._update_size_stats(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.nc
        )  # number of targets per class

        # 获取结果字典
        results_dict = self.metrics.results_dict

        # 保存结果为 CSV 文件（仿照训练过程）
        if not self.training:  # 只在独立验证时保存，不在训练过程中的验证保存
            results_dict.update(self._size_ap_results())
            csv_path = self.save_dir / "val_result.csv"
            keys = list(results_dict.keys())
            vals = list(results_dict.values())

            # 构建完整的列名和值
            headers = keys.copy()  # 指标列名
            values = [f"{v:.6f}" for v in vals]  # 指标值（保留6位小数）

            # 添加额外信息
            headers.append("Images")
            values.append(str(self.seen))

            headers.append("Instances")
            values.append(str(self.nt_per_class.sum()))

            for i, name in self.names.items():
                col_name = f"Class_{name}_instances"
                headers.append(col_name)
                values.append(str(self.nt_per_class[i]))

            # 写入 CSV（一行表头 + 一行数据）
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                f.write(",".join(headers) + "\n")
                f.write(",".join(values) + "\n")

            LOGGER.info(f"\nValidation results saved to: {csv_path}")

        return results_dict

    @staticmethod
    def _box_area(boxes):
        """Compute xyxy box areas."""
        if boxes.numel() == 0:
            return boxes.new_zeros(0)
        wh = (boxes[:, 2:4] - boxes[:, 0:2]).clamp(min=0)
        return wh[:, 0] * wh[:, 1]

    @staticmethod
    def _area_mask(area, area_range):
        """Return mask for a COCO-style area range."""
        low, high = area_range
        mask = area >= low
        if np.isfinite(high):
            mask &= area < high
        return mask

    def _update_size_stats(self, predn, gt_bboxes, gt_cls):
        """Collect AP statistics for small, medium and large objects."""
        predn = (
            predn
            if predn is not None
            else torch.zeros((0, 6), device=self.device, dtype=gt_bboxes.dtype if gt_bboxes.numel() else torch.float32)
        )
        pred_area = self._box_area(predn[:, :4])
        gt_area = self._box_area(gt_bboxes)

        for name, area_range in self.area_ranges.items():
            gt_mask = self._area_mask(gt_area, area_range)
            pred_mask = (
                self._area_mask(pred_area, area_range)
                if len(predn)
                else torch.zeros(0, dtype=torch.bool, device=self.device)
            )

            if len(predn) and len(gt_bboxes):
                gt_in, cls_in = gt_bboxes[gt_mask], gt_cls[gt_mask]
                gt_out, cls_out = gt_bboxes[~gt_mask], gt_cls[~gt_mask]
                in_overlap = torch.zeros(len(predn), dtype=torch.bool, device=self.device)

                if len(gt_in):
                    iou_in = box_iou(gt_in, predn[:, :4]) * (cls_in[:, None] == predn[:, 5]).float()
                    in_overlap = iou_in.max(0).values >= float(self.iouv[0])
                    pred_mask |= in_overlap

                if len(gt_out):
                    iou_out = box_iou(gt_out, predn[:, :4]) * (cls_out[:, None] == predn[:, 5]).float()
                    pred_mask &= ~((iou_out.max(0).values >= float(self.iouv[0])) & ~in_overlap)

            pred_eval = predn[pred_mask]
            stat = dict(
                conf=pred_eval[:, 4] if len(pred_eval) else torch.zeros(0, device=self.device),
                pred_cls=pred_eval[:, 5] if len(pred_eval) else torch.zeros(0, device=self.device),
                tp=torch.zeros(len(pred_eval), self.niou, dtype=torch.bool, device=self.device),
                target_cls=gt_cls[gt_mask],
            )
            if len(pred_eval) and gt_mask.any():
                stat["tp"] = self._process_batch(pred_eval, gt_bboxes[gt_mask], gt_cls[gt_mask])

            for k, v in stat.items():
                self.size_stats[name][k].append(v)

    def _size_ap_results(self):
        """Calculate AP_S, AP_M and AP_L from collected size-specific statistics."""
        results = {}
        for name, stat in self.size_stats.items():
            target_cls = torch.cat(stat["target_cls"], 0).cpu().numpy()
            if target_cls.size == 0:
                results[f"metrics/AP_{name}(B)"] = 0.0
                continue

            tp = torch.cat(stat["tp"], 0).cpu().numpy()
            conf = torch.cat(stat["conf"], 0).cpu().numpy()
            pred_cls = torch.cat(stat["pred_cls"], 0).cpu().numpy()
            ap = ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names=self.names)[5]
            results[f"metrics/AP_{name}(B)"] = float(ap.mean()) if ap.size else 0.0
        return results

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, "bbox")
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats["metrics/mAP50-95(B)"] = eval.stats[0]
                stats["metrics/mAP50(B)"] = eval.stats[1]
                stats["metrics/mAP75(B)"] = eval.stats[2]
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
