# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import itertools
import logging
import os
import re
from collections import OrderedDict

import imageio.v3
import pycocotools.mask
import torch
import numpy as np
from detectron2.structures import BoxMode
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
import xml.etree.ElementTree as ET

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator


def register_custom_voc(cfg):
    CLASS_NAMES = (
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    )

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    dirname = os.path.join(_root, "VOC2012")

    def _load_custom_voc_instances(dirname, split, class_names):
        """
            Load Pascal VOC instance segmentation annotations + gaze data to Detectron2 format.

            Args:
                dirname: Contain "Annotations", "ImageSets", "JPEGImages", "random_gaze"
                split (str): one of "train", "val"
                class_names: list or tuple of class names
            """
        base_folder = Path(dirname, "random_gaze", split)
        annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
        bbox_regex = re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")
        dicts_by_id = {}

        for class_path in base_folder.iterdir():
            cls_name = class_path.stem

            cls_instance_mask_files = list(Path(class_path, "masks").glob("*.png"))

            for cls_instance_mask_file in cls_instance_mask_files:
                fileid = "_".join(cls_instance_mask_file.stem.split("_")[:2])
                anno_file = os.path.join(annotation_dirname, fileid + ".xml")
                jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

                with PathManager.open(anno_file) as f:
                    tree = ET.parse(f)

                if fileid not in dicts_by_id.keys():
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                        "annotations": []
                    }
                    dicts_by_id[fileid] = r

                bbox_wrong_order = list(map(float, bbox_regex.match(str(cls_instance_mask_file.stem)).groups()))
                bbox = [bbox_wrong_order[2], bbox_wrong_order[0], bbox_wrong_order[3], bbox_wrong_order[1]]

                mask_array = imageio.v3.imread(cls_instance_mask_file)
                if len(mask_array.shape) == 3:
                    mask_array = np.sum(mask_array, axis=2, dtype=int)
                mask_array = mask_array.astype(bool).astype(np.uint8)
                segmentation_rle_dict = pycocotools.mask.encode(np.asarray(mask_array, order="F"))
                assert type(segmentation_rle_dict) == dict, type(segmentation_rle_dict)

                ann_dict = {"category_id": class_names.index(cls_name), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": segmentation_rle_dict,
                            }

                if split == "train":
                    pseudo_mask_file = Path(cls_instance_mask_file.parent.parent, cfg.MODEL.GAZEINST.GAZE_LOSS_LABEL,
                                            cls_instance_mask_file.name)
                    if not pseudo_mask_file.is_file():
                        pseudo_mask_array = np.zeros_like(mask_array)
                    else:
                        pseudo_mask_array = imageio.v3.imread(pseudo_mask_file)
                    if len(pseudo_mask_array.shape) == 3:
                        pseudo_mask_array = np.sum(pseudo_mask_array[:,:,:3], axis=2, dtype=int)
                    pseudo_mask_array = pseudo_mask_array.astype(bool).astype(np.uint8)
                    pseudo_segmentation_rle_dict = pycocotools.mask.encode(np.asarray(pseudo_mask_array, order="F"))

                    ann_dict["gaze_segmentation"] = pseudo_segmentation_rle_dict

                dicts_by_id[fileid]["annotations"].append(ann_dict)

        dicts = dicts_by_id.values()
        return dicts

    DatasetCatalog.register("voc_2012_gaze_train",
                            lambda: _load_custom_voc_instances(dirname, split="train", class_names=CLASS_NAMES))
    MetadataCatalog.get("voc_2012_gaze_train").set(
        thing_classes=list(CLASS_NAMES), dirname=dirname, year=2012, split="train"
    )
    MetadataCatalog.get("voc_2012_gaze_train").evaluator_type = "coco"
    DatasetCatalog.register("voc_2012_gaze_val",
                            lambda: _load_custom_voc_instances(dirname, split="val", class_names=CLASS_NAMES))
    MetadataCatalog.get("voc_2012_gaze_val").set(
        thing_classes=list(CLASS_NAMES), dirname=dirname, year=2012, split="val"
    )
    MetadataCatalog.get("voc_2012_gaze_val").evaluator_type = "coco"


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """

    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.MODEL.FCPOSE_ON:
            mapper = FCPoseDatasetMapper(cfg, True)
        else:
            mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)
    register_custom_voc(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)  # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
