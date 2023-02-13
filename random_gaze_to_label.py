import itertools
import json
import math
from pathlib import Path
import re

import scipy.signal
import skimage
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.segmentation import slic, random_walker
from skimage.color import rgb2lab
from tqdm import tqdm
from imageio.v3 import imread, imwrite
from skimage.morphology import skeletonize, disk
import matplotlib.pyplot as plt

import numpy as np
import scipy.spatial.distance as dist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix
from scipy.signal import find_peaks
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax
from sklearn.cluster import DBSCAN


def _plot_seg_statistics_from_json(base_path, ignore_dice=True, ignore_iou=True):
    plot_data_dict = {}

    for folder_path in [path for path in Path(base_path).glob("*") if path.is_dir()]:
        stat_json_dict = json.load(open(Path(folder_path, "statistics.json")))
        plot_data_dict[folder_path.name] = stat_json_dict

    num_plots = len(plot_data_dict.keys())
    key_list = list(plot_data_dict.keys())[::-1]

    num_cols = int(math.ceil(math.sqrt(num_plots)))
    num_rows = int(math.ceil(num_plots / float(num_cols)))

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(19.2, 10.8))

    for i in tqdm(range(num_plots)):
        row_idx = i // num_cols
        col_idx = i % num_cols

        cur_ax = ax[row_idx, col_idx]
        cls_str = key_list[i]
        data_dict = plot_data_dict[key_list[i]]

        for key, value in data_dict.items():
            if ignore_dice and "dice" in key:
                continue
            if ignore_iou and "iou" in key:
                continue
            if "gaze" not in key and "fix" not in key:
                continue
            cur_ax.hist(value, bins=np.linspace(0, 1., 41), label=f"{key} ({np.nanmean(value):.3f})", alpha=1.0, histtype="step")
        cur_ax.set_title(cls_str)
        cur_ax.legend(loc="upper left" if (ignore_iou and ignore_dice) else "upper center")

    plt.subplots_adjust(wspace=0.15, hspace=0.2, left=0.05, top=0.95, right=0.95, bottom=0.05)
    fig.savefig(Path(base_path, "class_base_statistics.pdf"))

    fig.clear()
    plt.close()

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(19.2, 10.8))

    for i in tqdm(range(num_plots)):
        row_idx = i // num_cols
        col_idx = i % num_cols

        cur_ax = ax[row_idx, col_idx]
        cls_str = key_list[i]
        data_dict = plot_data_dict[key_list[i]]

        for key, value in data_dict.items():
            if ignore_dice and "dice" in key:
                continue
            if ignore_iou and "iou" in key:
                continue
            if "mst" not in key:
                continue
            cur_ax.hist(value, bins=np.linspace(0, 1., 41), label=f"{key} ({np.nanmean(value):.3f})", alpha=1.0, histtype="step")
        cur_ax.set_title(cls_str)
        cur_ax.legend(loc="upper left" if (ignore_iou and ignore_dice) else "upper center")

    plt.subplots_adjust(wspace=0.15, hspace=0.2, left=0.05, top=0.95, right=0.95, bottom=0.05)
    fig.savefig(Path(base_path, "class_mst_statistics.pdf"))

    fig.clear()
    plt.close()

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(19.2, 10.8))

    for i in tqdm(range(num_plots)):
        row_idx = i // num_cols
        col_idx = i % num_cols

        cur_ax = ax[row_idx, col_idx]
        cls_str = key_list[i]
        data_dict = plot_data_dict[key_list[i]]

        for key, value in data_dict.items():
            if "crf" not in key or "rw" in key:
                continue
            cur_ax.hist(value, bins=np.linspace(0, 1., 41), label=f"{key} ({np.nanmean(value):.3f})", alpha=1.0, histtype="step")
        cur_ax.set_title(cls_str)
        cur_ax.legend(loc="upper center")

    plt.subplots_adjust(wspace=0.15, hspace=0.2, left=0.05, top=0.95, right=0.95, bottom=0.05)
    fig.savefig(Path(base_path, "class_crf_statistics.pdf"))

    fig.clear()
    plt.close()

    fig, ax = plt.subplots()
    metric_keys = list(set(list(itertools.chain(*[list(d.keys()) for d in plot_data_dict.values()]))))[::-1]

    for key in metric_keys:
        if ignore_dice and "dice" in key:
            continue
        if ignore_iou and "iou" in key:
            continue
        data = []
        for d in plot_data_dict.values():
            data += d.get(key, [])

        ax.hist(data, bins=100, label=key, alpha=1.0, histtype="step")
    ax.legend(loc="upper center")
    ax.set_title("Dataset-wide Statistics")

    fig.savefig(Path(base_path, "dataset_statistics.pdf"))


def _xiaoline(x0, y0, x1, y1):
    x = []
    y = []
    dx = x1 - x0
    dy = y1 - y0
    steep = abs(dx) < abs(dy)

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dy, dx = dx, dy

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    gradient = float(dy) / float(dx)  # slope

    """ handle first endpoint """
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xpxl0 = int(xend)
    ypxl0 = int(yend)
    x.append(xpxl0)
    y.append(ypxl0)
    x.append(xpxl0)
    y.append(ypxl0 + 1)
    intery = yend + gradient

    """ handles the second point """
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xpxl1 = int(xend)
    ypxl1 = int(yend)
    x.append(xpxl1)
    y.append(ypxl1)
    x.append(xpxl1)
    y.append(ypxl1 + 1)

    """ main loop """
    for px in range(xpxl0 + 1, xpxl1):
        x.append(px)
        y.append(int(intery))
        x.append(px)
        y.append(int(intery) + 1)
        intery = intery + gradient

    if steep:
        y, x = x, y
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    # move endpoint to end, such that points are in correct order
    start_x = x[:2]
    start_y = y[:2]
    end_x = x[2:4]
    end_y = y[2:4]
    x = start_x + x[4:] + end_x
    y = start_y + y[4:] + end_y

    x = list(np.clip(x, min(x0, x1), max(x0, x1)))
    y = list(np.clip(y, min(y0, y1), max(y0, y1)))

    coords = np.array(list(zip(y, x)))

    return coords


def _calc_mst_from_points(points, img):
    colors = img[points[:, 0], points[:, 1]]
    pwcolor_dists = dist.pdist(rgb2lab(colors), "seuclidean")
    color_dist_matrix = dist.squareform(pwcolor_dists)

    pwdists = dist.pdist(points, "seuclidean")
    eucl_dist_matrix = dist.squareform(pwdists)

    dist_matrix = eucl_dist_matrix + 0.5 * color_dist_matrix

    sparse_dist_matrix = csr_matrix(dist_matrix)

    mst = minimum_spanning_tree(sparse_dist_matrix)
    mst_matrix = mst.toarray()
    """
    mst_matrix[mst_matrix > 2 * np.std(mst_matrix[np.nonzero(mst_matrix)])] = 0.

    n_comps, label = connected_components(csr_matrix(mst_matrix), directed=False)
    counts = [np.count_nonzero(label[label == i]) for i in list(np.unique(label))]

    for idx in range(len(counts)):
        if counts[idx] <= 5:
            # delete all adjacencies for nodes whose conn. components are too small
            mst_matrix[np.argwhere(label == idx), :] = 0.
            mst_matrix[:, np.argwhere(label == idx)] = 0.
    """

    mst_entries = np.array(np.nonzero(mst_matrix)).T
    return mst_entries


def _get_edge_coords_from_mst_idx_pair(idx_pair, points, skip_small_edges=False):
    start_coord = points[idx_pair[0]]
    end_coord = points[idx_pair[1]]

    (start_y, start_x), (end_y, end_x) = start_coord, end_coord

    if skip_small_edges and np.sqrt(np.square(start_x - end_x) + np.square(start_y - end_y)) < 5:
        # fixations very close together, skip
        return None

    if start_x > end_x:
        start_x, start_y, end_x, end_y = end_x, end_y, start_x, start_y
        start_coord, end_coord = end_coord, start_coord

    unique_coords = _xiaoline(start_x, start_y, end_x, end_y)

    return unique_coords


def _calc_and_draw_mst_from_points(points, canvas, img, edges):
    mst_entries = _calc_mst_from_points(points, img)

    additional_points = []

    for idx_pair in mst_entries:
        unique_coords = _get_edge_coords_from_mst_idx_pair(idx_pair, points, skip_small_edges=True)
        # pair is too close together (eucl. dist of less than 5 in pixel space)
        if unique_coords is None:
            continue

        edge_intensities = edges[unique_coords[:, 0], unique_coords[:, 1]]

        peaks, properties = find_peaks(edge_intensities,
                                       prominence=(0.07, None),
                                       width=(5, 15),
                                       height=0.15,
                                       distance=5
                                       )

        additional_points += [unique_coords[idx] for idx in peaks]

    if len(additional_points) > 0:
        points = np.concatenate([points, np.array(additional_points)], axis=0)

    mst_entries = _calc_mst_from_points(points, img)

    if len(mst_entries) == 0:
        return mst_entries

    edge_avg_color_list = []
    for idx_pair in mst_entries:
        unique_coords = _get_edge_coords_from_mst_idx_pair(idx_pair, points)
        edge_colors = img[unique_coords[:, 0], unique_coords[:, 1]]

        edge_avg_color_list.append(np.array(edge_colors).mean(axis=0).astype(np.uint8))

    lab_colors = rgb2lab(np.array(edge_avg_color_list))
    # rescale dimensions of lab color space to same range [0, 255]
    lab_colors[:, 0] *= 1.55
    lab_colors[:, 1:] += 128
    color_dbscan = DBSCAN(eps=20, min_samples=max(int(len(lab_colors) * 0.15), 2))

    edge_labels = color_dbscan.fit_predict(lab_colors)

    valid_mst_entries = mst_entries[edge_labels != -1]

    for idx_pair in valid_mst_entries:
        start_coord = points[idx_pair[0]]
        end_coord = points[idx_pair[1]]

        cv2.line(canvas, start_coord[::-1], end_coord[::-1], color=[0, 255, 0])

    return mst_entries


def draw_spanning_tree_from_random_gaze_data(base_path, debug=True):
    """
    Draws spanning tree for all gaze points on canvas and saves it as a binary mask
    :param base_path: Path to class folder with random gaze
    :return: None
    """

    file_names = list(Path(base_path, "original").glob("*.png"))
    Path(base_path, "spanning_tree").mkdir(parents=True, exist_ok=True)
    if debug:
        Path(base_path, "spanning_tree_debug").mkdir(parents=True, exist_ok=True)

    bbox_regex = re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")

    for file_name in tqdm(file_names, total=len(file_names)):

        bbox_string = file_name.stem

        match = bbox_regex.match(bbox_string)
        # swapped around intentionally to match heuristic in other methods: y refers to vertical axis,
        # which is first axis in array display; x to horizontal
        y_min, y_max, x_min, x_max = map(int, match.groups())
        file_name = file_name.name

        img_path = Path(base_path, "original", file_name)
        gaze_path = Path(base_path, "gaze_images", file_name)

        img = imread(img_path)
        gaze_img = imread(gaze_path)

        gaze_points = np.array(np.nonzero(gaze_img)).T

        gaze_points = np.array(
            list(filter(lambda point: (y_min <= point[0] <= y_max) and (x_min <= point[1] <= x_max), gaze_points)))

        if len(gaze_points) != 0:
            tree_canvas = np.zeros(img.shape, dtype=np.uint8)
            _calc_and_draw_mst_from_points(gaze_points, tree_canvas, img)

            tree_mask = tree_canvas.sum(axis=2)
        else:
            print("WARNING: 0 gaze points in bbox detected. Saving empty mask.")
            tree_canvas = np.zeros(img.shape, dtype=np.uint8)
            tree_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        imwrite(Path(base_path, "spanning_tree", file_name), tree_mask)
        if debug:
            cv2.rectangle(tree_canvas, (x_min, y_min), (x_max, y_max), color=[255, 100, 0])
            plt.imsave(Path(base_path, "spanning_tree_debug", file_name),
                       cv2.addWeighted(tree_canvas, 1., img, 0.5, gamma=1))


def construct_fine_graph(base_path, debug=True, use_gt=False):
    """
        Constructs dense graph for all fixation points with intermediate nodes on color edges
        :param base_path: Path to class folder with random gaze
        :return: None
        """

    stat_dict = json.load(open(Path(base_path, "statistics.json"), "r")) if Path(base_path,
                                                                                 "statistics.json").exists() else {}

    file_names = list(Path(base_path, "original").glob("*.png"))
    out_path = Path(base_path, "dense_graph" if not use_gt else "gt_dense_graph")
    out_path.mkdir(parents=True, exist_ok=True)
    if debug:
        out_debug_path = Path(base_path, "dense_graph_debug" if not use_gt else "gt_dense_graph_debug")
        out_debug_path.mkdir(parents=True, exist_ok=True)

    bbox_regex = re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")

    precision_values = []
    fixation_precision_values = []
    gaze_precision_values = []

    for file_name in tqdm(file_names, total=len(file_names)):

        bbox_string = file_name.stem

        match = bbox_regex.match(bbox_string)
        # swapped around intentionally to match heuristic in other methods: y refers to vertical axis,
        # which is first axis in array display; x to horizontal
        y_min, y_max, x_min, x_max = map(int, match.groups())
        file_name = file_name.name

        img_path = Path(base_path, "original", file_name)
        fixation_path = Path(base_path, "fixation_images", file_name)
        gaze_path = Path(base_path, "gaze_images", file_name)
        mask_path = Path(base_path, "masks", file_name)

        img = imread(img_path)
        mask_img = imread(mask_path)
        mask = np.sum(mask_img, axis=2).astype(bool)

        bbox_mask = np.zeros_like(mask)
        bbox_mask[y_min:y_max, x_min:x_max] = True

        channel_image = img if img.ndim == 3 else np.stack([img] * 3, axis=2)
        edge_detector = cv2.ximgproc.createStructuredEdgeDetection(
            "/home/daniel/PycharmProjects/bbox_segmentation/structured_edge_detection_model.yml.gz")
        edges = edge_detector.detectEdges(np.float32(channel_image) / 255)
        fixation_img = imread(fixation_path)
        gaze_img = imread(gaze_path)

        dense_graph_canvas = np.zeros(channel_image.shape, dtype=np.uint8)

        if not use_gt:
            fixation_points = np.array(np.nonzero(fixation_img)).T
            gaze_points = np.array(np.nonzero(gaze_img)).T

            fixation_points = np.array(
                list(filter(lambda point: (y_min <= point[0] <= y_max) and (x_min <= point[1] <= x_max),
                            fixation_points)))
            gaze_points = np.array(
                list(filter(lambda point: (y_min <= point[0] <= y_max) and (x_min <= point[1] <= x_max), gaze_points)))

            gaze_tp = mask[gaze_points[:, 0], gaze_points[:, 1]].sum() if len(gaze_points.shape) == 2 else 0
            gaze_precision_values.append(gaze_tp / gaze_points.shape[0] if len(gaze_points) > 0 else 0)

            if len(fixation_points) == 0:
                print("WARNING: 0 gaze points in bbox detected. Saving empty mask.")
                tree_mask = np.zeros(img.shape[:2], dtype=np.uint8)

                imwrite(Path(base_path, "dense_graph", file_name), dense_graph_canvas.sum(axis=2).astype(np.uint8))

                tp = np.sum(np.logical_and(tree_mask, mask))
                fp = np.sum(tree_mask) - tp

                prec = tp / (tp + fp)

                precision_values.append(prec)
                continue

            fixation_colors = img[fixation_points[:, 0], fixation_points[:, 1]]
            fixation_colors = rgb2lab(fixation_colors)
            # rescale dimensions of lab color space to same range [0, 255]
            fixation_colors[:, 0] *= 1.55
            fixation_colors[:, 1:] += 128
            color_dbscan = DBSCAN(eps=15, min_samples=2)

            color_labels = color_dbscan.fit_predict(fixation_colors)

            # only remove 'outlier' if a cluster was found
            outlier_removed_fixations = fixation_points[color_labels != -1] if len(
                np.unique(color_labels)) > 1 else fixation_points

            fixation_points = outlier_removed_fixations

            # calculate precision before clustering, since this is only done to reduce computations
            # for finding new vertices on image edges
            tp = mask[fixation_points[:, 0], fixation_points[:, 1]].sum()
            fixation_precision_values.append(tp / fixation_points.shape[0])
        else:
            fixation_img = np.where(mask, fixation_img, 0.)
            fixation_points = np.array(np.nonzero(fixation_img)).T

        if debug:
            debug_canvas = np.zeros(channel_image.shape, dtype=np.uint8)
            for point in fixation_points:
                cv2.drawMarker(debug_canvas, (point[1], point[0]), [0, 255, 0], markerSize=5)

        """
        dbscan = DBSCAN(eps=5, min_samples=2)

        labels = dbscan.fit_predict(fixation_points)

        core_fixations = [np.mean(fixation_points[labels == i], axis=0).astype(int) for i in np.unique(labels) if i != -1]
        core_fixations.extend(list(fixation_points[labels == -1]))

        core_fixations = np.array(core_fixations)

        segment_dicts = []

        for (start_y, start_x), (end_y, end_x) in itertools.combinations(core_fixations, r=2):

            if np.sqrt(np.square(start_x - end_x) + np.square(start_y - end_y)) < 5:
                # fixations very close together, skip
                continue

            if start_x > end_x:
                start_x, start_y, end_x, end_y = end_x, end_y, start_x, start_y

            unique_coords = _xiaoline(start_x, start_y, end_x, end_y)

            edge_intensities = edges[unique_coords[:, 0], unique_coords[:, 1]]

            if debug:
                debug_canvas[unique_coords[:, 0], unique_coords[:, 1]] = [0, 100, 255]

            peaks, properties = find_peaks(edge_intensities,
                                           prominence=(0.1, None),
                                           width=(5, 15),
                                           )

            segment_dicts.append({"start": (start_y, start_x),
                                  "end": (end_y, end_x),
                                  "line": unique_coords,
                                  "edge_intensities": edge_intensities,
                                  "peaks": peaks,
                                  "peak_properties": properties})

        additional_graph_points = list(
            itertools.chain(*[[d["line"][idx] for idx in d["peaks"] if len(d["peaks"])] for d in segment_dicts]))

        
        all_points_array = np.array(list(fixation_points) + additional_graph_points)
        _calc_and_draw_mst_from_points(all_points_array, dense_graph_canvas, img, edges)
        """

        _calc_and_draw_mst_from_points(fixation_points, dense_graph_canvas, img, edges)

        tree_mask = dense_graph_canvas.sum(axis=2).astype(bool)
        imwrite(Path(out_path, file_name), dense_graph_canvas.sum(axis=2).astype(np.uint8))

        if debug:
            plt.imsave(Path(out_debug_path, file_name),
                       cv2.addWeighted(dense_graph_canvas, 1., img, 0.5, gamma=1))

        tp = np.sum(np.logical_and(tree_mask, mask))
        fp = np.sum(tree_mask) - tp

        prec = tp / (tp + fp)

        precision_values.append(prec)

        """
        superpixels = slic(channel_image, n_segments=1000, compactness=1, sigma=0.5, channel_axis=2, mask=bbox_mask)

        labels = np.unique(superpixels[tree_mask])
        if 0 in labels:
            labels_ = list(labels)
            labels_.remove(0)
            labels = np.array(labels_)

        super_pixel_segmentation = np.isin(superpixels, labels)
        super_pixel_segmentation_canvas = np.where(super_pixel_segmentation, 255, 0).astype(np.uint8)
        super_pixel_segmentation_canvas = np.stack([super_pixel_segmentation_canvas] * 3, axis=2)
        if debug:
            plt.imsave(Path(base_path, "spp_small_debug", file_name),
                       cv2.addWeighted(super_pixel_segmentation_canvas, 0.5, img, 0.5, gamma=1))

        plt.imsave(Path(base_path, "spp_small", file_name),
                   super_pixel_segmentation_canvas)

        dice = (2 * np.sum(super_pixel_segmentation * mask)) / (np.sum(super_pixel_segmentation) + np.sum(mask))
        seg_precision = np.sum(super_pixel_segmentation * mask) / (np.sum(super_pixel_segmentation))

        superpixel_dice.append(dice)
        superpixel_prec.append(seg_precision)
        """

    if debug:
        fig, ax = plt.subplots()
        ax.hist(precision_values, bins=25, label="Tree Precision")
        ax.hist(fixation_precision_values, bins=25, label="Fixation Precision")
        ax.hist(gaze_precision_values, bins=25, label="Gaze Precision")
        # ax.hist(superpixel_dice, bins=25, label="Superpixel Dice")
        # ax.hist(superpixel_prec, bins=25, label="Superpixel Precision")
        ax.legend()
        ax.set_title(
            f"Average Precision: {np.nanmean(precision_values):.3f} (Fixation: {np.nanmean(fixation_precision_values):.3f}, Gaze: {np.nanmean(gaze_precision_values):.3f})")
        fig.savefig(Path(base_path, "label_statistics_dense_to_mst.png"))
    stat_dict["mst_prec" if not use_gt else "gt_mst_prec"] = precision_values
    if len(fixation_precision_values):
        stat_dict["fix_prec"] = fixation_precision_values
    if len(gaze_precision_values):
        stat_dict["gaze_prec"] = gaze_precision_values

    json.dump(stat_dict, open(Path(base_path, "statistics.json"), "w"))


def _create_gt_skeletons(root_path):
    """
    root_path: Path to folder containing class folders
    """

    cls_paths = [path for path in Path(root_path).glob("*") if path.is_dir()]

    for cls_path in cls_paths:

        mask_base_path = Path(cls_path, "masks")
        skel_path = Path(cls_path, "gt_skeletons")
        skel_path.mkdir(parents=True, exist_ok=True)

        for mask_path in tqdm(mask_base_path.glob("*.png")):
            file_name = mask_path.name

            mask = imread(mask_path)
            mask = mask.sum(axis=2).astype(bool) if len(mask.shape) == 3 else mask.astype(bool)

            skel = skeletonize(mask)
            skel_img = np.stack([skel.astype(np.uint8) * 255] * 3, axis=2)

            imwrite(Path(skel_path, file_name), skel_img)


def _create_rw_segmentation_from_dense_graph(root_path):
    """
    root_path: Path to folder containing class folders
    """
    bbox_regex = re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")
    cls_paths = [path for path in Path(root_path).glob("*") if path.is_dir()]

    for cls_path in cls_paths:

        dense_graph_base_path = Path(cls_path, "dense_graph")
        masks_base_path = Path(cls_path, "masks")
        rw_path = Path(cls_path, "rw")
        rw_path.mkdir(parents=True, exist_ok=True)

        stat_dict = json.load(open(Path(cls_path, "statistics.json"), "r")) if Path(cls_path,
                                                                                    "statistics.json").exists() else {}

        rw_iou_values = []
        rw_prec_values = []

        for mask_path in tqdm(masks_base_path.glob("*.png")):
            file_name = mask_path.name

            mask = imread(mask_path)
            mask = mask.sum(axis=2).astype(bool) if len(mask.shape) == 3 else mask.astype(bool)

            dense_graph_mask = imread(Path(dense_graph_base_path, file_name))
            dense_graph_mask = dense_graph_mask.sum(axis=2).astype(bool) if len(
                dense_graph_mask.shape) == 3 else dense_graph_mask.astype(bool)

            bbox_string = mask_path.stem

            match = bbox_regex.match(bbox_string)
            # swapped around intentionally to match heuristic in other methods: y refers to vertical axis,
            # which is first axis in array display; x to horizontal
            y_min, y_max, x_min, x_max = map(int, match.groups())

            img_path = Path(cls_path, "original", file_name)
            img = imread(img_path)

            bbox_mask = np.zeros_like(mask)
            bbox_mask[y_min:y_max, x_min:x_max] = True

            rw_marker = np.zeros(mask.shape, dtype=np.uint8)
            rw_marker[np.invert(bbox_mask)] = 1
            rw_marker[dense_graph_mask] = 2

            if dense_graph_mask.sum() == 0:
                rw_mask = np.zeros(mask.shape, bool)
            else:
                rw_mask_probs = random_walker(img, channel_axis=2, labels=rw_marker, return_full_prob=True)
                rw_foreground_prob = rw_mask_probs[1]

                rw_mask = rw_foreground_prob >= 0.5

            intersection = (mask * rw_mask).sum()
            union = (mask + rw_mask).sum()
            iou = (intersection + 0.01) / (union + 0.01)
            precision = (intersection + 0.01) / (rw_mask.sum() + 0.01)

            rw_iou_values.append(iou)
            rw_prec_values.append(precision)

            rw_mask_image = np.stack([rw_mask.astype(np.uint8) * 255] * 3, axis=2)

            imwrite(Path(rw_path, file_name), rw_mask_image)

        stat_dict["rw_iou"] = rw_iou_values
        stat_dict["rw_prec"] = rw_prec_values

        json.dump(stat_dict, open(Path(cls_path, "statistics.json"), "w"))


def _create_crf_segmentation_from_dense_graph(root_path, use_rw_prelabeling=False, use_gt=False):
    """
    root_path: Path to folder containing class folders
    """
    bbox_regex = re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")
    cls_paths = [path for path in Path(root_path).glob("*") if path.is_dir()]

    for cls_path in cls_paths:

        dense_graph_base_path = Path(cls_path, "dense_graph" if not use_gt else "gt_dense_graph")
        masks_base_path = Path(cls_path, "masks")
        folder_str = "crf"
        if use_rw_prelabeling:
            folder_str = folder_str + "_rw"
        if use_gt:
            folder_str = "gt_" + folder_str
        crf_path = Path(cls_path, folder_str)
        crf_path.mkdir(parents=True, exist_ok=True)

        stat_dict = json.load(open(Path(cls_path, "statistics.json"), "r")) if Path(cls_path,
                                                                                    "statistics.json").exists() else {}

        crf_iou_values = []
        crf_prec_values = []

        for mask_path in tqdm(masks_base_path.glob("*.png")):
            file_name = mask_path.name

            mask = imread(mask_path)
            mask = mask.sum(axis=2).astype(bool) if len(mask.shape) == 3 else mask.astype(bool)

            dense_graph_mask = imread(Path(dense_graph_base_path, file_name))
            dense_graph_mask = dense_graph_mask.sum(axis=2).astype(bool) if len(
                dense_graph_mask.shape) == 3 else dense_graph_mask.astype(bool)

            bbox_string = mask_path.stem

            match = bbox_regex.match(bbox_string)
            # swapped around intentionally to match heuristic in other methods: y refers to vertical axis,
            # which is first axis in array display; x to horizontal
            y_min, y_max, x_min, x_max = map(int, match.groups())

            img_path = Path(cls_path, "original", file_name)
            img = imread(img_path)

            bbox_mask = np.zeros_like(mask)
            bbox_mask[y_min:y_max, x_min:x_max] = True
            background_marker = np.invert(binary_dilation(bbox_mask, structure=disk(3)))

            foreground_marker = binary_dilation(dense_graph_mask, structure=disk(2))

            crf_marker = np.zeros(mask.shape, dtype=np.int64)
            crf_marker[background_marker] = 1
            crf_marker[foreground_marker] = 2

            if use_rw_prelabeling and len(np.unique(crf_marker)) == 3:
                rw_marker = random_walker(img, crf_marker, channel_axis=2, return_full_prob=True)[1]

                rw_marker = rw_marker >= 0.7

                crf_marker[rw_marker] = 2

            if dense_graph_mask.sum() == 0:
                crf_mask = np.zeros(mask.shape, bool)
            else:
                unary = unary_from_labels(crf_marker, n_labels=2, gt_prob=.75)
                d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
                d.setUnaryEnergy(unary)

                # This adds the color-independent term, features are the locations only.
                d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                      normalization=dcrf.NORMALIZE_SYMMETRIC)

                # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
                d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                                       compat=10,
                                       kernel=dcrf.DIAG_KERNEL,
                                       normalization=dcrf.NORMALIZE_SYMMETRIC)

                q = d.inference(5)
                crf_labels = np.argmax(q, axis=0).reshape((img.shape[0], img.shape[1]))

                crf_mask = crf_labels

            intersection = (mask * crf_mask).sum()
            union = (mask + crf_mask).sum()
            iou = (intersection + 0.01) / (union + 0.01)
            precision = (intersection + 0.01) / (crf_mask.sum() + 0.01)

            crf_iou_values.append(iou)
            crf_prec_values.append(precision)

            crf_mask_image = np.stack([crf_mask.astype(np.uint8) * 255] * 3, axis=2)

            imwrite(Path(crf_path, file_name), crf_mask_image)

        stat_dict[f"{folder_str}_iou"] = crf_iou_values
        stat_dict[f"{folder_str}_prec"] = crf_prec_values

        json.dump(stat_dict, open(Path(cls_path, "statistics.json"), "w"))


def _create_crf_segmentation_from_gaze(root_path, use_gt=False):
    """
    root_path: Path to folder containing class folders
    """
    bbox_regex = re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")
    cls_paths = [path for path in Path(root_path).glob("*") if path.is_dir()]

    for cls_path in cls_paths:

        gaze_base_path = Path(cls_path, "gaze_images_blurred")
        masks_base_path = Path(cls_path, "masks")
        crf_path = Path(cls_path, "crf_gaze" if not use_gt else "gt_crf_gaze")
        crf_path.mkdir(parents=True, exist_ok=True)

        stat_dict = json.load(open(Path(cls_path, "statistics.json"), "r")) if Path(cls_path,
                                                                                    "statistics.json").exists() else {}

        crf_iou_values = []
        crf_prec_values = []

        for mask_path in tqdm(masks_base_path.glob("*.png")):
            file_name = mask_path.name

            mask = imread(mask_path)
            mask = mask.sum(axis=2).astype(bool) if len(mask.shape) == 3 else mask.astype(bool)

            gaze_mask = imread(Path(gaze_base_path, file_name))
            gaze_mask = gaze_mask.astype(np.float32) / 255.

            bbox_string = mask_path.stem

            match = bbox_regex.match(bbox_string)
            # swapped around intentionally to match heuristic in other methods: y refers to vertical axis,
            # which is first axis in array display; x to horizontal
            y_min, y_max, x_min, x_max = map(int, match.groups())

            img_path = Path(cls_path, "original", file_name)
            img = imread(img_path)

            bbox_mask = np.zeros_like(mask)
            bbox_mask[y_min:y_max, x_min:x_max] = True
            background_marker = np.invert(binary_dilation(bbox_mask, structure=disk(3)))

            if use_gt:
                # filter gaze data perfectly
                gaze_mask = np.where(mask, gaze_mask, 0.)
            else:
                # filter gaze to only use data within bbox
                gaze_mask = np.where(bbox_mask, gaze_mask, 0.)

            crf_marker = np.zeros(bbox_mask.shape, dtype=np.int64)
            crf_marker[background_marker] = 1
            crf_marker[gaze_mask.astype(bool)] = 2

            if gaze_mask.sum() == 0:
                crf_mask = np.zeros(mask.shape, bool)
            else:
                unary = unary_from_labels(crf_marker, 2, gt_prob=.75)
                d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
                d.setUnaryEnergy(unary)

                # This adds the color-independent term, features are the locations only.
                d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                      normalization=dcrf.NORMALIZE_SYMMETRIC)

                # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
                d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                                       compat=10,
                                       kernel=dcrf.DIAG_KERNEL,
                                       normalization=dcrf.NORMALIZE_SYMMETRIC)

                q = d.inference(5)
                crf_labels = np.argmax(q, axis=0).reshape((img.shape[0], img.shape[1]))

                crf_mask = crf_labels

            intersection = (mask * crf_mask).sum()
            union = (mask + crf_mask).sum()
            iou = (intersection + 0.01) / (union + 0.01)
            precision = (intersection + 0.01) / (crf_mask.sum() + 0.01)

            crf_iou_values.append(iou)
            crf_prec_values.append(precision)

            crf_mask_image = np.stack([crf_mask.astype(np.uint8) * 255] * 3, axis=2)

            imwrite(Path(crf_path, file_name), crf_mask_image)

        stat_dict["g_crf_iou" if not use_gt else "gt_g_crf_iou"] = crf_iou_values
        stat_dict["g_crf_prec" if not use_gt else "gt_g_crf_prec"] = crf_prec_values

        json.dump(stat_dict, open(Path(cls_path, "statistics.json"), "w"))


def _create_crf_segmentation_from_fixations(root_path, use_gt=False):
    """
    root_path: Path to folder containing class folders
    """
    bbox_regex = re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")
    cls_paths = [path for path in Path(root_path).glob("*") if path.is_dir()]

    for cls_path in cls_paths:

        fixation_base_path = Path(cls_path, "fixation_images")
        masks_base_path = Path(cls_path, "masks")
        crf_path = Path(cls_path, "crf_fixation" if not use_gt else "gt_crf_fixation")
        crf_path.mkdir(parents=True, exist_ok=True)

        stat_dict = json.load(open(Path(cls_path, "statistics.json"), "r")) if Path(cls_path,
                                                                                    "statistics.json").exists() else {}

        crf_iou_values = []
        crf_prec_values = []

        for mask_path in tqdm(masks_base_path.glob("*.png")):
            file_name = mask_path.name

            mask = imread(mask_path)
            mask = mask.sum(axis=2).astype(bool) if len(mask.shape) == 3 else mask.astype(bool)

            fixation_mask = imread(Path(fixation_base_path, file_name))
            fixation_mask = fixation_mask.astype(np.float32) / 255.

            bbox_string = mask_path.stem

            match = bbox_regex.match(bbox_string)
            # swapped around intentionally to match heuristic in other methods: y refers to vertical axis,
            # which is first axis in array display; x to horizontal
            y_min, y_max, x_min, x_max = map(int, match.groups())

            img_path = Path(cls_path, "original", file_name)
            img = imread(img_path)

            bbox_mask = np.zeros_like(mask)
            bbox_mask[y_min:y_max, x_min:x_max] = True
            background_marker = np.invert(binary_dilation(bbox_mask, structure=disk(3)))

            if use_gt:
                # filter fixation data perfectly
                fixation_mask = np.where(mask, fixation_mask, 0.)
            else:
                fixation_points = np.array(np.nonzero(fixation_mask)).T

                fixation_points = np.array(
                    list(filter(lambda point: (y_min <= point[0] <= y_max) and (x_min <= point[1] <= x_max),
                                fixation_points)))

                if len(fixation_points) != 0:

                    fixation_colors = img[fixation_points[:, 0], fixation_points[:, 1]]
                    fixation_colors = rgb2lab(fixation_colors)
                    # rescale dimensions of lab color space to same range [0, 255]
                    fixation_colors[:, 0] *= 1.55
                    fixation_colors[:, 1:] += 128
                    color_dbscan = DBSCAN(eps=15, min_samples=2)

                    color_labels = color_dbscan.fit_predict(fixation_colors)

                    # only remove 'outlier' if a cluster was found
                    outlier_removed_fixations = fixation_points[color_labels != -1] if len(
                        np.unique(color_labels)) > 1 else fixation_points

                    fixation_points = outlier_removed_fixations

                    fixation_mask[:] = 0
                    fixation_mask[fixation_points[:, 0], fixation_points[:, 1]] = 1

            fixation_mask = binary_dilation(fixation_mask, structure=disk(5))

            crf_marker = np.zeros(bbox_mask.shape, dtype=np.int64)
            crf_marker[background_marker] = 1
            crf_marker[fixation_mask] = 2

            if fixation_mask.sum() == 0:
                crf_mask = np.zeros(mask.shape, bool)
            else:
                unary = unary_from_labels(fixation_mask, 2, gt_prob=0.75)
                d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
                d.setUnaryEnergy(unary)

                # This adds the color-independent term, features are the locations only.
                d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                      normalization=dcrf.NORMALIZE_SYMMETRIC)

                # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
                d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                                       compat=10,
                                       kernel=dcrf.DIAG_KERNEL,
                                       normalization=dcrf.NORMALIZE_SYMMETRIC)

                q = d.inference(5)
                crf_labels = np.argmax(q, axis=0).reshape((img.shape[0], img.shape[1]))

                crf_mask = crf_labels

            intersection = (mask * crf_mask).sum()
            union = (mask + crf_mask).sum()
            iou = (intersection + 0.01) / (union + 0.01)
            precision = (intersection + 0.01) / (crf_mask.sum() + 0.01)

            crf_iou_values.append(iou)
            crf_prec_values.append(precision)

            crf_mask_image = np.stack([crf_mask.astype(np.uint8) * 255] * 3, axis=2)

            imwrite(Path(crf_path, file_name), crf_mask_image)

        stat_dict["g_crf_iou" if not use_gt else "gt_g_crf_iou"] = crf_iou_values
        stat_dict["g_crf_prec" if not use_gt else "gt_g_crf_prec"] = crf_prec_values

        json.dump(stat_dict, open(Path(cls_path, "statistics.json"), "w"))


if __name__ == '__main__':
    """
    for cls in [path.name for path in Path("/data/PascalVOC2012/VOC2012/random_gaze/train/").glob("*") if
                path.is_dir()]:
        base_path = f"/data/PascalVOC2012/VOC2012/random_gaze/train/{cls}"
        construct_fine_graph(base_path, debug=True)
    """
    """
    for cls in [path.name for path in Path("/data/PascalVOC2012/VOC2012/random_gaze/train/").glob("*") if path.is_dir()]:
        base_path = f"/data/PascalVOC2012/VOC2012/random_gaze/train/{cls}"
        construct_fine_graph(base_path, debug=True, use_gt=True)
    """
    # _create_rw_segmentation_from_dense_graph("/data/PascalVOC2012/VOC2012/random_gaze/train/")
    # _create_crf_segmentation_from_dense_graph("/data/PascalVOC2012/VOC2012/random_gaze/train/")
    # _create_crf_segmentation_from_dense_graph("/data/PascalVOC2012/VOC2012/random_gaze/train/", use_gt=True)
    # _create_crf_segmentation_from_dense_graph("/data/PascalVOC2012/VOC2012/random_gaze/train/", use_rw_prelabeling=True)
    # _create_crf_segmentation_from_dense_graph("/data/PascalVOC2012/VOC2012/random_gaze/train/", use_gt=True,use_rw_prelabeling=True)
    _create_crf_segmentation_from_gaze("/data/PascalVOC2012/VOC2012/random_gaze/train/")
    _create_crf_segmentation_from_gaze("/data/PascalVOC2012/VOC2012/random_gaze/train/", use_gt=True)
    _create_crf_segmentation_from_fixations("/data/PascalVOC2012/VOC2012/random_gaze/train/")
    _create_crf_segmentation_from_fixations("/data/PascalVOC2012/VOC2012/random_gaze/train/", use_gt=True)

    _plot_seg_statistics_from_json("/data/PascalVOC2012/VOC2012/random_gaze/train/")
