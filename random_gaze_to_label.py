import itertools
import json
import math
from pathlib import Path
import re

import scipy.signal
import skimage
from skimage.segmentation import slic
from skimage.color import rgb2lab
from tqdm import tqdm
from imageio.v3 import imread, imwrite
import matplotlib.pyplot as plt

import numpy as np
import scipy.spatial.distance as dist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix
from scipy.signal import find_peaks
import cv2


def _plot_seg_statistics_from_json(base_path, ignore_dice=True):
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
            cur_ax.hist(value, bins=25, label=key, alpha=1.0, histtype="step")
        cur_ax.set_title(cls_str)
        cur_ax.legend()

    plt.subplots_adjust(wspace=0.15, hspace=0.2, left=0.05, top=0.95, right=0.95, bottom=0.05)
    fig.savefig(Path(base_path, "class_statistics.pdf"))

    fig.clear()

    fig, ax = plt.subplots()
    metric_keys = list(set(list(itertools.chain(*[list(d.keys()) for d in plot_data_dict.values()]))))[::-1]

    for key in metric_keys:
        if ignore_dice and "dice" in key:
            continue
        data = []
        for d in plot_data_dict.values():
            data += d.get(key, [])

        ax.hist(data, bins=100, label=key, alpha=1.0, histtype="step")
    ax.legend()
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

    x = list(np.clip(x, min(x0, x1), max(x0, x1)))
    y = list(np.clip(y, min(y0, y1), max(y0, y1)))

    coords = np.array(sorted(list(zip(y, x))))

    return coords


def _calc_and_draw_mst_from_points(points, canvas, img):

    colors = img[points[:, 0], points[:, 1]]
    pwcolor_dists = dist.pdist(rgb2lab(colors), "seuclidean")
    color_dist_matrix = dist.squareform(pwcolor_dists)

    pwdists = dist.pdist(points, "seuclidean")
    eucl_dist_matrix = dist.squareform(pwdists)

    dist_matrix = eucl_dist_matrix + color_dist_matrix

    sparse_dist_matrix = csr_matrix(dist_matrix)

    mst = minimum_spanning_tree(sparse_dist_matrix)
    mst_matrix = mst.toarray()
    mst_matrix[mst_matrix > 2 * np.std(mst_matrix[np.nonzero(mst_matrix)])] = 0.

    n_comps, label = connected_components(csr_matrix(mst_matrix), directed=False)
    counts = [np.count_nonzero(label[label == i]) for i in list(np.unique(label))]

    for idx in range(len(counts)):
        if counts[idx] <= 5:
            # delete all adjacencies for nodes whose conn. components are too small
            mst_matrix[np.argwhere(label == idx), :] = 0.
            mst_matrix[:, np.argwhere(label == idx)] = 0.

    mst_entries = np.array(np.nonzero(mst_matrix)).T

    for idx_pair in mst_entries:
        start_coord = points[idx_pair[0]]
        end_coord = points[idx_pair[1]]

        cv2.line(canvas, start_coord[::-1], end_coord[::-1], color=[0, 255, 0])


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

        plt.imsave(Path(base_path, "spanning_tree", file_name), tree_mask)
        if debug:
            cv2.rectangle(tree_canvas, (x_min, y_min), (x_max, y_max), color=[255, 100, 0])
            plt.imsave(Path(base_path, "spanning_tree_debug", file_name),
                       cv2.addWeighted(tree_canvas, 1., img, 0.5, gamma=1))


def construct_fine_graph(base_path, debug=True):
    """
        Constructs dense graph for all fixation points with intermediate nodes on color edges
        :param base_path: Path to class folder with random gaze
        :return: None
        """

    file_names = list(Path(base_path, "original").glob("*.png"))
    Path(base_path, "dense_graph").mkdir(parents=True, exist_ok=True)
    Path(base_path, "spp_seg").mkdir(parents=True, exist_ok=True)
    if debug:
        Path(base_path, "dense_graph_debug").mkdir(parents=True, exist_ok=True)
        Path(base_path, "spp_seg_debug").mkdir(parents=True, exist_ok=True)

    bbox_regex = re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")

    precision_values = []
    superpixel_dice = []
    superpixel_prec = []

    for file_name in tqdm(file_names, total=len(file_names)):

        bbox_string = file_name.stem

        match = bbox_regex.match(bbox_string)
        # swapped around intentionally to match heuristic in other methods: y refers to vertical axis,
        # which is first axis in array display; x to horizontal
        y_min, y_max, x_min, x_max = map(int, match.groups())
        file_name = file_name.name

        img_path = Path(base_path, "original", file_name)
        fixation_path = Path(base_path, "fixation_images", file_name)
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

        fixation_points = np.array(np.nonzero(fixation_img)).T

        fixation_points = np.array(
            list(filter(lambda point: (y_min <= point[0] <= y_max) and (x_min <= point[1] <= x_max), fixation_points)))

        if len(fixation_points) == 0:
            print("WARNING: 0 gaze points in bbox detected. Saving empty mask.")
            tree_canvas = np.zeros(img.shape, dtype=np.uint8)
            tree_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            continue

        segment_dicts = []

        if debug:
            debug_canvas = np.zeros(channel_image.shape, dtype=np.uint8)
            for point in fixation_points:
                cv2.drawMarker(debug_canvas, (point[1], point[0]), [0, 255, 0], markerSize=5)

        for (start_y, start_x), (end_y, end_x) in itertools.combinations(fixation_points, r=2):
            if start_x > end_x:
                start_x, start_y, end_x, end_y = end_x, end_y, start_x, start_y

            unique_coords = _xiaoline(start_x, start_y, end_x, end_y)

            edge_intensities = edges[unique_coords[:, 0], unique_coords[:, 1]]

            if debug:
                debug_canvas[unique_coords[:, 0], unique_coords[:, 1]] = [0, 100, 255]

            peaks, properties = find_peaks(edge_intensities,
                                           prominence=(0.02, None),
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

        if debug:
            for point in additional_graph_points:
                cv2.drawMarker(debug_canvas, (point[1], point[0]), [255, 100, 0], markerSize=5)
            plt.imsave(Path(base_path, "dense_graph_debug", file_name),
                       cv2.addWeighted(debug_canvas, 1., img, 0.5, gamma=1))

        dense_graph_canvas = np.zeros(channel_image.shape, dtype=np.uint8)
        all_points_array = np.array(list(fixation_points) + additional_graph_points)
        _calc_and_draw_mst_from_points(all_points_array, dense_graph_canvas, img)
        plt.imsave(Path(base_path, "dense_graph", file_name),
                   cv2.addWeighted(dense_graph_canvas, 1., img, 0.5, gamma=1))
        tree_mask = dense_graph_canvas.sum(axis=2).astype(bool)

        tp = np.sum(np.logical_and(tree_mask, mask))
        fp = np.sum(tree_mask) - tp

        prec = tp / (tp + fp)

        precision_values.append(prec)

        superpixels = slic(channel_image, n_segments=400, compactness=10, sigma=0.5, channel_axis=2, mask=bbox_mask)

        labels = np.unique(superpixels[tree_mask])
        if 0 in labels:
            labels_ = list(labels)
            labels_.remove(0)
            labels = np.array(labels_)

        super_pixel_segmentation = np.isin(superpixels, labels)
        super_pixel_segmentation_canvas = np.where(super_pixel_segmentation, 255, 0).astype(np.uint8)
        super_pixel_segmentation_canvas = np.stack([super_pixel_segmentation_canvas] * 3, axis=2)
        if debug:
            plt.imsave(Path(base_path, "spp_seg_debug", file_name),
                       cv2.addWeighted(super_pixel_segmentation_canvas, 0.5, img, 0.5, gamma=1))

        plt.imsave(Path(base_path, "spp_seg", file_name),
                   super_pixel_segmentation_canvas)

        dice = (2 * np.sum(super_pixel_segmentation * mask)) / (np.sum(super_pixel_segmentation) + np.sum(mask))
        seg_precision = np.sum(super_pixel_segmentation * mask) / (np.sum(super_pixel_segmentation))

        superpixel_dice.append(dice)
        superpixel_prec.append(seg_precision)

    if debug:
        fig, ax = plt.subplots()
        ax.hist(precision_values, bins=25, label="Tree Precision")
        ax.hist(superpixel_dice, bins=25, label="Superpixel Dice")
        ax.hist(superpixel_prec, bins=25, label="Superpixel Precision")
        ax.legend()
        fig.savefig(Path(base_path, "label_statistics.png"))
        json.dump({"mst_prec": precision_values,
                   "spp_prec": superpixel_prec,
                   "spp_dice": superpixel_dice}, open(Path(base_path, "statistics.json"), "w"))


if __name__ == '__main__':

    _plot_seg_statistics_from_json("/data/PascalVOC2012/VOC2012/random_gaze/train/")
    exit(0)

    for cls in [path.name for path in Path("/data/PascalVOC2012/VOC2012/random_gaze/train/").glob("*") if
                path.is_dir()]:
        base_path = f"/data/PascalVOC2012/VOC2012/random_gaze/train/{cls}"
        construct_fine_graph(base_path, debug=True)
