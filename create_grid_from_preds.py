import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from imageio.v3 import imwrite, imread
from pathlib import Path

from tqdm import tqdm


def create_grid_from_preds(base_pred_folder):

    base_folder = Path(base_pred_folder)
    out_folder = Path(base_folder, "grid")

    out_folder.mkdir(exist_ok=True)

    model_folder_paths = [x for x in base_folder.glob("*") if "model" in x.name and x.is_dir()]

    file_names = sorted([x.name for x in Path(base_folder, "model_final").glob("*.png")])

    for file_name in tqdm(file_names):

        imgs = [imread(Path(folder, file_name)) for folder in model_folder_paths]

        ncols = int(np.ceil(np.sqrt(len(imgs))))
        nrows = int(np.ceil(float(len(imgs)) / ncols))

        fig = plt.figure(figsize=(16., 9.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(nrows, ncols),  #
                         axes_pad=0.3,  # pad between axes in inch.
                         )
        for idx, ax in enumerate(grid):
            if idx == len(imgs):
                break
            ax.imshow(imgs[idx])
            ax.set_title(model_folder_paths[idx].stem.split("_")[-1])
            ax.axis('off')

        fig.suptitle(base_folder.parent.stem)
        fig.tight_layout()
        fig.savefig(Path(out_folder, file_name))
        plt.close()


if __name__ == '__main__':

    create_grid_from_preds("/home/daniel/PycharmProjects/AdelaiDet_gaze/training_dir/BoxInst_MS_R_50_1x_VOC2012_run_2/preds")
    exit(0)

    base = "/home/daniel/PycharmProjects/AdelaiDet_gaze/training_dir/loss_conf_sweep/"
    runs = [x for x in Path(base).glob("*") if x.is_dir()]

    for run in runs:

        create_grid_from_preds(Path(run, "preds"))
