from pathlib import Path
import random

base = Path("/data/PascalVOC2012/VOC2012/random_gaze/val/")
path_list = []
fileids = []

for folder_path in list(base.glob("*")):
    img_list = list(Path(folder_path, "original").glob("*.png"))
    count = 0
    while count < 2:
        path_sample = random.choice(img_list)
        fileid = "_".join(path_sample.stem.split("_")[:2])
        if fileid not in fileids:
            fileids.append(fileid)
            path_list.append(path_sample)
            count +=1

path_list_str = " ".join([str(path) for path in path_list])
with open("test_inputs.txt", "w") as outfile:
    outfile.write(path_list_str)