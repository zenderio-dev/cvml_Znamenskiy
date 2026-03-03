import cv2
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.measure import label, regionprops

TASK_PATH = Path("./knn_ocr/task")
TRAIN_PATH = TASK_PATH / "train"

CHAR_LIST = []

def extract_features(image):
    if image.ndim == 3:
        gray = np.mean(image, 2).astype("u1")
    else:
        gray = image

    binary = gray > 0

    lb = label(binary)
    props = [p for p in regionprops(lb) if p.extent <= 0.8 and p.area > 5]

    if not props:
        return None

    prop = max(props, key=lambda p: p.area)

    feats = [
        prop.eccentricity,
        prop.solidity,
        prop.extent,
        prop.perimeter / (prop.area + 1e-5),
        prop.area_convex / (prop.area + 1e-5),
        (prop.bbox[3] - prop.bbox[1]) / (prop.bbox[2] - prop.bbox[0] + 1e-5)
    ]

    minr, minc, maxr, maxc = prop.bbox
    crop = binary[minr:maxr, minc:maxc]

    for h_slice in np.array_split(crop, 5, axis=0):
        feats.append(np.sum(h_slice) / h_slice.size)

    for v_slice in np.array_split(crop, 5, axis=1):
        feats.append(np.sum(v_slice) / v_slice.size)

    return np.array(feats, dtype="f4")


def make_train(path):
    train_feats = []
    train_labels = []
    CHAR_LIST.clear()

    cls_list = sorted([p for p in path.iterdir() if p.is_dir()])
    for idx, cls in enumerate(cls_list):
        if cls.name.startswith("s") and len(cls.name) == 2:
            char_name = cls.name[1]
        else:
            char_name = cls.name
        CHAR_LIST.append(char_name)

        for img_path in sorted(cls.glob("*.png")):
            feats = extract_features(imread(img_path))
            if feats is not None:
                train_feats.append(feats)
                train_labels.append(idx)
                # print(char_name, img_path)

    train_feats = np.array(train_feats, dtype="f4")
    train_labels = np.array(train_labels, dtype="f4").reshape(-1, 1)
    # print(len(train_feats))
    return train_feats, train_labels


def segment_image(image):
    if image.ndim == 3:
        gray = np.mean(image, 2).astype("u1")
    else:
        gray = image

    binary = gray > 0
    lb = label(binary)
    props = [p for p in regionprops(lb) if p.area > 5 and p.extent <= 0.8]

    props_sorted = sorted(props, key=lambda p: p.bbox[1])

    return [p.image for p in props_sorted]


def main():
    train_feats, train_labels = make_train(TRAIN_PATH)

    knn = cv2.ml.KNearest_create()
    knn.train(train_feats, cv2.ml.ROW_SAMPLE, train_labels)

    for i in range(7):
        image_path = TASK_PATH / f"{i}.png"
        image = imread(image_path)

        symbols = segment_image(image)
        # print(f"{i}.png: {len(symbols)}")

        feats_list = []
        for sym in symbols:
            feats = extract_features(sym)
            if feats is not None:
                feats_list.append(feats)
                # print(feats)

        if not feats_list:
            print(f"{i}.png: символы не распознаны")
            continue

        feats_array = np.array(feats_list, dtype="f4")
        ret, results, neighbours, dist = knn.findNearest(feats_array, k=5)

        text_chars = []
        for r in results:
            char_idx = int(r.item())
            char = CHAR_LIST[char_idx]
            text_chars.append(char)
        text = "".join(text_chars)

        # print(results)
        # print(neighbours)
        # print(dist)

        print(f"{i}.png: {text}")


if __name__ == "__main__":
    main()
