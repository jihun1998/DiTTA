"""VSPW data access: class metadata and a per-video clip loader.

The clip loader reproduces the CFFM test pipeline
(``LoadImageFromFile`` -> ``AlignedResize_clips`` -> ``Normalize_clips`` ->
``ImageToTensor_clips``) with plain cv2/numpy, so it is verified to be
bit-identical to the original mmseg pipeline
(see ``tools/verify_against_legacy.py``, check 1).

Directory layout expected under ``data_root`` (VSPW_480p):

    val.txt                     one video name per line
    data/<video>/origin/*.jpg   frames
    data/<video>/mask/*.png     ground truth (only used for evaluation)
"""

import math
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from PIL import Image

_CLASS_IDS = {"others": "0", "wall": "1", "ceiling": "2", "door": "3", "stair": "4", "ladder": "5", 
    "escalator": "6", "Playground_slide": "7", "handrail_or_fence": "8", "window": "9", 
    "rail": "10", "goal": "11", "pillar": "12", "pole": "13", "floor": "14",
    "ground": "15", "grass": "16", "sand": "17", "athletic_field": "18", "road": "19", "path": "20",
    "crosswalk": "21", "building": "22", "house": "23", "bridge": "24", "tower": "25", "windmill": "26",
    "well_or_well_lid": "27", "other_construction": "28", "sky": "29", "mountain": "30", "stone": "31",
    "wood": "32", "ice": "33", "snowfield": "34", "grandstand": "35", "sea": "36", "river": "37", 
    "lake": "38", "waterfall": "39", "water": "40", "billboard_or_Bulletin_Board": "41", "sculpture": "42",
    "pipeline": "43", "flag": "44", "parasol_or_umbrella": "45", "cushion_or_carpet": "46", "tent": "47",
    "roadblock": "48", "car": "49", "bus": "50", "truck": "51", "bicycle": "52", "motorcycle": "53",
    "wheeled_machine": "54", "ship_or_boat": "55", "raft": "56", "airplane": "57", "tyre": "58",
    "traffic_light": "59", "lamp": "60", "person": "61", "cat": "62", "dog": "63", "horse": "64",
    "cattle": "65", "other_animal": "66", "tree": "67", "flower": "68", "other_plant": "69", "toy": "70",
    "ball_net": "71", "backboard": "72", "skateboard": "73", "bat": "74", "ball": "75",
    "cupboard_or_showcase_or_storage_rack": "76", "box": "77", "traveling_case_or_trolley_case": "78",
    "basket": "79", "bag_or_package": "80", "trash_can": "81", "cage": "82", "plate": "83",
    "tub_or_bowl_or_pot": "84", "bottle_or_cup": "85", "barrel": "86", "fishbowl": "87", "bed": "88",
    "pillow": "89", "table_or_desk": "90", "chair_or_seat": "91", "bench": "92", "sofa": "93",
    "shelf": "94", "bathtub": "95", "gun": "96", "commode": "97", "roaster": "98", "other_machine": "99",
    "refrigerator": "100", "washing_machine": "101", "Microwave_oven": "102", "fan": "103", "curtain": "104",
    "textiles": "105", "clothes": "106", "painting_or_poster": "107", "mirror": "108", "flower_pot_or_vase": "109",
    "clock": "110", "book": "111", "tool": "112", "blackboard": "113", "tissue": "114", "screen_or_television": "115",
    "computer": "116", "printer": "117", "Mobile_phone": "118", "keyboard": "119", "other_electronic_product": "120",
    "fruit": "121", "food": "122", "instrument": "123", "train": "124"}

VSPW_PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0]
               ]


VSPW_CLASSES = tuple(list(_CLASS_IDS.keys())[1:])
NUM_CLASSES = len(VSPW_CLASSES)

IMG_MEAN = (123.675, 116.28, 103.53)
IMG_STD = (58.395, 57.12, 57.375)


def list_videos(data_root, split='val'):
    """Video names of a VSPW split, in file order."""
    with open(os.path.join(data_root, split + '.txt')) as f:
        return [line.strip() for line in f if line.strip()]


def _rescale_and_align(img, img_scale, size_divisor):
    """mmcv.imrescale(img, img_scale) followed by AlignedResize's _align."""
    h, w = img.shape[:2]
    scale = min(max(img_scale) / max(h, w), min(img_scale) / min(h, w))
    new_w, new_h = int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    align_h = int(math.ceil(new_h / size_divisor)) * size_divisor
    align_w = int(math.ceil(new_w / size_divisor)) * size_divisor
    return cv2.resize(img, (align_w, align_h), interpolation=cv2.INTER_LINEAR)


def _normalize(img, mean, std):
    """mmcv.imnormalize(img, mean, std, to_rgb=True)."""
    img = img.copy().astype(np.float32)
    mean = np.float64(np.array(mean, dtype=np.float32).reshape(1, -1))
    stdinv = 1 / np.float64(np.array(std, dtype=np.float32).reshape(1, -1))
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


class VSPWVideo:
    """One VSPW video, addressed by frame index.

    ``clip(i)`` returns the model input for frame ``i``: a ``[num_clips, 3, H, W]``
    tensor holding frames ``dilation + [i]`` (indices outside the video are
    dropped, so frame 0 of a video yields a single-frame clip).
    """

    #: how many decoded frames to keep; the warm-up prefix is revisited once per
    #: TTA iteration, so a small window avoids all of the re-decoding
    CACHE_SIZE = 32

    def __init__(self, data_root, name, dilation=(-1,), img_scale=(853, 480),
                 size_divisor=32, mean=IMG_MEAN, std=IMG_STD, device='cuda'):
        self.data_root = data_root
        self.name = name
        self.dilation = tuple(dilation)
        self.img_scale = img_scale
        self.size_divisor = size_divisor
        self.mean, self.std = mean, std
        self.device = device

        self.img_dir = os.path.join(data_root, 'data', name, 'origin')
        self.ann_dir = os.path.join(data_root, 'data', name, 'mask')
        self.frames = sorted(os.listdir(self.img_dir))
        self._cache = OrderedDict()
        with Image.open(self.frame_path(0)) as img:
            self._ori_shape = (img.size[1], img.size[0])

    def __len__(self):
        return len(self.frames)

    def frame_name(self, i):
        """Frame file name without suffix, e.g. ``00000543``."""
        return os.path.splitext(self.frames[i])[0]

    def frame_path(self, i):
        return os.path.join(self.img_dir, self.frames[i])

    @property
    def ori_shape(self):
        """(H, W) of the source frames."""
        return self._ori_shape

    def _load(self, i):
        """Normalised CHW tensor for a single frame (cached)."""
        if i not in self._cache:
            img = cv2.imread(self.frame_path(i))
            img = _rescale_and_align(img, self.img_scale, self.size_divisor)
            img = _normalize(img, self.mean, self.std)
            self._cache[i] = torch.from_numpy(
                np.ascontiguousarray(img.transpose(2, 0, 1)))
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)
        self._cache.move_to_end(i)
        return self._cache[i]

    def clip(self, i):
        idxs = [i + d for d in self.dilation if 0 <= i + d < len(self)] + [i]
        return torch.stack([self._load(j) for j in idxs]).to(self.device)

    def gt(self, i):
        """Raw VSPW mask (1..124 with 0 = unlabelled), as loaded from disk."""
        path = os.path.join(self.ann_dir, self.frame_name(i) + '.png')
        return np.asarray(Image.open(path))

    def drop_cache(self):
        self._cache.clear()


def warmup_length(num_frames, ratio):
    """Number of leading frames used for adaptation under the W2F protocol.

    Matches the original implementation, so the evaluated frame counts (and
    therefore the TensorBoard steps) line up with the reference runs.
    """
    return max(num_frames * ratio // 100, 1)
