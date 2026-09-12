import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _image_size(size):
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("image_size must be an integer or [height, width]")
    height, width = (int(value) for value in size)
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    return height, width


def _letterbox(image, size, fill, interpolation):
    target_height, target_width = _image_size(size)
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    resized = image.resize((round(width * scale), round(height * scale)), interpolation)
    canvas = Image.new(image.mode, (target_width, target_height), fill)
    left = (target_width - resized.width) // 2
    top = (target_height - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _resolve_file(directory, filename, mask=False):
    stem, _ = os.path.splitext(filename)
    candidates = [filename]
    if mask:
        candidates.extend((f"{stem}_mask.png", f"{stem}.png", f"{stem}.jpg"))
    for candidate in candidates:
        path = os.path.join(directory, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find '{filename}' in {directory}")


class VTONHDDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        pair_list=None,
        image_size=256,
        random_flip=False,
        random_shift_scale=False,
        shift_scale_prob=0.5,
        shift_limit=0.2,
        scale_limit=0.2,
        paired=True,
        preview_sample_id=None,
        garment_parse_labels=None,
        dense_pose_dir=None,
        garment_high_frequency=False,
        garment_high_frequency_canny_thresholds=(100, 200),
        garment_high_frequency_mode="canny",
        garment_high_frequency_dog_sigmas=(0.8, 2.4),
        garment_high_frequency_dog_gain=4.0,
        garment_high_frequency_gradient_gain=2.0,
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.image_size = _image_size(image_size)
        self.random_flip = bool(random_flip)
        self.random_shift_scale = bool(random_shift_scale)
        self.shift_scale_prob = float(shift_scale_prob)
        self.shift_limit = float(shift_limit)
        self.scale_limit = float(scale_limit)
        if not 0.0 <= self.shift_scale_prob <= 1.0:
            raise ValueError("shift_scale_prob must be in [0, 1]")
        if not 0.0 <= self.shift_limit <= 1.0:
            raise ValueError("shift_limit must be in [0, 1]")
        if not 0.0 <= self.scale_limit < 1.0:
            raise ValueError("scale_limit must be in [0, 1)")
        self.paired = bool(paired)
        self.garment_parse_labels = None if garment_parse_labels is None else tuple(int(x) for x in garment_parse_labels)
        if self.garment_parse_labels is not None and not self.garment_parse_labels:
            raise ValueError("garment_parse_labels must contain at least one clothing label")
        split_root = os.path.join(self.root, split)
        self.image_dir = os.path.join(split_root, "image")
        self.garment_dir = os.path.join(split_root, "cloth")
        self.agnostic_mask_dir = os.path.join(split_root, "agnostic-mask")
        self.garment_mask_dir = os.path.join(split_root, "cloth-mask")
        self.person_parse_dir = os.path.join(split_root, "image-parse-v3")
        self.dense_pose_dir = None if dense_pose_dir is None else os.path.join(split_root, str(dense_pose_dir))
        if self.dense_pose_dir is not None and not os.path.isdir(self.dense_pose_dir):
            raise FileNotFoundError(f"DensePose conditioning requires: {self.dense_pose_dir}")
        self.garment_high_frequency = bool(garment_high_frequency)
        thresholds = tuple(int(value) for value in garment_high_frequency_canny_thresholds)
        if len(thresholds) != 2 or not 0 <= thresholds[0] < thresholds[1] <= 255:
            raise ValueError(
                "garment_high_frequency_canny_thresholds must be [low, high] with "
                "0 <= low < high <= 255"
            )
        self.garment_high_frequency_canny_thresholds = thresholds
        self.garment_high_frequency_mode = str(garment_high_frequency_mode).lower()
        if self.garment_high_frequency_mode not in ("canny", "rgb_dog_gradient"):
            raise ValueError(
                "garment_high_frequency_mode must be 'canny' or 'rgb_dog_gradient'"
            )
        sigmas = tuple(float(value) for value in garment_high_frequency_dog_sigmas)
        if len(sigmas) != 2 or not 0 < sigmas[0] < sigmas[1]:
            raise ValueError(
                "garment_high_frequency_dog_sigmas must contain two increasing positive values"
            )
        self.garment_high_frequency_dog_sigmas = sigmas
        self.garment_high_frequency_dog_gain = float(garment_high_frequency_dog_gain)
        self.garment_high_frequency_gradient_gain = float(garment_high_frequency_gradient_gain)
        if min(
            self.garment_high_frequency_dog_gain,
            self.garment_high_frequency_gradient_gain,
        ) <= 0:
            raise ValueError("High-frequency gains must be positive")
        if self.garment_parse_labels is not None and not os.path.isdir(self.person_parse_dir):
            raise FileNotFoundError(f"Garment supervision requires parsing labels: {self.person_parse_dir}")
        pair_list = pair_list or os.path.join(self.root, f"{split}_pairs.txt")
        if not os.path.isfile(pair_list):
            raise FileNotFoundError(f"Pair list not found: {pair_list}")
        with open(pair_list, "r", encoding="utf-8") as handle:
            self.pairs = [tuple(line.split()[:2]) for line in handle if line.strip()]
        if not self.pairs:
            raise ValueError(f"Pair list is empty: {pair_list}")
        if preview_sample_id is not None:
            preview_sample_id = str(preview_sample_id)
            preview_index = next(
                (index for index, (person_name, _) in enumerate(self.pairs) if person_name == preview_sample_id),
                None,
            )
            if preview_index is None:
                raise ValueError(f"Preview sample '{preview_sample_id}' is not present in {pair_list}")
            self.pairs.insert(0, self.pairs.pop(preview_index))

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, path):
        image = Image.open(path).convert("RGB")
        return _letterbox(image, self.image_size, (127, 127, 127), Image.Resampling.BICUBIC)

    def _load_mask(self, path):
        image = Image.open(path).convert("L")
        return _letterbox(image, self.image_size, 0, Image.Resampling.NEAREST)

    def _sample_shift_scale(self):
        if not self.random_shift_scale or random.random() >= self.shift_scale_prob:
            return None
        height, width = self.image_size
        translate = [
            round(random.uniform(-self.shift_limit, self.shift_limit) * width),
            round(random.uniform(-self.shift_limit, self.shift_limit) * height),
        ]
        scale = random.uniform(1.0 - self.scale_limit, 1.0 + self.scale_limit)
        return translate, scale

    def _garment_high_frequency_map(self, garment, garment_mask):
        """Detail from the transformed cloth only, restricted to its own mask.

        ``canny`` is retained solely so old experiment configs remain reproducible.
        ``rgb_dog_gradient`` returns two RGB-like groups for the frozen VAE:
        signed RGB high-pass/DoG in channels 0:3 and luma/chroma/RGB gradient
        magnitudes in channels 3:6. Fixed gains retain absolute edge strength; unlike
        per-image normalisation they do not turn compression noise into a strong logo.
        """
        rgb = np.asarray(garment, dtype=np.uint8)
        mask = np.asarray(garment_mask, dtype=np.uint8) > 127
        if self.garment_high_frequency_mode == "rgb_dog_gradient":
            image = rgb.astype(np.float32) / 255.0
            sigma_fine, sigma_coarse = self.garment_high_frequency_dog_sigmas
            blur_fine = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_fine, sigmaY=sigma_fine)
            blur_coarse = cv2.GaussianBlur(
                image, (0, 0), sigmaX=sigma_coarse, sigmaY=sigma_coarse
            )
            # Preserve the sign independently in R/G/B. This is essential for coloured
            # text and colour-block boundaries, which grayscale Canny aliases together.
            signed_detail = 0.65 * (image - blur_fine) + 0.35 * (
                blur_fine - blur_coarse
            )
            signed_detail = np.clip(
                signed_detail * self.garment_high_frequency_dog_gain, -1.0, 1.0
            )

            def gradient(channel):
                dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3) / 8.0
                dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3) / 8.0
                return np.sqrt(np.square(dx) + np.square(dy))

            luma = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
            # Two opponent-colour axes retain isoluminant boundaries (e.g. red/green
            # blocks) which disappear from a luma-only edge detector.
            red_green = 0.5 * (image[..., 0] - image[..., 1])
            yellow_blue = 0.5 * (0.5 * (image[..., 0] + image[..., 1]) - image[..., 2])
            chroma = np.sqrt(
                np.square(gradient(red_green)) + np.square(gradient(yellow_blue))
            )
            rgb_gradient = np.maximum.reduce([gradient(image[..., channel]) for channel in range(3)])
            gradients = np.stack((gradient(luma), chroma, rgb_gradient), axis=-1)
            gradients = np.clip(
                gradients * self.garment_high_frequency_gradient_gain, 0.0, 1.0
            )

            # Do not spend branch capacity on the garment silhouette. Its geometry is
            # already supplied by garment_mask and the supervised warp; a two-pixel
            # interior keeps blur/Sobel support away from letterbox/background colours.
            interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if not interior.any():
                interior = mask
            signed_detail[~interior] = 0.0
            gradients[~interior] = 0.0
            # Signed absence is neutral 0.5 before the trainer maps it back to [-1, 1].
            signed_detail = 0.5 * (signed_detail + 1.0)
            features = np.concatenate((signed_detail, gradients), axis=-1)
            return torch.from_numpy(features.transpose(2, 0, 1).copy()).float()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        low, high = self.garment_high_frequency_canny_thresholds
        edges = cv2.Canny(gray, low, high)
        edges[~mask] = 0
        return torch.from_numpy(edges.copy()).unsqueeze(0).float().div_(255.0)

    @staticmethod
    def _apply_shift_scale(image, mask, params):
        if params is None:
            return image, mask
        translate, scale = params
        # The RGB image and its mask share one transform, while person and garment
        # receive independently sampled transforms. This is the StableVITON setup
        # that prevents correspondence from collapsing to absolute coordinates.
        image = TF.affine(
            image,
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BICUBIC,
            fill=[0, 0, 0],
        )
        mask = TF.affine(
            mask,
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        return image, mask

    def __getitem__(self, index):
        person_name, garment_name = self.pairs[index]
        if self.paired:
            garment_name = person_name
        person = self._load_rgb(_resolve_file(self.image_dir, person_name))
        garment = self._load_rgb(_resolve_file(self.garment_dir, garment_name))
        agnostic_mask = self._load_mask(_resolve_file(self.agnostic_mask_dir, person_name, mask=True))
        garment_mask = self._load_mask(_resolve_file(self.garment_mask_dir, garment_name, mask=True))
        dense_pose = None
        if self.dense_pose_dir is not None:
            dense_pose = self._load_rgb(_resolve_file(self.dense_pose_dir, person_name))
        person_garment_mask = None
        if self.garment_parse_labels is not None:
            # Preserve palette indices: L conversion gives brightness, not labels.
            parse_path = os.path.join(self.person_parse_dir, os.path.splitext(person_name)[0] + ".png")
            with Image.open(parse_path) as parse:
                labels = np.asarray(parse)
                if labels.ndim != 2:
                    raise ValueError(f"Expected indexed semantic labels in {parse_path}")
                binary = np.isin(labels, self.garment_parse_labels).astype(np.uint8) * 255
            person_garment_mask = _letterbox(
                Image.fromarray(binary), self.image_size, 0, Image.Resampling.NEAREST
            )
        flipped = self.random_flip and random.random() < 0.5
        if flipped:
            person = TF.hflip(person)
            garment = TF.hflip(garment)
            agnostic_mask = TF.hflip(agnostic_mask)
            garment_mask = TF.hflip(garment_mask)
            if dense_pose is not None:
                dense_pose = TF.hflip(dense_pose)
            if person_garment_mask is not None:
                person_garment_mask = TF.hflip(person_garment_mask)

        person_transform = self._sample_shift_scale()
        if dense_pose is not None:
            dense_pose, _ = self._apply_shift_scale(
                dense_pose, Image.new("L", dense_pose.size), person_transform
            )
        if person_garment_mask is not None:
            _, person_garment_mask = self._apply_shift_scale(person, person_garment_mask, person_transform)
        person, agnostic_mask = self._apply_shift_scale(
            person, agnostic_mask, person_transform
        )
        garment, garment_mask = self._apply_shift_scale(
            garment, garment_mask, self._sample_shift_scale()
        )

        garment_high_frequency = None
        if self.garment_high_frequency:
            # Compute after all garment augmentation: RGB, cloth mask and HF map then
            # describe the same target garment coordinates in paired and unpaired data.
            garment_high_frequency = self._garment_high_frequency_map(garment, garment_mask)

        person = TF.to_tensor(person) * 2 - 1
        garment = TF.to_tensor(garment) * 2 - 1
        agnostic_mask = (TF.to_tensor(agnostic_mask) > 0.5).float()
        garment_mask = (TF.to_tensor(garment_mask) > 0.5).float()
        person_agnostic = person * (1 - agnostic_mask)
        sample = {
            "image": person.clone(),
            "person": person,
            "person_agnostic": person_agnostic,
            "garment": garment,
            "agnostic_mask": agnostic_mask,
            "garment_mask": garment_mask,
            "has_ground_truth": torch.tensor(person_name == garment_name),
            "person_name": person_name,
            "garment_name": garment_name,
        }
        if person_garment_mask is not None:
            sample["person_garment_mask"] = (TF.to_tensor(person_garment_mask) > 0.5).float()
        if dense_pose is not None:
            sample["dense_pose"] = TF.to_tensor(dense_pose) * 2 - 1
        if garment_high_frequency is not None:
            sample["garment_high_frequency"] = garment_high_frequency
        return sample


class VTONValidationDataset(Dataset):
    """Fixed train/test reconstructions and test garment swaps sharing noise seeds."""

    def __init__(self, root, image_size=(512, 384), garment_parse_labels=(5, 6, 7),
                 dense_pose_dir=None, garment_high_frequency=False,
                 garment_high_frequency_canny_thresholds=(100, 200),
                 garment_high_frequency_mode="canny",
                 garment_high_frequency_dog_sigmas=(0.8, 2.4),
                 garment_high_frequency_dog_gain=4.0,
                 garment_high_frequency_gradient_gain=2.0, test_samples=8,
                 train_samples=4, preview_sample_id=None):
        if test_samples < 2 or train_samples < 0:
            raise ValueError("Validation needs at least two test samples and nonnegative train_samples")
        self.datasets = {}
        self.items = []
        for split, count in (("test", int(test_samples)), ("train", int(train_samples))):
            if count == 0:
                continue
            dataset = VTONHDDataset(
                root, split=split, image_size=image_size, paired=False,
                garment_parse_labels=garment_parse_labels,
                dense_pose_dir=dense_pose_dir,
                garment_high_frequency=garment_high_frequency,
                garment_high_frequency_canny_thresholds=garment_high_frequency_canny_thresholds,
                garment_high_frequency_mode=garment_high_frequency_mode,
                garment_high_frequency_dog_sigmas=garment_high_frequency_dog_sigmas,
                garment_high_frequency_dog_gain=garment_high_frequency_dog_gain,
                garment_high_frequency_gradient_gain=garment_high_frequency_gradient_gain,
                preview_sample_id=preview_sample_id if split == "test" else None,
            )
            if count > len(dataset):
                raise ValueError(f"Requested {count} {split} examples but only {len(dataset)} exist")
            indices = torch.linspace(0, len(dataset) - 1, count).long().tolist()
            names = [dataset.pairs[i][0] for i in indices]
            original_pairs = [dataset.pairs[i] for i in indices]
            dataset.pairs = []
            self.datasets[split] = dataset
            for index, (person, reference) in enumerate(original_pairs):
                dataset.pairs.append((person, person))
                self.items.append((split, len(dataset.pairs) - 1, f"{split}_paired", index))
                if split == "test":
                    if reference == person:
                        reference = names[(index + 1) % len(names)]
                    if reference == person:
                        raise ValueError("Validation swaps require distinct garment names")
                    dataset.pairs.append((person, reference))
                    self.items.append((split, len(dataset.pairs) - 1, "test_unpaired", index))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        split, item, group, seed = self.items[index]
        sample = self.datasets[split][item]
        sample["validation_group"] = group
        sample["validation_seed"] = torch.tensor(seed + (10000 if split == "train" else 0))
        return sample


class DummyVTONDataset(Dataset):
    def __init__(self, num_samples=1024, image_size=256):
        self.num_samples = int(num_samples)
        self.image_size = _image_size(image_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        height, width = self.image_size
        person = torch.rand((3, height, width), generator=generator) * 2 - 1
        garment = torch.rand((3, height, width), generator=generator) * 2 - 1
        mask = torch.zeros((1, height, width))
        mask[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
        garment_mask = mask.clone()
        return {
            "image": person.clone(),
            "person": person,
            "person_agnostic": person * (1 - mask),
            "garment": garment,
            "agnostic_mask": mask,
            "garment_mask": garment_mask,
        }
