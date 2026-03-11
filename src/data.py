import os
import h5py
import numpy as np
from math import ceil
from itertools import islice
from skimage.transform import resize

import odl
from dival.datasets.dataset import GroundTruthDataset
from dival.datasets.lodopab_dataset import NUM_SAMPLES_PER_FILE, LEN
from dival.config import get_config
from dival.reconstructors.odl_reconstructors import FBPReconstructor

# -------------------------------------------------------------------
# Basic configuration
# -------------------------------------------------------------------
DATA_PATH = "/home/simongoeppel/.dival/datasets/lodopab/"# get_config('lodopab_dataset/data_path')
OUT_DIR = "lodopab_256_dataset"

RECO_IM_SHAPE = (256, 256)
SIM_IM_SHAPE = (1000, 1000)

MIN_PT = [-0.13, -0.13]
MAX_PT = [0.13, 0.13]
NUM_ANGLES = 1000
IMPL = "astra_cuda"

NOISE_TYPE = "poisson"
NOISE_KWARGS = {"scaling_factor": 1e4}
NOISE_SEEDS = {"train": 4, "validation": 5, "test": 6}

# -------------------------------------------------------------------
# Ground truth dataset (resized & normalized)
# -------------------------------------------------------------------
class LoDoPaBGroundTruthDataset256(GroundTruthDataset):
    def __init__(self):
        self.shape = RECO_IM_SHAPE
        self.train_len = LEN["train"]
        self.validation_len = LEN["validation"]
        self.test_len = LEN["test"]
        self.random_access = False
        self.space = odl.uniform_discr(MIN_PT, MAX_PT, self.shape, dtype=np.float32)
        super().__init__(space=self.space)

    def generator(self, part="train"):
        num_files = ceil(self.get_len(part) / NUM_SAMPLES_PER_FILE)
        for i in range(num_files):
            file_path = os.path.join(
                DATA_PATH,
                f"ground_truth_{part}_{i:03d}.hdf5",
            )
            with h5py.File(file_path, "r") as f:
                gts = f["data"][:]
            num_samples = min((i + 1) * NUM_SAMPLES_PER_FILE, LEN[part]) - i * NUM_SAMPLES_PER_FILE
            for gt_arr in islice(gts, num_samples):
                gt_resized = resize(
                    gt_arr, self.shape, order=1, mode="reflect",
                    anti_aliasing=True, preserve_range=True
                ).astype(np.float32)
                vmin, vmax = gt_resized.min(), gt_resized.max()
                if vmax > vmin:
                    gt_resized = (gt_resized - vmin) / (vmax - vmin)
                else:
                    gt_resized[:] = 0.0
                yield self.space.element(gt_resized)

# -------------------------------------------------------------------
# Forward and reconstruction operators
# -------------------------------------------------------------------
class _ResizeOperator(odl.Operator):
    def __init__(self, reco_space, sim_space):
        super().__init__(reco_space, sim_space)
        self.sim_shape = sim_space.shape
        self.sim_space = sim_space

    def _call(self, x, out):
        arr = resize(x, self.sim_shape, order=1, mode="reflect",
                     anti_aliasing=True, preserve_range=True).astype(np.float32)
        out.assign(self.sim_space.element(arr))

def get_forward_and_fbp():
    reco_space = odl.uniform_discr(MIN_PT, MAX_PT, RECO_IM_SHAPE, dtype=np.float32)
    sim_space = odl.uniform_discr(MIN_PT, MAX_PT, SIM_IM_SHAPE, dtype=np.float32)

    reco_geom = odl.tomo.parallel_beam_geometry(reco_space, num_angles=NUM_ANGLES)
    sim_geom = odl.tomo.parallel_beam_geometry(sim_space, num_angles=NUM_ANGLES,
                                               det_shape=reco_geom.detector.shape)

    ray_trafo = odl.tomo.RayTransform(sim_space, sim_geom, impl=IMPL)
    resize_op = _ResizeOperator(reco_space, sim_space)
    forward_op = ray_trafo * resize_op

    reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geom, impl=IMPL)
    fbp_reconstructor = FBPReconstructor(
        reco_ray_trafo,
        hyper_params={"filter_type": "Ram-Lak", "frequency_scaling": 1.0}
    )

    return forward_op, fbp_reconstructor, reco_geom


def build_split(split="train"):
    os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

    gt_dataset = LoDoPaBGroundTruthDataset256()
    forward_op, fbp_reconstructor, reco_geom = get_forward_and_fbp()

    dataset = gt_dataset.create_pair_dataset(
        forward_op=forward_op,
        noise_type=NOISE_TYPE,
        noise_kwargs=NOISE_KWARGS,
        noise_seeds=NOISE_SEEDS,
    )

    total = gt_dataset.get_len(split)
    num_files = ceil(total / NUM_SAMPLES_PER_FILE)

    det_shape = reco_geom.detector.shape
    num_angles = NUM_ANGLES

    gen = dataset.generator(part=split)

    # for file_idx in range(num_files):
    for file_idx in range(1):
        remaining = num_files - file_idx - 1
        print(f"\nProcessing file {file_idx+1}/{num_files} ({remaining} remaining)")
        count = min(NUM_SAMPLES_PER_FILE, total - file_idx * NUM_SAMPLES_PER_FILE)
        out_path = os.path.join(OUT_DIR, split, f"data_{file_idx:03d}.h5")
        print(f"Writing {out_path} ({count} samples)")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("gt", (count, *RECO_IM_SHAPE), dtype="f4")
            f.create_dataset("sino", (count, num_angles, det_shape[0]), dtype="f4")
            f.create_dataset("fbp", (count, *RECO_IM_SHAPE), dtype="f4")

            for i in range(count):
                obs, gt = next(gen)
                sino = np.asarray(obs, dtype=np.float32)
                reco = np.asarray(fbp_reconstructor.reconstruct(obs), dtype=np.float32)

                f["gt"][i] = gt
                f["sino"][i] = sino
                f["fbp"][i] = reco

                print(f"  Sample {i+1}/{count} written", end="\r", flush=True)

        print(f"Saved file: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in ["test"]:
        print(f"\n=== Processing split: {split} ===")
        build_split(split)
    print("\nAll splits processed successfully!")

if __name__ == "__main__":
    main()