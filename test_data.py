import matplotlib.pyplot as plt
import numpy as np
import torch

from src.visualization.visualization import visualization_with_zoom

which = "train"
name = "synthetic"

print(f"Loading {name} ...")

index = 1366
index = 2

x = np.load("data/data_synthetic/phantom.npy")[index, :, :]
y = np.load("data/data_synthetic/fbp.npy")[index, :, :]
z = np.load("data/data_synthetic/init_regul_tv.npy")[index, :, :]
a = np.load("data/data_synthetic/init_regul_ell1.npy")[index, :, :]
q = np.load("data/data_synthetic/landweber.npy")[index, :, :]

print(f"Plotting {name} ...")

plt.figure()
plt.imshow(x)
plt.colorbar()
plt.savefig(f"plots/{name}_phantom.png")

plt.figure()
plt.imshow(y)
plt.colorbar()
plt.savefig(f"plots/{name}_fbp.png")

plt.figure()
plt.imshow(z)
plt.colorbar()
plt.savefig(f"plots/{name}_tv.png")

plt.figure()
plt.imshow(a)
plt.colorbar()
plt.savefig(f"plots/{name}_ell1.png")

plt.figure()
plt.imshow(q)
plt.colorbar()
plt.savefig(f"plots/{name}_landweber.png")

print(f"Finished {name} ...")

print(f"Norm: {np.load('data/data_synthetic/norm.npy', allow_pickle=True)}")

index = 1363
name = "lodopab"
which = "train"

x = np.load(
    f"data/data_lodopab/data_processed/{which}/single_files/phantom_{index}.npy"
).squeeze()
y = np.load(
    f"data/data_lodopab/data_processed/{which}/single_files/fbp_{index}.npy"
).squeeze()
z = np.load(
    f"data/data_lodopab/data_processed/{which}/single_files/init_regul_tv_{index}.npy"
).squeeze()
a = np.load(
    f"data/data_lodopab/data_processed/{which}/single_files/init_regul_ell1_{index}.npy"
).squeeze()
q = np.load(
    f"data/data_lodopab/data_processed/{which}/single_files/landweber_{index}.npy"
).squeeze()

print(f"Plotting {name} ...")

plt.figure()
plt.imshow(x, cmap="gray")
plt.colorbar()
plt.savefig(f"plots/{name}_phantom.png")

plt.figure()
plt.imshow(y, cmap="gray")
plt.colorbar()
plt.savefig(f"plots/{name}_fbp.png")

plt.figure()
plt.imshow(z, cmap="gray")
plt.colorbar()
plt.savefig(f"plots/{name}_tv.png")

visualization_with_zoom("lodopab", z, True, True, f"plots/{name}_tv.pdf")

plt.figure()
plt.imshow(a, cmap="gray")
plt.colorbar()
plt.savefig(f"plots/{name}_ell1.png")

plt.figure()
plt.imshow(q, cmap="gray")
plt.colorbar()
plt.savefig(f"plots/{name}_landweber.png")

print(
    f"Norm: {np.load('data/data_lodopab/data_processed/train/norm.npy', allow_pickle=True)}"
)

print(f"Finished {name} ...")

print("Finished.")
