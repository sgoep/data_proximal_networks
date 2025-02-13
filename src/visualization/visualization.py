import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def visualization_with_zoom(example: str, x: np.ndarray, zoom: bool, colorbar: bool, name: str) -> None:
    """
    Visualizes a 2D array with optional zoomed-in region and colorbar.

    This function displays a grayscale image of a 2D numpy array `x`.
    It optionally adds a zoomed-in section of the image, outlined by a red
    rectangle, and can also include a colorbar.
    The resulting image can be saved to a file if a filename is provided.

    Parameters:
    -----------
    x : numpy.ndarray
        A 2D numpy array representing the image data to be visualized.

    zoom : bool
        If True, a zoomed-in region of the image is displayed in a separate
        subplot.

    colorbar : bool
        If True, a colorbar is displayed alongside the image to indicate the
        intensity values.

    name : str
        The filename to save the image. If an empty string is provided, the
        image is not saved.

    Returns:
    --------
    None
        This function does not return any value. It displays the image and
        optionally saves it to a file.
    """
    if example == "lodopab":
        a = 40
        b = a + 50  # height
        c = 125
        d = c + 100  # with
    else:
        a = 48
        b = a + 30
        c = 73
        d = c + 30

    vmin = np.min(x)
    vmax = np.max(x)
    vmin = 0
    vmax = np.max(x)

    fig, ax = plt.subplots()
    plt.imshow(x, cmap="gray", vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar()
    plt.axis("off")

    if zoom:
        rect = patches.Rectangle((c, a), d - c, b - a, linewidth=1.5, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        if example == "lodopab":
            sub_axes = plt.axes([0.23, 0.075, 0.25, 0.25])  # type: ignore
        else:
            sub_axes = plt.axes([0.2, 0.12, 0.25, 0.25])  # type: ignore
        for axis in ["top", "bottom", "left", "right"]:
            sub_axes.spines[axis].set_linewidth(1.5)
        sub_axes.imshow(x[a:b, c:d], cmap="gray", vmin=vmin, vmax=vmax)
        sub_axes.spines["bottom"].set_color("red")
        sub_axes.spines["top"].set_color("red")
        sub_axes.spines["left"].set_color("red")
        sub_axes.spines["right"].set_color("red")
        sub_axes.set_xticks([])
        sub_axes.set_yticks([])

    if len(name) > 0:
        plt.savefig(name, bbox_inches="tight", dpi=1200)
