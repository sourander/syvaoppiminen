import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## HOG

    You can play around with this Notebook as much as you want.

    **Note**: the cat image is not in the repository. Use an image of your own. Don't add large images to your repo.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from skimage.feature import hog
    from skimage import exposure, io, transform, color
    from PIL import Image
    return color, exposure, hog, io, plt, transform


@app.cell
def _():
    # CHANGE THESE THREE VARIABLES!
    image_path = "data/images/505_cat.png"
    ppc = 10 # pixels per cell
    cpb = 2  # cells per block
    return cpb, image_path, ppc


@app.cell(hide_code=True)
def _(color, cpb, hog, image_path, io, ppc, transform):
    # Calculate the largest square that fits (using the smaller of height/width)
    image = io.imread(image_path)
    height, width = image.shape[:2]
    square_size = min(height, width)

    # Crop from the center
    center_y, center_x = height // 2, width // 2
    half_size = square_size // 2
    cropped = image[
        center_y - half_size : center_y + half_size,
        center_x - half_size : center_x + half_size,
    ]

    # Resize to 200x200 pixels with anti-aliasing
    resized = transform.resize(cropped, (200, 200), anti_aliasing=True)

    # Convert to grayscale
    gray = color.rgb2gray(resized)

    # Apply HOG with good default settings
    hog_features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(ppc, ppc),
        cells_per_block=(cpb, cpb),
        visualize=True,
        block_norm="L1",
    )

    print(f"HOG feature vector shape: {hog_features.shape}")
    return gray, hog_image, resized


@app.cell(hide_code=True)
def _(cpb, exposure, gray, hog_image, mo, plt, ppc, resized):
    # Rescale HOG image for better visibility
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_image_rescaled = exposure.adjust_gamma(hog_image_rescaled, gamma=0.5)

    # Create visualization with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # 1. Display the 200x200 color image
    axes[0].imshow(resized)
    axes[0].set_title("200x200 Color Image")
    axes[0].axis("off")

    # 2. Display grayscale with cell and block grid overlaid
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale with HOG Grid")

    # Overlay cell grid
    for i in range(0, 200, ppc):
        axes[1].axhline(i, color="cyan", linewidth=0.5, alpha=0.75)
        axes[1].axvline(i, color="cyan", linewidth=0.5, alpha=0.75)

    # Draw first 3 blocks with increasing opacity to show stride
    block_size = ppc * cpb
    positions = [(0, 0), (ppc, 0), (ppc * 2, 0)]  # First 3 horizontal positions
    alphas = [0.2, 0.3, 1.0]

    for (x, y), alpha in zip(positions, alphas):
        rect = plt.Rectangle(
            (x, y), block_size, block_size,
            linewidth=2, edgecolor='yellow', facecolor='yellow',
            alpha=alpha
        )
        axes[1].add_patch(rect)

    axes[1].axis("off")

    # 3. Display the HOG image
    axes[2].imshow(hog_image_rescaled, cmap="gray")
    axes[2].set_title("HOG Features")
    axes[2].axis("off")

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
