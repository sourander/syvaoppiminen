import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CIFAR10 Investigation with PyTorch

    Let's get used to loading images with PyTorch
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    return DataLoader, datasets, plt, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset and DataLoader
    """)
    return


@app.cell
def _(datasets, transforms):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR10 mean and std
    ])

    # Download and load the training data
    print("[INFO] accessing CIFAR10...")
    trainset = datasets.CIFAR10('./data', download=True, train=True, transform=transform)
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform)
    return (trainset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect

    Print out the following:

    * How many pictures?
    * What is a shape of a single picture? Color channels? Width x Height?
    * What are the labels? How do they map to class numbers?
    """)
    return


@app.cell
def _(
    channels,
    class_to_idx,
    classes,
    height,
    shape,
    test_len,
    total_len,
    train_len,
    width,
):
    # IMPLEMENT

    print(f"Number of pictures in trainset: {train_len}")
    print(f"Number of pictures in testset: {test_len}")
    print(f"Total number of pictures: {total_len}")

    print(f"Shape of a single picture (C, H, W): {shape}")
    print(f"Color channels: {channels}")
    print(f"Width x Height: {(width, height)}")

    print("Class labels:", classes)
    print("Class to index mapping:", class_to_idx)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize a Single Picture

    Hint: the `plt.imshow()` assumes shape: `(w, h, c)`
    """)
    return


@app.cell
def _():
    # IMPLEMENT
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize Chosen Label in Grid

    If the user chooses label 0, there should be e.g. 4x5 images of the class 0. Figure out what is the class 0 in this dataset.
    """)
    return


@app.cell
def _():
    # IMPLEMENT
    # Teacher implemented two functions here:
    # Signatures:
    #   def get_class_examples(chosen) -> list:
    #   def five_by_four_fig(class_examples) -> Fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use DataLoader to Visualize a Mini-Batch

    Below is an implementation that works if you happen to have a function `five_by_four_fig()` defined that returns a subplot of 4x5 images.
    """)
    return


@app.cell
def _(DataLoader, five_by_four_fig, plt, trainset):
    train_loader = DataLoader(trainset, batch_size=20, shuffle=True)

    for imgs, labels in train_loader:
        break

    new_fig = five_by_four_fig(imgs)
    new_fig.suptitle("Random Mini-Batch", fontsize=16)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
