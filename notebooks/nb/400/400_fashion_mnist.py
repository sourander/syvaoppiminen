import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Naive MLP for Fashion MNIST

    This Marimo notebook contains only the first cells of the solution. You may use this as a starting point of start from scratch.
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import datetime

    from pathlib import Path
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import classification_report
    from torch.utils.tensorboard import SummaryWriter
    return (
        DataLoader,
        SummaryWriter,
        datasets,
        datetime,
        plt,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration
    """)
    return


@app.cell
def _(SummaryWriter, datetime, torch):
    # Configuration
    USE_GPU = True  # Toggle this to False to use CPU instead

    # Hyperparameters
    LEARNING_RATE = 0.01
    EPOCHS = 100
    BATCH_SIZE = 128

    # Device selection
    if USE_GPU and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    elif USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # Option B: TensorBoard setup for logging metrics to disk
    device_name = str(device)
    run_name = f"fashion_mlp_{device_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"TensorBoard logging to: runs/{run_name}")

    # Option B: Log configuration information to TensorBoard
    writer.add_text('config/device', device_name)
    writer.add_text('config/hyperparameters', 
                    f'LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}')
    return (BATCH_SIZE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and prepare data

    It will be about 82 MB.
    """)
    return


@app.cell
def _(BATCH_SIZE, DataLoader, datasets, transforms):
    transform = transforms.Compose([
        # Convert to tensor and scale to [0, 1]
        transforms.ToTensor(),
        # Fashion MNIST mean and std to normalize
        # (computed using the train dataset)
        transforms.Normalize((0.286041,), (0.353024,))
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize labels
    """)
    return


@app.cell
def _(mosaic):
    {idx: mos for idx, mos in zip(range(10), sum(mosaic, []))}
    return


@app.cell
def _(plt, train_dataset):
    def flatten(x: list):
        return sum(x, [])

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    # Find first occurrence of each class
    class_examples = {}
    for img, label in train_dataset:
        if label not in class_examples:
            class_examples[label] = img
        if len(class_examples) == 10:
            break

    # Create a mosaic layout with letters A-J (10 classes, 2 rows x 5 cols)
    mosaic = [
        ["A", "B", "C", "D", "E"],
        ["F", "G", "H", "I", "J"],
    ]
    # Map letters to class indices
    letter_to_class = {mos: idx for idx, mos in zip(range(10), flatten(mosaic))}

    # Create figure
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 5))

    for letter, ax in axes.items():
        class_idx = letter_to_class[letter]
        img = class_examples[class_idx]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(labels_map[class_idx])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return (mosaic,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
