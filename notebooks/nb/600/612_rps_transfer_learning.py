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
    ## Transfer Learning: Rock, Paper, Scissors

    ... or some other data, whatever you decided to create!
    """)
    return


@app.cell
def _():
    import torch
    import torchvision.transforms.v2 as transforms
    import matplotlib.pyplot as plt
    import numpy as np

    from pathlib import Path
    from torchvision import models
    from torchvision.datasets import ImageFolder
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    return (
        ImageFolder,
        Path,
        Subset,
        models,
        np,
        plt,
        torch,
        train_test_split,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Device
    """)
    return


@app.cell
def _(torch):
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data
    """)
    return


@app.cell
def _(ImageFolder, Path, Subset, torch, train_test_split, transforms):
    DATA_DIR = Path("data/rps/")
    assert DATA_DIR.exists()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(342),  # Mimic Inception V3 resize
        transforms.CenterCrop(299), # Central 299x299 patch to match V3 input size
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Inverse normalization for plotting purposes
    inv_transform = transforms.Normalize(
        mean=[-m/s for m, s in zip(imagenet_mean, imagenet_std)],
        std=[1/s for s in imagenet_std]
    )

    full_dataset = ImageFolder(root=DATA_DIR, transform=transform)

    # Stratified 80/20 split by label
    targets = full_dataset.targets
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=42,
        shuffle=True
    )

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    return (
        full_dataset,
        imagenet_mean,
        imagenet_std,
        inv_transform,
        train_dataset,
    )


@app.cell(hide_code=True)
def _(
    full_dataset,
    imagenet_mean,
    imagenet_std,
    inv_transform,
    np,
    plt,
    torch,
    train_dataset,
):
    def get_class_examples(train_dataset, chosen):
        class_examples = []
        for img, label in train_dataset:
            if label == CHOSEN_LABEL:
                class_examples.append(img)

            if len(class_examples) == 20:
                break
        return class_examples

    def five_by_four_fig(images, img_transform=None):
        fig, axes = plt.subplots(4, 5, figsize=(10,8), constrained_layout=True)
        axes = axes.flatten()
        mean = torch.tensor(imagenet_mean).view(3, 1, 1)
        std = torch.tensor(imagenet_std).view(3, 1, 1)
    
        for ax, img in zip(axes, images):
            img_disp = img.clone()

            if img_transform:
                img_disp = img_transform(img_disp)
            
            # img_disp = img_disp * std + mean
            img_disp = img_disp.permute(1, 2, 0).numpy()
            img_disp = np.clip(img_disp, 0.0, 1.0)

            ax.imshow(img_disp)
            ax.axis("off")

        return fig, axes

    CHOSEN_LABEL = 0
    examples = get_class_examples(train_dataset, CHOSEN_LABEL)
    fig, axes = five_by_four_fig(examples, img_transform=inv_transform)
    class_name = full_dataset.classes[CHOSEN_LABEL]
    fig.suptitle(f"Class #{CHOSEN_LABEL}: {class_name}")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model

    This example uses the MobiletNet V3. You can use something else, if you like. For example, Inception V3 can be a good challenge. You need to battle with something called auxiliary logits.
    """)
    return


@app.cell
def _(device, models):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1

    pre_trained_model = models.inception_v3(weights=weights)
    pre_trained_model = pre_trained_model.to(device)
    return (pre_trained_model,)


@app.cell
def _(pre_trained_model):
    pre_trained_model.fc
    return


@app.cell
def _(pre_trained_model):
    pre_trained_model.AuxLogits
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
