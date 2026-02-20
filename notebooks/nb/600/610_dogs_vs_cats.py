import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Transfer Learning

    We will train a DenseNet 121 classifier as a binary classifier to detect if an image contains a dog or a cat. The dataset is from Hugging Face: [microsoft/cats_vs_dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs).
    """)
    return


@app.cell
def _():
    import torch
    import torchvision.transforms.v2 as transforms
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import torch.optim as optim

    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from torchvision import models
    from pathlib import Path
    from torchmetrics.classification import MulticlassAccuracy

    return (
        DataLoader,
        MulticlassAccuracy,
        Path,
        load_dataset,
        models,
        nn,
        np,
        optim,
        plt,
        sns,
        torch,
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
    ## Get Dataset

    Remember that the dataset will be stored to `~/.cache/huggingface/datasets`. It is about 700 MB.
    """)
    return


@app.cell
def _(load_dataset):
    # The original dataset has only "train" split available
    ds = load_dataset("microsoft/cats_vs_dogs", split="train")

    # Set format to PyTorch. This converts the PIL image to typical Torch
    # ds = ds.with_format("torch", device=device)

    # Get the mapping from:
    # 0 -> cat
    # 1 -> dig
    classidx_to_label = ds.features["labels"].names

    # Train test split
    ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)

    print(ds)
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convert to DataLoader

    Each item in the dataset will be of size: `(32, 3, 224, 224)`. The original dataset contains various resolution sizes and aspect ratios.

    We have to SOMEHOW do this rescaling/cropping. Simple `transforms.Resize((224, 224))` would work, but for non-square images it would squeeze or stretch the poor dog or cat. We don't want that.

    Note that we could use separate set of transforms for training and testing. For training, we might want to utilize randomized image augmentation like horizontal flip, rotation or mild hue/saturation shifts.
    """)
    return


@app.cell
def _(torch, transforms):
    def collate_fn(batch):
        images = torch.stack([transform(item["image"]) for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])
        return {"image": images, "labels": labels}

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(lambda x: x[:3] if x.shape[0] == 4 else x),  # Drop alpha
        transforms.Resize(256),     # Shorter side to 256px
        transforms.CenterCrop(224), # Central 224x224 patch
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        )
    ])
    return (collate_fn,)


@app.cell
def _(DataLoader, collate_fn, ds):
    ds_train = ds["train"]
    df_test = ds["test"]

    dataloader_train = DataLoader(ds_train, batch_size=32, collate_fn=collate_fn)
    dataloader_test = DataLoader(df_test, batch_size=32, collate_fn=collate_fn)
    return dataloader_test, dataloader_train, ds_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PIL vs Tensor

    The Hugging Face datasets keeps the original data in PIL image format. Marimo can display that as-is.

    For training, we need a Tensor with float32. To do this, we use the `collate_fn()` helper function.
    """)
    return


@app.cell
def _(ds_train):
    ds_train[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Note about data quality

    Note that the images are in various shapes.
    """)
    return


@app.cell
def _(ds_train, np, sns):
    aspect_ratios = []
    for item in ds_train:
        img = item["image"]
        w, h = img.size
        aspect_ratios.append(w / h)

    aspect_ratios = np.array(aspect_ratios)

    sns.histplot(aspect_ratios, bins=80)
    return (aspect_ratios,)


@app.cell
def _(aspect_ratios, np):
    # aspect_ratios is already a NumPy array
    counts, bin_edges = np.histogram(aspect_ratios, bins=80)

    # Index of the most populated bin
    max_bin_idx = np.argmax(counts)

    # Bin range
    bin_left = bin_edges[max_bin_idx]
    bin_right = bin_edges[max_bin_idx + 1]

    # A representative value for the bin (midpoint)
    midpoint = (bin_left + bin_right) / 2

    # Percentage of common common vs all
    percentage = counts[max_bin_idx] / counts.sum()

    print(f"Most common bin range: {bin_left:.2f} ... {bin_right:.2f}")
    print(f"Representative value: {midpoint:.2f}")
    print("Count within range:", counts[max_bin_idx])
    print(f"Percentage of all: ", percentage)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Widest Image
    """)
    return


@app.cell
def _(aspect_ratios, ds_train):
    ds_train[aspect_ratios.argmax()]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Tallest Image
    """)
    return


@app.cell
def _(aspect_ratios, ds_train):
    ds_train[aspect_ratios.argmin()]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get Model
    """)
    return


@app.cell
def _(models):
    [x for x in models.list_models() if 'dense' in x]
    return


@app.cell
def _(device, models):
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    pretrained_model = models.densenet121(weights=weights)

    pretrained_model = pretrained_model.to(device)

    preprocess = weights.transforms()

    print(preprocess)
    return pretrained_model, weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluate Before Fine-Tune

    Note that the ImageNet classes do not include a simple "cat" and "dog". However, there are specific cat and dog species/races. They can be found in ranges:

    * 281 to 285: Tabby to Egyptian cat
    * 151 to 268: Chihuahua to Mexican hairless
    """)
    return


@app.cell
def _():
    cats_in_imagenet = range(281, 286)
    dogs_in_imagenet = range(151, 269)
    return cats_in_imagenet, dogs_in_imagenet


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using this naive assumption, let's do some rough evaluation:
    """)
    return


@app.cell(hide_code=True)
def _(
    cats_in_imagenet,
    dataloader_test,
    device,
    dogs_in_imagenet,
    pretrained_model,
    torch,
    weights,
):
    def evaluate_model(model):
        model.eval()
        accurate = 0
        total = 0
        total_true_cat = 0
        total_true_dog = 0
        correct_true_cat = 0
        correct_true_dog = 0
        pred_counts = {"cat": 0, "dog": 0, "other": 0}
        confusion = {
            "true_cat": {"pred_cat": 0, "pred_dog": 0, "pred_other": 0},
            "true_dog": {"pred_cat": 0, "pred_dog": 0, "pred_other": 0},
        }

        # Track mislabeled predictions with their ImageNet class names
        cats_mislabeled_as = {}
        dogs_mislabeled_as = {}

        # Get ImageNet category labels
        imagenet_categories = weights.meta["categories"]

        with torch.no_grad():
            for batch in dataloader_test:
                images = batch["image"].to(device)
                labels = batch["labels"]  # on CPU
                outputs = model(images)
                preds_idx = outputs.argmax(dim=1).cpu()

                for pred_idx, true_label in zip(
                    preds_idx.tolist(), labels.tolist()
                ):
                    total += 1
                    if true_label == 0:
                        total_true_cat += 1
                    else:
                        total_true_dog += 1

                    if pred_idx in cats_in_imagenet:
                        pred = "cat"
                    elif pred_idx in dogs_in_imagenet:
                        pred = "dog"
                    else:
                        pred = "other"

                    pred_counts[pred] += 1

                    if true_label == 0:  # true cat
                        confusion["true_cat"][f"pred_{pred}"] += 1
                        if pred == "cat":
                            correct_true_cat += 1
                            accurate += 1
                        else:
                            # Track mislabeled cats
                            class_name = imagenet_categories[pred_idx]
                            cats_mislabeled_as[class_name] = (
                                cats_mislabeled_as.get(class_name, 0) + 1
                            )
                    else:  # true dog
                        confusion["true_dog"][f"pred_{pred}"] += 1
                        if pred == "dog":
                            correct_true_dog += 1
                            accurate += 1
                        else:
                            # Track mislabeled dogs
                            class_name = imagenet_categories[pred_idx]
                            dogs_mislabeled_as[class_name] = (
                                dogs_mislabeled_as.get(class_name, 0) + 1
                            )

        eval_accuracy_strict = accurate / total if total > 0 else 0.0
        preds_catdog = pred_counts["cat"] + pred_counts["dog"]
        eval_accuracy_conditional = (
            (accurate / preds_catdog) if preds_catdog > 0 else 0.0
        )

        # Get top 10 mislabeled classes for each
        cats_mislabeled_top10 = dict(
            sorted(cats_mislabeled_as.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
        )
        dogs_mislabeled_top10 = dict(
            sorted(dogs_mislabeled_as.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
        )

        eval_results = {
            "total_samples": total,
            "total_true_cat": total_true_cat,
            "total_true_dog": total_true_dog,
            "pred_counts": pred_counts,
            "confusion": confusion,
            "correct_true_cat": correct_true_cat,
            "correct_true_dog": correct_true_dog,
            "accuracy_strict": eval_accuracy_strict,
            "accuracy_when_pred_cat_or_dog": eval_accuracy_conditional,
            "cats_mislabeled_as": cats_mislabeled_top10,
            "dogs_mislabeled_as": dogs_mislabeled_top10,
        }

        return eval_results


    eval_results = evaluate_model(pretrained_model)
    return (eval_results,)


@app.cell
def _(eval_results):
    eval_results
    return


@app.cell(hide_code=True)
def _(eval_results, np, plt, sns):
    _conf = eval_results["confusion"]

    pred_labels = ["pred_cat", "pred_dog", "pred_other"]
    x_tick_labels = ["cat", "dog", "other"]
    y_tick_labels = ["true_cat", "true_dog"]

    matrix = np.array(
        [
            [_conf["true_cat"][p] for p in pred_labels],
            [_conf["true_dog"][p] for p in pred_labels],
        ],
        dtype=int,
    )

    row_sums = matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    pct_matrix = matrix / row_sums_safe

    annot = np.empty(matrix.shape, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            annot[i, j] = f"{matrix[i, j]}\n({pct_matrix[i, j]:.1%})"

    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=x_tick_labels,
        yticklabels=["cat (true)", "dog (true)"],
        linewidths=0.5,
        linecolor="white",
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fine Tune DenseNet

    If you check what the `model.classifier` contains, you will see:

    ```
    Linear(in_features=1024, out_features=1000, bias=True)
    ```

    Remember that this is the head on top of `model.features` feature extractor CNN. The CNN layers create a feature vector that describes the image contents in numbers.  The classifier knows nothing about images: it is a simple classifier. We could use e.g. scikit-learns Logistic Regression or any other traditional ML classifier.
    """)
    return


@app.cell
def _(torch):
    def train_epoch(model, train_loader, criterion, optimizer, metric, device, epoch=None, num_epochs=None):
        """Train the model for one epoch.

        Returns:
            tuple: (epoch_loss, epoch_acc)
        """
        model.train()
        metric.reset()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Statistics
            metric.update(outputs, labels)

            if (batch_idx + 1) % 50 == 0:
                current_acc = metric.compute().item()
                print(f"Epoch [{epoch}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {total_loss/(batch_idx+1):.4f}, "
                      f"Acc: {100.*current_acc:.2f}%")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = metric.compute().item()
        return epoch_loss, epoch_acc

    def evaluate(model, data_loader, criterion, metric, device):
        """Evaluate the model on the given data loader.

        Returns:
            tuple: (average_loss, accuracy)
        """
        model.eval()
        metric.reset()
        total_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = metric.compute().item()
        return avg_loss, accuracy

    return (train_epoch,)


@app.cell
def _(
    MulticlassAccuracy,
    Path,
    dataloader_train,
    device,
    models,
    nn,
    optim,
    pretrained_model,
    torch,
    train_epoch,
):
    # Try to load the model first
    model_path = Path("models/densenet121_catdog.pth")

    NEEDS_SAVING = False

    if model_path.exists():
        # Load checkpoint
        checkpoint = torch.load(model_path)

        # Recreate model with custom classifier
        tuned_model = models.densenet121(weights=None)
        tuned_model.classifier = torch.nn.Linear(1024, 2)

        # Load trained weights
        tuned_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        tuned_model = models.densenet121(weights=None)
        tuned_model.load_state_dict(pretrained_model.state_dict())
        tuned_model = tuned_model.to(device)

        # Replace the classifier head for binary classification
        tuned_model.classifier = nn.Linear(1024, 2).to(device)

        # Freeze the feature extractor (optional - only train the classifier)
        for param in tuned_model.features.parameters():
            param.requires_grad = False

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(tuned_model.classifier.parameters(), lr=0.001)

        accuracy_metric = MulticlassAccuracy(num_classes=2).to(device)

        # Training loop
        num_epochs = 5

        for epoch in range(1, num_epochs + 1):
            epoch_loss, epoch_acc = train_epoch(
                tuned_model, 
                dataloader_train, 
                criterion, 
                optimizer,
                accuracy_metric,
                device,
                epoch=epoch,
                num_epochs=num_epochs
            )
            print(f"Epoch [{epoch}/{num_epochs}] Summary: "
                  f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%\n")

        print("Training complete!")
        NEEDS_SAVING = True
    return NEEDS_SAVING, model_path, tuned_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Materialize Model for later Eval

    The Model most likely trained for 10 minutes or more on GPU (5 epochs). Materialize it. You will need it again.

    The model will be about 28 MB in size. We could save some space by simply saving the new classifier head, but this solution works for now.
    """)
    return


@app.cell
def _(NEEDS_SAVING, model_path, torch, tuned_model):
    if NEEDS_SAVING:
        # Create models directory if it doesn't exist
        model_path.parent.mkdir(exist_ok=True)

        # Save the entire model state
        torch.save({
            'model_state_dict': tuned_model.state_dict(),
            'model_architecture': 'densenet121',
            'num_classes': 2,
            'class_names': ['cat', 'dog']
        }, model_path)

        print(f"Model saved to: {model_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How to Use the Model

    To load the fine-tuned model later:

    ```python
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    # Use for inference
    with torch.no_grad():
        image = transform(your_image).unsqueeze(0).to(device)
        output = model(image)
        pred_class = output.argmax(dim=1).item()
        class_name = checkpoint['class_names'][pred_class]
        print(f"Prediction: {class_name}")
    ```

    The model is now ready for inference!

    ## Your Part of the Exercise

    Make evaluation showing the poorly classified images. Also, run the evaluation against the test data.
    """)
    return


if __name__ == "__main__":
    app.run()
