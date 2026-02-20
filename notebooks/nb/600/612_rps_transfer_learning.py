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
    ## Transfer Learning: Rock, Paper, Scissors

    ... or some other data, whatever you decided to create!
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms.v2 as transforms
    import matplotlib.pyplot as plt
    import numpy as np

    from pathlib import Path
    from torch.utils.data import DataLoader
    from torchvision import models
    from torchvision.datasets import ImageFolder
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    from torchmetrics.classification import MulticlassAccuracy

    return (
        DataLoader,
        ImageFolder,
        MulticlassAccuracy,
        Path,
        Subset,
        models,
        nn,
        np,
        optim,
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
def _(
    DataLoader,
    ImageFolder,
    Path,
    Subset,
    torch,
    train_test_split,
    transforms,
):
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

    # Training DataLoader
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    # Optional: Test DataLoader (for validation)
    dataloader_test = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )
    return (
        dataloader_test,
        dataloader_train,
        full_dataset,
        inv_transform,
        train_dataset,
    )


@app.cell(hide_code=True)
def _(full_dataset, inv_transform, np, plt, train_dataset):
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

    This example uses the Inception V3 (alias GoogLeNet). Inception uses "auxiliary logits" during training. It may loosely resemle a residual connection, but it is not quite that, since it bypasses the next modules and shortcuts to output loss. In the loss functions, these can then be added together with a weight. You can see it's usage in the source code of Inception: [gh:pytorch/torchvision/../inception:L103-L155](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py#L103-L155)

    ```python
        def forward:
            # Forward pass contains layers up to Mixed_6e
            x = BEFORE(x)

            # The aux logits and handled here conditionally
            if aux_logits:
                aux = AuxLogits(x)

            # Forward pass continues with Mixed_7a to 7c
            x = AFTER(x)

            # And finally, tuple of typical classifier
            # and the aux classifier is returned
            return x, aux
    ```

    The actual aux logit head is:

    ```
    InceptionAux(
      (conv0): BasicConv2d(
        (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv1): BasicConv2d(
        (conv): Conv2d(128, 768, kernel_size=(5, 5), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (fc): Linear(in_features=768, out_features=1000, bias=True)
    )
    ```

    You can also see these in the original paper [Going Deeper with Convolutions](https://doi.org/10.48550/arXiv.1409.4842).  Look at the Figure 3 (pg. 7). You can see two classification heads branching off from the *main road*. They are *"put on top of the output of the Inception (4a) and (4d) modules"*

    However, this is something we need to handle when training the model and computing the loss. Original paper explains it as:

    > "During training, their loss gets added to the total loss of the
    > network with a discount weight (the losses of the auxiliary classifiers were weighted by 0.3). At
    > inference time, these auxiliary networks are discarded."
    """)
    return


@app.cell
def _(device, models):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1

    pre_trained_model = models.inception_v3(weights=weights)
    pre_trained_model = pre_trained_model.to(device)
    return (pre_trained_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create new FC
    """)
    return


@app.cell
def _(full_dataset, nn):
    # You can list model's child modules them like this
    # [x for x in pre_trained_model.named_children()]

    n_classes = len(full_dataset.classes)
    new_head = nn.Linear(in_features=2048, out_features=n_classes)
    return n_classes, new_head


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Function

    Note that we have the auxiliary logits, and we must weight them in to the loss. This affects the training function. An example of how this can be done is seen in the repository for book **AI and ML for Coders in PyTorch** in a file [Chapter03.ipynb](https://github.com/lmoroney/PyTorch-Book-FIles/blob/main/Chapter03/PyTorch_Chapter_3.ipynb). The key part is:

    ```python
        # Forward pass
        outputs = model(inputs)
        # Handle multiple outputs for training with auxiliary logits
        if isinstance(outputs, tuple):
            output, aux_output = outputs
            loss1 = criterion(output, labels)
            loss2 = criterion(aux_output, labels)
            loss = loss1 + 0.4 * loss2  # Scale the auxiliary loss as is standard for Inception
        else:
            loss = criterion(outputs, labels)
    ```
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

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)

                # Handle auxiliary logits for Inception V3
                if isinstance(outputs, tuple):
                    main_output, aux_output = outputs
                    loss1 = criterion(main_output, labels)
                    loss2 = criterion(aux_output, labels)
                    loss = loss1 + 0.4 * loss2  # Weight auxiliary loss by 0.4
                else:
                    main_output = outputs
                    loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Statistics (use main output for accuracy)
                metric.update(main_output, labels)

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
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                metric.update(outputs, labels)

        avg_loss = total_loss / len(data_loader)
        accuracy = metric.compute().item()
        return avg_loss, accuracy

    return evaluate, train_epoch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Loop

    Note that we are not saving the model this time. Since our dataset is tiny, the model will likely train in a few seconds.
    """)
    return


@app.cell
def _(
    MulticlassAccuracy,
    dataloader_test,
    dataloader_train,
    device,
    evaluate,
    models,
    n_classes,
    new_head,
    nn,
    optim,
    pre_trained_model,
    train_epoch,
):
    tuned_model = models.inception_v3(weights=None)
    tuned_model.load_state_dict(pre_trained_model.state_dict())
    tuned_model = tuned_model.to(device)

    # Replace the classifier heads (main + auxiliary)
    tuned_model.fc = new_head.to(device)
    tuned_model.AuxLogits.fc = nn.Linear(in_features=768, out_features=n_classes).to(device)

    # Freeze the feature extractor
    for param in tuned_model.parameters():
        param.requires_grad = False

    for param in tuned_model.fc.parameters():
        param.requires_grad = True

    for param in tuned_model.AuxLogits.fc.parameters():
        param.requires_grad = True

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(tuned_model.fc.parameters()) + list(tuned_model.AuxLogits.fc.parameters()),
        lr=0.001
    )

    metric = MulticlassAccuracy(num_classes=n_classes).to(device)

    # Training loop
    num_epochs = 6

    for epoch in range(1, num_epochs + 1):
        epoch_loss, epoch_acc = train_epoch(
            tuned_model, 
            dataloader_train, 
            criterion, 
            optimizer,
            metric,
            device,
            epoch=epoch,
            num_epochs=num_epochs
        )
        val_loss, val_acc = evaluate(
            tuned_model,
            dataloader_test,
            criterion,
            metric,
            device,
        )
        print(f"Epoch [{epoch}/{num_epochs}] Summary: "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {100.*epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {100.*val_acc:.2f}%\n")

    print("Training complete!")
    return


if __name__ == "__main__":
    app.run()
