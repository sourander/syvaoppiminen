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
    # LeNet-5 - MNIST Version

    This implements the LeNet-5 architecture (Adrian Rosebrock's version) for the MNIST dataset.

    Architecture, if using the Rosebrock adaptation of LeNet-5.

    | Layer Type | Output Size | Filter Size / Stride |
    | ---------- | ----------- | -------------------- |
    | Input      | 28x28x1     |                      |
    | Conv1      | 28x28x20    | 5x5 / K = 20         |
    | Act1       | 28x28x20    | ReLU                 |
    | Pool1      | 14x14x20    | 2x2 / S = 2          |
    | Conv2      | 14x14x50    | 5x5 / K = 50         |
    | Act2       | 14x14x50    | ReLU                 |
    | Pool2      | 7x7x50      | 2x2 / S = 2          |
    | FC1        | 500         |                      |
    | Act3       | 500         | ReLU                 |
    | FC2        | 10          |                      |
    | Softmax    | 10          |                      |
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import os
    import torchvision.transforms.v2 as transforms
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import math

    from pathlib import Path
    from datetime import datetime
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torch.utils.tensorboard import SummaryWriter
    from torchmetrics.classification import MulticlassAccuracy
    return (
        DataLoader,
        MulticlassAccuracy,
        SummaryWriter,
        datasets,
        datetime,
        nn,
        os,
        plt,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set Device
    """)
    return


@app.cell
def _(os, torch):
    USE_GPU = True
    NUM_CPU = os.cpu_count()

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
    return NUM_CPU, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define Model
    """)
    return


@app.cell
def _(device, nn):
    class LeNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()

            # Conv1: 28x28x1 -> 28x28x20
            # Padding=2 to maintain 28x28 size with 5x5 kernel
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=..., kernel_size=..., padding=...)
            self.act1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Conv2: 14x14x20 -> 14x14x50
            # Padding=2 to maintain 14x14 size with 5x5 kernel
            self.conv2 = nn.Conv2d(in_channels=20, out_channels=..., kernel_size=..., padding=...)
            self.act2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # FC1: 7x7x50 -> 500
            self.fc1 = nn.Linear(in_features=..., out_features=...)
            self.act3 = nn.ReLU()

            # FC2: 500 -> 10
            self.fc2 = nn.Linear(in_features=..., out_features=num_classes)

        def forward(self, x):
            # Block 1
            x = self.conv1(x)
            x = self.act1(x)
            x = self.pool1(x)

            # Block 2
            x = self.conv2(x)
            x = self.act2(x)
            x = self.pool2(x)

            # Flatten
            x = x.view(x.size(0), -1)

            # FC Layers
            x = self.fc1(x)
            x = self.act3(x)
            x = self.fc2(x)

            return x

    model = LeNet(num_classes=10)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize Model Insides
    """)
    return


@app.cell
def _(model, torch):
    def debug_forward_pass(model, input_size=(1, 1, 28, 28)):
        # 1. Detect device
        device = next(model.parameters()).device
        print(f"--- Debugging Forward Pass on {device} (Input: {input_size}) ---")

        # 2. Create input
        x = torch.randn(*input_size).to(device)
        print(f"Input: {x.shape}")

        # Block 1
        x = model.conv1(x)
        print(f"-> Conv1: {x.shape}")
        x = model.act1(x)
        x = model.pool1(x)
        print(f"-> Pool1: {x.shape}")

        # Block 2
        x = model.conv2(x)
        print(f"-> Conv2: {x.shape}")
        x = model.act2(x)
        x = model.pool2(x)
        print(f"-> Pool2: {x.shape}")

        # Flatten
        x = x.view(x.size(0), -1)
        print(f"-> Flatten: {x.shape}")

        # FC Layers
        x = model.fc1(x)
        print(f"-> FC1: {x.shape}")
        x = model.act3(x)
        x = model.fc2(x)
        print(f"-> FC2 (Output): {x.shape}")

    # Run it
    debug_forward_pass(model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperparameters and Tensorboard
    """)
    return


@app.cell
def _(SummaryWriter, datetime, device):
    LEARNING_RATE = ...
    EPOCHS = ...
    BATCH_SIZE = ...

    # TensorBoard setup for logging metrics to disk
    device_name = str(device)
    run_name = f"mnist_lenet_{device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"TensorBoard logging to: runs/{run_name}")

    # Log configuration information to TensorBoard
    writer.add_text('config/device', device_name)
    writer.add_text('config/hyperparameters', 
                    f'LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}')
    return BATCH_SIZE, EPOCHS, LEARNING_RATE, device_name, run_name, writer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data
    """)
    return


@app.cell
def _(BATCH_SIZE, DataLoader, NUM_CPU, datasets, torch, transforms):
    # MNIST stats (mean and std for normalization)
    MNIST_MEAN = (0.1307,)  # Single channel
    MNIST_STD = (0.3081,)

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),  # Slight rotation for augmentation
            transforms.ToImage(),  # convert PIL/ndarray -> v2 image
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )

    # Download and load the training data
    print("[INFO] accessing MNIST...")
    trainset = datasets.MNIST(
        "./data", download=True, train=True, transform=train_transform
    )
    testset = datasets.MNIST(
        "./data", download=True, train=False, transform=test_transform
    )

    # Create data loaders
    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        persistent_workers=True,
        num_workers=NUM_CPU,
    )
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        persistent_workers=True,
        num_workers=NUM_CPU,
    )
    return testloader, trainloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train and Eval Helpers
    """)
    return


@app.cell
def _(torch):
    def train_epoch(model, train_loader, criterion, optimizer, metric, device, writer=None, epoch=None):
        """Train the model for one epoch.

        Args:
            metric: A torchmetrics metric instance (e.g., MulticlassAccuracy)

        Returns:
            tuple: (average_loss, accuracy)
        """
        model.train()
        metric.reset()
        total_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # TODO: Zero the gradients
            ...
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # TODO: Perform backpropagation
            ...
            
            # TODO: Update the model parameters
            ...

            total_loss += loss.item()

            # TODO: Update metric with predictions
            ...

            # Optional: Log batch-level loss to TensorBoard
            if writer and epoch is not None and batch_idx % 50 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        avg_loss = total_loss / len(train_loader)
        accuracy = metric.compute().item()
        return avg_loss, accuracy

    def evaluate(model, data_loader, criterion, metric, device):
        """Evaluate the model on the given data loader.

        Args:
            metric: A torchmetrics metric instance (e.g., MulticlassAccuracy)

        Returns:
            tuple: (average_loss, accuracy)
        """
        model.eval()
        metric.reset()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                # Update metric with predictions
                metric.update(outputs, labels)

        avg_loss = total_loss / len(data_loader)
        accuracy = metric.compute().item()
        return avg_loss, accuracy
    return evaluate, train_epoch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Loop
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MulticlassAccuracy,
    datetime,
    device,
    device_name,
    evaluate,
    model,
    nn,
    testloader,
    torch,
    train_epoch,
    trainloader,
    writer,
):
    # For printing run duration
    start = datetime.now()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=...)

    # Initialize torchmetrics accuracy metric for 10 classes (MNIST digits)
    accuracy_metric = MulticlassAccuracy(num_classes=10).to(device)

    # Initialize local history dictionary to store metrics in memory
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print("[INFO] training network...")
    for epoch in range(EPOCHS):
        # Training phase
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, accuracy_metric, device, writer, epoch
        )

        # Validation phase
        val_loss, val_acc = evaluate(model, testloader, criterion, accuracy_metric, device)

        # Append epoch metrics to local history lists
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Log comparative metrics (train vs val) to TensorBoard
        writer.add_scalars('Loss/train_vs_val', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        writer.add_scalars('Accuracy/train_vs_val', {
            'train': train_acc,
            'val': val_acc
        }, epoch)

        # Print progress every epoch since
        # the model is expected to find a proper solution fairly quickly.
        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print("[INFO] training complete!")

    # Log final hyperparameters with results for comparison across runs
    writer.add_hparams(
        {'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'device': device_name},
        {'hparam/final_train_acc': train_acc, 'hparam/final_val_acc': val_acc,
         'hparam/final_train_loss': train_loss, 'hparam/final_val_loss': val_loss}
    )

    # Close the TensorBoard writer to flush all remaining data
    writer.close()
    print("TensorBoard logs saved!")

    end = datetime.now()
    delta = end - start
    print("The duration of the training:", delta)
    return history, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the Training History
    """)
    return


@app.cell
def _(EPOCHS, history, plt):
    import numpy as np

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, EPOCHS), history["train_loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), history["train_acc"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy - MNIST LeNet")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save the Model
    """)
    return


@app.cell
def _(EPOCHS, history, model, optimizer, run_name, torch):
    # Define a path for the full checkpoint
    checkpoint_path = f'models/{run_name}_checkpoint_epoch{EPOCHS}.pth'

    # Save a dictionary containing all necessary state
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': history['train_loss'][-1], # Save the last loss
        'accuracy': history['val_acc'][-1], # Save the last validation accuracy
    }, checkpoint_path)

    print(f"Full checkpoint saved to: {checkpoint_path}")
    return


if __name__ == "__main__":
    app.run()
