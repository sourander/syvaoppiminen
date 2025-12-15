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
    # Fractional Max-Pooling

    See: https://arxiv.org/abs/1412.6071

    This replicates the model in "4.4 CIFAR-10 with dropout and training data augmentation", but without training data augmentation. Also, BatchNormalization has been added.
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
        F,
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
        device = torch.device("cpu")
        print("Sorry! The FractionalMaxPool2d is not implemented on MPS backend.")
        print(f"Using device: {device}")
    elif USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    return NUM_CPU, device


@app.cell
def _(mo):
    mo.md(r"""
    ## Calculate sizes
    """)
    return


@app.cell
def _():
    def get_fmp_sizes(alpha=2**(1/3), layers=12):
        """
        Calculates the target spatial sizes for each FMP layer by walking 
        backwards from the network output to the input.
        """
        # Start: The paper ends with C2-C1-Output. 
        # C2 (2x2 Conv) reduces a 2x2 volume to 1x1. 
        # Therefore, the input to the 'Tail' must be 2x2.
        size = 2 

        sizes = []

        # Walk backwards through the 12 blocks
        for _ in range(layers):
            # This 'size' is the target output for the current FMP layer
            sizes.append(size)

            # Inverse FMP: Increase size by factor alpha, round to nearest integer 
            size = int(round(size * alpha))

            # Inverse C2 Conv: Add 1 (because 2x2 conv with valid padding reduces size by 1)
            size = size + 1

        # 'sizes' is currently [Tail_Input, Block12_Input, ..., Block2_Input]
        # We need to reverse it to get [Block1_Output, Block2_Output, ...]
        return size, sizes[::-1]

    # Generate the list
    input_dim, fmp_targets = get_fmp_sizes()
    print(f"Calculated Input Size: {input_dim}")
    print(f"Target Sizes: {fmp_targets}")
    return (fmp_targets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define Model

    Before dropping the learning rate to 0.0001 and adding the Batch Normalization, the learning process was an absolute mess. Analyse what this kind of learning pattern means:

    ```
    [INFO] training network...
    Epoch [10/100] - Train Loss: 24982.7228, Train Acc: 0.1024, Val Loss: 1398.8729, Val Acc: 0.0940
    Epoch [20/100] - Train Loss: 32.5940, Train Acc: 0.1048, Val Loss: 3.1006, Val Acc: 0.1376
    Epoch [30/100] - Train Loss: 883.8111, Train Acc: 0.1005, Val Loss: 93.8354, Val Acc: 0.1001
    Epoch [40/100] - Train Loss: 142335.4038, Train Acc: 0.1001, Val Loss: 8272.5112, Val Acc: 0.0999
    Epoch [50/100] - Train Loss: 131.5608, Train Acc: 0.1080, Val Loss: 16.9567, Val Acc: 0.1003
    Epoch [60/100] - Train Loss: 472.5742, Train Acc: 0.1016, Val Loss: 388.8641, Val Acc: 0.1000
    Stopped training here
    ```
    """)
    return


@app.cell
def _(F, device, fmp_targets, nn):
    class FMPNet(nn.Module):
        def __init__(self, target_sizes, filter_growth_rate, num_classes=10):
            super().__init__()
            self.blocks = nn.ModuleList()
            in_channels = 3

            # We iterate 1..12 and zip with the pre-calculated target sizes
            # n determines filters, target_size determines FMP output
            for n, target_size in zip(range(1, 13), target_sizes):
                out_channels = filter_growth_rate * n 
                p_dropout = ((n - 1) / 11) * 0.5

                block = nn.Sequential(
                    # C2: 2x2 Conv, Valid Padding
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=0),
                    # This is potentially my own addition
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(p=p_dropout),

                    # FMP: Explicitly enforce the output size
                    nn.FractionalMaxPool2d(kernel_size=2, output_size=target_size)
                )
                self.blocks.append(block)
                in_channels = out_channels

            # "Tail" of the network
            # Input here is guaranteed to be 2x2 because the last FMP target was 2
            self.convC2 = nn.Conv2d(in_channels, in_channels, kernel_size=2) # 2x2 -> 1x1
            self.bnC2 = nn.BatchNorm2d(in_channels) # My own addition
            self.convC1 = nn.Conv2d(in_channels, num_classes, kernel_size=1) 

        def forward(self, x):
            # 1. Pad Input dynamically to match the calculated input_dim (94)
            # CIFAR is 32x32, so we need 62 padding total (31 per side)
            # This calculation makes it robust to different input sizes
            pad_total = 94 - x.size(2)
            if pad_total > 0:
                pad_val = pad_total // 2
                x = F.pad(x, (pad_val, pad_val, pad_val, pad_val), "constant", 0)

            for block in self.blocks:
                x = block(x)

            # Head Forward Pass
            x = self.convC2(x)
            x = self.bnC2(x) # Apply BN
            x = F.leaky_relu(x, 0.2)

            x = self.convC1(x)
            x = F.leaky_relu(x, 0.2)

            x = x.view(x.size(0), -1)
            return x

    filter_growth_rate = 64
    model = FMPNet(fmp_targets, filter_growth_rate, num_classes=10)
    model = model.to(device) # If using GPU

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
def _(F, model, nn, torch):
    def debug_forward_pass(model, input_size=(1, 3, 32, 32)):
        # 1. Detect device
        device = next(model.parameters()).device
        print(f"--- Debugging Forward Pass on {device} (Input: {input_size}) ---")

        # 2. Create input
        x = torch.randn(*input_size).to(device)

        # 3. Dynamic Padding (matches the model's logic)
        # The model expects 94x94.
        pad_total = 94 - x.size(2)
        if pad_total > 0:
            pad_val = pad_total // 2
            x = F.pad(x, (pad_val, pad_val, pad_val, pad_val), "constant", 0)
        print(f"After Padding: {x.shape}")

        # 4. Iterate Blocks
        for i, block in enumerate(model.blocks):
            print(f"\n[Block {i+1}] Input: {x.shape}")

            # Breakdown the block to see inside
            for layer in block:
                try:
                    x = layer(x)
                    # If it's FMP, we can print the target size it was aiming for
                    if isinstance(layer, nn.FractionalMaxPool2d):
                        print(f"   -> FMP (Target {layer.output_size}): {x.shape}")
                    else:
                        print(f"   -> {layer.__class__.__name__}: {x.shape}")
                except RuntimeError as e:
                    print(f"   !!! CRASH at {layer.__class__.__name__} !!!")
                    print(f"   Error: {e}")
                    return 

        # 5. Head (The "Fully Convolutional" Classifier)
        print(f"\n[Tail] Input: {x.shape} (Expected 2x2)")

        try:
            # C2: 2x2 Conv -> Reduces 2x2 input to 1x1
            x = model.convC2(x)
            x = F.leaky_relu(x, 0.2)
            print(f"   -> ConvC2 (2x2 kernel): {x.shape} (Should be 1x1)")

            # C1: 1x1 Conv -> Projects to Num_Classes
            x = model.convC1(x)
            x = F.leaky_relu(x, 0.2)
            print(f"   -> ConvC1 (1x1 kernel): {x.shape} (Channels = Num Classes)")

            # Flatten (This is the final output)
            x = x.view(x.size(0), -1)
            print(f"   -> Final Output (Flattened): {x.shape}")

        except Exception as e:
            print(f"   !!! CRASH in Head !!!")
            print(f"   Error: {e}")

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
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    BATCH_SIZE = 32

    # Option B: TensorBoard setup for logging metrics to disk
    device_name = str(device)
    run_name = f"cifar10_fractionalmaxp_{device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"TensorBoard logging to: runs/{run_name}")

    # Option B: Log configuration information to TensorBoard
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
    # CIFAR-10 stats
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToImage(),  # convert PIL/ndarray -> v2 image
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


    # Download and load the training data
    print("[INFO] accessing CIFAR10...")
    trainset = datasets.CIFAR10(
        "./data", download=True, train=True, transform=train_transform
    )
    testset = datasets.CIFAR10(
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

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update metric with predictions
            metric.update(outputs, labels)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize torchmetrics accuracy metric for 10 classes (MNIST digits)
    accuracy_metric = MulticlassAccuracy(num_classes=10).to(device)

    # Option A: Initialize local history dictionary to store metrics in memory
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

        # Option A: Append epoch metrics to local history lists
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Option B: Log comparative metrics (train vs val) to TensorBoard
        writer.add_scalars('Loss/train_vs_val', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        writer.add_scalars('Accuracy/train_vs_val', {
            'train': train_acc,
            'val': val_acc
        }, epoch)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print("[INFO] training complete!")

    # Option B: Log final hyperparameters with results for comparison across runs
    writer.add_hparams(
        {'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'device': device_name},
        {'hparam/final_train_acc': train_acc, 'hparam/final_val_acc': val_acc,
         'hparam/final_train_loss': train_loss, 'hparam/final_val_loss': val_loss}
    )

    # Option B: Close the TensorBoard writer to flush all remaining data
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
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save the Last Epoch
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
        'loss': history['train_loss'][-1], # Optional: save the last loss
    }, checkpoint_path)

    print(f"Full checkpoint saved to: {checkpoint_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize Training Curves
    """)
    return


@app.cell
def _(TOTAL_EPOCHS, history, np, plt):
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, TOTAL_EPOCHS), history["train_loss"], label="train_loss")
    plt.plot(np.arange(0, TOTAL_EPOCHS), history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, TOTAL_EPOCHS), history["train_acc"], label="train_acc")
    plt.plot(np.arange(0, TOTAL_EPOCHS), history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
