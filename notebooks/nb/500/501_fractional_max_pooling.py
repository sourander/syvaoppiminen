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

    This replicates the model in "4.4 CIFAR-10 with dropout and training data augmentation", but without training data augmentation.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import os
    import torchvision.transforms.v2 as transforms
    import matplotlib.pyplot as plt

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define Model
    """)
    return


@app.cell
def _():
    1 / 2**(1/3)
    return


@app.cell
def _(device, nn, torch):
    class FMPNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.blocks = nn.ModuleList()
            in_channels = 3  # RGB input

            # 12 repeated blocks of (160nC2 - FMP √3 2)
            # Use padding=1 with kernel=3 to preserve spatial dims, let FMP do downsampling
            # Adjust output_ratio to ~0.85 so we don't shrink too fast over 12 layers
            for _ in range(12):
                # The convolutional layer
                conv = nn.Conv2d(in_channels, 160, kernel_size=3, padding=1, bias=True)
            
                # FMP with output_ratio = 1 / 2^(1/3) ≈ 0.794
                fmp = nn.FractionalMaxPool2d(kernel_size=2, output_ratio=(0.83, 0.83))
                self.blocks.append(nn.Sequential(conv, fmp))
                in_channels = 160  # output of conv

            # Additional conv layers
            self.convC2 = nn.Conv2d(160, 160, kernel_size=1, bias=True)
            self.convC1 = nn.Conv2d(160, 50, kernel_size=1, bias=True)  # final conv before FC

            # Activation and pooling modules
            self.relu = nn.ReLU()
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            # Classifier (global average pool → linear)
            self.fc = nn.Linear(50, num_classes)

        def forward(self, x):
            for block in self.blocks:
                x = self.relu(block(x))
            x = self.relu(self.convC2(x))
            x = self.relu(self.convC1(x))
            # Global average pool to 1x1
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    # Instantiate model
    model = FMPNet(num_classes=10)
    model = model.to(device)
    return (model,)


@app.cell
def _(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    return


@app.cell
def _(device, model, torch):
    def print_layer_sizes(model, input_size=(1, 3, 32, 32)):
        """Print spatial dimensions through the network."""
        x = torch.randn(input_size).to(device)
        print(f"Input: {x.shape}")
    
        for i, block in enumerate(model.blocks):
            x = model.relu(block(x))
            print(f"Block {i+1}: {x.shape}")
    
        x = model.relu(model.convC2(x))
        print(f"ConvC2: {x.shape}")
    
        x = model.relu(model.convC1(x))
        print(f"ConvC1: {x.shape}")
    
        x = model.global_avg_pool(x)
        print(f"Global Avg Pool: {x.shape}")
    
        x = torch.flatten(x, 1)
        x = model.fc(x)
        print(f"Output: {x.shape}")

    # Run it
    print("[INFO] Investigating layer sizes")
    print("=" * 42)
    print_layer_sizes(model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperparameters and Tensorboard
    """)
    return


@app.cell
def _(SummaryWriter, datetime, device):
    LEARNING_RATE = 0.01
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
    CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToImage(),  # convert PIL/ndarray -> v2 image
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])


    # Download and load the training data
    print("[INFO] accessing CIFAR10...")
    trainset = datasets.CIFAR10('./data', download=True, train=True, transform=train_transform)
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=test_transform)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, persistent_workers=True, num_workers=NUM_CPU)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, persistent_workers=True, num_workers=NUM_CPU)
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
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save the Model
    """)
    return


@app.cell
def _(model, run_name, torch):
    from pathlib import Path

    # Create models directory if it doesn't exist
    Path('models').mkdir(parents=True, exist_ok=True)

    # Save the model as state dict only. 
    # We will use the runname to differentiate models.
    model_path = f'models/{run_name}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the Training History
    """)
    return


@app.cell
def _(EPOCHS, history, np, plt):
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
    return


if __name__ == "__main__":
    app.run()
