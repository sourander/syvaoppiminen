import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Naive MLP for CIFAR10 with PyTorch

    This notebook implements a simple MLP for CIFAR10 image classification using PyTorch. The architecture is adapted from the MNIST implementation (3072-??-??-10) to handle color images. This implementation uses both two different ways to visualize training progress:

    * Option A: Plot training history using matplotlib (stored in local variables)
    * Option B: Log training progress to TensorBoard (stored in files in `runs/` folder)
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import os
    import torch.nn as nn

    from datetime import datetime
    from pathlib import Path
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report
    from torch.utils.tensorboard import SummaryWriter

    NUM_CPU = os.cpu_count()
    return (
        DataLoader,
        NUM_CPU,
        Path,
        SummaryWriter,
        classification_report,
        datasets,
        datetime,
        nn,
        np,
        optim,
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
    USE_GPU = False  # Toggle this to False to use CPU instead

    # Hyperparameters
    LEARNING_RATE = 0.01
    EPOCHS = 100
    BATCH_SIZE = 32

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
    run_name = f"cifar10_mlp_{device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"TensorBoard logging to: runs/{run_name}")

    # Option B: Log configuration information to TensorBoard
    writer.add_text('config/device', device_name)
    writer.add_text('config/hyperparameters', 
                    f'LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}')
    return (
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        device,
        device_name,
        run_name,
        writer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load and Prepare Data
    """)
    return


@app.cell
def _(BATCH_SIZE, DataLoader, NUM_CPU, datasets, transforms):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR10 mean and std
    ])

    # Download and load the training data
    print("[INFO] accessing CIFAR10...")
    trainset = datasets.CIFAR10('./data', download=True, train=True, transform=transform)
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, persistent_workers=True, num_workers=NUM_CPU)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, persistent_workers=True, num_workers=NUM_CPU)
    return testloader, trainloader, trainset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define the Model

    Choose your architecture here.
    """)
    return


@app.cell
def _(device, nn, torch, writer):
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(..., ...)
            self.fc2 = nn.Linear(..., ...)
            self.fc3 = nn.Linear(..., ...)

        def forward(self, x):
            # Flatten the input
            x = x.view(-1, ...)
            # First layer with sigmoid activation
            x = torch.sigmoid(self.fc1(x))
            # Second layer with sigmoid activation
            x = torch.sigmoid(self.fc2(x))
            # Output layer with softmax (will use cross entropy loss which includes softmax)
            x = self.fc3(x)
            return x

    # Initialize the model and move to device
    model = MLP().to(device)
    print(model)

    # Option B: Log model architecture graph to TensorBoard
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    writer.add_graph(model, dummy_input)
    print("Model graph logged to TensorBoard")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.cell
def _(torch):
    def train_epoch(model, train_loader, criterion, optimizer, device):
        """Train the model for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # Move data to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc


    def evaluate(model, data_loader, criterion, device):
        """Evaluate the model on validation/test data."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                # Move data to device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, dim=1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    return evaluate, train_epoch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train the Model

    The loss function is Cross Entropy Loss. You may see some materials using `NLLLoss` instead. Difference is that `NLLLoss` expects log-probabilities as input, while `CrossEntropyLoss` expects raw logits. Since our model outputs raw logits, we use `CrossEntropyLoss`. If we swapped to `NLLLoss`, we would need to add a `LogSoftmax` layer at the end of the model.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    datetime,
    device,
    device_name,
    evaluate,
    model,
    nn,
    optim,
    testloader,
    train_epoch,
    trainloader,
    writer,
):
    start = datetime.now()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Add scheduler - reduces LR when test loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=0.2,          # reduce by multiplying LR with this
        patience=5,          # wait 5 epochs before reducing
    )

    # Option A: Initialize local history dictionary to store metrics in memory
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print("[INFO] training network...")
    for epoch in range(EPOCHS):
        # Training phase using helper function
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)

        # Validation phase using helper function
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

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
def _(Path, model, run_name, torch):
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
    ## Evaluate the Model
    """)
    return


@app.cell
def _(classification_report, device, model, np, testloader, torch, trainset):
    print("[INFO] evaluating network...")
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for ev_inputs, ev_labels in testloader:
            ev_inputs = ev_inputs.to(device)
            ev_outputs = model(ev_inputs)
            _, ev_predicted = torch.max(ev_outputs.data, 1)

            all_predictions.extend(ev_predicted.cpu().numpy())
            all_labels.extend(ev_labels.numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Print classification report using class names from the dataset
    print(classification_report(all_labels, all_predictions,
                              target_names=trainset.classes))
    return all_labels, all_predictions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion Matrix
    """)
    return


@app.cell
def _(all_labels, all_predictions, trainset):
    from sklearn.metrics import confusion_matrix

    # Get class names from dataset
    class_names = trainset.classes

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Confusion Matrix
    print("             ", "  ".join(f"{i:4}" for i in range(10)))
    print("          " + "-" * 65)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:10} | ", "  ".join(f"{val:4}" for val in row))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize Training History
    """)
    return


@app.cell
def _(EPOCHS, history, np, plt):
    # Option A: Plot training history from local variables using matplotlib
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
