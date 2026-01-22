import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ySkills Dataset - PyTorch Neural Network

    This notebook demonstrates training a fully connected neural network on the ySkills longitudinal dataset using PyTorch. The goal is to predict the binary risk indicator `RISK101`.

    The dataset has been preprocessed with one-hot encoding for categorical variables and standardization for ordinal variables. A baseline Logistic Regression achieves ~75% accuracy.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
    from torch.utils.data import DataLoader, TensorDataset

    from huggingface_hub import hf_hub_download
    return (
        DataLoader,
        TensorDataset,
        hf_hub_download,
        nn,
        np,
        optim,
        pd,
        plt,
        torch,
        train_test_split,
    )


@app.cell
def _(np, torch):
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Check for MPS (Apple Silicon) or CUDA availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"PyTorch version: {torch.__version__}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download the dataset

    Note that this uses a Hugging Face download function. This is a sneak peak for the upcoming lessons. We will end up using some datasets and/or models from Hugging Face.
    """)
    return


@app.cell
def _(hf_hub_download, pd):
    filepath = hf_hub_download(
        repo_id="sourander/yskills",
        repo_type="dataset",
        filename="ySKILLS_longitudinal_dataset_teacher_processed.csv",
    )

    print(f"The file was downloaded to Hugging Face 🤗 cache directory: {filepath}")
    df = pd.read_csv(filepath)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the Neural Network Architecture

    Here are interactive Marimo sliders that you can use to modify the major hyperparameters. Notice that you are free to modify whatever you want. Don't feel that you are limited by these sliders: you can even destroy them and do as you like.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n = mo.ui.slider(1, 10, label="no. of hidden layers")

    D = mo.ui.dropdown(
        options={"4": 4, "8": 8, "16": 16, "32": 32},
        value="4",
        label="no.  of neurons per hidden layer",
    )

    batch_size = mo.ui.dropdown(
        options={"1": 1, "16": 16, "32": 32, "64": 64, "128": 128, "256": 256},
        value="32",
        label="Batch size",
    )

    num_epochs = mo.ui.slider(10, 100, label="no. of epochs to train")

    run_button = mo.ui.run_button()

    lr = mo.ui.dropdown(
        options={"0.0001": 0.0001, "0.001": 0.001, "0.01": 0.01},
        value="0.001",
        label="Learning rate",
    )

    mo.vstack(
        [
            mo.hstack([lr, batch_size]),
            mo.hstack([n, D, num_epochs, run_button])
        ]
    )
    return D, batch_size, lr, n, num_epochs, run_button


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare Data for PyTorch
    """)
    return


@app.cell
def _(
    DataLoader,
    TensorDataset,
    batch_size,
    device,
    df,
    torch,
    train_test_split,
):
    # Split the dataset into features (X) and target variable (y)
    X = df.drop('RISK101', axis=1).values
    y = df['RISK101'].values

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {X.shape[1]}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # Create DataLoader for batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size.value, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size.value, shuffle=False)

    print(f"Batch size: {batch_size.value}")
    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")
    return X_train, test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Model itself
    """)
    return


@app.cell(hide_code=True)
def _(D, X_train, device, lr, n, nn, optim):
    class BinaryClassifier(nn.Module):
        def __init__(self, input_size, hidden_sizes):
            """
            Fully connected neural network for binary classification.

            Args:
                input_size: Number of input features
                hidden_sizes: List of hidden layer sizes
                dropout_rate: Dropout probability for regularization
            """
            super(BinaryClassifier, self).__init__()

            # Build the network layers
            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size

            # Output layer (single neuron for binary classification)
            layers.append(nn.Linear(prev_size, 1))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # Get input size from the data
    input_size = X_train.shape[1]

    # Define network architecture
    hidden_sizes = [D.value] + n.value * [16]
    print(hidden_sizes)

    # Create the model
    model = BinaryClassifier(input_size, hidden_sizes)
    model = model.to(device)

    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss function - Binary Cross Entropy
    criterion = nn.BCEWithLogitsLoss(
        # torch.tensor([0.6911], dtype=torch.float32).to(device)
    )

    # Optimizer - Adam with learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr.value)

    # Add scheduler - reduces LR when test loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=0.5,          # reduce by multiplying LR with this
        patience=5,          # wait 5 epochs before reducing
    )

    print(f"Loss function: {criterion}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {lr.value}")
    return criterion, model, optimizer, scheduler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Helper Functions
    """)
    return


@app.cell(hide_code=True)
def _(torch):
    def train_epoch(model, train_loader, criterion, optimizer, device):
        """Train the model for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * X_batch.size(0)
            predicted = (outputs >= 0.5).float()
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
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                predicted = (outputs >= 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    return evaluate, train_epoch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Loop
    """)
    return


@app.cell(hide_code=True)
def _(
    criterion,
    device,
    evaluate,
    mo,
    model,
    num_epochs,
    optimizer,
    run_button,
    scheduler,
    test_loader,
    train_epoch,
    train_loader,
):
    mo.stop(not run_button.value, mo.md("Click 👆 to run this cell"))

    # Track history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    print("Starting training...\n")

    for epoch in range(num_epochs.value):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs.value}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")
            print()
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize Training History
    """)
    return


@app.cell
def _(history, plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['test_acc'], label='Test Accuracy', linewidth=2)
    ax2.axhline(y=0.745, color='r', linestyle='--', label='Logistic Regression Baseline (75%)', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate Model Performance

    This is where your work begins.

    * Change the architecture and model parameters (above)
    * Write the code for model evaluation (below)

    What evaluation? It might be a good idea to simply copy-paste code from your Johdatus koneoppimiseen courses repository. Classification report, confusion matrix, ROC Curve and ROC AUC score might be a good set.

    Hint: you will most likely need something that looks like:

    ```python
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor).squeeze()
        y_pred_probs = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_pred_probs >= 0.5).astype(int)
    ```
    """)
    return


@app.cell
def _():
    # IMPLEMENT: Your stuff
    return


if __name__ == "__main__":
    app.run()
