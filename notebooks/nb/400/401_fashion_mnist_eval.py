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
    # Evaluate Fashion MNIST Model
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np

    from sklearn.metrics import classification_report, confusion_matrix
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    return (
        DataLoader,
        classification_report,
        confusion_matrix,
        datasets,
        nn,
        np,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
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
    ## Need to redefine the Model class

    You could also import it if you were writing modular code.

    **NOTE**! Replace this with your own implementation if it differs from mine.
    """)
    return


@app.cell
def _(nn):
    class FashionMLP(nn.Module):
        def __init__(self, input_size=784, output_size=10, hidden_layers=[256, 128]):
            super(FashionMLP, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.hidden_layers = hidden_layers

            # Build the network layers dynamically
            layers = []
            layer_sizes = [input_size] + hidden_layers + [output_size]

            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                # Add ReLU activation for all layers except the output layer
                if i < len(layer_sizes) - 2:
                    layers.append(nn.ReLU())

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            # Flatten the input
            x = x.view(-1, self.input_size)
            # Pass through the network
            x = self.network(x)
            return x
    return (FashionMLP,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Instantiate with Saved State

    **NOTE**! You may have used completely different kind of saving strategy. Modify as needed.
    """)
    return


@app.cell
def _(FashionMLP, device, torch):
    # This will obviously need to point to the model YOU saved.
    checkpoint = torch.load('models/fashion_mlp_mps_20251209_140953_model.pth')

    model = FashionMLP(
        input_size=checkpoint['input_size'],
        output_size=checkpoint['output_size'],
        hidden_layers=checkpoint['hidden_layers']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the Test Dataset

    No that the used Transformers were not saved (nor their parameters). This might be an issue in production, don't you think?
    """)
    return


@app.cell
def _(DataLoader, datasets, transforms):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.286041,), (0.353024,))
    ])

    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return test_dataset, test_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation begins
    """)
    return


@app.cell
def _(device, model, np, test_dataset, test_loader, torch):
    print("[INFO] evaluating network...")
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for ev_inputs, ev_labels in test_loader:
            ev_inputs = ev_inputs.to(device)
            ev_outputs = model(ev_inputs)
            _, ev_predicted = torch.max(ev_outputs.data, 1)

            all_predictions.extend(ev_predicted.cpu().numpy())
            all_labels.extend(ev_labels.numpy())

    # Convert to numpy arrays
    y_hat = np.array(all_predictions)
    y = np.array(all_labels)
    target_names=[test_dataset.classes[i] for i in range(10)]
    return all_labels, all_predictions, target_names, y, y_hat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Classification Report
    """)
    return


@app.cell
def _(classification_report, target_names, y, y_hat):
    # Print classification report
    print(classification_report(
        y, y_hat,
        target_names=target_names)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrix
    """)
    return


@app.cell
def _(all_labels, all_predictions, confusion_matrix, target_names):
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Text-based confusion matrix (most compact)
    print("Confusion Matrix (rows=true, cols=predicted):")
    print("             ", "  ".join(f"{i:4}" for i in range(10)))
    print("          " + "-" * 65)
    for i, row in enumerate(cm):
        print(f"{target_names[i]:10} | ", "  ".join(f"{val:4}" for val in row))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Top-k Accuracy

    We have only 10 classes, so you might want to investigate whether the top-k accuracy makes ANY sense. However, practice implementing it. Use `k=2`.
    """)
    return


@app.cell
def _():
    # IMPLEMENT
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check the Raw Output

    Run inference on a single image. Investigate what the output looks like.

    Try also running SoftMax on it.
    """)
    return


@app.cell
def _():
    # IMPLEMENT
    return


if __name__ == "__main__":
    app.run()
