import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Auto MPG

    The idea of this exercise is to lower the threshold of using PyTorch instantly. We already know Pandas, but PyTorch Dataset Loader is unknown to us, so... um.. let's use Pandas!

    ## Dataset

    We will predict the mpg (miles per gallon) of a car given the following features:

    1. cylinders:     multi-valued discrete
    2. displacement:  continuous
    3. horsepower:    continuous
    4. weight:        continuous
    5. acceleration:  continuous
    6. (model )year:    multi-valued discrete
    7. origin:        multi-valued discrete
    8. (car )name:      string (unique for each instance)

    Two of the columns have shortened names in the dataset we will be using. The *model year* is simply a *year*. Same applies to *car name*. In total, we will have **44 columns** after the One-Hot Encoding (or which one is the $y$.)

    ## Credits

    The idea for this exercise is from this video [PyTorch Network Definition Class or Sequence? (3.5)](https://www.youtube.com/watch?v=NOu8jMZx3LY) by Jeff Heaton.

    The original dataset can be downloaded from [UC Irvine Machine Learning Repository](
    https://archive.ics.uci.edu/dataset/9/auto+mpg) and it has the CC-BY 4.0 License. However, we will download it from Jeff Heaton's given URI unless it goes down.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    return F, nn, pd, torch


@app.cell
def _(torch):
    USE_GPU = True

    if USE_GPU and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    elif USE_GPU and torch.cuda.is_available():
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

    Load the CSV and make it into a DataFrame defined above.

    ### Extra Task

    Practice using the Marimo's DataFrame features! Instead of simply browsing the tabular data, you can use drag&drop features to create various plots.

    1. Plot the horsepower vs. mpg as scatter plot (without mean aggregation)
    2. Plot the horsepower vs. mpg as scatter plot (without NO aggregation)
    3. Plot whatever you want. Play with the tool.

    **WARNING!** Please keep in mind that the chart is not saved. You have to copy-paste to code into a new Cell to save it.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", na_values=["NA", "?"])

    # Extract make (first word) from name column
    df["make"] = df["name"].str.split().str[0]

    # One-hot encode the make column
    df = pd.get_dummies(df, columns=["make"], prefix="make", dtype=float, drop_first=True)

    # Drop the original name column (optional, if you don't need it)
    df = df.drop(columns=["name"])

    # List all columns start start with 'make_'
    make_columns = [col for col in df.columns if col.startswith("make_")]

    df
    return df, make_columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Engineering
    """)
    return


@app.cell
def _(df):
    # Simple median fill for missing data
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pandas to PyTorch Tensor
    """)
    return


@app.cell
def _(device, df, make_columns, torch):
    # Convert pandas DataFrame to PyTorch tensors
    x = torch.from_numpy(
        df[
            [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "year",
                "origin",
                *make_columns
            ]
        ].values,
    ).to(device, dtype=torch.float32)

    # Very simple standard scaling
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True)
    x_scaled = (x - x_mean) / (x_std + 1e-8)

    y = torch.from_numpy(df["mpg"].values).to(device, dtype=torch.float32)
    return x, x_scaled, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Most Naive Test Split Ever

    Here we make a strong assumption that the dataset is not ordered in some meaningful way.
    """)
    return


@app.cell
def _(x_scaled, y):
    # Simple 80-20 split using indexing
    split_idx = int(x_scaled.shape[0] * 0.8)

    x_train = x_scaled[:split_idx]
    x_test = x_scaled[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}")
    return x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Model

    Let's use identical model than what Jeff Heaton is using. Note that you can try modifying the model. You can treat this like the Tensorflow Playground. For example, try making layers smaller (in terms of neuron count) but adding one extra layer.

    It is a good idea to document the network structure on each trial and some key metrics like MAE (Mean Absolute Error). Example:

    | Arch          | MAE    | # params | epochs |
    | ------------- | ------ | -------: | ------ |
    | 43-50-25-1    | 6.0134 | 3501     | 1000   |
    | 43-64-32-1    | ?      | ???      |  ???   |
    | 43-32-16-16-1 | ?      | ???      |  ???   |

    **NOTE!** We have locked the PyTorch seed, making the weight initialization deterministic. Try removing it and see what happens to the results between consecutive Notebook runs. How easily can you trust the single MAE value in the table above?
    """)
    return


@app.cell
def _(F, device, nn, torch, x):
    class Net(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_dim, 50)
            self.fc2 = nn.Linear(50, 25)
            self.fc3 = nn.Linear(25, output_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    # Let's make model initialization deterministic
    torch.manual_seed(42)

    # Instantiate
    model = Net(x.shape[1], 1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Print for clarity
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print("TOTAL PARAMETERS: ", total_params)
    return loss_fn, model, optimizer


@app.cell
def _(loss_fn, model, optimizer, x_train, y_train):
    # Training loop
    for epoch in range(1000):
        # Zero gradients
        optimizer.zero_grad()
    
        # Forward pass
        outputs = model(x_train).flatten()

        # Compute loss
        loss = loss_fn(outputs, y_train)
    
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss.item()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results
    """)
    return


@app.cell
def _(loss_fn, model, torch, x_test, x_train, y_test, y_train):
    # Make predictions on both sets
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        y_train_pred = model(x_train).flatten()
        y_test_pred = model(x_test).flatten()

    # Calculate metrics for both sets
    train_rmse = torch.sqrt(loss_fn(y_train_pred, y_train))
    train_mae = torch.mean(torch.abs(y_train - y_train_pred))

    test_rmse = torch.sqrt(loss_fn(y_test_pred, y_test))
    test_mae = torch.mean(torch.abs(y_test - y_test_pred))

    print("=" * 50)
    print(f"Train RMSE: {train_rmse.item():>8.4f} | Train MAE: {train_mae.item():>8.4f}")
    print(f"Test RMSE:  {test_rmse.item():>8.4f} | Test MAE:  {test_mae.item():>8.4f}")
    print("=" * 50)
    return


if __name__ == "__main__":
    app.run()
