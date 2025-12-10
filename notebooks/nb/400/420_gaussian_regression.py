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
    # Gaussian Regression

    Multi-head model for predicting mean and variance in one go.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import seaborn as sns
    return nn, optim, plt, sns, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate Data
    """)
    return


@app.cell
def _(torch):
    # Generate synthetic data
    n_samples = 1000
    X = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    print(f"Generated X with shape: {X.shape}")

    # True function with varying noise
    true_mean = 0.5 * X.squeeze() ** 2 + 0.1 * torch.sin(5 * X.squeeze())

    # Noise that increases with |x|
    noise_std = 0.1 + 0.2 * torch.abs(X.squeeze())
    noise = torch.randn(n_samples) * noise_std
    y = true_mean + noise
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the Original Data

    The data exhibits **heteroscedastic noise**, meaning the variance of the noise is not constant across all values of X. In this case, the noise is lower when X is close to 0 and increases proportionally to |X|. This creates a funnel-shaped pattern in the scatter plot, where data points are tightly clustered near X=0 and become more spread out as |X| increases.
    """)
    return


@app.cell
def _(X, plt, sns, y):
    def generate_plot():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X.squeeze().numpy(), y=y.numpy(), alpha=0.6, ax=ax)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('Synthetic Data with Heteroscedastic Noise')
        return fig

    generate_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model
    """)
    return


@app.cell
def _(nn, torch):
    class GaussianRegressionModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64):
            super(GaussianRegressionModel, self).__init__()

            # Shared layers
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            # Mean prediction head
            self.mean_head = nn.Linear(hidden_dim, 1)

            # Variance prediction head (outputs log-variance for numerical stability)
            self.log_var_head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            # Shared feature extraction
            features = self.shared(x)

            # Predict mean
            mean = self.mean_head(features)

            # ensure variance is positive, numerically stable
            var = torch.nn.functional.softplus(self.log_var_head(features))

            return mean.squeeze(), var.squeeze()
    return (GaussianRegressionModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training
    """)
    return


@app.cell
def _(GaussianRegressionModel, X, nn, optim, y):
    # Initialize model, loss, and optimizer
    model = GaussianRegressionModel()
    criterion = nn.GaussianNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    n_epochs = 150
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        pred_mean, pred_var = model(X)

        # Compute loss
        loss = criterion(pred_mean, y, pred_var)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    return losses, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eval

    Note that `standard_deviation == sqrt(variance)`
    """)
    return


@app.cell
def _(X, model, torch):
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_hat_mean, y_hat_var = model(X)
        y_hat_std = torch.sqrt(y_hat_var)
    return y_hat_mean, y_hat_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot Results

    Note that `±2σ confidence` refers to two standard deviations (added or subtracted). Under a Gaussian assumption, roughly 95% of the population (or 95% of future observations) are expected to fall within this range.”
    """)
    return


@app.cell
def _(X, losses, plt, sns, y, y_hat_mean, y_hat_std):
    def generate_results_plot():
        # Set seaborn style for better aesthetics
        sns.set_style("whitegrid")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Predictions with confidence intervals
        X_np = X.numpy().flatten()
        y_np = y.numpy()
        y_hat_mean_np = y_hat_mean.numpy()
        y_hat_std_np = y_hat_std.numpy()

        ax1.scatter(X_np, y_np, alpha=0.3, label='Data', s=20)
        ax1.plot(X_np, y_hat_mean_np, 'r-', label='Predicted Mean', linewidth=2)
        ax1.fill_between(X_np,
                        y_hat_mean_np - 2*y_hat_std_np,
                        y_hat_mean_np + 2*y_hat_std_np,
                        alpha=0.3, color='red', label='±2σ Confidence')
        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        ax1.set_title('Gaussian Regression Results')
        ax1.legend()

        # Right plot: Training loss
        ax2.plot(losses, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')

        plt.tight_layout()
        return fig

    generate_results_plot()
    return


if __name__ == "__main__":
    app.run()
