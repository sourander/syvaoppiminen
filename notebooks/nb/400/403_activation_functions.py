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
    ## Activation Functions

    We will plot the following activation functions:

    * Step function (aka Heaviside)
    * Logistic Function (aka Sigmoid)
    * Tanh
    * Arctan
    * ReLU
    * Leavy ReLU
    * exp-LU
    * Softplus

    We will plot each for input (`x`) range -10 … 10.
    """)
    return


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt
    return plt, torch


@app.cell
def _(torch):
    x = torch.linspace(-10.0, 10.0, steps=1000)
    x.shape
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting Helper Functions
    """)
    return


@app.cell
def _(torch):
    def heaviside(x):
        """Step function (Heaviside)"""
        return torch.heaviside(x, torch.tensor(0.5))

    def sigmoid(x):
        """Logistic Function (Sigmoid)"""
        return torch.sigmoid(x)

    def tanh(x):
        """Hyperbolic Tangent"""
        return torch.tanh(x)

    def arctan(x):
        """Arctan"""
        return torch.atan(x)

    def relu(x):
        """Rectified Linear Unit"""
        return torch.relu(x)

    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU"""
        return torch.nn.functional.leaky_relu(x, negative_slope=alpha)

    def elu(x, alpha=1.0):
        """Exponential Linear Unit"""
        return torch.nn.functional.elu(x, alpha=alpha)

    def softplus(x):
        """Softplus"""
        return torch.nn.functional.softplus(x)
    return arctan, elu, heaviside, leaky_relu, relu, sigmoid, softplus, tanh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting Logic

    The diagrams are plotted in the same order as in book **Essential Math for AI** by Hala Nelson (O'Reilly, 2023). Thus, this matches to the Figure 4-5 in the Chapter 4. Nelson explains, that: *"first row consists of sigmoidal-type activation functions, shaped like the letter S. These saturate (become flat and output the same values) for inputs large in magnitude. The second row consists of ReLU-type activation functions, which do not saturate."*
    """)
    return


@app.cell(hide_code=True)
def _(
    arctan,
    elu,
    heaviside,
    leaky_relu,
    plt,
    relu,
    sigmoid,
    softplus,
    tanh,
    x,
):
    # Create 2x4 grid (2 rows, 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Activation Functions', fontsize=16, y=1.00)

    # Define functions and titles in order
    functions = [
        (heaviside, 'Step Function (Heaviside)'),
        (sigmoid, 'Logistic Function (Sigmoid)'),
        (tanh, 'Tanh'),
        (arctan, 'Arctan'),
        (relu, 'ReLU'),
        (leaky_relu, 'Leaky ReLU'),
        (elu, 'ELU'),
        (softplus, 'Softplus'),
    ]

    # Plot each function
    for idx, (func, title) in enumerate(functions):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        y = func(x)
        ax.plot(x.numpy(), y.numpy(), linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extra: Swift

    You will meet this later on in the course, as it is introduced by Aurélien Géron as a good default one to use for deep models.
    """)
    return


@app.cell
def _(torch):
    def swift(x):
        """Swift (alias SiLU)"""
        return torch.nn.functional.silu(x)
    return (swift,)


@app.cell
def _(plt, swift, x):
    def plot_swift(x, y):
        # Create a single plot for Swift (SiLU)
        fig2, ax = plt.subplots(1, 1, figsize=(6, 5))
    
        ax.plot(x.numpy(), y.numpy(), linewidth=2)
        ax.set_title('Swift (SiLU)', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
    
        plt.tight_layout()
        return plt.gca()

    plot_swift(x, swift(x))
    return


if __name__ == "__main__":
    app.run()
