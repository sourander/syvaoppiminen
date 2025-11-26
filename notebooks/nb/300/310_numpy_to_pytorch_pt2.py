import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From NumPy to PyTorch

    In the previous notebook, we implemented a 2-2-1 neural network from scratch using NumPy. We had to:

    1. Manually implement the forward pass
    2. Manually calculate gradients for backpropagation
    3. Manually implement the weight updates

    In this notebook, we'll take a closer look at step 2: calculating gradients for backpropagation. We will use PyTorch to automatically compute these gradients for us, but, we will print out all the intermediate results and compare those to the manual calculations we did in the previous notebook.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np

    from dataclasses import dataclass
    return dataclass, nn, np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the PyTorch Model

    Note that this is from the previous notebook. Last time, we trained it and compared it to NumpyNN. Now, we no longer want to fully train it: we simply inspect how PyTorch handles the forward and backward passes.
    """)
    return


@app.cell
def _(nn, torch):
    class PyTorchNNInspectable(nn.Module):
        """Same as PyTorchNN but stores intermediate activations. 
    
        We also needed to fiddle with the sigmoid slightly to access the pre-activation values."""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.fc2 = nn.Linear(2, 1)
        
            # Store intermediate values
            self.Z1 = None
            self.A1 = None
            self.Z2 = None
    
        def forward(self, x):
            self.Z1 = self.fc1(x)
            self.A1 = torch.sigmoid(self.Z1)
            self.Z2 = self.fc2(self.A1)
            output = torch.sigmoid(self.Z2)
            return output
    return (PyTorchNNInspectable,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prep 1: Create a Fresh Model and Single Sample

    Let's use a single sample from our XOR dataset to trace the backward pass clearly.
    """)
    return


@app.cell
def _(PyTorchNNInspectable, torch):
    # Use a single sample: [0, 1] -> 1 (should output 1 for XOR)
    x_sample = torch.tensor([[0.0, 1.0]], requires_grad=False)
    y_sample = torch.tensor([[1.0]], requires_grad=False)

    # Instantiate the inspectable model
    inspect_model = PyTorchNNInspectable()

    # Store references to forward pass values for manual gradient calculation 
    A0 = x_sample  # Input layer activations (our input features)
    W1 = inspect_model.fc2.weight  # Output layer weights (for calculating dA1)

    # Forward pass - this populates the intermediate values we'll need
    A2 = inspect_model(x_sample)  # Final output (prediction)

    # Extract intermediate values that were stored during forward pass
    A1 = inspect_model.A1  # Hidden layer activations (after sigmoid)
    Z1 = inspect_model.Z1  # Hidden layer pre-activations (before sigmoid)
    Z2 = inspect_model.Z2  # Output layer pre-activations (before sigmoid)
    return A0, A1, A2, W1, inspect_model, x_sample, y_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note**: We'll compute gradients with respect to these Z values (dZ1, dZ2) during backpropagation to match our manual NumPy implementation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prep 2: Compute Loss and Backward Pass

    Now let's compute the loss and call `backward()` to let PyTorch compute all the gradients.

    Running this cell will populate the `.grad` attributes of all parameters in `inspect_model`. This will allow us to later on compare these gradients to our manual calculations.
    """)
    return


@app.cell
def _(A2, nn, y_sample):
    # Compute loss
    criterion = nn.BCELoss()
    loss = criterion(A2, y_sample)

    print(f"Loss: {loss.item():.7f}")

    # Perform backward pass
    loss.backward()
    return (loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prep 3: Mimic our NumPy Implementation

    Let's manually compute the gradients to verify PyTorch got it right! Or maybe.. that we got it right.

    For reference, the relating functions are:

    ```python
    def backward(self, target):
        self.dZ2 = self.A2 - target
        dA1 = self.dZ2.dot(self.W1.T)
        self.dZ1 = dA1 * self.sigmoid_derivative(self.A1)

    def optimize(self):
        self.W1 -= self.learning_rate * self.A1.T.dot(self.dZ2)
        self.b1 -= self.learning_rate * self.dZ2

        self.W0 -= self.learning_rate * self.A0.T.dot(self.dZ1)
        self.b0 -= self.learning_rate * self.dZ1
    ```

    Let's mimic this as closely as possible. Since we are not using the class, the syntax would end up lacking the `self.` part. We will mock the instance variables with a dataclass.
    """)
    return


@app.cell
def _(A0, A1, A2, W1, dataclass, np, y_sample):
    # This cell is just for mockery purposes
    # Focus on understanding the next cell!
    @dataclass
    class MockedInstanceVariables:
        A0: np.ndarray
        A1: np.ndarray
        A2: np.ndarray
        # W0: np.ndarray  # <- We are calculating gradients w.r.t. these weights
        W1: np.ndarray
        dZ1: np.ndarray | None = None
        dZ2: np.ndarray | None = None

        def sigmoid_derivative(self, x):
            return x * (1 - x)

    self = MockedInstanceVariables(
        A0=A0.detach().numpy(),
        A1=A1.detach().numpy(), # type: ignore
        A2=A2.detach().numpy(),
        # W0=W0.detach().numpy(),
        W1=W1.detach().numpy()
    )

    # --- This was a given argument for the backward pass ---
    target = y_sample.detach().numpy()
    return self, target


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backpropagation

    **Goal**: Manually compute gradients to verify PyTorch's automatic differentiation.

    We'll work backwards through the network (output → input), computing how much each
    parameter contributed to the loss. This follows the notation that e.g. Andrew Ng's notation uses, where `dZ`  represents the gradient with respect to pre-activation values.

    Compute gradients layer by layer (output → input).

    The LaTeX formulas below have been generated with Sonnet 4.5 from Anthropic.

    ### Step 1: Output Layer Error

    Compute the gradient at the output layer. For Binary Cross-Entropy loss with sigmoid activation:

    $$\frac{\partial L}{\partial Z_2} = A_2 - y$$

    This is the "error signal" at the output layer, giving a value for: how far our prediction is from the target.

    ### Step 2: Backpropagate to Hidden Layer

    Propagate the error backward through the network using the chain rule:

    1. First, propagate through the weights: $\frac{\partial L}{\partial A_1} = \frac{\partial L}{\partial Z_2} \cdot W_1$
    2. Then apply the activation derivative: $\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \sigma'(A_1)$

    where $\odot$ denotes element-wise multiplication.

    ### Step 3: Compute Output Layer Weight Gradients

    Calculate how much each weight and bias in the output layer contributed to the loss:

    $$\frac{\partial L}{\partial W_1} = A_1^T \cdot \frac{\partial L}{\partial Z_2}$$

    $$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial Z_2}$$

    ### Step 4: Compute Hidden Layer Weight Gradients

    Calculate how much each weight and bias in the hidden layer contributed to the loss:

    $$\frac{\partial L}{\partial W_0} = A_0^T \cdot \frac{\partial L}{\partial Z_1}$$

    $$\frac{\partial L}{\partial b_0} = \frac{\partial L}{\partial Z_1}$$

    Note: We transpose the results to match PyTorch's weight matrix shape convention (out_features, in_features).
    """)
    return


@app.cell
def _(np, self, target):
    # Step 1
    self.dZ2 = self.A2 - target

    # Step 2
    dA1 = np.dot(self.dZ2, self.W1)
    self.dZ1 = dA1 * self.sigmoid_derivative(self.A1)

    # Step 3
    dW1_manual = self.A1.T.dot(self.dZ2)
    dW1_manual = dW1_manual.T # Transpose to match PyTorch shape
    db1_manual = self.dZ2     # For bias, the partial derivative is just dZ

    # Step 4
    dW0_manual = self.A0.T.dot(self.dZ1) # type: ignore
    dW0_manual = dW0_manual.T
    db0_manual = self.dZ1

    # Step N
    # If we had more layers, we would continue propagating dZ backwards
    # through the network.

    print("\n=== MANUAL GRADIENTS (should match PyTorch) ===")
    print(f"\ndW1 (dL/dW1): \n{dW1_manual}")
    print(f"\ndb1 (dL/db1): \n{db1_manual}")
    print(f"\ndW0 (dL/dW0): \n{dW0_manual}")
    print(f"\ndb0 (dL/db0): \n{db0_manual}")
    return dW0_manual, dW1_manual, db0_manual, db1_manual


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finally: Compare PyTorch vs Manual Gradients

    Let's verify that PyTorch's automatic gradients match our manual calculations!
    """)
    return


@app.cell
def _(dW0_manual, dW1_manual, db0_manual, db1_manual, inspect_model, np):
    # Compare fc2 (W1, b1)
    pytorch_dW1 = inspect_model.fc2.weight.grad.numpy()
    pytorch_db1 = inspect_model.fc2.bias.grad.numpy()

    print("fc2.weight gradient:")
    print(f"  PyTorch:  {pytorch_dW1}")
    print(f"  Manual:   {dW1_manual}")
    print(f"  Match: {np.allclose(pytorch_dW1, dW1_manual)}")

    print("\nfc2.bias gradient:")
    print(f"  PyTorch:  {pytorch_db1}")
    print(f"  Manual:   {db1_manual.flatten()}")
    print(f"  Match: {np.allclose(pytorch_db1, db1_manual.flatten())}")

    # Compare fc1 (W0, b0)
    pytorch_dW0 = inspect_model.fc1.weight.grad.numpy()
    pytorch_db0 = inspect_model.fc1.bias.grad.numpy()

    print("\nfc1.weight gradient (flatted for readability):")
    print(f"  PyTorch:  {pytorch_dW0.flatten()}")
    print(f"  Manual:   {dW0_manual.flatten()}")
    print(f"  Match: {np.allclose(pytorch_dW0, dW0_manual)}")

    print("\nfc1.bias gradient:")
    print(f"  PyTorch:  {pytorch_db0}")
    print(f"  Manual:   {db0_manual}")
    print(f"  Match: {np.allclose(pytorch_db0, db0_manual.flatten())}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary: What We Learned

    This summary has been written with Sonnet 4.5 from Anthropic.

    **PyTorch automates backpropagation, but we can verify it matches our manual calculations!**

    1. **Capturing intermediate activations**: By storing values (Z1, A1, Z2) as instance variables in our custom `forward()` method, we can inspect what happens during PyTorch's forward pass without manual computation.

    2. **Accessing gradients**: After calling `loss.backward()`, PyTorch stores gradients in each parameter's `.grad` attribute:
       - `model.fc1.weight.grad` → gradient with respect to W0
       - `model.fc1.bias.grad` → gradient with respect to b0
       - `model.fc2.weight.grad` → gradient with respect to W1
       - `model.fc2.bias.grad` → gradient with respect to b1

    3. **Verification**: We manually computed the same gradients using NumPy (exactly like our previous implementation) and confirmed PyTorch's automatic gradients match perfectly.

    4. **The key insight**: PyTorch's `loss.backward()` performs the exact same chain rule calculations we coded manually:
       - `dZ2 = A2 - y` (for BCE + Sigmoid)
       - `dA1 = dZ2 @ W1.T`
       - `dZ1 = dA1 * sigmoid'(A1)`
       - Then computes weight/bias gradients from these

    **The power of PyTorch: It handles all gradient calculations automatically and correctly, no matter how complex your architecture becomes!**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus: GraphViz Visualization of the Model

    Note: this requires `graphviz` to be installed on your system. On macOS, you can install it via Homebrew:

    ```bash
    brew install graphviz
    ```

    On Windows, you would need to download and install it from the [Graphviz website](https://graphviz.org/download/) and make sure to add it to your system PATH.

    The graph shows the computational flow from inputs → loss, with nodes representing
    operations and edges showing dependencies. This visualizes what PyTorch tracks
    internally for automatic differentiation.
    """)
    return


@app.cell
def _(inspect_model, loss, mo, x_sample):
    from torchviz import make_dot

    # After forward pass
    output = inspect_model(x_sample)
    # loss = criterion(output, y_sample)

    # Create visualization
    dot = make_dot(loss, params=dict(inspect_model.named_parameters()))
    # dot.render("computation_graph", format="png")
    svg_string = dot.pipe(format='svg').decode('utf-8')
    mo.Html(svg_string)
    return


if __name__ == "__main__":
    app.run()
