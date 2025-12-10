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
    ## Function Chain and Derivation

    Create the following computation in PyTorch and perform the backpropagation:

    $$
    \begin{aligned}
    a &= x^3 \\
    b &= y^2 \\
    c &= a \odot b  \\
    d &= |\sin(\frac{c}{b})| \\
    e &= \sqrt{\frac{d}{z}} \\
    f &= \frac{1}{n}\sum_i e_i
    \end{aligned}
    $$
    """)
    return


@app.cell
def _():
    import torch
    return (torch,)


@app.cell
def _(torch):

    def compute(x):
        x = torch.tensor(x, requires_grad=True)
        y = torch.tensor([1.5, 1.0, 1.5], requires_grad=True)
        z = torch.tensor(0.5, requires_grad=True)

        a = ...
        b = ...
        c = ...
        d = ...
        e = ...
        f = torch.tensor(0.123)

        # Backward pass here maybe?

        return f.detach().clone()
    return (compute,)


@app.cell
def _(compute):
    f1 = compute([1.0, 2.0, 3.0])
    f2 = compute([1.0, 2.0, 3.00001])
    return f1, f2


@app.cell
def _(f1, f2):
    delta = (f2 - f1).item()

    # The first value her should be x.grad[2].
    # Print it in compute() function or return that too?
    calculated = -1.9011 * .00001 
    return calculated, delta


@app.cell
def _(calculated, delta):
    print(f"Actual delta:    {delta}")
    print(f"Predicted delta: {calculated}")
    print(f"Relative error:  {(delta - calculated) / calculated * 100:.2f}%")
    return


if __name__ == "__main__":
    app.run()
