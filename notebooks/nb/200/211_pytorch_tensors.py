import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tensors - Official Tutorial

    ## Attribution

    This notebook is sourced from PyTorch's official tutorial "Learn the Basics" by Suraj Subramanian, Seth Juarez, Cassie Breviu, Dmitry Soshnikov, and Ari Bornstein.

    - **Original source**: https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
    - **License**: BSD 3-Clause License
    - **Copyright**: (c) 2017-2022, PyTorch contributors
    - **Modifications**: Stylistic changes and conversion to Marimo.

    Full license text: https://github.com/pytorch/tutorials/blob/main/LICENSE

    Reason for resharing: ease of access and integration into a larger collection of educational resources.

    ---
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    return np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Initializing a Tensor
    =====================

    Tensors can be initialized in various ways. Take a look at the following
    examples:

    **Directly from data**

    Tensors can be created directly from data. The data type is
    automatically inferred.
    """)
    return


@app.cell
def _(torch):
    data = [[1, 2],[3, 4]]
    x_data = torch.tensor(data)
    return data, x_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **From a NumPy array**

    Tensors can be created from NumPy arrays and vice versa.
    """)
    return


@app.cell
def _(data, np, torch):
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **From another tensor:**

    The new tensor retains the properties (shape, datatype) of the argument
    tensor, unless explicitly overridden.
    """)
    return


@app.cell
def _(torch, x_data):
    x_ones = torch.ones_like(x_data) # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **With random or constant values:**

    `shape` is a tuple of tensor dimensions. In the functions below, it
    determines the dimensionality of the output tensor.
    """)
    return


@app.cell
def _(torch):
    shape = (2,3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ------------------------------------------------------------------------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Attributes of a Tensor
    ======================

    Tensor attributes describe their shape, datatype, and the device on
    which they are stored.
    """)
    return


@app.cell
def _(torch):
    tensor = torch.rand(3,4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    return (tensor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ------------------------------------------------------------------------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Operations on Tensors
    =====================

    Over 1200 tensor operations, including arithmetic, linear algebra,
    matrix manipulation (transposing, indexing, slicing), sampling and more
    are comprehensively described
    [here](https://pytorch.org/docs/stable/torch.html).

    Each of these operations can be run on the CPU and
    [Accelerator](https://pytorch.org/docs/stable/torch.html#accelerators)
    such as CUDA, MPS, MTIA, or XPU. If you're using Colab, allocate an
    accelerator by going to Runtime \> Change runtime type \> GPU.

    By default, tensors are created on the CPU. We need to explicitly move
    tensors to the accelerator using `.to` method (after checking for
    accelerator availability). Keep in mind that copying large tensors
    across devices can be expensive in terms of time and memory!
    """)
    return


@app.cell
def _(tensor, torch):
    # We move our tensor to the current accelerator if available
    if torch.accelerator.is_available():
        tensor_accelerated = tensor.to(torch.accelerator.current_accelerator())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try out some of the operations from the list. If you're familiar with
    the NumPy API, you'll find the Tensor API a breeze to use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Standard numpy-like indexing and slicing:**
    """)
    return


@app.cell
def _(torch):
    tensor2 = torch.ones(4, 4)
    print(f"First row: {tensor2[0]}")
    print(f"First column: {tensor2[:, 0]}")
    print(f"Last column: {tensor2[..., -1]}")
    tensor2[:,1] = 0
    print(tensor2)
    return (tensor2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Joining tensors** You can use `torch.cat` to concatenate a sequence of
    tensors along a given dimension. See also
    [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html),
    another tensor joining operator that is subtly different from
    `torch.cat`.
    """)
    return


@app.cell
def _(tensor2, torch):
    t1 = torch.cat([tensor2, tensor2, tensor2], dim=1)
    print(t1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Arithmetic operations**
    """)
    return


@app.cell
def _(tensor2, torch):
    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    # ``tensor.T`` returns the transpose of a tensor
    y1 = tensor2 @ tensor2.T
    y2 = tensor2.matmul(tensor2.T)

    y3 = torch.rand_like(y1)
    torch.matmul(tensor2, tensor2.T, out=y3)


    # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor2 * tensor2
    z2 = tensor2.mul(tensor2)

    z3 = torch.rand_like(tensor2)
    torch.mul(tensor2, tensor2, out=z3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Single-element tensors** If you have a one-element tensor, for example
    by aggregating all values of a tensor into one value, you can convert it
    to a Python numerical value using `item()`:
    """)
    return


@app.cell
def _(tensor2):
    agg = tensor2.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In-place operations** Operations that store the result into the
    operand are called in-place. They are denoted by a `_` suffix. For
    example: `x.copy_(y)`, `x.t_()`, will change `x`.
    """)
    return


@app.cell
def _(tensor2):
    print(f"{tensor2} \n")
    tensor2.add_(5)
    print(tensor2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="background-color: #54c7ec; color: #fff; font-weight: 700; padding-left: 10px; padding-top: 5px; padding-bottom: 5px"><strong>NOTE:</strong></div>

    <div style="background-color: #f3f4f7; padding-left: 10px; padding-top: 10px; padding-bottom: 10px; padding-right: 10px">

    <p>In-place operations save some memory, but can be problematic when computing derivatives because of an immediate lossof history. Hence, their use is discouraged.</p>

    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ------------------------------------------------------------------------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Bridge with NumPy
    =================

    Tensors on the CPU and NumPy arrays can share their underlying memory
    locations, and changing one will change the other.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tensor to NumPy array
    =====================
    """)
    return


@app.cell
def _(torch):
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")
    return n, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A change in the tensor reflects in the NumPy array.
    """)
    return


@app.cell
def _(n, t):
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    NumPy array to Tensor
    =====================
    """)
    return


@app.cell
def _(np, torch):
    n2 = np.ones(5)
    t2 = torch.from_numpy(n2)
    return n2, t2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Changes in the NumPy array reflects in the tensor.
    """)
    return


@app.cell
def _(n2, np, t2):
    np.add(n2, 1, out=n2)
    print(f"t: {t2}")
    print(f"n: {n2}")
    return


if __name__ == "__main__":
    app.run()
