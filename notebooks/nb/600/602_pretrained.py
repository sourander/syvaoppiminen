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
    ## Using Pretrained Models

    We will use MobileNet V3 Small.
    """)
    return


@app.cell
def _():
    import torch
    import requests
    import matplotlib.pyplot as plt

    from pathlib import Path
    from torchvision import models
    from torchvision.io import decode_image
    return Path, decode_image, models, plt, requests, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Device
    """)
    return


@app.cell
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
    ## List Models

    You should look into the models by browsing the PyTorch docs, e.g. [Models and pre-trained weights](https://docs.pytorch.org/vision/main/models.html). As can be read from the docs, there is a `list_models` helper function that returns the string-names for all models.
    """)
    return


@app.cell
def _(models):
    [x for x in models.list_models() if 'mobile' in x]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load with Weights

    We will need the weights, since they are the parameters that have been trained. The model itself has been trained on ImageNet.
    """)
    return


@app.cell
def _(device, models):
    # The weights
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1

    # Model itself
    pretrained_model = models.mobilenet_v3_small(weights=weights)

    # Do try commenting this out! You will notice that the model behaves VERY differenly
    # in training and in inference mode. Your prediction will not be correct without this.
    pretrained_model.eval()

    # We want to also move it to GPU
    pretrained_model = pretrained_model.to(device)

    # The transforms for inference
    preprocess = weights.transforms()

    # Print what is going on
    print(preprocess)
    return preprocess, pretrained_model, weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose a Known Class

    Obviously, the model can only classify images to labels that have been part of the original training set. There are 1000 labels, of which one is a magpie (fin. harakka). I will choose that, but you can choose any.
    """)
    return


@app.cell
def _(weights):
    weights.meta["categories"][18]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data
    """)
    return


@app.cell
def _(Path, decode_image, requests):
    img_path = Path("data/images/magpie.jpg")

    if not img_path.exists():
        print(f"Downloading the image to {img_path}")
        img_path.parent.mkdir(exist_ok=True)
        url = "https://upload.wikimedia.org/wikipedia/commons/c/cc/Eurasian_magpie_%28Pica_pica%29.jpg"
        headers = {'User-Agent': 'DeepLearningBasics KAMK'}
        r = requests.get(url, headers=headers)
        if r.ok:
            img_path.write_bytes(r.content)
        else:
            r.raise_for_status()
    else:
        print("Already downloaded")

    img = decode_image(img_path)
    print("Loaded image with shape", img.shape)
    return (img,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Display image
    """)
    return


@app.cell
def _(img, plt):
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0))
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference
    """)
    return


@app.cell
def _(device, img, preprocess, pretrained_model, weights):
    batch = preprocess(img).unsqueeze(0)
    batch = batch.to(device)

    prediction = pretrained_model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")
    return (prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Most Similar Labels

    You may want to also investigate the $k$ most similar labels.
    """)
    return


@app.cell
def _(prediction, torch, weights):
    def print_k_best(prediction, k=5):
        top_vals, top_idx = torch.topk(prediction, k)
        for i in range(k):
            class_id = top_idx[i].item()
            score = top_vals[i].item()
            category_name = weights.meta["categories"][class_id]
            print(f"{category_name}: {100 * score:.1f}%")

    print_k_best(prediction)
    return


if __name__ == "__main__":
    app.run()
