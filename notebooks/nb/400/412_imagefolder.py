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
    # ImageFolder

    TODO
    """)
    return


@app.cell
def _():
    import torch

    from pathlib import Path
    from PIL import Image
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    return DataLoader, Image, ImageFolder, Path, plt, torch, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate Dataset

    It will be 3x3 images (3 of each colors red, green, blue)
    """)
    return


@app.cell
def _(Image, Path, torch):
    def generate_images(out_dir_path, num_images=3, img_size=(100,100)):

        colors = {
        'red': torch.tensor([1.0, 0.0, 0.0]),
        'green': torch.tensor([0.0, 1.0, 0.0]),
        'blue': torch.tensor([0.0, 0.0, 1.0])
        }
    
        for color_name, color_tensor in colors.items():
            class_dir_path = out_dir_path / color_name
            class_dir_path.mkdir(parents=True, exist_ok=True)
    
            for idx in range(1, num_images + 1):
                # Generate random image and push toward the color
                img = torch.rand(3, *img_size) * 0.3 + color_tensor.view(3, 1, 1) * 0.7
                img = (img * 255).clamp(0, 255).byte()  # convert to 0-255
                img_pil = Image.fromarray(img.permute(1, 2, 0).numpy())  # C,H,W -> H,W,C
    
                # Save image
                save_path = class_dir_path / f"{idx:03d}.png"
                img_pil.save(save_path)

        print("Done! Images saved to ./data/colors/{red,green,blue}/")

    out_dir_path = Path("./data/colors/")

    # Create if not exists
    if out_dir_path.exists():
        print("Directory already exists. No need to re-generate.")
    else:
        generate_images(out_dir_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the Dataset

    **Task**: fix the following cell so that instead of alphabetic ordering, you will use your own `custom_class_to_idx` map.
    """)
    return


@app.cell
def _(ImageFolder, transforms):
    # TODO: Re-implement this cell! 

    transform = transforms.ToTensor()
    dataset = ImageFolder(root='./data/colors/', transform=transform)
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize as Grid

    If you have done your work correcly, the labels will match the colors you are seeing. If Red and Blue are reversed, check the previous cell. You cannot use alphabetic ordering.
    """)
    return


@app.cell
def _(DataLoader, dataset, plt):
    should_be_colors = {0: "red", 1: "green", 2: "blue"}

    dataloader = DataLoader(dataset, batch_size=9, shuffle=True)
    for images, labels in dataloader:
        print(images.shape, labels)
        break

    # Create a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(5, 5))
    axes = axes.flatten()  # make it easier to iterate

    for idx, ax in enumerate(axes):
        img = images[idx]  # [3,H,W]
        label = labels[idx].item()
    
        # Convert tensor to HWC numpy for plt
        img_np = img.permute(1, 2, 0).numpy()
    
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(should_be_colors[label], fontsize=10)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
