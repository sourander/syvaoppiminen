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
    ## Fractional Max-Pooling Autopsy

    In this document, we will look what the trained model actually looks like. Key parts will be plotting the model architecture and plotting the n'th CONV block's feature map. What goes in, what comes out?
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torchvision.transforms.v2 as transforms
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np

    from torchvision import datasets
    from torchvista import trace_model
    from pathlib import Path
    return F, Path, datasets, nn, np, plt, torch, trace_model, transforms


@app.cell
def _(torch):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define Model

    This should be identical to the one you trained.
    """)
    return


@app.cell
def _(F, device, nn):
    class FMPNet(nn.Module):
        def __init__(self, target_sizes, filter_growth_rate, num_classes=10):
            super().__init__()
            self.blocks = nn.ModuleList()
            in_channels = 3

            # We iterate 1..12 and zip with the pre-calculated target sizes
            # n determines filters, target_size determines FMP output
            for n, target_size in zip(range(1, 13), target_sizes):
                out_channels = filter_growth_rate * n 
                p_dropout = ((n - 1) / 11) * 0.5

                block = nn.Sequential(
                    # C2: 2x2 Conv, Valid Padding
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=0),
                    # This is potentially my own addition
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(p=p_dropout),

                    # FMP: Explicitly enforce the output size
                    nn.FractionalMaxPool2d(kernel_size=2, output_size=target_size)
                )
                self.blocks.append(block)
                in_channels = out_channels

            # "Tail" of the network
            # Input here is guaranteed to be 2x2 because the last FMP target was 2
            self.convC2 = nn.Conv2d(in_channels, in_channels, kernel_size=2) # 2x2 -> 1x1
            self.bnC2 = nn.BatchNorm2d(in_channels) # My own addition
            self.convC1 = nn.Conv2d(in_channels, num_classes, kernel_size=1) 

        def forward(self, x):
            # 1. Pad Input dynamically to match the calculated input_dim (94)
            # CIFAR is 32x32, so we need 62 padding total (31 per side)
            # This calculation makes it robust to different input sizes
            pad_total = 94 - x.size(2)
            if pad_total > 0:
                pad_val = pad_total // 2
                x = F.pad(x, (pad_val, pad_val, pad_val, pad_val), "constant", 0)

            for block in self.blocks:
                x = block(x)

            # Head Forward Pass
            x = self.convC2(x)
            x = self.bnC2(x) # Apply BN
            x = F.leaky_relu(x, 0.2)

            x = self.convC1(x)
            x = F.leaky_relu(x, 0.2)

            x = x.view(x.size(0), -1)
            return x

    fmp_targets=[74, 58, 45, 35, 27, 21, 16, 12, 9, 6, 4, 2]
    num_classes=10
    filter_growth_rate = 64

    model = FMPNet(fmp_targets, filter_growth_rate, num_classes=10)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the Model
    """)
    return


@app.cell
def _(device, model, torch):
    # Load the trained model checkpoint
    checkpoint_path = "./gitlfs-store/502_cifar10_fractionalmaxp.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()  # Set to evaluation mode

    print(f"Model loaded from: {checkpoint_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load one image from the test set
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    victim_idx = mo.ui.number(start=0, stop=9_999, label="Choose the nth item in test set.")
    victim_idx
    return (victim_idx,)


@app.cell(hide_code=True)
def _(datasets, torch, transforms, victim_idx):
    # CIFAR-10 stats
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    test_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


    testset = datasets.CIFAR10(
        "./data", download=True, train=False, transform=test_transform
    )


    autopsy_victim, victim_label = testset[victim_idx.value]
    victim_label_srt = testset.classes[victim_label]
    print("Loaded image of a", victim_label_srt)
    return CIFAR10_MEAN, CIFAR10_STD, autopsy_victim, testset, victim_label_srt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Torchvista Magic

    Torchvista stores the file with random UUID4 name. Thus, we Monkeypatch it to deterministic file name.

    Open the file in browser.
    """)
    return


@app.cell
def _(Path, autopsy_victim, model, trace_model):
    if not Path("torchvista_graph_fixed.html").exists:

        from unittest.mock import patch, MagicMock

        input_tensor = autopsy_victim.unsqueeze(0)

        # Mock uuid.uuid4() to return a fixed string
        mock_uuid = MagicMock()
        mock_uuid.__str__ = MagicMock(return_value="fixed")

        with patch('torchvista.tracer.uuid.uuid4', return_value=mock_uuid):
            trace_model(model, input_tensor, export_format="html")
    else:
        print("The 'torchvista_graph_fixed.html' already exists. Look at it plz.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Display Image
    """)
    return


@app.cell(hide_code=True)
def _(CIFAR10_MEAN, CIFAR10_STD, autopsy_victim, plt, victim_label_srt):
    def display_image(image_tensor, label):
        """Display a normalized CIFAR-10 image."""

        img = image_tensor.clone()

        # Denormalize the image (using global scope vars)
        for i in range(3):
            img[i] = img[i] * CIFAR10_STD[i] + CIFAR10_MEAN[i]

        # Clip values to [0, 1]
        img = img.clamp(0, 1)

        # Convert CHW to HWC for matplotlib
        img_np = img.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np)
        ax.set_title(f"Label: {label}")
        ax.axis("off")
        plt.tight_layout()
        return fig


    fig = display_image(autopsy_victim, victim_label_srt)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Display nth batch ouputs

    Also known as: feature maps.

    Remember that the 1st batch has layers like this:

    ```
    [Block 1] Input: torch.Size([1, 3, 94, 94])
       -> Conv2d: torch.Size([1, 64, 93, 93])
       -> BatchNorm2d: torch.Size([1, 64, 93, 93])
       -> LeakyReLU: torch.Size([1, 64, 93, 93])
       -> Dropout2d: torch.Size([1, 64, 93, 93])
       -> FMP (Target (74, 74)): torch.Size([1, 64, 74, 74])
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    BLOCK_IDX = mo.ui.slider(1, 12, label="The block index to inspect (1-12)")

    BLOCK_LAYER = mo.ui.radio(
        options=["Conv2d", "BatchNorm2d", "LeakyReLU", "FMP"], value="Conv2d", label="The layer within the block to inspect"
    )

    COLORMAP = mo.ui.radio(
        options=["viridis", "gray", "magma"],
        value="viridis",
        label="Colormap"
    )

    mo.hstack([BLOCK_IDX, BLOCK_LAYER, COLORMAP], justify="start")
    return BLOCK_IDX, BLOCK_LAYER, COLORMAP


@app.cell(hide_code=True)
def _(BLOCK_IDX, BLOCK_LAYER, COLORMAP, F, autopsy_victim, device, model, plt):
    def get_feature_maps(model, image_tensor, block_idx, layer_name="Conv2d"):
        """
        Get feature maps from a specific layer in a block.

        Args:
            model: The FMPNet model
            image_tensor: Input image tensor (C, H, W)
            block_idx: Index of the block (1-12, user-facing)
            layer_name: Specific layer name like 'Conv2d', 'BatchNorm2d', 'LeakyReLU',
                       'Dropout2d', or 'FractionalMaxPool2d'

        Returns:
            Tensor of feature maps after the specified layer (shape: 1, C, H, W)
        """
        x = image_tensor.unsqueeze(0).to(device)

        # Pad input to match model's expected size (94x94)
        pad_total = 94 - x.size(2)
        if pad_total > 0:
            pad_val = pad_total // 2
            x = F.pad(x, (pad_val, pad_val, pad_val, pad_val), "constant", 0)

        # Convert 1-indexed to 0-indexed
        target_block_idx = block_idx - 1

        # Process through blocks up to and including the target block
        for i, block in enumerate(model.blocks):
            if i < target_block_idx:
                x = block(x)
            elif i == target_block_idx:
                # Process through specific layers in the target block
                for layer in block:
                    x = layer(x)
                    if layer.__class__.__name__ == layer_name:
                        return x.detach()
                # If layer not found, return block output
                return x.detach()


    def visualize_feature_maps(feature_maps, max_channels=16, figsize=(15, 10), cmap="viridis"):
        """
        Visualize feature maps from a convolutional layer.

        Args:
            feature_maps: Tensor of shape (1, C, H, W)
            max_channels: Maximum number of channels to display
            figsize: Figure size
            cmap: Matplotlib colormap name
        """
        feature_maps = feature_maps.squeeze(0)
        num_channels = min(feature_maps.shape[0], max_channels)

        # Calculate grid size
        cols = 4
        rows = (num_channels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_channels > 1 else [axes]

        for idx in range(num_channels):
            ax = axes[idx]
            fmap = feature_maps[idx].cpu().numpy()

            im = ax.imshow(fmap, cmap=cmap)
            ax.set_title(f"Channel {idx}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(num_channels, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        return fig


    # Get feature maps using 1-indexed block number from UI
    fmaps = get_feature_maps(
        model,
        autopsy_victim,
        block_idx=BLOCK_IDX.value,
        layer_name=BLOCK_LAYER.value,
    )
    print(
        f"Feature maps after Block {BLOCK_IDX.value} {BLOCK_LAYER.value}: {fmaps.shape}"
    )

    # Visualize
    fig_maps = visualize_feature_maps(fmaps, max_channels=16, cmap=COLORMAP.value)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Display the Head

    The head takes the 12th block's output as input.

    ```
    [Head] Input: torch.Size([1, 768, 2, 2])
       -> ConvC2 (2x2 kernel): torch.Size([1, 768, 1, 1]) (Channels = Features)
       -> ConvC1 (1x1 kernel): torch.Size([1, 10, 1, 1])  (Channels = Num Classes)
       -> Final Output (Flattened): torch.Size([1, 10])
    ```
    """)
    return


@app.cell(hide_code=True)
def _(F, autopsy_victim, device, model, np, plt, testset, torch):
    def visualize_2x2_to_1x1(input_2x2, output_1x1, channel_idx=0):
        """
        Visualize how a 2x2 Conv reduces spatial dimensions from 2x2 to 1x1.

        Args:
            input_2x2: Tensor of shape [1, C, 2, 2]
            output_1x1: Tensor of shape [1, C, 1, 1]
            channel_idx: Which channel to visualize
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Extract the specific channel
        input_grid = input_2x2[0, channel_idx].detach().cpu().numpy()
        output_val = output_1x1[0, channel_idx, 0, 0].detach().cpu().item()

        # Plot 1: Input 2x2 grid
        im1 = ax1.imshow(input_grid, cmap='RdBu_r', interpolation='nearest')
        ax1.set_title(f'Input: 2×2 Grid (Channel {channel_idx})', fontsize=12, fontweight='bold')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['0', '1'])
        ax1.set_yticklabels(['0', '1'])

        # Add values to cells
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, f'{input_grid[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Plot 2: Output 1x1 value
        # Create a 1x1 "grid" for visualization
        output_grid = np.array([[output_val]])
        im2 = ax2.imshow(output_grid, cmap='RdBu_r', interpolation='nearest',
                        vmin=input_grid.min(), vmax=input_grid.max())
        ax2.set_title(f'Output: 1×1 (Conv2d k=2)', fontsize=12, fontweight='bold')
        ax2.set_xticks([0])
        ax2.set_yticks([0])
        ax2.set_xticklabels(['0'])
        ax2.set_yticklabels(['0'])

        # Add value to cell
        ax2.text(0, 0, f'{output_val:.3f}',
                ha="center", va="center", color="black", fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Add arrow annotation
        fig.text(0.5, 0.05, '2×2 Conv (kernel_size=2) collapses spatial dimensions', 
                ha='center', fontsize=11, style='italic')

        plt.tight_layout()
        return fig

    def process_head(model, image_tensor):
        """
        Process the head  of the network and visualize the transformations.

        The FMPNet head  uses a "fully convolutional" approach instead of traditional
        fully connected (FC) layers. This is achieved by:
        1. Using a 2x2 Conv to reduce the 2x2 spatial input to 1x1
        2. Using a 1x1 Conv to project features to the number of classes
        """
        import numpy as np

        x = image_tensor.unsqueeze(0).to(device)

        # Pad input to match model's expected size (94x94)
        pad_total = 94 - x.size(2)
        if pad_total > 0:
            pad_val = pad_total // 2
            x = F.pad(x, (pad_val, pad_val, pad_val, pad_val), "constant", 0)

        # Process through all 12 blocks
        for block in model.blocks:
            x = block(x)

        # Save the input for visualization
        input_to_head  = x.clone()
        print(f"\n[1] Input to Head (Block 12 output): {x.shape}")
        print(f"    2×2 spatial grid with 768 channels")

        # Apply ConvC2 (2x2 kernel) - Reduces spatial dimensions to 1x1
        x_before_bn = model.convC2(x)
        print(f"\n[2] After ConvC2 (2×2 Conv, reduces 2×2 → 1×1): {x_before_bn.shape}")
        print(f"    Spatial dimensions collapsed to 1×1, keeping 768 features")

        # Visualize the transformation for first 3 channels
        print(f"\n    Visualizing transformation for channels 0, 1, 2:")
        figs = []
        for ch in range(3):
            fig = visualize_2x2_to_1x1(input_to_head, x_before_bn, channel_idx=ch)
            figs.append(fig)

        # Apply BatchNorm and activation (covered in previous cells)
        x = model.bnC2(x_before_bn)
        x = F.leaky_relu(x, 0.2)

        # At this point, x is [1, 768, 1, 1] - a 768-dimensional feature vector
        feature_vector = x.squeeze().detach().cpu().numpy()
        print(f"\n[3] After BatchNorm + LeakyReLU: {x.shape}")
        print(f"    This is now a 768-dimensional feature vector")
        print(f"    First 16 values: {feature_vector[:16]}")
        print(f"    ... and {len(feature_vector) - 16} more values")

        # Apply ConvC1 (1x1 kernel) - Projects to number of classes
        x = model.convC1(x)
        x = F.leaky_relu(x, 0.2)
        print(f"\n[4] After ConvC1 (1×1 Conv, 768 → 10 classes) + LeakyReLU: {x.shape}")

        # Flatten for final output
        x = x.view(x.size(0), -1)
        print(f"\n[5] Final Output (Flattened): {x.shape}")
        print(f"    Class scores (logits): {x.squeeze().detach().cpu().numpy()}")

        # Apply softmax to get probabilities
        probs = torch.softmax(x, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

        print(f"\n[6] After Softmax:")
        # print(f"    Probabilities: {probs.squeeze().tolist()}")
        n_classes = range(len(testset.classes))
        for i, prob in zip(n_classes, probs.squeeze().tolist()):
            print(f"    {testset.classes[i]:12} {prob:.3f}")
        print(f"    Predicted class: {predicted_class} ({testset.classes[predicted_class]})")

        return x, probs, predicted_class, figs

    # Run the head  processing demonstration
    logits, probabilities, pred_class, visualization_figs = process_head(model, autopsy_victim)

    # Display the visualizations
    visualization_figs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Insights

    No Fully Connected Layers!

    Traditional CNNs flatten the feature maps and use FC layers:

    - Flatten: [1, 768, 2, 2] → [1, 3072]
    - FC1: [1, 3072] → [1, 512] (1.57M parameters!)
    - FC2: [1, 512] → [1, 10]  (5K parameters)

    FMPNet uses convolutions instead:

    - ConvC2: 2×2 kernel → (768 × 768 × 4) + 768 = 2.36M parameters
    - ConvC1: 1×1 kernel → (768 × 10 × 1) + 10 = 7.7K parameters
    """)
    return


if __name__ == "__main__":
    app.run()
