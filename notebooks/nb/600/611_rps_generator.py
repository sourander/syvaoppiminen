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
    ## Generate Rock, Paper, Scissors Data

    Note: You may also decide to capture some other data. Maybe 3-10 different kind of toys? 3 different facial gestures? 3-5 different family members?

    ### How to use?

    * Choose the **label** from dropdown menu.
    * Press **Capture**
    * Image will be stored to `data/label/rps_timestamp.png`

    You may want to put on the **Auto-capture**. It will store one image per second until toggled off.
    """)
    return


@app.cell(hide_code=True)
def _():
    import time

    from pathlib import Path
    from wigglystuff import WebcamCapture
    return Path, WebcamCapture, time


@app.class_definition(hide_code=True)
class ImageMemory():
    """
    By default, Marimo runs the dependent cells when the Dropdown value is changed. The cell that
    saves images would thus save the PREVIOUSLY captured image, since it remains in the buffer. 

    This naughtly behaviour, but easily fixed by using an instance of this class as a memory store. 
    Since this is upper-level DAG dependency, Marimo will not recreate this when changing dropdown 
    value.
    """
    def __init__(self):
        self.prev_bytes = None

    def has_changed(self, current_bytes):
        """Check if current image is different from previous one."""
        if self.prev_bytes is None:
            return True
        return current_bytes != self.prev_bytes

    def update(self, current_bytes):
        """Store current image bytes for next comparison."""
        self.prev_bytes = current_bytes


@app.cell(hide_code=True)
def _():
    image_memory = ImageMemory()
    return (image_memory,)


@app.cell(hide_code=True)
def _(WebcamCapture, mo):
    dropdown = mo.ui.dropdown(
        options=["rock", "paper", "scissors"], value="rock", label="Label to save:"
    )

    widget = mo.ui.anywidget(WebcamCapture(interval_ms=1000))
    return dropdown, widget


@app.cell(hide_code=True)
def _(dropdown, mo, widget):
    mo.hstack([dropdown, widget], justify="center")
    return


@app.cell(hide_code=True)
def _(Path, dropdown, image_memory, mo, time, widget):
    img = widget.get_pil()

    # Mask the image version prefix from unix time, keeping tenths.
    #   time: 1767773242.3092108
    # prefix: ----7732423-------
    prefix = str(int(time.time() * 10))[4:]

    save_path = Path(f"data/rps/{dropdown.value}/rps_{prefix}.jpg")
    save_path.parent.mkdir(exist_ok=True, parents=True)

    if img:
        current_bytes = img.tobytes()

        # Only save if image has changed (new capture)
        if image_memory.has_changed(current_bytes):
            # Save the PIL image
            img.convert("RGB").save(save_path)

            # Print info about the saved image
            txt = f"Saved image rps_{prefix}.jpg as {dropdown.value} label with resolution {img.size}"

            # Update memory with current image
            image_memory.update(current_bytes)
        else:
            # Same image as before - skip saving
            txt = "Waiting for new image capture..."
    else:
        txt = f"Warning! The img does not contain image, but: {img}"

    mo.md(txt)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
