import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perplexity

    **COPYRIGHT NOTE:** This Marimo Notebook is based on **Transformers - The definitive Guide** repository: [gh:/Nicolepcx/transformers-the-definitive-guide](https://github.com/Nicolepcx/transformers-the-definitive-guide). Look for file `ch01_perplexity.ipynb`. Code files in the repo follow Apache License 2.0.

    Key changes:

    * Using Marimo over Jupyter Notebook
    * Using Viking Model over Falcon
    * Using Finnish Wikipedia quote over English
    """)
    return


@app.cell
def _():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from functools import partial

    return AutoModelForCausalLM, AutoTokenizer, partial, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download model

    Note: at least on teacher's computer, the cell output was not updating. If you want to see how the download progresses, open a terminal, and run:

    ```bash
    hf download LumiOpen/Viking-7B
    ```

    The model will be downloaded to `~/.cache/huggingface/hub/models--LumiOpen--Viking-7B/` and will take about 14 GB.
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, device):
    model = AutoModelForCausalLM.from_pretrained("LumiOpen/Viking-7B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("LumiOpen/Viking-7B")
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose your device

    Note that running the model requires a fair amount of memory – assuming it has to **fully fit** into your memory. However, the safetensor format allows lazy-loading and partial data loading, meaning that the model is sequantially loaded into memory layer by layer.... **when utilizing CPU**. It may be that CUDA can be made to work with this too, but without any code modification, this causes the following rule:

    * if `< 16 GB` memory on your graphics card: `USE_GPU = False`
    * if `>= 16 GB` memory on your graphics card: `USE_GPU = True`

    Nevertheless, a curious mind tries both settings. Your GPU won't break. It will just cry a bit.
    """)
    return


@app.cell
def _(torch):
    USE_GPU = False

    # Device selection
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
    ## Helper functions

    We are using **partial function application** here from the functional programming paradigm.
    """)
    return


@app.cell
def _(device, model, partial, tokenizer, torch):
    def _calculate_perplexity(model, tokenizer, device, wiki_quote):
        model.eval()
        with torch.no_grad():
            # 1. Tokenize (creates CPU tensors by default)
            wiki_text = tokenizer(wiki_quote, return_tensors="pt")

            # 2. Move input tensors to the GPU (MPS)
            input_ids = wiki_text["input_ids"].to(device)

            # 3. accurate calculation
            loss = model(input_ids = input_ids,
                         labels = input_ids).loss

        return torch.exp(loss)

    # Create the wrapper with dependencies pre-filled 
    # to make next cells less verbose
    Perplexity = partial(_calculate_perplexity, model, tokenizer, device)
    return (Perplexity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Playground

    Find out how perplexed the model is with various input. Copy-pasting from Finnish wikipedia should keep the scores low (close to 1). Modifying those copy-pasted quotes should increase it. Writing utter nonsense should increase it even higher.
    """)
    return


@app.cell
def _(Perplexity):
    Perplexity("Ainola on säveltäjä Jean Sibeliuksen ja hänen puolisonsa Aino Sibeliuksen asuintalo, joka on vuodesta 1974 alkaen toiminut kotimuseona")
    return


@app.cell
def _(Perplexity):
    Perplexity("Ainola on puisto Oulusa, jossa taskulämmin sinikylkinen maistuu maukkaimmalta.")
    return


if __name__ == "__main__":
    app.run()
