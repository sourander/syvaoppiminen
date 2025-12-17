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
    # Hugging Face Hello World

    This exercise is based on Dr. Noureddin Sadawi's [O'Reilly Course: Hugging Face Fundamentals for Machine Learning (2024)](https://www.oreilly.com/live-events/hugging-face-fundamentals-for-machine-learning/0642572010327/). You can find other relating Hugging Face notebooks from that courses repository: [gh:nsadawi/huggingface-course](https://github.com/nsadawi/huggingface-course).

    Also ideas from Hugging Faces [Pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines) docs have been used here.
    """)
    return


@app.cell
def _():
    import pandas as pd

    from datasets import load_dataset
    from transformers import pipeline
    from time import perf_counter
    return load_dataset, perf_counter, pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the Model

    The model will be stored into your home directory. The path is `~/.cache/huggingface`. The tilde symbol means your `$HOME` or `%homePath` depending on your OS.

    You can use Python to check the contents of that dir any time:

    ```
    from huggingface_hub import scan_cache_dir
    print(scan_cache_dir().export_as_table())
    ```

    Models (and datasets) require a lot of storage, so you may want to prune old content from your cache once in a while. You can do this using either Python or CLI (or by using any File Explorer). To get started with CLI, try running: `uvx hf cache --help`.
    """)
    return


@app.cell
def _(pipeline):
    model = pipeline("sentiment-analysis")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference on 1 line
    """)
    return


@app.cell
def _(model):
    sentence = "I just love handling dependencies in Python! What is better than dep conflicts?"

    model(sentence)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference on a Dataset

    This dataset will be about 900 KB.
    """)
    return


@app.cell
def _(load_dataset):
    # Load train, test and dev
    reviews = load_dataset('rotten_tomatoes')

    # Create DataFrame out of test
    df = reviews['test'].to_pandas()
    n = len(df)
    return df, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Method A: Pandas Apply
    """)
    return


@app.cell
def _(df, model, n, perf_counter):
    def label(review, model):
        label = model(review)[0]['label']
        if label == 'POSITIVE':
            return 1
        else:
            return 0

    df_apply = df.copy()

    t1_start = perf_counter() 
    df_apply['predicted_label'] = df['text'].apply(label, args=(model,))
    t1_stop = perf_counter()

    print(f"Classifying {n} reviews took seconds: {t1_stop-t1_start:.2f}")
    return (df_apply,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Method B: Batching

    Let's also try out the Pipeline batching as suggested by the Hugging Face documentation.
    """)
    return


@app.cell
def _(df, model, perf_counter):
    for batch_size in [1, 8, 64, 256]:
        df_batch = df.copy()
    
        t2_start = perf_counter()
    
        # Run pipeline on all texts with batching
        results = []
        for out in model(df['text'].tolist(), batch_size=batch_size):
            predicted = 1 if out['label'] == 'POSITIVE' else 0
            results.append(predicted)
    
        df_batch['predicted_label'] = results
        t2_stop = perf_counter()
    
        print(f"Batch size: {batch_size}, seconds: {t2_stop-t2_start:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the Accuracy of the Model

    The original dataset has a column `label` which contains the true $y$. Let's compare our BERT-created $y_{hat}$ to this.
    """)
    return


@app.cell
def _(df_apply, model, n):
    name= model.model.name_or_path
    acc = sum(df_apply['label']==df_apply['predicted_label'])/n*100
    print(f"Classification with {name}, accuracy: {acc:.1f}")
    return


if __name__ == "__main__":
    app.run()
