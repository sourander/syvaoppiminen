import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Play with SpaCy

    You can use any SpaCy model/pipeline here. Download it first:

    ```bash
    uv add "fi-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/fi_core_news_sm-3.8.0/fi_core_news_sm-3.8.0-py3-none-any.whl"
    ```
    """)
    return


@app.cell
def _():
    import spacy

    from spacy import displacy
    return (spacy,)


@app.cell
def _(spacy):
    nlp = spacy.load("fi_core_news_sm")
    return (nlp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Do Your Thing

    This is your Notebook now. Do whatever helps you ingest the topic.
    """)
    return


@app.cell
def _(nlp):
    doc = nlp("Pizza on ravitsevaa.")
    for token in doc:
        print(f"{token.text:>12}: ", end="")
        for value in token.vector[:3]:
            print(f"{value:>5.2f}", end=" | ")
        print(f"... | {value:>5.2f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
