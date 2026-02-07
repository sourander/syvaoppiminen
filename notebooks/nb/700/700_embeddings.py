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
    # Investigate Embeddings
    """)
    return


@app.cell
def _():
    import spacy
    import polars as pl
    import altair as alt

    from pathlib import Path
    from scipy.spatial import distance
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    return PCA, Path, TSNE, alt, distance, pl, spacy


@app.cell
def _():
    import duckdb
    conn = duckdb.connect()
    conn.execute("INSTALL vss; LOAD vss;")
    return (conn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing Word Vectors OvO
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load Pipeline and fetch the word vectors (embeddings)

    We will use the LARGE model for this. When using uv, you need to install it like this:

    ```bash
    uv add "fi-core-news-lg @ https://github.com/explosion/spacy-models/releases/download/fi_core_news_lg-3.8.0/fi_core_news_lg-3.8.0-py3-none-any.whl"
    ```
    """)
    return


@app.cell
def _(spacy):
    nlp = spacy.load("fi_core_news_lg")
    return (nlp,)


@app.cell
def _(distance, nlp, pd):
    def compare_vectors(pairs):
        results = []
        for left, right in pairs:
            vec1 = nlp(left).vector
            vec2 = nlp(right).vector
            cos_sim = 1 - distance.cosine(vec1, vec2)
            results.append({"Pair": f"{left}:{right}", "Cosine Sim": cos_sim})
        return pd.DataFrame(results)

    pairs = [
        ("kuningas", "kuningatar"),
        ("setä", "täti"),
        ("mies", "nainen"),
        ("sielu", "teräs"),
    ]
    compare_vectors(pairs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Access 10k most common Finnish words

    The dataset has been loaded from [CSC](http://www.csc.fi/tutkimus/alat/kielitiede/taajuussanasto-B9996/view) site. The site is archived in WaybackMachine if you need to access the old site for some reason. You should not have the needs, since the file is in the Syväoppiminen I repository. Here is the CC BY-NC-ND 1.0 Finnish copyright note:

    > "Suomen sanomalehtikielen 9996 yleisintä lemmaa (käytettävissä Commons Nimeä-Epäkaupallinen-Ei muutettuja teoksia 1.0 Suomi-lisenssin ehdoin). Sanasto koottiin vuonna 2004 ja lähdeaineistossa oli 43999826 sanaa. Listan laatimisessa hyödynnettiin Kielipankin Lemmie-käyttöliittyymää."

    The first rows in the file look like this:

    ```
       N        Abs   Rel    Uppslagsord
       1    2716396 4,614851 olla (verbi)
       2    1566108 2,660641 ja (konjunktio)
       3     593462 1,008225 ei (verbi)
       4     538609 0,915036 se (pronomini)
    ```

    Let's create an in-memory database -- using DuckDB -- that utilizes this data. The idea is described in the [The Vector Similarity Search (VSS) Extension](https://duckdb.org/2024/05/03/vector-similarity-search-vss#the-vector-similarity-search-vss-extension) blog post. Running this cell will likely take 20-30 seconds.
    """)
    return


@app.cell
def _(Path, conn, mo, nlp):
    sanasto = Path("./gitlfs-store/suomen-sanomalehtikielen-taajuussanasto-utf8.txt")

    # Create table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (word VARCHAR, vec FLOAT[300])"
    )

    # Read and parse the file
    words_with_vectors = []
    with open(sanasto, 'r', encoding='utf-8') as f:
        # Skip the header line
        next(f)

        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Extract the word (Uppslagsord column, index 3)
                # Remove the part in parentheses, e.g., "olla (verbi)" -> "olla"
                word_with_pos = ' '.join(parts[3:])
                word = word_with_pos.split('(')[0].strip()

                # Get the embedding vector for the word
                doc = nlp(word)
                if doc.vector.shape[0] > 0:  # Check if vector exists
                    words_with_vectors.append((word, doc.vector.tolist()))

    # Insert data in batches
    conn.executemany(
        "INSERT INTO embeddings (word, vec) VALUES (?, ?)",
        words_with_vectors
    )

    # Create HNSW index
    conn.execute("CREATE INDEX IF NOT EXISTS idx ON embeddings USING HNSW (vec) WITH (metric = 'cosine')")

    result = conn.execute("SELECT COUNT(*) as count FROM embeddings").fetchone()
    mo.md(f"Processed and inserted **{result[0]}** words into the embeddings table")
    return


@app.cell
def _(conn, embeddings, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM embeddings
        """,
        engine=conn
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analogy playground

    The original analogy here is `pariisi - ranska + suomi = ???`, where expected result is some Finnish town. Helsinki would make sense intuitively. Try changing these to something like:

    * `ihminen - sielu + keinotekoinen`
    * `leipä - paistaminen + hiiva`
    * `auto - renkaat + ohjaustanko`
    """)
    return


@app.cell
def _(nlp):
    query_item = nlp("pariisi").vector - nlp("ranska").vector + nlp("suomi").vector
    return (query_item,)


@app.cell
def _(conn, embeddings, mo, query_item):
    _df = mo.sql(
        f"""
        SELECT *
        FROM embeddings
        ORDER BY array_distance(vec, {query_item.tolist()}::FLOAT[300])
        LIMIT 10;
        """,
        engine=conn
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Least similar top k
    """)
    return


@app.cell
def _(conn, embeddings, mo):
    _df = mo.sql(
        f"""
        SELECT 
            e1.word as word1, 
            e2.word as word2,
            array_distance(e1.vec, e2.vec) as distance
        FROM embeddings e1
        CROSS JOIN embeddings e2
        WHERE e1.word < e2.word  -- Avoid duplicates and self-comparisons
        ORDER BY distance ASC
        LIMIT 20;
        """,
        engine=conn
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Most similar top k
    """)
    return


@app.cell
def _(conn, embeddings, mo):
    _df = mo.sql(
        f"""
        SELECT 
            e1.word as word1, 
            e2.word as word2,
            array_distance(e1.vec, e2.vec) as distance
        FROM embeddings e1
        CROSS JOIN embeddings e2
        WHERE e1.word < e2.word
          AND length(e1.word) > 2
          AND length(e2.word) > 2
        ORDER BY distance DESC
        LIMIT 10;
        """,
        engine=conn
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize with reduced dims
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Perform PCA or t-SNE

    PCA is faster, but the end result may lack proper clusters.
    """)
    return


@app.cell
def _(PCA, TSNE, mo):
    mo_algo = mo.ui.radio(
        options={"PCA": PCA, "t-SNE": TSNE},
        value="PCA",
        label="Choose the Algorithm",
    )

    mo_algo
    return (mo_algo,)


@app.cell
def _(conn, mo_algo, pl):
    # Polars DataFrame
    df = conn.execute("""
        SELECT word, vec
        FROM embeddings
    """).pl()

    pca = mo_algo.value(n_components=2)
    two_components = pca.fit_transform(df["vec"])

    df = (
        df
        .with_columns(pl.Series(name="a", values=two_components[:, 0]))
        .with_columns(pl.Series(name="b", values=two_components[:, 1]))
        .drop("vec")
    )
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(alt, df, mo):
    chart = mo.ui.altair_chart(alt.Chart(df).mark_point().encode(
        x='a',
        y='b',
        tooltip='word'
    ))
    return (chart,)


@app.cell
def _(chart, mo):
    mo.vstack([
        chart,
        mo.ui.table(chart.value)
    ])
    return


if __name__ == "__main__":
    app.run()
