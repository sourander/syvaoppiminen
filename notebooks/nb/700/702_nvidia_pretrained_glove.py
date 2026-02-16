import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nvidia Learning Deep Learning: Using Pretrained GloVe Embeddings

    ## Attribution

    This notebook is sourced from Nvidia's Learning Deep Learning, [LDL](https://github.com/NVDLI/LDL), repository. The repository contains code examples associated with a similarly named book and video series by Magnus Ekman (Book ISBN: 9780137470358; Video: 9780138177553)

    - **Original source**: [gh:NVDLI/LDL/blob/main/stand_alone/v6_6_pretrained_glove_embeddings.ipynb](https://github.com/NVDLI/LDL/blob/main/stand_alone/v6_6_pretrained_glove_embeddings.ipynb)
    - **License**: BSD 3-Clause License
    - **Copyright**: (c) 2021 NVIDIA
    - **Modifications**:
        - Stylistic changes and conversion to Marimo
        - Swapped data to `glove.2024.wikigiga.50d.zip` or similar that student will download from [GloVe website](https://nlp.stanford.edu/projects/glove/).

    Full MIT license text is in the cell below.

    Reason for resharing: ease of access and integration into a larger collection of educational resources.
    """)
    return


@app.cell
def _():
    """
    The MIT License (MIT)
    Copyright (c) 2021 NVIDIA
    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This code example explores properties of GloVe word embeddings and word vector arithmetics. More context for this code example can be found in video 6.6 "Programming Example: Using Pretrained GloVe Embeddings" in the video series "Learning Deep Learning: From Perceptron to Large Language Models" by Magnus Ekman (Video ISBN-13: 9780138177614).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first code snippet contains two import statements and a function to read the embeddings. The function simply opens the file and reads it line by line. It splits each line into its elements. It extracts the first element, which represents the word itself, and then creates a vector from the remaining elements and inserts the word and the corresponding vector into a dictionary, which serves as the return value of the function.

    Comparison of original 6B and a newer embeddings file:

    | Embedding                | Corpus | Tokens | Vocab size | Dim   | MB   |
    | ------------------------ | ------ | ------ | ---------- | ----- | ---- |
    | glove.6B.100d            | 2014   |  6B    | 400 k      | 100-d | 347  |
    | glove.2024.wikigiga.50d  | 2024   | 12B    | 1.2 M      |  50-d | 842  |
    | glove.2024.wikigiga.100d | 2024   | 12B    | 1.2 M      | 100-d | 1680 |
    """)
    return


@app.cell
def _():
    import numpy as np
    import scipy.spatial

    # Read embeddings from file.
    def read_embeddings(file_path, DIM=50):
        """
        Read GloVe embeddings from a file. Note that the original implementation reads an older embedding file (data/glove.6B.100d.txt). You are encouraged to download a never file from GloVe website, e.g. glove.2024.wikigiga.50d.

        Difference is quickly compared in the next cell.
        """
        embeddings = {}
        file = open(file_path, 'r', encoding='utf-8')
        for line in file:
            parts = line.rstrip().split()
            word = " ".join(parts[:-DIM])
            vec = np.array(parts[-DIM:], dtype=np.float32)
            embeddings[word] = vec
        file.close()
        print('Read %s embeddings.' % len(embeddings))
        return embeddings

    return read_embeddings, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next code snippet implements a function that computes the cosine distance between a specific embedding and all other embeddings. It then prints the n closest ones. Euclidean distance would also have worked fine, but the results would sometimes be different because the GloVe vectors are not normalized.
    """)
    return


@app.cell
def _(scipy):
    def print_n_closest(embeddings, vec0, n):
        word_distances = {}
        for (word, vec1) in embeddings.items():
            distance = scipy.spatial.distance.cosine(
                vec1, vec0)
            word_distances[distance] = word
        # Print words sorted by distance.
        for distance in sorted(word_distances.keys())[:n]:
            word = word_distances[distance]
            print(word + ': %6.3f' % distance)

    return (print_n_closest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using these two functions, we can now retrieve word embeddings for arbitrary words and print out words that have similar embeddings. This is shown below, where we first read call read_embeddings() and then retrieve the embeddings for hello, precisely, and dog and call print_n_closest() on each of them.

    | Word | 6B 2014 (Orig) | 50d 2024 | 100d 2024 (Ref) |
    | :--- | :--- | :--- | :--- |
    | **hello** | hello: 0.000<br>goodbye: 0.209<br>hey: 0.283 | ??? | hello: 0.000<br>goodbye: 0.186<br>hey: 0.230 |
    | **precisely** | precisely: 0.000<br>exactly: 0.147<br>accurately: 0.293 | ??? | precisely: 0.000<br>exactly: 0.142<br>accurately: 0.223 |
    | **dog** | dog: 0.000<br>cat: 0.120<br>dogs: 0.166 | ??? | dog: 0.000<br>cat: 0.104<br>dogs: 0.139 |
    """)
    return


@app.cell
def _(print_n_closest, read_embeddings):
    from pathlib import Path

    file_path = Path("./data/glove/wiki_giga_2024_50_MFT20.txt")
    if not file_path.exists():
        print("[INFO] Download the file from: "
              " https://nlp.stanford.edu/projects/glove/")

    embeddings = read_embeddings(file_path, DIM=50)

    _lookup_word = 'hello'
    print('\nWords closest to ' + _lookup_word)
    print_n_closest(embeddings, embeddings[_lookup_word], 3)

    _lookup_word = 'precisely'
    print('\nWords closest to ' + _lookup_word)
    print_n_closest(embeddings, embeddings[_lookup_word], 3)

    _lookup_word = 'dog'
    print('\nWords closest to ' + _lookup_word)
    print_n_closest(embeddings, embeddings[_lookup_word], 3)
    return (embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using NumPy, it is also trivial to combine multiple vectors using vector arithmetic and then print out words that are similar to the resulting vector. This is demonstrated in the code snippet below, which first prints the words closest to the word vector for king and then prints the words closest to the vector resulting from computing (king − man + woman).

    | Word | 6B 2014 (Orig) | 50d 2024 | 100d 2024 (Ref) |
    | :--- | :--- | :--- | :--- |
    | **king** | king: 0.000<br>prince: 0.232<br>queen: 0.249 | ??? | king: 0.000<br>prince: 0.218<br>queen: 0.245 |
    | **king - man + woman** | king: 0.145<br>queen: 0.217<br>monarch: 0.307 | ??? | king: 0.150<br>queen: 0.188<br>throne: 0.274 |
    """)
    return


@app.cell
def _(embeddings, print_n_closest):
    _lookup_word = 'king'
    print('\nWords closest to ' + _lookup_word)
    print_n_closest(embeddings, embeddings[_lookup_word], 3)

    _lookup_word = '(king - man + woman)'
    print('\nWords closest to ' + _lookup_word)
    _vec = embeddings['king'] - embeddings['man'] + embeddings['woman']
    print_n_closest(embeddings, _vec, 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another example is shown below where we print the vector resulting from subtracting Spain and adding Sweden to the word Madrid.

    | Word | 6B 2014 (Orig) | 50d 2024 | 100d 2024 (Ref) |
    | :--- | :--- | :--- | :--- |
    | **sweden** | sweden: 0.000<br>denmark: 0.138<br>norway: 0.193 | ??? | sweden: 0.000<br>denmark: 0.155<br>norway: 0.212 |
    | **madrid** | madrid: 0.000<br>barcelona: 0.157<br>valencia: 0.197 | ??? | madrid: 0.000<br>barcelona: 0.139<br>valencia: 0.169 |
    | **madrid - spain + sweden** | stockholm: 0.271<br>sweden: 0.300<br>copenhagen: 0.305 | ??? | stockholm: 0.209<br>sweden: 0.271<br>copenhagen: 0.283 |
    """)
    return


@app.cell
def _(embeddings, print_n_closest):
    _lookup_word = 'sweden'
    print('\nWords closest to ' + _lookup_word)
    print_n_closest(embeddings, embeddings[_lookup_word], 3)

    _lookup_word = 'madrid'
    print('\nWords closest to ' + _lookup_word)
    print_n_closest(embeddings, embeddings[_lookup_word], 3)

    _lookup_word = '(madrid - spain + sweden)'
    print('\nWords closest to ' + _lookup_word)
    _vec = embeddings['madrid'] - embeddings['spain'] + embeddings['sweden']
    print_n_closest(embeddings, _vec, 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Playground

    Don't forget to have fun and play around with the tools you have. The 2024 embeddings are trained on Wikipedia, so surely it has cool words in it like `pytorch` or `linux` or `meme` ?

    Can you find anything surprising from the embeddings? For example, there are phone numbers and emails.
    """)
    return


@app.cell
def _(embeddings, print_n_closest):
    _lookup_word = ". . ."
    print('\nWords closest to ' + _lookup_word)
    print_n_closest(embeddings, embeddings[_lookup_word], 20)
    return


if __name__ == "__main__":
    app.run()
