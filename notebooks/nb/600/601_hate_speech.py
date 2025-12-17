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
    ## Hate Speech detection

    We had this same data and same task in Johdatus koneoppimiseen course. Last time, we downloaded the data from [gh:t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language). This time, let's use Hugging Face datasets.

    Let's use some model that has not been trained on this data - for all we know, at least. Then, we shall compare to our old results.

    * Data (1.6 MB): https://huggingface.co/datasets/tdavidson/hate_speech_offensive?library=datasets
    * Model (500 MB): https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target

    Remember, our task was **binary classification** whether a text is hatespeech or not. With Naive Bayes, it was suggested to try our keeping or dropping the *offensive language* label, since it makes the class imbalance worse, and also it makes the task more nuanced. We will do the same here.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns

    from datasets import load_dataset
    from transformers import pipeline
    from time import perf_counter
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
    return (
        classification_report,
        confusion_matrix,
        load_dataset,
        perf_counter,
        pipeline,
        plt,
        roc_auc_score,
        roc_curve,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data
    """)
    return


@app.cell
def _(load_dataset):
    ds = load_dataset("tdavidson/hate_speech_offensive", split="train")

    # Keep only the tweet and the 0,1,2 class
    df = ds.to_pandas()[["tweet", "class"]]

    # Remap to 0 and 1
    df["class"] = df["class"].map(
        {
            0: 1,    # Hate speech -> 1
            1: None, # Offensive -> drop
            2: 0     # Neither -> 0
        }
    )

    # Drop offensive language rows
    df = df.dropna()  
    n = len(df)
    return df, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model
    """)
    return


@app.cell
def _(mo, pipeline):
    pipe = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")

    monty = pipe("Shut your festering gob, you tit! Your type makes me puke! You vacuous, toffee-nosed, malodorous pervert!")
    mo.md(f"The Monty Python insult is **{monty[0]["label"]}** (score: {monty[0]["score"]:.3f})")
    return (pipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict into a DataFrame

    Make predictions using the full dataset, where all 3 labels are present:

    * class = 0 = hate speech
    * class = 1 = offensive
    * class = 2 = neither

    The predictions use the `tweet` columns. All other columns can be ignored. We will add our prediction to `prediction` column.
    """)
    return


@app.cell
def _(df, perf_counter, pipe):
    batch_size = 16

    # Start timing
    t_start = perf_counter()

    # Run pipeline on all tweets with batching
    predictions = []
    scores = []
    for out in pipe(df['tweet'].tolist(), batch_size=batch_size):

        # Convert to 1 if hate, otherwise 0
        predicted = ... # IMPLEMENNT

        # Always probability of "hate"
        prob_hate = ... # IMPLEMENT

        # Append both
        predictions.append(predicted)
        scores.append(prob_hate)

    df_pred = df.copy()
    df_pred['prediction'] = predictions
    df_pred['score'] = scores
    t_stop = perf_counter()

    print(f"Batch size: {batch_size}, classified {len(df_pred)} tweets in {t_stop - t_start:.2f} seconds")

    df_pred
    return (df_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get y and y_hat
    """)
    return


@app.cell
def _(df_pred, n):
    y = df_pred["class"]
    y_hat = df_pred["prediction"]

    acc = sum(df_pred['class']==df_pred['prediction'])/n*100
    print(f"Accuracy: {acc:.1f}")
    return y, y_hat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classification Report
    """)
    return


@app.cell
def _(classification_report, y, y_hat):
    print(classification_report(y, y_hat))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion Matrix
    """)
    return


@app.cell(hide_code=True)
def _(confusion_matrix, plt, sns, y, y_hat):
    cm = confusion_matrix(y, y_hat)
    labels = ["neither", "hate"]
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ROC
    """)
    return


@app.cell(hide_code=True)
def _(df_pred, plt, roc_auc_score, roc_curve, y):
    # Calculate the ROC curve
    y_pred_proba = df_pred["score"]
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    # Calculate the AUC
    print("AUC:", roc_auc_score(y, y_pred_proba))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Investigate Wrong Classifications
    """)
    return


@app.cell
def _(df_pred):
    df_pred.query("`class` != prediction")
    return


if __name__ == "__main__":
    app.run()
