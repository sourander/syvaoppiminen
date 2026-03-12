import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn

    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Metro Interstate Traffic Volume

    Data dataset has been downloaded from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) and is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. Below is the APA format reference:

    > Hogue, J. (2019). Metro Interstate Traffic Volume [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X60B.

    /// attention | Data
    You need to extract the zip before accessing the dataset. The zip includes a `.gz`-compressed file, but Pandas is able to open that. Unzip that gz file into proper directory with the following directory...

    ```bash
    cd notebooks
    unzip gitlfs-store/metro_interstate_traffic_volume.zip -d data/metro
    ```

    ///
    """)
    return


@app.cell
def _():
    df = pd.read_csv("data/metro/Metro_Interstate_Traffic_Volume.csv.gz", low_memory=False)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect Data

    | Column                  | Type              | Description              |
    |-------------------------|-------------------|--------------------------|
    |`holiday`                |Categorical        |US National holidays|
    |`temp`                   |Numeric            |Avg temp in Kelvin|
    |`rain_1h`                |Numeric            |Amount in mm of rain that occurred in the hour|
    |`snow_1h`                |Numeric            |Amount in mm of snow that occurred in the hour|
    |`clouds_all`             |Numeric (Integer)  |Percentage of cloud cover|
    |`weather_main`           |Categorical        |Short textual description of the current weather|
    |`weather_description`    |Categorical        |Longer textual description of the current weather|
    |`date_time`              |DateTime           |Hour of the data collected in local CST time|
    |`traffic_volume`         |Numeric            |Hourly I-94 ATR 301 reported westbound traffic volume|

    The data is fairly clean. It has no missing values in any other field than the holiday. Those are missing on purpose, since... it was not a holiday. You can verify this with `df.isna().sum()`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Not so fast!

    Remember, this is TIME data. Missing values are no necessary NULL values, but simply gaps in the timeline. This is hourly data, so we should have `len(df)` count of continuous hourly values.
    """)
    return


@app.function
def plot_timedataframe(df, split_date=None):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["traffic_volume"], linewidth=0.5, color="steelblue")
    if split_date:
        ax.axvspan(split_date, df.index[-1], alpha=0.35, color="gray")
    ax.set_title("Metro Interstate Traffic Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Traffic Volume")
    plt.tight_layout()
    return plt.gcf()


@app.cell
def _(df):
    # Utilize DatetimeIndex to make plotting nice'n'easy 
    df_time = df.copy()
    df_time["date_time"] = pd.to_datetime(df_time["date_time"])
    df_time = df_time.set_index("date_time")

    split_idx = int(len(df_time) * 0.8)
    split_date = df_time.index[split_idx]

    plot_timedataframe(df_time, split_date).gca()
    return (df_time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspect smaller time frame
    """)
    return


@app.cell
def _(df_time):
    start = "2016-10-01"
    end = pd.to_datetime(start) + pd.Timedelta(days=7)

    df_snapshot = df_time.loc[start:end]

    plot_timedataframe(df_snapshot).gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Identify each Gap
    """)
    return


@app.cell
def _(df):
    dt = pd.to_datetime(df["date_time"]).sort_values().reset_index(drop=True)
    delta = dt.diff()

    mask = delta > pd.Timedelta("1h")
    gap_starts = dt[mask.shift(-1, fill_value=False)].reset_index(drop=True)
    gap_ends = dt[mask].reset_index(drop=True)
    gap_durations = delta[mask].reset_index(drop=True)

    gaps_df = pd.DataFrame({
        "gap_start": gap_starts,
        "gap_end": gap_ends,
        "gap_duration_hours": gap_durations.dt.total_seconds() / 3600,
        "missing_hours": (gap_durations.dt.total_seconds() / 3600).astype(int) - 1,
    })

    expected_hours = int((dt.iloc[-1] - dt.iloc[0]).total_seconds() / 3600) + 1
    pct_missing = gaps_df["missing_hours"].sum() / expected_hours * 100

    print(f"Total rows     : {len(dt):,}")
    print(f"Expected hours : {expected_hours:,}")
    print(f"Missing hours  : {gaps_df['missing_hours'].sum():,}  ({pct_missing:.1f}% of span)")
    print(f"Gaps found     : {len(gaps_df)}")

    gaps_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature engineering

    Plan...

    * Drop textual weather fields. We *could* use an AI model to turn this into somewhat numerical value, but we won't.
    * Convert date time into...
        * sin_hour, cos_hour
        * sin_dow, cos_dow (day of week)
        * sin_month, cos_month
    * One Hot Encode the holiday (fit on train, transform both).
    * Normalize the numeric features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Non-Leaking Features

    These can be added to train and test, since all this information is available -- cyclical features are deterministicly defined from timestamp, now and forever.
    """)
    return


@app.cell
def _(df):
    def cyclical_encode(series, max_val):
        sin = np.sin(2 * np.pi * series / max_val)
        cos = np.cos(2 * np.pi * series / max_val)
        return sin, cos

    df_eng = df.copy()
    df_eng["date_time"] = pd.to_datetime(df_eng["date_time"])

    sin_hour, cos_hour = cyclical_encode(df_eng["date_time"].dt.hour, 24)
    sin_dow, cos_dow = cyclical_encode(df_eng["date_time"].dt.dayofweek, 7)
    sin_month, cos_month = cyclical_encode(df_eng["date_time"].dt.month, 12)

    df_engineered = (
        df_eng.assign(
            sin_hour=sin_hour,
            cos_hour=cos_hour,
            sin_dow=sin_dow,
            cos_dow=cos_dow,
            sin_month=sin_month,
            cos_month=cos_month,
        )
        .drop(columns=["weather_main", "weather_description", "date_time"])
    )
    df_engineered["holiday"] = df_engineered["holiday"].fillna("None")

    df_engineered
    return (df_engineered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train/Test Split

    We will not be doing hyperparameter tuning, so no need for validation. Otherwise... we would need it.
    """)
    return


@app.cell
def _(df_engineered):
    TRAIN_SIZE = 0.8
    train_index = int(len(df_engineered) * TRAIN_SIZE)

    train_df = df_engineered.iloc[:train_index]
    test_df = df_engineered.iloc[train_index:]

    train_df
    return test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Features That Leak

    These features would leak information into our test set if we would fit the pipeline before split.
    """)
    return


@app.cell
def _(test_df, train_df):
    TARGET = "traffic_volume"
    cat_cols = ["holiday"]
    num_cols = [c for c in train_df.columns if c not in cat_cols + [TARGET]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("scaler", StandardScaler(), num_cols),
        ]
    )
    feature_pipeline = Pipeline([("preprocessor", preprocessor)])

    X_train = feature_pipeline.fit_transform(train_df.drop(columns=[TARGET]))
    X_test = feature_pipeline.transform(test_df.drop(columns=[TARGET]))

    # The model
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(train_df[[TARGET]])
    y_test = scaler_y.transform(test_df[[TARGET]])
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Windowing
    """)
    return


@app.class_definition
class PandasMetroDataset(Dataset):
    def __init__(self, X, y, n_steps, m_steps):
        """
        Args:
            X: Scaled feature array of shape (N, num_features).
            y: Scaled target array of shape (N, 1).
            n_steps: Number of input time steps.
            m_steps: Number of future steps to predict.
        """
        self.n_steps = n_steps
        self.m_steps = m_steps

        num_windows = len(X) - n_steps - m_steps + 1
        self.windows = [
            (
                torch.tensor(X[i : i + n_steps], dtype=torch.float32),
                torch.tensor(y[i + n_steps : i + n_steps + m_steps], dtype=torch.float32),
            )
            for i in range(num_windows)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


@app.cell
def _(X_test, X_train, y_test, y_train):
    N_STEPS = 48  # n-hour lookback window
    M_STEPS = 3   # forecast horizon, predict m hours ahead 

    train_dataset = PandasMetroDataset(X_train, y_train, N_STEPS, M_STEPS)
    test_dataset = PandasMetroDataset(X_test, y_test, N_STEPS, M_STEPS)

    # Inspect the first few materialised windows
    train_dataset.windows[:3]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
