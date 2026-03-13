import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import altair as alt
    import seaborn as sns

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from torch.utils.data import Dataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torchmetrics import MetricCollection
    from torchmetrics.regression import (
        MeanSquaredError,
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        SymmetricMeanAbsolutePercentageError
    )
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from datetime import datetime

    alt.data_transformers.enable("vegafusion")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Configuration
    USE_GPU = True  # Toggle this to False to use CPU instead

    # Hyperparameters
    LEARNING_RATE = 0.01
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 50


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
    return BATCH_SIZE, EPOCHS, LEARNING_RATE, device


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
    df["date_time"] = pd.to_datetime(df["date_time"])
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
    df_time = df_time.set_index("date_time")

    split_idx = int(len(df) * 0.8)
    split_date = df_time.index[split_idx]

    plot_timedataframe(df_time, split_date).gca()
    return (df_time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspect smaller time frame

    This is to feed our curiosity.
    """)
    return


@app.cell
def _(df_time):
    start = "2016-10-01"
    end = pd.to_datetime(start) + pd.Timedelta(days=7)

    plot_timedataframe(df_time.loc[start:end]).gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect distribution

    For some counts, we might want to utilize Poisson if the counts are all the time near-zero. Note however, that this distribution is fairly far away from typical Poisson. The resulting histogram looks more like multi-modal (multiple camel backs or peaks) gaussian. Probably the time-of-day or day-of-week are causing multi-modality.
    """)
    return


@app.cell
def _(df):
    # 1. Clean the raw data by dropping duplicate rows
    df_clean = df.drop_duplicates().copy()

    # 2. Extract the traffic volume and calculate the mean
    raw_traffic = df_clean["traffic_volume"]
    mean_traffic = raw_traffic.mean()

    # 3. Create the distribution plot
    plt.figure(figsize=(10, 5))

    sns.histplot(
        raw_traffic, 
        bins=50, 
        kde=True, 
        alpha=0.7
    )

    # Add a line for the Mean
    plt.axvline(
        mean_traffic, 
        linestyle='dashed', 
        linewidth=2.5, 
        label=f'Mean Volume: {mean_traffic:.0f} cars'
    )

    plt.title("Traffic Volume Distribution")
    plt.xlabel("Traffic Volume (Cars per Hour)")
    plt.ylabel("Frequency (Number of Hours)")
    plt.legend()
    plt.tight_layout()

    # Display the figure in Marimo
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspect each gap
    """)
    return


@app.cell
def _(df):
    # Analyze only the date_time Series
    dt = df["date_time"].sort_values().reset_index(drop=True)
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
    print(f"Gaps >1 hour : {(gaps_df['missing_hours'] > 1).sum()}")
    print(f"Duplicate hours: {dt.duplicated().sum():,}")

    gaps_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspect each duplicate
    """)
    return


@app.cell
def _(df):
    # Find all rows that share a 'date_time' with another row
    duplicate_mask = df.duplicated(subset=['date_time'], keep=False)

    # Filter the dataframe and sort by time so duplicates are adjacent
    df_duplicates = df[duplicate_mask].sort_values(by='date_time')

    # Display the dataframe to browse the duplicates
    df_duplicates
    return (df_duplicates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Double check lines where holiday

    We would not wan to use FIRST() our our duplicate fixing aggregation function if every other field had values.
    """)
    return


@app.cell
def _(df_duplicates):
    df_duplicates[df_duplicates["holiday"].notna()]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fix it all!

    Here we will...

    1. Use group by to merge duplicates using FIRST(). This keeps the first value within the group for each column.
    2. Force the strict hourly frequency. This fills the missing values with NaNs.

    The step 2 might not sound like a fix, but this will let us handle the missing values in multiple ways, like...

    * Impute where only 1-2 values are missing
    * Slice the data into segments/windows that contain no NULLs. This way, our segments won't include hidden "time jumps" as they would if we would've used the original data.
    """)
    return


@app.cell
def _(df):
    # GROUPBY using the date_time to get rid on duplicates.
    # By inspection, it has been proven that FIRST() is a viable solution here.
    # Mean would work, but only with number fields.
    df_continuous = df.groupby("date_time").first()

    # Force the strict hourly frequency ('h'). 
    # This expands the index to cover every single hour, filling missing hours with NaNs.
    df_continuous = df_continuous.asfreq('h')

    print(f"\n--- After Forcing Hourly Grid ---")
    print(f"New dataset shape : {df_continuous.shape}")
    print(f"Total NaN rows    : {df_continuous['traffic_volume'].isna().sum():,} (should be same as before)")

    df_continuous
    return (df_continuous,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fix Part Two: Impute

    This will make each..

    * Gap of size 1 -> no gap
    * Gap of size 2 -> 1
    * Gap of size 3 -> 2
    * ... and so on
    """)
    return


@app.cell
def _(df_continuous):
    # Forward fill (LAG 1 / keep previous) strictly limited to 1 consecutive hour
    df_imputed = df_continuous.ffill(limit=1)

    # Quick check to see how many NaNs were successfully filled
    filled_count = df_continuous['traffic_volume'].isna().sum() - df_imputed['traffic_volume'].isna().sum()
    print(f"1-hour gaps filled: {filled_count:,}")
    print(f"Remaining NaN rows (large gaps): {df_imputed['traffic_volume'].isna().sum():,}")
    df_imputed
    return (df_imputed,)


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
def _(df_imputed):
    def cyclical_encode(series, max_val):
        sin = np.sin(2 * np.pi * series / max_val)
        cos = np.cos(2 * np.pi * series / max_val)
        return sin, cos

    # Reset the index so 'date_time' becomes a regular column again
    df_eng = df_imputed.reset_index()

    # Calculate cyclical features using the date_time column
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
        # Dropping weather columns, but explicitly keeping 'date_time'
        .drop(columns=["weather_main", "weather_description"])
    )

    # Fill missing holidays
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
    return test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Features That Leak

    These features would leak information into our test set if we would fit the pipeline before split.

    /// attention | Date time
    We are keeping the datetime columns matching the ordering or `X_train` and `X_test`. Why? For educational purposes. Later on, we want to be able to window our dataset into small samples without losing the time information. This will allow us to plot e.g. 5 consecutive windows including their original source timestamps.
    ///
    """)
    return


@app.cell
def _(test_df, train_df):
    TARGET = "traffic_volume"
    DATE_COL = "date_time"

    DATE_train = train_df[DATE_COL].reset_index(drop=True)
    DATE_test = test_df[DATE_COL].reset_index(drop=True)

    cat_cols = ["holiday"]
    num_cols = [c for c in train_df.columns if c not in cat_cols + [DATE_COL]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("scaler", StandardScaler(), num_cols),
        ]
    )
    feature_pipeline = Pipeline([("preprocessor", preprocessor)])

    X_train = feature_pipeline.fit_transform(train_df.drop(columns=[DATE_COL]))
    X_test = feature_pipeline.transform(test_df.drop(columns=[DATE_COL]))

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(train_df[[TARGET]])
    y_test = scaler_y.transform(test_df[[TARGET]])
    return DATE_test, DATE_train, X_test, X_train, scaler_y, y_test, y_train


@app.cell
def _(X_test, X_train, y_test, y_train):
    print("Missing Values in Training Set")
    print(f"X_train rows with NaNs: {np.isnan(X_train).any(axis=1).sum():,}")
    print(f"y_train : {np.isnan(y_train).sum():,} of {len(y_train)}")

    print("\nMissing Values in Testing Set")
    print(f"X_train rows with NaNs: {np.isnan(X_test).any(axis=1).sum():,}")
    print(f"y_test  : {np.isnan(y_test).sum():,} of {len(y_test)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset (Windowing)

    Read the Datasets methods and arguments carefully. Notice how we handle the datetimes.
    """)
    return


@app.class_definition
class PandasMetroDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, dates: pd.Series, n_steps: int, m_steps: int):
        """
        Args:
            X: Scaled feature array of shape (N, num_features).
            y: Scaled target array of shape (N, 1).
            dates: Pandas Series or array of datetime objects.
            n_steps: Number of input time steps.
            m_steps: Number of future steps to predict.
        """
        # Store references to the full arrays
        self.X = X
        self.y = y
        self.dates = np.array(dates) 
        self.n_steps = n_steps
        self.m_steps = m_steps

        # Store only the starting integer indices of valid windows
        self.valid_indices = []

        # Theoretical maximum of these "n-grams" that fit the data
        num_windows = len(X) - n_steps - m_steps + 1

        for i in range(num_windows):
            x_window = X[i : i + n_steps]
            y_window = y[i + n_steps : i + n_steps + m_steps]

            # Check if there are ANY NaNs in the input or target windows
            if not (np.isnan(x_window).any() or np.isnan(y_window).any()):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Standard PyTorch method for training loops. Returns only tensors."""
        start_idx = self.valid_indices[idx]

        x_window = self.X[start_idx : start_idx + self.n_steps]
        y_window = self.y[start_idx + self.n_steps : start_idx + self.n_steps + self.m_steps]

        return (
            torch.tensor(x_window, dtype=torch.float32),
            torch.tensor(y_window, dtype=torch.float32)
        )

    def get_window_with_dates(self, idx):
        """Educational helper to extract window data alongside timestamps for plotting."""
        start_idx = self.valid_indices[idx]

        # Get the standard tensors
        x_tensor, y_tensor = self.__getitem__(idx)

        # Slice the dates
        x_dates = self.dates[start_idx : start_idx + self.n_steps]
        y_dates = self.dates[start_idx + self.n_steps : start_idx + self.n_steps + self.m_steps]

        return x_tensor, y_tensor, x_dates, y_dates


@app.cell
def _(DATE_test, DATE_train, X_test, X_train, y_test, y_train):
    N_STEPS = 48  # n-hour lookback window
    M_STEPS = 3   # forecast horizon, predict m hours ahead 

    # Initialize the datasets
    train_dataset = PandasMetroDataset(X_train, y_train, DATE_train, n_steps=N_STEPS, m_steps=M_STEPS)
    test_dataset = PandasMetroDataset(X_test, y_test, DATE_test, n_steps=N_STEPS, m_steps=M_STEPS)

    print(f"Valid training windows: {len(train_dataset):,}")
    print(f"Valid testing windows: {len(test_dataset):,}")
    return test_dataset, train_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot consecutive windows

    **Note:** The plotting function has been created with Gemini 3.1 Pro.

    /// note | Tip
    Investigate what happens e.g. in window range #58-60
    ///
    """)
    return


@app.function(hide_code=True)
def plot_overlapping_windows(dataset, start_idx=0, num_windows=3, figsize=(14, 8)):
    """Educational helper to visualize overlapping sliding windows and forecasts.
    Creates a grid of stacked subplots for clarity.
    """

    # Helper function for common plotting tasks
    def format_subplot(ax, window_dates, window_y, title_text, label_prefix):
        # a. Plot the background "context" data
        ax.plot(
            overall_dates_converted, 
            dataset.y[base_start:last_end], 
            color="gray", 
            linestyle="--", 
            alpha=0.5
        )

        # b. Plot the input window portion (n_steps)
        input_dates = window_dates[:dataset.n_steps]
        current_idx = dataset.valid_indices[start_idx + i] # Grabbing the current starting index
        input_y = dataset.y[current_idx : current_idx + dataset.n_steps]

        ax.plot(
            input_dates,
            input_y,
            color="steelblue",
            linestyle="-",
            linewidth=1.5,
            label=f"{label_prefix} Input Context (n={dataset.n_steps})"
        )

        # c. Overlay the forecast horizon (m_steps)
        # Stitch the last input point to the forecast arrays to close the visual gap
        forecast_dates_connected = np.concatenate([input_dates[-1:], window_dates[dataset.n_steps:]])
        forecast_y_connected = np.concatenate([input_y[-1:], window_y])

        ax.plot(
            forecast_dates_connected,
            forecast_y_connected,
            color="tomato",
            linestyle="-.",
            marker="o",
            markersize=6,
            linewidth=2,
            label=f"{label_prefix} Forecast Horizon (m={dataset.m_steps})"
        )

        ax.set_title(title_text)
        ax.set_ylabel("Traffic Vol. (Scaled)")
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')


    # Create subplots, one for each window. 
    # 'sharex=True' ensures all plots use the exact same time axis.
    fig, axes = plt.subplots(nrows=num_windows, ncols=1, sharex=True, figsize=figsize)

    # Determine the overall date range for the subplots
    # Pull the *first* full window's context dates for reference
    base_idx = dataset.valid_indices[start_idx]

    # Range of data to plot in each subplot for background context
    # Adjust this as needed. Here we plot from base start - 2 hours 
    # to last target + 2 hours to add some buffer.
    base_start = max(0, base_idx - 2)
    last_window_idx = dataset.valid_indices[start_idx + num_windows - 1]
    last_target_end = last_window_idx + dataset.n_steps + dataset.m_steps
    last_end = min(len(dataset.dates), last_target_end + 2)

    overall_dates = dataset.dates[base_start : last_end]
    # Handle potentially complex date array to ensure matplotlib compatibility
    try:
        overall_dates_converted = mdates.date2num(overall_dates)
    except:
        overall_dates_converted = overall_dates

    # Iterate and plot each window in its assigned subplot
    for i in range(num_windows):
        _, y_tensor, _, y_dates = dataset.get_window_with_dates(start_idx + i)

        # Extract corresponding full window dates (n_steps + m_steps)
        current_idx = dataset.valid_indices[start_idx + i]
        full_window_dates = dataset.dates[current_idx : current_idx + dataset.n_steps + dataset.m_steps]

        # Convert the numpy.datetime64 to a Pandas Timestamp to use strftime
        start_time = pd.Timestamp(dataset.dates[current_idx])
        title_str = f"Window {start_idx + i} (start hour: {start_time.strftime('%Y-%m-%d %H:00')})"
        label_pre = f"Window {start_idx + i}"
        format_subplot(axes[i], full_window_dates, y_tensor.numpy(), title_str, label_pre)

    # Final visual polish
    # Only show axis labels on the bottom subplot
    axes[-1].set_xlabel("Date Time (Hourly)")

    # Improve date formatting on shared X-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
    fig.autofmt_xdate(rotation=30)

    fig.suptitle(f"Consecutive Overlapping Sliding Windows Visualization (num_windows={num_windows})")
    plt.tight_layout()
    # Add space above the subplots for the suptitle
    fig.subplots_adjust(top=0.92)

    return plt.gcf()


@app.cell
def _(mo):
    start_idx_slider = mo.ui.slider(start=1, stop=500, step=1, value=200, label="Start Index for Window Plot", full_width=True)
    return (start_idx_slider,)


@app.cell
def _(start_idx_slider):
    start_idx_slider
    return


@app.cell
def _(start_idx_slider, train_dataset):
    num_windows = 3
    h = num_windows * 3
    fig = plot_overlapping_windows(train_dataset, start_idx=start_idx_slider.value, num_windows=num_windows, figsize=(16, h))
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Loaders

    Here is a minor danger of confusion of when and how to keep the data in time order. With models like ARIMA, the algorithm relies on the continuous, **unbroken progression of the entire dataset** to calculate its internal stats. But an LSTM treats each 48-hour window as an independent "sentence."

    Because the strict chronological order is locked inside the window itself, the order in which you feed those windows to the model during training does not matter. In fact, shuffling your training data is a good idea.
    """)
    return


@app.cell
def _(BATCH_SIZE, test_dataset, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model
    """)
    return


@app.cell
def _(X_train, device):
    class MetroModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, m_steps=3, dropout=0.2):
            """
            Args:
                input_size: Number of features in the input X arrays.
                hidden_size: Number of features in the hidden state.
                num_layers: Number of recurrent layers.
                m_steps: Number of future time steps to predict.
                dropout: Dropout probability for LSTM layers (if num_layers > 1).
            """
            super(MetroModel, self).__init__()

            self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

            # Maps the final LSTM hidden state to the forecast horizon
            self.fc = nn.Linear(hidden_size, m_steps)

        def forward(self, x):
            # x shape: (batch_size, n_steps, input_size)

            # lstm_out shape: (batch_size, n_steps, hidden_size)
            lstm_out, (h_n, c_n) = self.lstm(x)

            # Extract the hidden state from the final time step of the sequence
            last_hidden_state = lstm_out[:, -1, :] 

            # Generate the forecast. Shape: (batch_size, m_steps)
            forecast = self.fc(last_hidden_state)

            # Add the feature dimension back to match the target shape: (batch_size, m_steps, 1)
            return forecast.unsqueeze(-1)

    # Let's instantiate it immediately to verify it matches our data
    # We get the input_size directly from the X_train array's shape
    INPUT_SIZE = X_train.shape[1]

    model = MetroModel(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, m_steps=3)
    model = model.to(device)
    model
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper Functions
    """)
    return


@app.cell(hide_code=True)
def _():
    def train_epoch(model, train_loader, criterion, optimizer, metric, device, writer=None, epoch=None):
        """Train the model for one epoch.

        Args:
            metric: A torchmetrics regression metric instance (e.g., MeanAbsoluteError)

        Returns:
            tuple: (average_loss, metric_value)
        """
        model.train()
        metric.reset()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update regression metric with predictions
            metric.update(outputs, targets)

            # Optional: Log batch-level loss to TensorBoard
            if writer and epoch is not None and batch_idx % 50 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        avg_loss = total_loss / len(train_loader)
        metric_value = metric.compute().item()

        return avg_loss, metric_value

    def evaluate(model, data_loader, criterion, metric, device):
        """Evaluate the model on the given data loader.

        Args:
            metric: A torchmetrics regression metric instance (e.g., MeanAbsoluteError)

        Returns:
            tuple: (average_loss, metric_value)
        """
        model.eval()
        metric.reset()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()

                # Update regression metric with predictions
                metric.update(outputs, targets)

        avg_loss = total_loss / len(data_loader)
        metric_value = metric.compute().item()

        return avg_loss, metric_value

    return evaluate, train_epoch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Loop
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    device,
    evaluate,
    model,
    scaler_y,
    test_loader,
    train_epoch,
    train_loader,
):
    # TensorBoard Setup
    device_name = str(device)
    run_name = f"metro_lstm_{device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text('config/hyperparameters', 
                    f'LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}')
    print(f"TensorBoard logging to: runs/{run_name}")

    # Loss, Optimizer, and Metrics
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mae_metric = MeanAbsoluteError().to(device)

    # Initialize local history dictionary to store metrics in memory
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': []
    }

    print("[INFO] training network...")
    for epoch in range(EPOCHS):
        # Training phase
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, mae_metric, device, writer, epoch
        )

        # Validation phase
        val_loss, val_mae = evaluate(model, test_loader, criterion, mae_metric, device)

        # Append epoch metrics to local history lists
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        # Log comparative metrics (train vs val) to TensorBoard
        writer.add_scalars('Loss/train_vs_val', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        writer.add_scalars('MAE/train_vs_val', {
            'train': train_mae,
            'val': val_mae
        }, epoch)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Convert scaled MAE to original units (cars) using the scaler itself.
            # Works for both MinMaxScaler and StandardScaler
            zero_scaled = np.array([[0.0]])
            train_mae_scaled = np.array([[train_mae]])
            val_mae_scaled = np.array([[val_mae]])

            zero_real = scaler_y.inverse_transform(zero_scaled)[0, 0]
            real_train_mae = abs(scaler_y.inverse_transform(train_mae_scaled)[0, 0] - zero_real)
            real_val_mae = abs(scaler_y.inverse_transform(val_mae_scaled)[0, 0] - zero_real)

            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - "
                  f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} ({real_train_mae:.0f} cars) | "
                  f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f} ({real_val_mae:.0f} cars)")

    print("[INFO] training complete!")

    # Log final hyperparameters with results for comparison across runs
    writer.add_hparams(
        {'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'device': device_name},
        {'hparam/final_train_mae': train_mae, 'hparam/final_val_mae': val_mae,
         'hparam/final_train_loss': train_loss, 'hparam/final_val_loss': val_loss}
    )

    # Close the TensorBoard writer to flush all remaining data
    writer.close()
    print("TensorBoard logs saved!")

    # Let's help Marimo DAG
    trained_model = model
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot History
    """)
    return


@app.cell
def _(EPOCHS, history):
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, EPOCHS), history["train_loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), history["train_mae"], label="train_mae")
    plt.plot(np.arange(0, EPOCHS), history["val_mae"], label="val_mae")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return


@app.cell
def _(device, model, scaler_y, test_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Store the full m_steps. Squeeze out the last dimension so it's (batch_size, m_steps)
            all_preds.append(outputs.squeeze(-1).cpu())
            all_targets.append(targets.squeeze(-1).cpu())

    # Concatenate all batches into large tensors of shape (Total_Samples, m_steps)
    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # --- The Scaler Trick ---
    # Flatten to (Total_Samples * m_steps, 1), inverse transform, then reshape back to (Total_Samples, m_steps)
    preds_real = scaler_y.inverse_transform(preds_tensor.numpy().reshape(-1, 1)).reshape(-1, model.fc.out_features)
    targets_real = scaler_y.inverse_transform(targets_tensor.numpy().reshape(-1, 1)).reshape(-1, model.fc.out_features)

    # Convert back to PyTorch tensors for torchmetrics
    preds_real_tensor = torch.tensor(preds_real, dtype=torch.float32)
    targets_real_tensor = torch.tensor(targets_real, dtype=torch.float32)
    return preds_real, preds_real_tensor, targets_real, targets_real_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics

    /// note | Guide: How to Read MAPE and SMAPE
    When evaluating a model, raw numbers like MAE (Mean Absolute Error) tell us exactly how many cars we missed by. However, missing by 50 cars is terrible if the actual traffic is 10 cars. With 5,000 cars, it is quite the opposite. This is where the **percentage errors** fit into the picture.

    **MAPE (Mean Absolute Percentage Error)**

    If your MAPE is 10%, a real-world count of 100 cars means your model usually guesses between 90 and 110. The potential problem is that MAPE divides the error by the *actual* value. If actual traffic drops to near zero at 3:00 AM, dividing by a tiny number causes the MAPE to explode. No exploding gradients, but exploding cars.

    **SMAPE (Symmetric Mean Absolute Percentage Error)**

    SMAPE solves the "exploding percentage" problem. By dividing the error by the average of both the actual *and* predicted numbers, it naturally caps the maximum possible error at 200%.  This is easiest to explain with an example:

    ```
    actual = 112
    forecast = 85
    difference = abs(112 - 85)        # = 27
    average = (112 + 85) / 2          # = 98.5
    smape_single = (27 / 98.5) * 100  # = 27.4%
    ```

    Thus, if your data never gets close to zero, MAPE is easier to explain to a business stakeholder. But for datasets like traffic – where night-time volume drops close to zero – SMAPE makes sense.

    On the other hand, if there is no clear long-time upwards trend, MAE might be a good option.
    ///
    """)
    return


@app.cell
def _(mo, preds_real_tensor, targets_real_tensor):
    # Define the Metric Collection
    metrics = MetricCollection({
        'MSE': MeanSquaredError(),
        'RMSE': MeanSquaredError(squared=False),
        'MAE': MeanAbsoluteError(),
        'MAPE': MeanAbsolutePercentageError(),
        'SMAPE': SymmetricMeanAbsolutePercentageError()
    })

    # 1. Calculate Overall Metrics (Flattening to evaluate all steps together)
    overall_results = metrics(preds_real_tensor.flatten(), targets_real_tensor.flatten())

    # 2. Initialize the Markdown table string with headers
    performance_metrics_table = (
        "| Horizon | MSE | RMSE (cars) | MAE (cars) | MAPE (%) | SMAPE (%) |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )

    # Append the overall results row
    performance_metrics_table += (
        f"| **Overall (All 3 Steps)** | {overall_results['MSE']:.2f} | {overall_results['RMSE']:.2f} | "
        f"{overall_results['MAE']:.2f} | {overall_results['MAPE'] * 100:.2f} | {overall_results['SMAPE'] * 100:.2f} |\n"
    )

    # 3. Calculate Per-Step Metrics (Degradation over time) and append to table
    m_steps = preds_real_tensor.shape[1]
    for step in range(m_steps):
        metrics.reset() # Reset before calculating the new slice
        step_results = metrics(preds_real_tensor[:, step], targets_real_tensor[:, step])

        performance_metrics_table += (
            f"| Step {step + 1} (Hour +{step + 1}) | {step_results['MSE']:.2f} | {step_results['RMSE']:.2f} | "
            f"{step_results['MAE']:.2f} | {step_results['MAPE'] * 100:.2f} | {step_results['SMAPE'] * 100:.2f} |\n"
        )

    # 4. Render the Markdown table in the Marimo notebook
    mo.md(performance_metrics_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot forecast

    To create a clean, continuous line chart we will just use a 1/2/3-hour forecast per diagram.
    """)
    return


@app.cell
def _(preds_real, targets_real):
    def plot_forecast_degradation(targets_real, preds_real, PLOT_LENGTH=200):
        m_steps = preds_real.shape[1]
        fig, axes = plt.subplots(nrows=m_steps, ncols=1, figsize=(14, 8), sharex=True)

        for step in range(m_steps):
            ax = axes[step]

            # Plot actual traffic for this specific future step
            ax.plot(
                targets_real[:PLOT_LENGTH, step], 
                label="Actual Traffic", 
                color="gray", 
                alpha=0.6,
                linewidth=2
            )

            # Plot predicted traffic for this specific future step
            ax.plot(
                preds_real[:PLOT_LENGTH, step], 
                label=f"Predicted Traffic (+{step + 1}h ahead)", 
                color="tomato", 
                linestyle="--",
                linewidth=1.5
            )

            ax.set_title(f"Step {step + 1} Forecast (+{step + 1} Hour)")
            ax.set_ylabel("Traffic Vol.")
            ax.legend(loc="upper left")

        axes[-1].set_xlabel("Hours (Sequential)")
        plt.tight_layout()
        return fig

    # Generate and display the plot in Marimo
    fig_degrade = plot_forecast_degradation(targets_real, preds_real)
    fig_degrade
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Altair Plot

    /// note | Can we loop this into an Autoregressive Forecast?
    You might wonder if we could take our 3-hour predictions, feed them back into the model's input history, and predict the *next* 3 hours—looping indefinitely like a text AI generating a story word-by-word.

    While our model *does* use past traffic as an input feature, in a real-world production environment, we face one major roadblock: **Future Unknowns (Exogenous Features).** To predict the next window of traffic, the model requires a *complete* set of features for that future time period. We can easily feed our predicted traffic volume back into the `traffic_volume` slot, and we know future dates and holidays perfectly. However, we **do not** have ground-truth data for future weather. To recursively forecast traffic days into the future using this specific model, we would first need to supply it with a separate, accurate forecast for temperature, rain, and snow! And, as everyone knows from experience, the weather forecast is not accuracy past a day or two.
    ///
    """)
    return


@app.cell
def _(DATE_test, preds_real, targets_real, test_dataset):
    target_indices = [idx + test_dataset.n_steps for idx in test_dataset.valid_indices]
    target_dates = DATE_test.iloc[target_indices]
    # Find the midpoint to slice the most recent half
    midpoint = len(target_dates) // 2

    # Combine to DF
    df_results = pd.DataFrame({
        "Date": pd.to_datetime(target_dates.values[midpoint:]),
        "Actual Traffic": targets_real[midpoint:, 0],
        "Predicted (1-h ahead)": preds_real[midpoint:, 0]
    })

    # Set Date as index, force an hourly grid, and reset index
    # this is to make sure that line cuts when there are no true Y values
    df_results = df_results.set_index("Date").asfreq('h').reset_index()

    # Melt the DataFrame into the long format required by Altair
    df_melted = df_results.melt(
        id_vars=["Date"], 
        value_vars=["Actual Traffic", "Predicted (1-h ahead)"],
        var_name="Type", 
        value_name="Volume"
    )

    # Create the interactive Altair chart
    chart = alt.Chart(df_melted).mark_line(opacity=0.8, strokeWidth=1.5).encode(
        x=alt.X("Date:T", title="Date Time"), 
        y=alt.Y("Volume:Q", title="Traffic Volume (Cars)"),
        color=alt.Color(
            "Type:N", 
            title="Legend",
            scale=alt.Scale(
                domain=["Actual Traffic", "Predicted (1-h ahead)"],
                range=["gray", "tomato"]
            )
        ),
        tooltip=["Date:T", "Type:N", "Volume:Q"] 
    ).properties(
        title="Real-Time Traffic Volume Forecast (Most Recent Half) - Zoomable",
        width=800,
        height=400
    ).interactive(bind_y=False) 

    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TensorBoard

    Remember, you can open the Tensorboard with...

    ```bash
    cd notebooks
    uv run tensorboard --logdir=runs
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
