# Notebooks for Course

## Naming Convention

This directory contains Marimo Notebooks used in the course. Unlike the theory site, these Notebooks are written in English, not in Finnish. Each notebook corresponds to a specific topic or module covered in the curriculum. If you check the `docs/.nav.yml`, you will notice that each course category/week has a hundred-starting identifier:

```yaml
nav:
    - index.md
    - neuroverkot    # 100
    - tensorit       # 200
    - ...
    - kieli          # 700
    - aikasarjat     # 800
```

Underneath each category, each entry will have their own ten-starting identifier in the header part of the Markdown file, like the first entry in `neuroverkot` category is 100, and the next one is 110, and so on. For example, the `docs/neuroverkot/neuroverkot_101.md` file will contain a header with a value `priority: 101`. Just go and check the files to see how it works, if you are curious.

The Notebooks will match to this numbering scheme. For example, if the lesson `docs/neuroverkot/syvaoppiminen_FC.md` has a priority of `110`, and the lesson would have three Notebooks, they would be named as follows:

```
notebooks/nb/100/110_lorem.py
notebooks/nb/100/111_ipsum.py
notebooks/nb/100/112_dolor.py
```

## Using Marimo

The `uv` project exists so that you can use Marimo either in Browser or using VS Code Extension for Marimo. This guide focuses on the browser usage, but there is a section about VS Code as well. Note that teacher will use it in the browser during the lessons.

### Using the Teacher's UV Environment

To use teacher's UV environment, simply run:

```bash
# Go to the directory that contains the pyproject.toml
# for MARIMO configuration. Notice that there is also a 
# pyproject.toml in the root of this repository. It is 
# for the documentation site, not for Marimo!
cd notebooks

# This will create `.venv/` in the current directory
# with same versions of all packages as the teacher has.
# They are listed in uv.lock file.
uv sync
```

You can check what the teacher's environment contains by inspecting the `pyproject.toml` file. If you a wondering what the recommended is in `marimo[recommended]`, it is called *extras*. As of 2025 November, it contains following packages:

* duckdb
* altair
* polars
* sqlglot
* openai
* google-genai
* ruff
* nbformat

You can check those by reading the [pyproject.toml](https://github.com/marimo-team/marimo/blob/main/pyproject.toml) file in Marimo repository. Look for line starting with `recommended = [`.

<details>
<summary>🤓 Click to expand: If you want your own UV Environment</summary>

### Creating your own UV Environment

```bash
# To create your own,
# delete uv project files
rm pyproject.toml  uv.lock

# Then create your own bare uv environment with a chosen name
uv init --bare syvaoppiminen

# Add marimo with recommended extras
uv add "marimo[recommended]"

# Optional: Add other packages
uv add torchvision matplotlib scikit-learn # etc.

# When you need teacher's notebooks, copy them over 
# from nb/ to your own workspace.
mkdir syvaoppiminen/100/
cp nb/100/110_first_model.py syvaoppiminen/100/110_first_model.py

# Start working in Browser
uv run marimo edit
```
</details>


## Optional: Using Marimo VS Code Extension

It is completely possibly to use Marimo right inside VS Code. However, I think that the browser experience is better in terms of usability. If you still want to use VS Code Extension, follow these steps:

1. Add this to your `notebooks/.vscode/settings.json`:

    ```json
    {
    "marimo.marimoPath": "uv run marimo",
    "marimo.sandbox": true // optional
    }
    ```
2. Install the Marimo VS Code Extension.
3. Then, run in terminal:

    ```bash
    uv run marimo vscode install-extension
    ```

The last command will open VS Code in the `notebooks` directory. You want this, since the VS Code will use the `{{workspaceFolder}}/.venv/**/*` as the Python interpreter for everything, including the Marimo Notebooks.

## 🤖 Using Github Copilot with Marimo

To enable Github Copilot in Marimo Notebooks, follow these steps:

1. Install Github CLI (`gh`)
2. Install extension: `gh extension install https://github.com/github/gh-models`
3. Login: `gh auth login`
4. Get a token: `gh auth token`
5. Modify the `~/.config/marimo/marimo.toml` file to include your token:

```toml
[ai.models]
chat_model = "github/gpt-4o-mini"

[ai.github]
api_key = "gho_..."
```

You can also add other models like `github/gpt-5`. To identify suitable models, run `gh models list`. You can either edit the `marimo.toml` file directly or use the Marimo settings in the Web UI. For reference, adding the `gpt-5` model would create this kind of entries in the `marimo.toml`:

```toml
[ai.models]
custom_models = ["github/gpt-5"]
chat_model = "github/gpt-4o-mini"
edit_model = "github/gpt-5"
displayed_models = ["github/gpt-4o-mini", "github/gpt-5"]
```


## Teacher 👨‍🏫: How to handle solution backups

Some of the Notebooks contain exercises. Example solutions are in a subdirectory `solutions`. You can backup the current state of that directory by running:

```bash
./backup_solutions.sh
```

It will be saved in a timestamped `tar.gz` file into a `${ONEDRIVE}/__SOLUTIONS_BACKUPS/syvaoppiminen/yyyy-mm-dd-solutions.tar.gz`.