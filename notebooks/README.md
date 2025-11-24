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

Underneath each category, each entry will have their own ten-starting identifier in the header part of the Markdown file, like this:

```markdown
---
priority: 100
---

# Neuroverkot 101

The rest of the lesson material goes here ...
```

The Notebooks will match to this numbering scheme. For example, if the lesson `docs/neuroverkot/syvaoppiminen_FC.md` has a priority of `110`, and the lesson would have three Notebooks, they would be named as follows:

```
notebooks/100/110_lorem.ipynb
notebooks/100/111_ipsum.ipynb
notebooks/100/112_dolor.ipynb
```

## Notebooks with Exercises

Some of the Notebooks contain exercises. These are also listed in the theory site.

Example solutions are in a subdirectory `solutions`. You can backup the current state of that directory by running:

```bash
./backup_solutions.sh
```

It will be saved in a timestamped `tar.gz` file into a `${ONEDRIVE}/__SOLUTIONS_BACKUPS/syvaoppiminen/yyyy-mm-dd-solutions.tar.gz`.

## Using Marimo

The `uv` project exists so that you can use Marimo either in Browser or using VS Code Extension for Marimo.

### Warning: Using nested UV Envs if challenging

To avoid problems, please open the Notebooks directory as a separate VS Code workspace. This is because using nested `uv` environments (one inside another) can lead to complications. By opening the Notebooks directory separately, you ensure that the VS Code Python interpreter matches to the venv that `uv` creates for Marimo, preventing potential conflicts and issues.

```bash
code notebooks
```

You have been warned! 👻

### Using the Teacher's UV Environment

To use teacher's UV environment, simply run:

```bash
# This will create `.venv/` in the current directory
uv sync
```

### Creating your own UV Environment

```bash
# To create your own:
rm pyproject.toml   # delete uv project files
rm uv.lock          # if any exists

# Then create your own bare uv environment with a chosen name
uv init --bare syvaoppiminen

# If you did, add marimo to uv
uv add "marimo[recommended]"

# When you need teacher's notebooks, copy them over from nb/ to your own workspace
mkdir syvaoppiminen/100/
cp nb/100/110_first_model.py syvaoppiminen/100/110_first_model.py

# Then, launch marimo server
uv run marimo edit
```

## Using Marimo VS Code Extension

Add this to your `.vscode/settings.json`:

```json
{
  "marimo.marimoPath": "uv run marimo",
  "marimo.sandbox": true // optional
}
```

Now you can launch Marimo in Browser like this:

```bash
uv run marimo edit
```

Rest of the guides will appears in the videos teacher has provided (and in the Marimo documentation).