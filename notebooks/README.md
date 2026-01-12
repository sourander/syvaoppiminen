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

Note that we are using GPU, so assuming you have a local Nvidia GPU available, you want to install:

* Latest Nvidia Drivers
* Run `nvidia-smi` to verify installation and check CUDA version (e.g. `12.7`)
    * The nvidia-smi CUDA version shows the maximum CUDA Toolkit version your installed NVIDIA driver supports.
* Install CUDA Toolkit matching the version shown by `nvidia-smi`
    * This can be detected how !?!?!?
* 

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
uv sync --frozen
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


## Teacher 👨‍🏫: How to handle Solutions

Some of the Notebooks contain exercises. Example solutions are in a subdirectory `solutions`. Old hack was backing them up to OneDrive using a script. New solution is using `git-crypt` to encrypt the solutions directory, so that only teachers with the decryption key can access them.

Here is a guide for setup, I will most likely end up doing this on other courses too.

### Part A: Encrypt (macOS)

This setup is done only on one machine. I will start it on my macOS. The folloging command initializes git-crypt in the repository. The key file is created into `.git/git-crypt/keys/default`. Run this only once per repository.

```bash
brew install git-crypt
git-crypt init
# Output: Generating key...
```

Modify the Git's dotiles as follows:

```
# .gitattributes Add this:
notebooks/nb/solutions/** filter=git-crypt diff=git-crypt

# .gitignore Remove if exists
notebooks/nb/solutions/
```

Finally, push to GitHub:

```bash
# Usual commands
git add .gitattributes .gitignore notebooks/nb/solutions
git commit -m "Add encrypted solutions using git-crypt"

# Verify - list files and view one of them (should be GITCRYPT + binary data)
git show HEAD:notebooks/nb/solutions
git show HEAD:notebooks/nb/solutions/213_tensor_exercises.ipynb | head -n 2

# Push to GitHub
git push
```

### Part B: Pull (Ubuntu)

On another machine, Ubuntu Desktop in this, I ran these commands:

```bash
# Install
sudo apt install git-crypt
cd syvaoppiminen
git pull

# Verify
head -n 2 notebooks/nb/solutions/213_tensor_exercises.ipynb
# Output: GITCRYPT + binary data
```

### Part C: Share key (macOS)

Instead of GPG keys, I will be using symmetric key encryption for simplicity. These solutions can be found using LLM's anyways, so no need to overengineer this. In order to allow decryption on other machines, the key must be shared. On macOS machine, run:

```bash
# Export the repo’s symmetric key
git-crypt export-key gh-solutions.git-crypt.key

# Copy it to the Ubuntu machine
scp gh-solutions.git-crypt.key poytakone:~

# Delete local copy
rm gh-solutions.git-crypt.key
```

### Part D: Decrypt (Ubuntu)

On Ubuntu machine, run:

```bash
cd syvaoppiminen

# Unlock the repo using the symmetric key
git-crypt unlock ~/gh-solutions.git-crypt.key

# Verify
head -n 2 notebooks/nb/solutions/213_tensor_exercises.ipynb
# Output: { "cells": [ ...}
```

Finally, store the key to a password manager and delete the local file: `rm ~/gh-solutions.git-crypt.key`.

How does it work? After these commands, the Ubuntu PC will have exactly the same `.git/git-crypt/keys/default` file as the macOS machine. The "repo is the key" principle applies: as long as the same key file is used, the encryption and decryption will work seamlessly. The `unlock` command simply copied the key file to the correct location.



