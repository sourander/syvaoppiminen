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

The `uv` project exists so that you can use Marimo either in Browser or using VS Code Extension for Marimo. This guide focuses on the browser usage, but there is a section about VS Code as well. Note that teacher will use it in the browser during the lessons. You **should** have the Marimo installed as being the 2nd year AI student, but if not, follow the official [Installing uv](https://docs.astral.sh/uv/getting-started/installation/) guide.

Note that we are using GPU, so this setup requires extra steps compared to Johdatus koneoppimiseen (ML Basics) course setup. If you have a compatible Nvidia GPU or Apple Silicon GPU, you can run Marimo locally following the guide below. If you have a different setup, like an ADM GPU, contact the teacher for help if needed.

### On Windows

<details>
<summary>Click to expand</summary>

<br>

Install:

* Latest Nvidia Drivers
* Docker Desktop with WSL2 backend

You can then launch the Compose Project with:

```bash
docker compose -f docker-compose-marimo.yml up -d
```

After that, you can access the Marimo service at localhost:2718 and TensorBoard at port 6006. Read the [GPU support in Docker Desktop for Windows](https://docs.docker.com/desktop/features/gpu/) for more information.

</details>

### On Ubuntu (native)

<details>
<summary>Click to expand</summary>

<br>

Install:

* Latest Nvidia Drivers, following a guide: [Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html) 
* CUDA Toolkit.

Read the **Prepare Ubuntu** and **Network Repository Installation** sections from the [CUDA Installation Guide for Linux: Ubuntu](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu) guide. After those steps, do the [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#post-installation-actions) as well.

After that, simply utilize the `uv` command to run Marimo in browser:

```bash
# Marimo
uv run marimo edit

# TensorBoard
uv run tensorboard --logdir=runs
```

</details>

### On Ubuntu (Docker)

<details>
<summary>Click to expand</summary>

<br>
Install

* Latest Nvidia Drivers, following a guide: [Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html) 
* Docker Engine (see [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/))
* Nvidia Container Toolkit

When using Docker, you do not need to install CUDA Toolkit, since the Docker image contains it already. Read the [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for more information. After that, the launch command is same as on Windows:

```bash
docker compose -f docker-compose-marimo.yml up -d
```
</details>

### On macOS

Assuming you have M1 or other Apple Silicon GPU, it just works. You do not need Docker. Simply run `uv run marimo edit`. 

#### Information about the Teacher's UV Environment

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

You can check what the teacher's environment contains by inspecting the `pyproject.toml` file. If you are wondering what the term *recommended* is in `marimo[recommended]`, it is called *extras*. As of 2025 November, it contains following packages:

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

#### Creating your own UV Environment

```bash
# To create your own,
# delete uv project files
rm pyproject.toml  uv.lock

# Delete existing .venv if exists
rm -rf .venv/

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

## Teacher 👨‍🏫: How to handle Solutions

Some of the Notebooks contain exercises. Example solutions are in a subdirectory `solutions`. Old hack was backing them up to OneDrive using a script. New solution is using `git-crypt` to encrypt the solutions directory, so that only teachers with the decryption key can access them.

Here is a guide for setup. I might move this guide to `How to Git` one day. For now, it is here.

### Prerequisites

The `git-crypt` key has been originally created only on one machine and then distributed to other machines utilizing a password manager.

Here we assume that: 

1. the `.gitignore` file contains a filter that we need. Check the file in `../.gitignore` for details.
2. the `gh-solutions.git-crypt.key` has been downloaded to `$HOME` directory.

The key file is copied into `.git/git-crypt/keys/default` after running the commands below. Run this only once per repository (or again if you need to clone the repository again).

```bash
# Navigate to whereever the repo root is
cd ~/Code/sourander/syvaoppiminen

# Unlock
git-crypt unlock ~/gh-solutions.git-crypt.key

# Remove the key
rm ~/gh-solutions.git-crypt.key

# Check
git-crypt status -e
```

This should list all files in `notebooks/nb/solutions` as encrypted.
