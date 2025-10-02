# Notebooks for Course

## Naming Convention

This directory contains Jupyter notebooks used in the course. Unlike the theory site, these Notebooks are written in English, not in Finnish. Each notebook corresponds to a specific topic or module covered in the curriculum. If you check the `docs/.nav.yml`, you will notice that each course category/week has a hundred-starting identifier:

```
nav:
    - index.md
    - neuroverkot    # 100
    - tensorit       # 200
    - ...
    - kieli          # 700
    - aikasarjat     # 800
```

Underneath each category, each MMark will have their own ten-starting identifier in the header part of the Markdown file, like this:

```
---
priority: 100
---
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

## Using Python Kernel

The `uv` project exists only to make a kernel available for Jupyter Notebooks in VS Code within this repository. The kernel was created as a globally available kernel like this:

```bash
uv add ipykernel
uv run python -m ipykernel install --user --name nb-env --display-name "Syvaoppiminen I"
```

It will install the kernel spec to: `~/Library/Jupyter/kernels/nb-env`