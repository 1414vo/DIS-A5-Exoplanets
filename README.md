# A5: Exoplanets - ivp24

This repository contains my approach towards the Exoplanets minor module.

![Static Badge](https://img.shields.io/badge/build-passing-lime)
![Static Badge](https://img.shields.io/badge/logo-gitlab-blue?logo=gitlab)

## Table of contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Running the solver](#running-the-solver)
4. [Features](#features)
5. [Frameworks](#frameworks)
6. [Build status](#build-status)
7. [Credits](#credits)

## Requirements

The user should preferrably have a version of Docker installed in order to ensure correct setup of environments. If that is not possible, the user is recommended to have Conda installed, in order to set up the requirements. If Conda is also not available, make sure that the packages described in `environment.yml` are available and installed.

## Setup

We provide two different set up mechanisms using either Docker or Conda. The former is recommended, as it ensures that the environment used is identical to the one used in the development in the project.

### Using Docker

To correctly set up the environment, we utilise a Docker image. To build the image before creating the container, you can run.

```docker build -t ivp24_sudoku .```

The setup image will also add the necessary pre-commit checks to your git repository, ensuring the commits work correctly. You need to have the repository cloned beforehand, otherwise no files will be into the working directory.

Afterwards, any time you want to use the code, you can launch a Docker container using:

```docker run --name <name> --rm -ti ivp24_exop```

If you want to make changes to the repository, you would likely need to use your Git credentials. A safe way to load your SSH keys was to use the following command:

```docker run --name <name> --rm -v <ssh folder on local machine>:/root/.ssh -ti ivp24_exop```

This copies your keys to the created container and you should be able to run all required git commands.

### Using Conda

The primary concern when using Conda is to install the required packages. In this case **make sure to specify an environment name**. Otherwise, you risk overriding your base environment. Installation and activation can be done using the commands:

```conda env create --name <envname> -f environment.yml ```
```conda activate <envname> ```

For the sake of consistency and for plot rendering, please also run the following commands (on a UNIX system):

```
apt-get update && apt-get install -y git vim unzip gcc pkg-config g++ texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

For radial velocity running, please set up the following environment (as the versions of NumPy are inconsistent with the previous one).

```
conda create -n radvel python=3.9
conda activate radvel
pip install -r requirements.txt
pip install radvel scikit-learn
conda deactivate radvel
```

## Running the scripts

### Transits
We include 2 scripts for reproducing the transit results:

For GP fitting and removing stellar activity:
```python -m src.remove_activity ./data ./out```

For transit fitting and plotting:
```python -m src.fit_transit ./data ./out <--print_progress>```

### Radial velocities
For this section, please activate the `radvel` environment:

```conda activate radvel```

To reproduce the model selection results please run the following script:

```python -m src.rv_model_inference ./data ./out```

To reproduce the model inference, the following sequence of commands should be used:
```
radvel fit -s src/rv_inference.py
radvel mcmc -s src/rv_inference.py
radvel report -s src/rv_inference.py
radvel derive -s src/rv_inference.py
```

## Build status
Currently, the build is complete and the program can be used to its full capacity.

## Credits

The `.pre-commit-config.yaml` configuration file content has been adapted from the Research Computing lecture notes.
