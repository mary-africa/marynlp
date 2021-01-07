# `marynlp` - python

## Overview

This is the python version for `marynlp`. It uses python version `3.7`
## Installation Guide

This installation guide is can be followed by different classes of users in the project. This can be users, researches / developers / contributors


### Getting Started

Make sure you get the project in your machine

```bash
$ git clone https://github.com/inspiredideas/marynlp.git
```


To install only the `marynlp` package, move over the files needed for the package installation.

```bash
$ mkdir marynlp-main && mv ./marynlp/python ./marynlp-main
$ rm -rf marynlp-main && cd marynlp-main
```

To install the `marynlp` package **[for development sake]**, after cloning, only navigate to the `python` folder:
```bash
$ cd marynlp/python
```

### For using an environment

**(BECAUSE OF THE VOLATILE STATE OF THE PROJECT, DEVELOPERS SHOULD CONSIDER THIS)**<br />
If you want install the package in an environment, follow these steps below


- ### Install via virtualenv

    Make sure you have installed [`virtualenv`](https://github.com/pypa/virtualenv) in your computer. If you have not, try checking for solutions online (Try googling: "Installing virtualenv")

    Since the project uses `python==3.7.*`, you will need to install `python==3.7` according to your machine.

    #### For linux
    ```bash
    $ sudo apt install python3.7 
    ```
    
    To install
    ```bash
    $ python3.7 -m pip install virtualenv
    $ virtualenv -p python3.7 venv
    ```
    And to run the environment
    ```bash
    $ source venv/bin/activate 
    ```

    When that happens you should see `(venv)` in your shell like this:
    ```bash
    (venv) $
    ```

    This indicated that you have activated your shell. To leave the virtual environment. At any point, you can simply type `deactivate`. and the virtual environment will close

    ```bash
    (venv) $ deactivate
    $
    ```

- ### Install via conda
    Make sure you have installed anaconda in your machine. If not, be sure to check online for solutions (Try googling: "Downloading and Installing anaconda")

    Create an environment named `maryenv` (you can chose anyname if you want) that you will the package on. Using the following scripts, you get that going.

    ```bash
    $ conda create --name maryenv python=3.7
    ```

    Then activate the conda environment

    ```bash
    $ conda activate maryenv
    (maryenv) $
    ```

    To leave the conda environment, type in `conda deactivate` at any point in the terminal

    ```bash
    (maryenv) $ conda deactivate
    $
    ```

### Proceeding with the installation
Whether or not you have installed an environment for the project, you are to proceed with these steps. 

Just make sure that if you have chosen to install the virtual environment, **these steps are done when the virtual environment is running**

There are 2 types of users who are expected to interact with the project. 

- ### For users
    This is intended only for those who are using the `marylp` package (like the AI server / basic user of `marynlp`)

    ```bash
    $ pip install -r requirements.txt
    ```

- ### For developers
    You will be required to install the development assets

    ```bash
    $ pip install -r requirements.txt
    $ pip install -r requirements-dev.txt
    ```

After you have installed the project dependencies (according to the user type). Get to installing the package itself

```bash
pip install .
```

**That's all folks**

At this point, you'd have already installed the package.

**DEVELOPER NOTICE:** If you want make changes to the code and want to install the prevised changes, simply type `pip install .`. If that doesn't work, uninstall the `marynlp` package and install again.
