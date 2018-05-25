# Coral: A Hybrid Static and Statistical Analysis System for Weak Supervision with Non-Text Data

*Installation instructions and package requirements copied from [Snorkel](https://github.com/HazyResearch/snorkel)*
## Installation
Snorkel uses Python 2.7 and requires [a few python packages](python-package-requirement.txt) which can be installed using [`conda`](https://www.continuum.io/downloads) and `pip`.

### Setting Up Conda
Installation is easiest if you download and install [`conda`](https://www.continuum.io/downloads).
If you are running multiple version of Python, you might need to run:
```
conda create -n py2Env python=2.7 anaconda
```
And then run the correct environment:
```
source activate py2Env
```

### Installing dependencies
First install [NUMBA](https://numba.pydata.org/), a package for high-performance numeric computing in Python via Conda:
```bash
conda install numba
```

Then install the remaining package requirements:
```bash
pip install --requirement python-package-requirement.txt
```
**Note that though [numbskull](https://github.com/HazyResearch/numbskull) is installed via the above command, you have to clone the numbskull repo separately so you can switch to the [coral branch](https://github.com/HazyResearch/numbskull/tree/coral). You also have to add this numbskull path to your pythonpath so it uses the right version**

Finally, enable `ipywidgets`:
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

_Note: Currently the `Viewer` is supported on the following versions:_
* `jupyter`: 4.1
* `jupyter notebook`: 4.2

### Frequently Asked Questions
See [this FAQ](https://hazyresearch.github.io/snorkel/install_faq) for help with common questions that arise. 

*Note: if you have an issue with the matplotlib install related to the module `freetype`, see [this post](http://stackoverflow.com/questions/20533426/ubuntu-running-pip-install-gives-error-the-following-required-packages-can-no); if you have an issue installing ipython, try [upgrading setuptools](http://stackoverflow.com/questions/35943606/error-on-installing-ipython-for-python-3-sys-platform-darwin-and-platform)*

## Jupyter Notebook Best Practices

The Coral experiments are built specifically with usage in **Jupyter/IPython notebooks** in mind; an incomplete set of best practices for the notebooks:

It's usually most convenient to write most code in an external `.py` file, and load as a module that's automatically reloaded; use:
```python
%load_ext autoreload
%autoreload 2
```
A more convenient option is to add these lines to your IPython config file, in `~/.ipython/profile_default/ipython_config.py`:
```
c.InteractiveShellApp.extensions = ['autoreload']     
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
```
