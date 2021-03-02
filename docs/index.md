# bmorph
Bias correction for streamflow time series

## Installation

We provide a conda environment in `environment.yml`. You can build the environment by running:

`conda env create -f environment.yml`

Then, to install `bmorph` run,

```
conda activate bmorph
python setup.py develop
python -m ipykernel install --user --name bmorph
```

## Usage

### Notebook
This will be explored in a later release of `bmorph`

### Command line
`./scripts/bmorph_tip304` contains an implementation of the method for the
TIP304 project. The script can be run as:

`bmorph_tip304 --cfg <configuration file> --verbose`

There is a sample configuration file in `./config`
