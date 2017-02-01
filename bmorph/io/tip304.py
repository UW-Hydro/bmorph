"""
IO utilities for the files associated with the BPA co-funded project:
TIP 304: Predicting the Hydrologic Response of the Columbia River System to
         Climate Change

Import and export file formats as required by the project. Provide the data to
the rest of bmorph as panda data frames and/or time series.
"""

import pandas as pd
import xarray as xr


def construct_file_name(sitedata, file_template):
    '''Construct a file name from a dictionary and template'''
    return file_template.format(**sitedata)


def get_metadata(infilename, comment='#'):
    '''Get commented out section from file as metadata'''
    metadata = ''
    with open(infilename, 'r') as f:
        for line in f:
            if line.startswith(comment):
                metadata += line
            else:
                break
    return metadata


def get_model_ts(infilename, na_values='-9999', comment='#',
                 rename_columns=None, column='streamflow'):
    '''Retrieve modeled time series from file by site index'''
    ts = pd.read_csv(infilename, comment=comment, na_values=na_values,
                     index_col=0, parse_dates=True)
    # renaming of columns may seem superfluous if we are converting to a Series
    # anyway, but it allows all the Series to have the same name
    if rename_columns:
        ts = ts.rename(columns=rename_columns)
    return pd.Series(ts[column])


def get_nrni_ts_nc(site_index, nrni_file,
                rename_columns={'Streamflow': 'streamflow'},
                column='streamflow'):
    '''Retrieve NRNI streamflow from file by site index'''
    nrni = xr.open_dataset(nrni_file)
    nrni.coords['time'] = nrni.Time
    nrni = nrni.Streamflow[nrni.IndexNames == site_index, :]
    nrni = nrni[0, ].drop('index')
    nrni = nrni.to_dataframe()
    # renaming of columns may seem superfluous if we are converting to a Series
    # anyway, but it allows all the Series to have the same name
    if rename_columns:
        nrni = nrni.rename(columns=rename_columns)
    return pd.Series(nrni[column])

def get_nrni_ts_csv(site_index, nrni_file,
                rename_columns={'Streamflow': 'streamflow'}):
    nrni = pd.read_csv(nrni_file, index_col=0, skiprows=[1,2,3,4,5,6])
    nrni.drop('Unnamed: 1', axis=1, inplace=True)
    nrni.index = pd.date_range(start='1928-07-01', end='2008-09-30')
    for suffix in ['5N', '_QD', '_QN', '_QM']:
        nrni.columns = nrni.columns.str.replace(suffix, '')
    nrni = nrni[site_index]
    if rename_columns:
        nrni.columns = ['streamflow']
    return pd.Series(nrni)

def put_bmorph_ts(outfilename, ts, metadata=''):
    '''Write bias-corrected output to file'''
    buffer = metadata
    buffer += ts.to_csv()
    with open(outfilename, 'w') as f:
        f.write(buffer)
    return
