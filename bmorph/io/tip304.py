"""
IO utilities for the files associated with the BPA co-funded project:
TIP 304: Predicting the Hydrologic Response of the Columbia River System to
         Climate Change

Import and export file formats as required by the project. Provide the data to
the rest of bmorph as pandas.Series.
"""

import pandas as pd
import xarray as xr


def construct_file_name(file_info, file_template):
    '''Construct a file name from a dictionary and template

    Parameters
    ----------
    file_info : dict
        Dictionary with keys that match entries in `file_template`
    file_template : str
        Template for constructing path names for files

    Returns
    -------
    str
        Pathname for file
    '''
    return file_template.format(**file_info)


def get_metadata(infilename, comment='#'):
    '''Get commented out header section from ascii file as metadata

    Parameters
    ----------
    infilename : str
        Pathname for file
    comment : str, optional
        Comment indicator at the start of the line. Default value is '#'.

    Returns
    -------
    metadata : str
        The header section of the file as a single string (with line breaks)
    '''
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
    '''Retrieve modeled time series from ASCII file

    Parameters
    ----------
    infilename : str
        Pathname for file
    na_values : str, optional
        Values that should be converted to `NA`. Default value is '-9999'
    comment : str, optional
        Comment indicator at the start of the line. Default value is '#'=
    rename_columns: dict or None, optional
        Dictionary to rename columns. Default value is None
    column = str, optional
        Name of the column that will be returned. Default value is 'streamflow'

    Returns
    -------
    pandas.Series
        Column from file as a pandas.Series
    '''
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
    '''Retrieve NRNI streamflow from NetCDF file by site index

    Parameters
    ----------
    site_index : str
        Site index for NRNI site
    nrni_file : str
        Pathname for NRNI file
    rename_columns: dict or None, optional
        Dictionary to rename columns. Default value is
        `{'Streamflow': 'streamflow'}`
    column = str, optional
        Name of the column that will be returned. Default value is 'streamflow'

    Returns
    -------
    pandas.Series
        Column from file as a pandas.Series
    '''
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


def get_nrni_ts_csv(site_index, nrni_file, date_column=1,
                    series_name='streamflow'):
    '''Retrieve NRNI streamflow from ASCII file by site index'''
    nrni = pd.read_csv(nrni_file, skiprows=list(range(1, 7)))
    nrni.index = pd.date_range(
        start=parse_csv_date(nrni.iloc[0, date_column]),
        end=parse_csv_date(nrni.iloc[-1, date_column]))
    for suffix in ['5N', '_QD', '_QN', '_QM']:
        nrni.columns = nrni.columns.str.replace(suffix, '')
    nrni = pd.Series(nrni[site_index], dtype='float32')
    if series_name:
        nrni.name = series_name
    return nrni


def parse_csv_date(date_string):
    '''Fix the time stamp for the 2-digit years in the NRNI file'''
    (day, month, year) = date_string.split('-')
    year = int(year)
    if year < 10:
        year += 2000
    else:
        year += 1900
    return pd.to_datetime('{}-{}-{}'.format(day, month, year),
                          format="%d-%b-%Y")


def put_bmorph_ts(outfilename, ts, metadata=''):
    '''Write bias-corrected output to file

    Parameters
    ----------
    outfilename : str
        Pathname for output file
    ts : pandas.Series
        Series to be written to file
    metadata : str or None, optional
        Metadata that will be written at the start of the file. This is one
        long string with line breaks. Default value is ''

    Returns
    -------
    nothing
    '''
    buffer = metadata
    buffer += ts.to_csv()
    with open(outfilename, 'w') as f:
        f.write(buffer)
    return
