import bmorph
import numpy as np
from numpy import testing
import pandas as pd
import pytest


@pytest.fixture()
def truth_ts(scope='module'):
    return pd.Series(np.arange(31)*0.1,
                     index=pd.date_range('1/1/2000', '1/31/2000'))


@pytest.fixture()
def train_ts(scope='module'):
    return pd.Series(np.arange(31)*0.2,
                     index=pd.date_range('1/1/2000', '1/31/2000'))


@pytest.fixture()
def raw_ts(scope='module'):
    return pd.Series(np.arange(31)*0.4,
                     index=pd.date_range('1/1/2001', '1/31/2001'))


@pytest.fixture()
def edcdfm_expected_ts(scope='module'):
    return pd.Series(np.arange(31)*0.2,
                     index=pd.date_range('1/1/2001', '1/31/2001'))


def test_edcdfm(raw_ts, train_ts, truth_ts, edcdfm_expected_ts):
    edcdfm_multipliers = bmorph.edcdfm(raw_ts, raw_ts.sort_values(),
                                       train_ts.sort_values(),
                                       truth_ts.sort_values())
    edcdfm_actual_ts = edcdfm_multipliers * raw_ts
    testing.assert_allclose(edcdfm_actual_ts, edcdfm_expected_ts)
