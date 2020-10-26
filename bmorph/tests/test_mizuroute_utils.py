import pytest

import numpy as np
import pandas as pd
import xarray as xr

import bmorph
from bmorph.util import mizuroute_utils as mizutil

reference = xr.open_dataset("./bmorph/tests/data/test_reference.nc")
routed = xr.open_dataset("./bmorph/tests/data/test_routed.nc")
topo = xr.open_dataset("./bmorph/tests/data/test_topo.nc")
true_fill = xr.open_dataset("./bmorph/tests/data/true_fill_segs.nc")
true_results = xr.open_dataset("./bmorph/tests/data/true_results.nc")

test_fill_methods = ['kge', 'kldiv', 'r2', 'leave_null']

gauge_flows = xr.Dataset(
    {
        'reference_flow' : (('seg', 'time'), reference['reference_flow'].transpose().values)
    },
    {"seg": reference['seg'].values, "time": reference['time'].values},
)

def test_map_headwater_sites(routed=routed.copy()):
    routed['down_seg'] = true_results['down_seg']
    test_routed = mizutil.map_headwater_sites(routed)
    assert 'is_headwaters' in test_routed.var()
    for truth, test in zip(true_results['is_headwaters'].values, test_routed['is_headwaters']):
        assert truth == test
        
def test_find_up(routed=routed.copy()):
    test_routed = routed
    test_routed['down_seg'] = true_results['down_seg']
    test_routed['is_headwaters'] = true_results['is_headwaters']
    for seg, true_up_seg in zip(test_routed['seg'].values, true_results['up_seg'].values):
        test_up_seg = mizutil.find_up(test_routed, seg)
        if np.isnan(true_up_seg):
            assert np.isnan(test_up_seg)
        else:
            assert true_up_seg == test_up_seg
        
def test_find_max_r2(routed=routed.copy()):
    true_r2_fill = true_fill.sel(fill_method='r2')['true_seg']
    for true_fill_seg, test_flow in zip(true_r2_fill.values, routed['flow'].values):
        test_fill_seg = mizutil.find_max_r2(gauge_flows, test_flow)[1]
        assert true_fill_seg == test_fill_seg
        
def test_find_max_kge(routed=routed.copy()):
    true_kge_fill = true_fill.sel(fill_method='kge')['true_seg']
    for true_fill_seg, test_flow in zip(true_kge_fill.values, routed['flow'].values):
        test_fill_seg = mizutil.find_max_kge(gauge_flows, test_flow)[1]
        assert true_fill_seg == test_fill_seg
        
def test_find_min_kldiv(routed=routed.copy()):
    true_kldiv_fill = true_fill.sel(fill_method='kldiv')['true_seg']
    for true_fill_seg, test_flow in zip(true_kldiv_fill.values, routed['flow'].values):
        test_fill_seg = mizutil.find_min_kldiv(gauge_flows, test_flow)[1]
        assert true_fill_seg == test_fill_seg
        
def test_map_ref_sites(routed=routed.copy(), fill_methods=test_fill_methods):
    test_routed = routed
    test_routed['down_seg'] = true_results['down_seg']
    test_routed['is_headwaters'] = true_results['is_headwaters']
    for fill_method in fill_methods:
        test_routed = mizutil.map_ref_sites(routed=test_routed, gauge_reference=reference,
                                            route_var = 'flow', fill_method = fill_method
                                           )
        for true_up_ref_seg, test_up_ref_seg in zip(true_fill.sel(fill_method=f"{fill_method}_up")['true_seg'].values, 
                                                    test_routed['up_ref_seg'].values):
            assert true_up_ref_seg == test_up_ref_seg
        for true_down_ref_seg, test_down_ref_seg in zip(true_fill.sel(fill_method=f"{fill_method}_down")['true_seg'].values, 
                                                        test_routed['down_ref_seg'].values):
            assert true_down_ref_seg == test_down_ref_seg
