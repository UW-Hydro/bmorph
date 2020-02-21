KACimport numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import bmorph

sns.set_context('talk')
mpl.style.use('seaborn-bright')
mpl.rcParams['figure.figsize'] = (12, 9)
CFS_TO_CMS = 35.314666212661

sites = ['KEE','KAC','CLE','YUMW','BUM', 'AMRW','CLFW','NACW','UMTW','AUGW','YGVW','YAK'] #list of gauge sites

#converting obs into a workable file and then indexing by datetime
obs = pd.read_csv('../data/observed.csv');
obs = obs.drop('A', axis=1)[6:]
index = pd.to_datetime(obs['Unnamed: 1'])
mask = index.dt.year > 2010
index[mask] = index[mask] - pd.Timedelta('100Y')
obs.index = index #pd.DatetimeIndex(index)
obs.index.name = 'Date'
obs = obs.drop('Unnamed: 1', axis=1)
obs = obs.astype('float64')
obs.columns = [o.split('_')[0].replace('5N', '') for o in obs.columns]
cols = []
obs = obs / CFS_TO_CMS
obs = obs.loc[:,['KEE','KAC','CLE','YUMW','BUM', 'AMRW','CLFW','NACW','UMTW','AUGW','YGVW','YAK']]

unc_sim = pd.read_csv('../data/uncorrected_simulation.csv'); #reading data into pandas
unc_sim.index = pd.to_datetime(unc_sim['time']) #indexing based on datetime data type
unc_sim = unc_sim.loc[:,['KEE','KAC','CLE','YUMW','BUM', 'AMRW','CLFW','NACW','UMTW','AUGW','YGVW','YAK']]

st_biasC = pd.read_csv('../data/standard_bias_corrected.csv');
st_biasC.index = pd.to_datetime(st_biasC['time'])
st_biasC = st_biasC.loc[:,['KEE','KAC','CLE','YUMW','BUM', 'AMRW','CLFW','NACW','UMTW','AUGW','YGVW','YAK']]

new_biasC = pd.read_csv('../data/new_bias_corrected.csv');
new_biasC.index = pd.to_datetime(new_biasC['time'])
new_biasC = new_biasC.loc[:,['KEE','KAC','CLE','YUMW','BUM', 'AMRW','CLFW','NACW','UMTW','AUGW','YGVW','YAK']]

obs = obs.loc['19550101':'19751231']
unc_sim = unc_sim.loc['19550101':'19751231']
st_biasC = st_biasC.loc['19550101':'19751231']
new_biasC = new_biasC.loc['19550101':'19751231']

pointlist = pd.read_csv('../data/crcc_pointlist.txt',delimiter=';')

weekEndP = pd.read_csv('../data/prec.csv');
weekEndP.index = pd.to_datetime(weekEndP['time'])
weekEndP = weekEndP.loc[:,['KEE','KAC','CLE','YUMW','BUM', 'AMRW','CLFW','NACW','UMTW','AUGW','YGVW','YAK']]
weekEndP = weekEndP.loc['19550101':'19751231']

weekAvgT = pd.read_csv('../data/temp.csv');
weekAvgT.index = pd.to_datetime(weekAvgT['time'])
weekAvgT = weekAvgT.loc[:,['KEE','KAC','CLE','YUMW','BUM', 'AMRW','CLFW','NACW','UMTW','AUGW','YGVW','YAK']]
weekAvgT = weekAvgT.loc['19550101':'19751231']