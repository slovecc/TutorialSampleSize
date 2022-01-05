## import function
import numpy as np
import pandas as pd

from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind,norm, zscore

import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *

sig=0.05/4
power=0.8
z = norm.isf([sig/2])  
zp = -1 * norm.isf([power])

2*(z+zp)**2/15.698

20000*1.421


