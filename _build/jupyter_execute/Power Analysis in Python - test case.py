# Sensitivity analysis for the sample size

## Introduction

Let's recall from the previous session the two main formula for the sample size in the case of the independent two-sample (with equal sizes and variances). The version for the continous metrics:

$$n_i = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 }{ \big(\frac{\mu_0 -\mu_1}{\sigma}\big)^2 } $$

and for the fractional metrics: 

$$n_i = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 }{ \big(\frac{p_1 -p_2}{p_1 (1-p_1)}\big)^2 } $$


Remember that the denominator is called effect size. For the continuos metric:

$$ \textrm{effect size} = \frac{\mu_0 -\mu_1}{\sigma} $$


In this section, I will focus on the continous version of the formula (providing also the way to compute for the fractional one) and I will show some of the insights discussed in the previous section. 


First of all, let's implement the formula: in the python package of **statsmodels** some of the function for the power analysis have been implemented ([ref here](https://www.statsmodels.org/devel/stats.html#power-and-sample-size-calculations) ).

I will use that package in the following but in order to check if the results is exactly the one expected by implementing the formula above, let's compare a test case.

## import function
import numpy as np
import pandas as pd

from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind,norm, zscore

import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()

#Ex using the stats package
mu=4.61
perc_increase = 0.02
p = 0.8
sigma=9.25
sig=0.05
diff_mu = (1+perc_increase)*mu - mu
size = diff_mu/sigma

n=TTestIndPower().solve_power(effect_size = size, 
                                             power = p, 
                                             alpha = sig)

print("n from the package: "+str(round(n)))

def sample_power_continous(mu, sigma, perc_increase, power=0.8, sig=0.05):
    ### raw implementation of sample size formula for continuous metric
    z = norm.isf([sig/2])  
    zp = -1 * norm.isf([power]) 

    diff_mu = (1+perc_increase)*mu - mu
    size = diff_mu/sigma
    n = 2 * ((zp + z)**2) / (size**2)
    return int(round(n[0]))

def sample_power_fractional(p1, p2, power=0.8, sig=0.05):
    ## raw implementation of sample size formula  for fractional metrics
    z = norm.isf([sig/2])  
    zp = -1 * norm.isf([power]) 
    d = (p1-p2)
    s =2*((p1+p2) /2)*(1-((p1+p2) /2))
    n = s * ((zp + z)**2) / (d**2)
    return int(round(n[0]))

mu=4.61
perc_increase = 0.02
p = 0.8
sigma=9.25
signific=0.05

n = sample_power_continous(mu, sigma, perc_increase, power=p, sig=signific)
print("n from the formula directly: "+str(n))

# Power analysis in python (adapted from [ref 5])


In this example, I carry out power analysis. Let’s start with an easy example by assuming that we would like to know how big a sample we need to collect for our experiment, if we accept power at the level of 80%, the significance level of 5% and the expected effect size is 0.8.

Then, we need to run the following commands and arrive at the required sample size of 25.

# parameters for the analysis 
effect_size = 0.8
alpha = 0.05 # significance level
power = 0.8

power_analysis = TTestIndPower()
sample_size = power_analysis.solve_power(effect_size = effect_size, 
                                         power = power, 
                                         alpha = alpha)

print('Required sample size: {0:.2f}'.format(sample_size))

Having done that, it is time to take it a step further. 

We would like to see how does the power change when we modify the rest of the building blocks. To do so we plot power with respect to the other parameters.

I begin the analysis by inspecting how does the sample size influence the power (while keeping the significance level and the effect size at certain levels). I have chosen [0.2, 0.5, 0.8] as the considered effect size values, as these correspond to the thresholds for small/medium/large, as defined in the case of Cohen’s d.

### Power vs Numb. of observation (given signific. and effect size)

# power vs. number of observations 

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
fig = TTestIndPower().plot_power(dep_var='nobs',
                                 nobs= np.arange(2, 200),
                                 effect_size=np.array([0.2, 0.5, 0.8]),
                                 alpha=0.01,
                                 ax=ax, title='Power of t-Test' + '\n' + r'$\alpha = 0.01$')
ax.get_legend().remove()
ax = fig.add_subplot(1,2,2)
fig = TTestIndPower().plot_power(dep_var='nobs',
                                 nobs= np.arange(2, 200),
                                 effect_size=np.array([0.2, 0.5, 0.8]),
                                 alpha=0.05,
                                 ax=ax, title=r'$\alpha = 0.05$') 
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.subplots_adjust(right = 2)

From the plots, we can infer that an increase in the sample/effect size leads to an increase in power. In other words, the bigger the sample, the higher the power, keeping other parameters constant. 


Below I also present the plots for two remaining building blocks on the x-axis and the results are pretty self-explanatory.

### Power vs effect size (given signific. and numb. of observations)

# power vs. effect size 

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
fig = TTestIndPower().plot_power(dep_var='effect_size',
                                 nobs= np.array([10,20,50,70,100,200]),
                                 effect_size=np.arange(0, 1, 0.001),
                                 alpha=0.01,
                                 ax=ax, title='Power of t-Test' + '\n' + r'$\alpha = 0.01$')
ax.get_legend().remove()
ax = fig.add_subplot(1,2,2)
fig = TTestIndPower().plot_power(dep_var='effect_size',
                                 nobs= np.array([10,20,50,70,100,200]),
                                 effect_size=np.arange(0, 1, 0.001),
                                 alpha=0.05,
                                 ax=ax, title=r'$\alpha = 0.05$')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.subplots_adjust(right = 2)

### Power vs signific. (given effect size  and numb. of observations)

# power vs. alpha

fig = plt.figure()
ax = fig.add_subplot(1,3,1)
fig = TTestIndPower().plot_power(dep_var='alpha',
                                 nobs= np.array([50,100,500]),
                                 effect_size=0.2,
                                 alpha=np.arange(0.01, 0.1, 0.001),
                                 ax=ax, title='Power of t-Test' + '\n' + r'es $ = 0.2$')
ax.get_legend().remove()
ax = fig.add_subplot(1,3,2)
fig = TTestIndPower().plot_power(dep_var='alpha',
                                 nobs= np.array([50,100,500]),
                                 effect_size=0.5,
                                 alpha=np.arange(0.01, 0.1, 0.001),
                                 ax=ax, title='Power of t-Test' + '\n' + r'es $ = 0.5$')

ax.get_legend().remove()
ax = fig.add_subplot(1,3,3)
fig = TTestIndPower().plot_power(dep_var='alpha',
                                 nobs= np.array([50,100,500]),
                                 effect_size=0.8,
                                 alpha=np.arange(0.01, 0.1, 0.001),
                                 ax=ax, title='Power of t-Test' + '\n' + r'es $ = 0.8$')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.subplots_adjust(right = 3)

Finally, I would like to expand the analysis to three dimensions. To do so, I fix the significance level at 5% (which is often used in practice) and create a grid of possible combinations of the sample and effect sizes. Then I need to obtain the power values for each combination. To do this I use NumPy's meshgrid and vectorize.

# for this part I assume significance level of 0.05

@np.vectorize
def power_grid(x,y):
    power = TTestIndPower().solve_power(effect_size = x, 
                                        nobs1 = y, 
                                        alpha = 0.05)
    return power

X,Y = np.meshgrid(np.linspace(0.01, 1, 51), 
                  np.linspace(10, 1000, 100))
X = X.T
Y = Y.T

Z = power_grid(X, Y) # power

data = [Surface(x = X, y= Y, z = Z)]

layout = Layout(
    title='Power Analysis',
    scene = dict(xaxis = dict(title='effect size'),
                 yaxis = dict(title='number of observations'),
                 zaxis = dict(title='power'),)
)

fig = Figure(data=data, layout=layout)
iplot(fig,filename = 'figure_1.html')

## Sample size

Let's focus now on extracting the sample size for a generic case and after for some practical cases.

Initially let's understand the dependence of the sample size from the effect size defined above as: 

$$ \textrm{effect size} = \frac{\mu_0 -\mu_1}{\sigma} $$


effect_size=np.arange(0.001,0.5,0.002)
power=np.arange(0.7,0.9,0.05)
alpha=np.arange(0.01,0.08,0.01)

dn = []

for size in effect_size :
    for p in power : 
        for sig in alpha:
            n=TTestIndPower().solve_power(effect_size = size, 
                                             power = p, 
                                             alpha = sig)
            
            dn.append((size, p, sig,n))

pn=pd.DataFrame(dn, columns=('size', 'power', 'significance','sample'))
pn.power=pn.power.astype(np.float16)
    


powerr = np.array([0.7,0.8,0.9])
signn = np.array([0.01,0.02,0.05])

    
plt.figure(figsize=(8,7))     
sns.set(font_scale = 2)
plt.xlabel("effect size")

sns.lineplot(data=pn[(pn.power == 0.7) & (pn.significance == 0.05)], 
             x="size", y="sample",color='blue', linewidth = 2.5, marker='o',label='pow = 0.7, '+r'$\alpha=0.05$')

sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.05)], 
             x="size", y="sample",color='red', linewidth = 2.5, marker='o',label='pow = 0.8, '+r'$\alpha=0.05$')

sns.lineplot(data=pn[(pn.power == 0.85) & (pn.significance == 0.05)], 
             x="size", y="sample",color='green', linewidth = 2.5, marker='o',label='pow = 0.85, '+r'$\alpha=0.05$')


sns.lineplot(data=pn[(pn.power == 0.7) & (pn.significance == 0.01)], 
             x="size", y="sample",color='blue', linewidth = 2.5, marker='s',label='pow = 0.7, '+r'$\alpha=0.01$')

sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.01)], 
             x="size", y="sample",color='red', linewidth = 2.5, marker='s',label='pow = 0.8, '+r'$\alpha=0.01$')

sns.lineplot(data=pn[(pn.power == 0.85) & (pn.significance == 0.01)], 
             x="size", y="sample",color='green', linewidth = 2.5, marker='s',label='pow = 0.85, '+r'$\alpha=0.01$')

#plt.title('sign:0.01' )
plt.legend(fontsize='12')
plt.ylim(1.0, 100000000)
plt.xlim(0, 0.5)

plt.yscale("log")

plt.show()


### KPI 1

mu=0.0244
perc_increase = np.arange(0.005,0.05,0.0005)
sigma=0.158


power=np.arange(0.8,0.95,0.05)
alpha=np.arange(0.02,0.06,0.01)


dn = []

for increase in perc_increase :
    for p in power : 
        for sig in alpha:
            diff_mu = (1+increase)*mu - mu
            size = diff_mu/sigma
            n=TTestIndPower().solve_power(effect_size = size, 
                                             power = p, 
                                             alpha = sig)
            
            dn.append((increase,size, p, sig,n))

pn=pd.DataFrame(dn, columns=('increase','size', 'power', 'significance','sample'))
pn.head()

pn.power=pn.power.astype(np.float16)
pn.significance=pn.significance.astype(np.float16)

# pn.loc[(pn.power==0.9) & (pn.significance ==0.05)]

powerr = np.array([0.8,0.9])
signn = np.array([0.02,0.05])

plt.figure(figsize=(8,7)) 
sns.set(font_scale = 2)
plt.xlabel("lift [%] ")

pn.increase = pn.increase *100
sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.05)], 
             x="increase", y="sample",color='red', linewidth = 2.5, marker='o',label='pow = 0.8, '+r'$\alpha=0.05$')

sns.lineplot(data=pn[(pn.power == 0.9) & (pn.significance == 0.05)], 
             x="increase", y="sample",color='blue', linewidth = 2.5, marker='o',label='pow = 0.9, '+r'$\alpha=0.05$')

sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.02)], 
             x="increase", y="sample",color='red', linewidth = 2.5, marker='s',label='pow = 0.8, '+r'$\alpha=0.02$')


sns.lineplot(data=pn[(pn.power == 0.9) & (pn.significance == 0.02)], 
             x="increase", y="sample",color='blue', linewidth = 2.5, marker='s',label='pow = 0.9, '+r'$\alpha=0.02$')


plt.legend(fontsize='12')
plt.ylim(100000, 100000000)
#plt.xlim(0, 0.5)

plt.yscale("log")

plt.show()

dau_each_group = 17487/2.
 
pn['numb_week'] = (pn['sample']/ dau_each_group)/ 7.
    
    
    
powerr = np.array([0.8,0.9])
signn = np.array([0.02,0.05])

 
plt.figure(figsize=(8,7)) 
sns.set(font_scale = 2)
plt.xlabel("lift [%] ")


sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.05)], 
             x="increase", y="numb_week",color='red', linewidth = 2.5, marker='o',label='pow = 0.8, '+r'$\alpha=0.05$')

sns.lineplot(data=pn[(pn.power == 0.9) & (pn.significance == 0.05)], 
             x="increase", y="numb_week",color='blue', linewidth = 2.5, marker='o',label='pow = 0.9, '+r'$\alpha=0.05$')



sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.02)], 
             x="increase", y="numb_week",color='red', linewidth = 2.5, marker='s',label='pow = 0.8, '+r'$\alpha=0.02$')

sns.lineplot(data=pn[(pn.power == 0.9) & (pn.significance == 0.02)], 
             x="increase", y="numb_week",color='blue', linewidth = 2.5, marker='s',label='pow = 0.9, '+r'$\alpha=0.02$')

plt.legend(fontsize='12')
plt.ylim(0.1, 1000)
plt.axhline(y = 3, color = 'black', linestyle = '--')
plt.text(1, 4, "3 weeks")

plt.yscale("log")

plt.show()

### KPI 2

mu=29.16
perc_increase = np.arange(0.005,0.05,0.0005)
sigma=88.59


power=np.arange(0.8,0.95,0.05)
alpha=np.arange(0.02,0.06,0.01)


dn = []

for increase in perc_increase :
    for p in power : 
        for sig in alpha:
            diff_mu = (1+increase)*mu - mu
            size = diff_mu/sigma
            n=TTestIndPower().solve_power(effect_size = size, 
                                             power = p, 
                                             alpha = sig)
            
            dn.append((increase,size, p, sig,n))

pn=pd.DataFrame(dn, columns=('increase','size', 'power', 'significance','sample'))
pn.head()

pn.power=pn.power.astype(np.float16)
pn.significance=pn.significance.astype(np.float16)

# pn.loc[(pn.power==0.9) & (pn.significance ==0.05)]

dau_each_group = 17487/2.
 
pn['numb_week'] = (pn['sample']/ dau_each_group)/ 7.
    
    
    
powerr = np.array([0.8,0.9])
signn = np.array([0.02,0.05])

 
plt.figure(figsize=(8,7)) 
sns.set(font_scale = 2)
plt.xlabel("lift [%] ")
pn.increase = pn.increase *100


sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.05)], 
             x="increase", y="numb_week",color='red', linewidth = 2.5, marker='o',label='pow = 0.8, '+r'$\alpha=0.05$')

sns.lineplot(data=pn[(pn.power == 0.9) & (pn.significance == 0.05)], 
             x="increase", y="numb_week",color='blue', linewidth = 2.5, marker='o',label='pow = 0.9, '+r'$\alpha=0.05$')



sns.lineplot(data=pn[(pn.power == 0.8) & (pn.significance == 0.02)], 
             x="increase", y="numb_week",color='red', linewidth = 2.5, marker='s',label='pow = 0.8, '+r'$\alpha=0.02$')

sns.lineplot(data=pn[(pn.power == 0.9) & (pn.significance == 0.02)], 
             x="increase", y="numb_week",color='blue', linewidth = 2.5, marker='s',label='pow = 0.9, '+r'$\alpha=0.02$')

plt.legend(fontsize='12')
plt.ylim(0.1, 1000)
plt.axhline(y = 3, color = 'black', linestyle = '--')
plt.text(1, 4, "3 weeks")
plt.axhline(y = 2, color = 'black', linestyle = '--')
plt.text(1, 1, "2 weeks")

plt.yscale("log")

plt.show()

