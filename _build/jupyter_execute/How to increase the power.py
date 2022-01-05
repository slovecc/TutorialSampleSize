# Power Analysis

### Definition of power

From the previous section we have introduced the statistical power of a binary hypothesis test as the probability that the test correctly rejects the null hypothesis ($ H_{0} $ ) when a specific alternative hypothesis ($ H_{A} $ ) is true. 

### Can we increase the power?

It is clear that we would need a "good" power in the test in order to accept the $H_A$ hypthesis when it is true. 

In the nice article in [Ref 8](https://towardsdatascience.com/5-ways-to-increase-statistical-power-377c00dd0214) it is described clearly which are the factor that influence the power and can be modified in order to increase it. 


Here I will do a brief summary and show schematically some of those.


#### Decrease the significativity
If we increase the threshold for the significance level ($\alpha$) we artificially increase the power of the test. This is strongly not recommended since the error type 1 will increase. The current value suggested for the significance is $95\%$.

#### 1 tails vs 2 tails

If we move from 2 tails (figure left below) to 1 tail (figure right below), the vertical dashed line will be moved to the left increasing the power. This is because the critical p value for each tail is half of alpha, while the critical p equals alpha in a 1-tailed test.
Whether the test is with 1 tail or 2, should be decided at the design phase of the test.


```{figure} img/schema8.png
---
height: 400px
name: log-figure
---
```

#### Increase the difference 
Keeping constant all the other parameters, we could increase the power of the test if the mean difference (the distance between the null and alternative hypothesis) expected is increased.

```{figure} img/schema9.png
---
height: 400px
name: log-figure
---
```

#### Decrease the sigma or increase the sample
Another way to increase the power is to change the shape of the distribution. In particular we know that the sigma of the distribution is proportional to the standard deviation and inversely proportional to the sample size: 

$$\sigma^{(p)} \propto \frac{\sigma}{\sqrt{N}}.$$ 

In this context the way to change the sigma would be either to change the KPI and take into account another metric with different standard deviation, either increase the sample size which is actually the most accurate way to influence the power and is what we are going to inspect in the next sessions.

```{figure} img/schema10.png
---
height: 400px
name: log-figure
---
```