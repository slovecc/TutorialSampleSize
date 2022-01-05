# Sample size

Before to jump into the mathematical formulation for the sample size derivation, the previous discussion about all the factor that influence the power can be applied also to the sample size : below a list of those components that we will discuss in the next section.


                                        
```{figure} img/sampleSizeDep.png
---
height: 300px
name: log-figure
---
```

It is evident that the sample size is highly sensitive to these parameters: having a control of how the change of one of those affects the sample size is of crucial importance to the determination of the size that would make the experiments meaningful.

## Formulation

Different formulations can be found in literature, some of those with strong assumption other with less clear assumptions and fundation. In any case, all those derivations conduct to a similar results and can be classified in two category: sample size for 1 sample (similar to 2 matched samples) and sample size for 2 independent samples.


Let's recap quickly what each of those two categories would cover: 

- 1 sample test: The one sample z test compares the mean of your sample data to a known value. For example, you might want to know how your sample mean compares to the population mean ($\mu_0$). For this test typically we compute the z statistics as :  $ z= \frac{\bar{x}- \mu_0}{\sigma / \sqrt{n}} $

where n is the size of the sample and $\sigma$ the estimated standard deviation of the population

<div class="alert alert-block alert-warning"> <b> Example: </b> A recent report indicated that 26% of people free of cardiovascular disease had elevated LDL cholesterol levels, defined as LDL > 159 mg/dL. An investigator hypothesizes that a higher proportion of patients with a history of cardiovascular disease will have elevated LDL cholesterol. How many patients should be studied to ensure that the power of the test is 90% to detect a 5% difference in the proportion with elevated LDL cholesterol? A two sided test will be used with a 5% level of significance.   
</div>



- 2 sample test: Suppose the two groups are A and B, and we collect a sample from both groups, i.e. we have two samples. We perform a two-sample test to determine whether the mean in group A, $\mu_A$ , is different from the mean in group B, $\mu_B$. The z statistics in this case is : $z= \frac{\mu_A- \mu_B}{\sigma_p / \sqrt{1/n_A + 1/n_B}}$ where the standard error in the most generical case is the pooled standard error $\sigma_p = \sqrt{\frac{(n_A-1) \sigma_A^2 +(n_B-1) \sigma_B^2}{n_A+n_B-2} }$.In the most common case in which $n_A=n_B=n_i$ and assuming that the two standard error $\sigma_A=\sigma_B=\sigma_i$ the previous z statistics reduces to $ z= \frac{\mu_A- \mu_B}{\sigma_i / \sqrt{2/n_i}} $

We will focus on the sample size derived from the 2 sample test since is the one we have to apply in case of AB test.

In **[REF 2]** there is an overview of some formulas applied in different context, in the following we will focus on the main one which seems to be generally adopted in the community and has been derived also in **[REF 1, 4]**.

In general the formulation is splitted in two main scenarios:
- continuous 
- proportion

<div class="alert alert-block alert-danger"> <b> Assumption: </b> The following formulation is obtained for the case of two sample with the same standard deviation and equal number of observations for the two distributions. In case those hypothesis are not satisfied the formulation can be generalised.  </div>


### Derivation of formula 

There are different ways to  derive the previous formula. One of the most concise and graphical way is proposed in  [Ref 1: ](http://www.vanbelle.org/chapters/webchapter2.pdf) 

```{figure} img/derivation.png
---
height: 300px
name: log-figure
---
```

The critical value definesthe boundary between the rejection and nonrejection
regions. This value must be the same under the null and alternative hypotheses. This
then leads to the fundamental equation for the two-sample situation:

$$ 0 + z_{1-\alpha/2} \sigma \sqrt{2/n} = \delta -z_{1-\beta}\sigma \sqrt{2/n}$$

solving it for $n$ will lead to the previous formula

### Continuos metrics
The formula for the sample size required to compare two population means, $\mu_0$ and $\mu_1$, with common variance, $\sigma^2$ and supposing an equal size $n_A=n_B=n_i$:

$$n_i = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 }{ \big(\frac{\mu_0 -\mu_1}{\sigma}\big)^2 } $$

where :
- $\mu_0 -\mu_1$ is the expected difference between the two treatments 
- $z_{1-\alpha/2}$, $ z_{1-\beta}$ are the Normal values for power and significance
- $\sigma$  is the estimated sample variance (this will be extracted from historical data for the control variant)

In the case ratio between the sample sizes of the two groups is : $k=n_A/n_B$:

$$n_i = \big(1+ \frac{1}{k}\big) \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 }{ \big(\frac{\mu_0 -\mu_1}{\sigma}\big)^2 } $$


### Proportional metrics
The equivalent formula for the proportional metric is

$$n_i = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 }{ \big(\frac{p_1 -p_2}{p_1 (1-p_1)}\big)^2 } $$


### Rule of thumb for the two-sample case
The recommended values are for $\alpha=0.05$ and $1-\beta = 0.8$ and this means that consequently $z_{1-\alpha/2}=1.96$ and $ z_{1-\beta}=0.84$. The numerator of both the continuos and proportional metrics would be $2(z_{1-\alpha/2} + z_{1-\beta})^2 =15.68$ which can be rounded up to 16 producing the simple rule of thumb:

$$n_i = \frac{16}{\Delta^2} $$

The $\Delta$ is called also effect size and in the case of the continous metric is equals to :

$$ \Delta = \frac{\mu_0 -\mu_1}{\sigma} = \frac{\delta}{\sigma} $$

and is the treatment difference to be detected in units of the standard deviation–the
standardized difference.