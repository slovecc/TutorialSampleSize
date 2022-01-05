# Introduction 

In statistical hypothesis testing, generally two hypothesis are formulated. If we have two variants, the null hypotesis is that the two distributions for the metric $M$ of variant A ($ M_A $) and variant B ($ M_B $ ) are the same. The sampling distribution (and the corresponding normal distribution)  obtained from the difference of the distribution for $ M_A $ and $ M_B $ would be centered on 0 and with a standard deviation given by the pooled standard deviation $ \sigma^{(p)}$ like in the next fig.:

```{figure} img/schema6.png
---
height: 300px
name: log-figure
---
```

From its normal distribution, we can find the $z$ critical value that corresponds to extreme events that could occour at a probability of $\alpha$. 


The alternative hypothesis ($ H_A $) if that there is an uplift of metric $ M_B $ with the respect to the metric $ M_A $ (for example a difference of $ \delta$ ). In this case, the normal distribution of the difference of the two metrics will be centered in the difference between the two metrics ( the value of the $ z$ would be $z_\delta$ ) and let's assume that it will have the same standard deviation of before.  

```{figure} img/schema7.png
---
height: 300px
name: log-figure
---
```

If we overlap the distributions for the two hypothesis ($H_0$ and $H_A$) we can see that (considering the case of the two tails hypotesis for $ H_0 $) the value corresponding to the $ \alpha/2 $ probability (the $ z^{*}_{critic} $ ) for $ H_0 $ will intersect the distribution of the alternative hypothesis ( $ H_A $ at the position of $ z_{\beta} $ ). The area of the $H_A$ to the left of $ z_{\beta} $ will be equal to the value $ \beta$. 


```{figure} img/schema1.png
---
height: 500px
name: log-figure
---
```

We don't know if the hypothesis correct is $H_0$ or $H_A$. What we can do is to have some measure and see which distribution of those measures, fits our hypothesis. 


Now let's consider all the scenarios that can arise from a measure.

### Scenario 1:
We reject the $ H_0 $. 

#### Case 1:

We do a measure and the corresponding $z$ is greater than the $ z^{*}_{critic} $, the decision would be to reject $ H_0 $. Imagine that we know the true distribution below and that in this case actually the null hypothesis is correct. By rejecting the $ H_0 $ we do a mistake: this would happen with a frequency given by the value of $ \alpha $. This source of error is the known $\textbf{ Error Type 1 }$.

```{figure} img/schema2.png
---
height: 500px
name: log-figure
---
```

#### Case 2:

Same scenario as before: We do a measure and the corresponding $z$ is greater than the $ z^{*}_{critic} $, the decision would be to reject $ H_0 $. But in this case, the null hypothesis is incorrect. Rejecting the $ H_0 $ is correct: this would happen with a frequency given by the value of $ 1 -\beta $. 

```{figure} img/schema3.png
---
height: 500px
name: log-figure
---
```



### Scenario 2:
We don't reject the $ H_0 $. 

#### Case 1:

We do a measure and the corresponding $z$ is lower than the $ z^{*}_{critic} $, the decision would be to not reject $ H_0 $. Imagine that we know the true distribution below and that in this case actually the null hypothesis is incorrect. By not rejecting the $ H_0 $ we do a mistake: this would happen with a frequency given by the value of $ \beta $. This source of error is the known $\textbf{ Error Type 2 }$.

```{figure} img/schema4.png
---
height: 500px
name: log-figure
---
```

#### Case 2:

Same scenario as before: We do a measure and the corresponding $z$ is lower than the $ z^{*}_{critic} $, the decision would be to not reject $ H_0 $ and actually the null hypothesis is correct.
```{figure} img/schema5.png
---
height: 500px
name: log-figure
---
```


### Summary
Below it is reported the summary of all the scenarios:

```{figure} img/summary.png
---
height: 500px
name: log-figure
---
```