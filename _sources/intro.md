# Welcome to Sample Size tutorial 

In this tutorial we will give some overview of the sample size definition and define some guideline for the computation of the same in the experiments.

The reasoning to deep dive on this topic is because in general to run AB test we rely on different online calculators which rarely gave the same results and report the formula and hypothesis assumed.

We will see how the sample size computation is highly sensible and dependent on different parameters and knowing how to model it is of crucial importance to avoid a wrong setup in the experiment.

Furthermore an preliminary offline evaluation of what would be the sample size needed to reach a statistical significance level can drive also to the choise of the main metric!

<!-- #region -->
### Example of online calculator
Let's imagine the following scenario: we want to compute the sample size for an experiment with a continuos variable for which

- the mean baseline of the kpi is 4.61
- the standard deviation is 9.25
- we would like to see the effect of a +2% of the mean (the mean of the test would be 4.7022, or equivalently the difference of the mean is 0,0922)

We access to some online calculators: example

1) [calculator 1 ](https://select-statistics.co.uk/calculators/sample-size-calculator-two-means/) -> results $n=17082$

Let's double check with another calculator 

2) [calculator 2 ](http://powerandsamplesize.com/Calculators/Compare-2-Means/2-Sample-Equality) -> results $n=157677$


What is happening here?

What is the formula and the hypothesis behind? 

Can we control and implement by ourself the formula and trust it?
<!-- #endregion -->

### Literature
Some references used in the following tutorial:

[Ref 1: Chapt 2 on Sample size by prof. van Belle: BOOK Statistical Rules of Thumb  ](http://www.vanbelle.org/chapters/webchapter2.pdf)

[Ref 2: Quick overview of some formulation ](https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_power/bs704_power_print.html)

[Ref 3: Fundamental Sample size ](https://www.researchgate.net/publication/303312866_Fundamentals_of_Estimating_sample_size)

[Ref 4: Quick derivation of the main formula ](https://www.youtube.com/watch?v=JEAsoUrX6KQ)

[Ref 5: Nice article from medium about power analysis - reproduced below ](https://towardsdatascience.com/introduction-to-power-analysis-in-python-e7b748dfa26)

[Ref 6: Some exercize with the different formulas ](https://www.statstutor.ac.uk/resources/uploaded/stcp-rothwell-samplesize.pdf)

[Ref 7: Power analysis made easy](https://towardsdatascience.com/power-analysis-made-easy-dfee1eb813a)

[Ref 8: 5 ways to Increase Statistical Power](https://towardsdatascience.com/5-ways-to-increase-statistical-power-377c00dd0214)

[Ref 9 Book: Statistical methods by G.Z.Georgiev](https://www.abtestingstats.com/)

#### Example of online calculators

[Calc 1: we currently use this calculator](http://powerandsamplesize.com/Calculators/Test-1-Mean/1-Sample-1-Sided) 

[Calc 2: difference between mean](https://select-statistics.co.uk/calculators/sample-size-calculator-two-means/)

[Calc 3: Evan Miller Calculator](https://www.evanmiller.org/ab-testing/sample-size.html#!20;90;5;0.1;0)


```python

```
