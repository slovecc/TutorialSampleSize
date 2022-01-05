# Advanced Topic not covered

This tutorial is far from being exhaustive. Below a first list of topics that are not covered:

- $\textbf{Correction for ABn test}$  ( Ref 9 : chapter 6 [book: Statistical methods by G.Z.Georgiev](https://www.abtestingstats.com/))
In the case of ABn test, the sample size computation have to be adjusted in accordance to the strategy used to correct the pvalue. In fact, with more than 1 variant, the Type 1 increases and the type 1 error rate is called family-wise error rate (FWER): the  probability of falsely rejecting any of the pairwise null hypothesis if all are in fact true. 
There are different strategy to take into account this. Just a brief mention (from the most simple and conservative to the most accurate and advanced):

    - Bonferroni correction : $ \alpha_{adj}=\alpha/m$ where m is the number of variants
    - Holm-Bonferroni step-down methos 
    - Dunnet's correction: The Dunnet's correction is the most powerful one but it can involve some computational expensive simulation to be computed
    
- $\textbf{Impact of returning users in the sample size final computation}$

- $\textbf{"Conjoint" Sample size for more than 1 variable}$ [[nice article]](https://towardsdatascience.com/the-third-ghost-of-experimentation-multiple-comparisons-65af360169a1)