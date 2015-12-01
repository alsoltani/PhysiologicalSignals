## Applied Statistics Project, ENSAE 2013.

*Alain Soltani, Loic Tudela, Adil Salim, Laurent Lambert*

## Denoising physiological signals using greedy methods

_Note :_ Summary and complete report are available on GitHub.

### Abstract

One recurrent problem in physiological signal acquisition is the presence of additive noise, inherent to the experiment ; subject movements, devices whiffs all deteriorate the signal we intend to study. Our aim here is to develop efficient denoising algorithms, to extract clear signals from noisy electro-encephalographic, magneto-encephalographic measurements.

We use in this study a famous signal processing algorithm : the Matching Pursuit algorithm, based on iterative research of highest correlations between the original signal and a dictionary of explanatory functions ; retained projections overlap to form a signal properly describing the original data. 
We present a procedure for explanatory variables' selection - and particularly its stopping criterion - by drawing comparisons between our iterative algorithm and elementary regression models. We test the robustness of our model by various means - by ensuring that extracted noise is consistent with the assumptions of normality issued, and that our algorithm resists to simulated noise additions.

Various improvements are then detailed ; from additional orthogonal projections on the selected atoms (Orthogonal Matching Pursuit) to dictionary orthonormalization, we present successively the effects on our results in statistical and computational terms. Similitudes with classical signal processing tools, such as hard thresholding, are drawn.
We finish by extending our work to the case of multichannel acquisitions, and developing a method for selecting the best dictionary.