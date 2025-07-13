# Global Optimization with a Power-Transformed Objective and Gaussian Smoothing
**Author:** Chen Xu

## Abstract
Abstract: We propose a novel method, namely Gaussian Smoothing with a Power-Transformed Objective (GS-PowerOpt), that solves global optimization problems in two steps: (1) perform a (exponential) power- $N$
transformation to the not necessarily differentiable objective $f:\mathbb{R}^d\rightarrow \mathbb{R}$ and get $f_N$, and (2) optimize the Gaussian-smoothed 
$f_N$ with stochastic approximations. Under mild conditions on $f$, for any $\delta>0$, we prove that with a sufficiently large power $N_\delta$, this method converges to a solution in the 
$\delta$-neighborhood of $f$'s global optimum point, at the iteration complexity of $O(d^{4}\epsilon^{-2})$. If we require that $f$ is differentiable and further assume the Lipschitz condition on 
$f$ and its gradient, the iteration complexity reduces to $O(d^2\epsilon^{-2})$, which is significantly faster than the standard homotopy method. In most of the experiments performed, our method produces better solutions than other algorithms that also apply the smoothing technique.

## Paper
You can find the full paper at https://openreview.net/pdf?id=6ojzpDczIY .

## Code
All the codes for the paper can be found in this repo.
