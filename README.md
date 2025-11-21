# EM for Gaussian Mixture Models (GMM)

This repository implements the Expectation–Maximization (EM) algorithm for GMMs on a 2D “Mickey Mouse” dataset. 


## Quick start
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --k 3 --max-iter 200 --tol 1e-4 --seed 42


### 1. Gaussian Mixture Model
The probability density function is defined as a weighted sum of $K$ Gaussian components:

$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$

Where:
- $\pi_k$: Weight of component $k$ (must sum to 1)
- $\mu_k$: Mean vector of component $k$
- $\Sigma_k$: Covariance matrix of component $k$
