# EM for Gaussian Mixture Models (GMM)

This repository implements the Expectation–Maximization (EM) algorithm for GMMs on a 2D “Mickey Mouse” dataset. 


## Quick start
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --k 3 --max-iter 200 --tol 1e-4 --seed 42
