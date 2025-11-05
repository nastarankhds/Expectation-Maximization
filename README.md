# EM for Gaussian Mixture Models (GMM)

This repository implements the Expectation–Maximization (EM) algorithm for GMMs on a 2D “Mickey Mouse” dataset. 
If `data/mouse.txt` is present, it is used; otherwise a synthetic dataset with the same structure is generated.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --k 3 --max-iter 200 --tol 1e-4 --seed 42
