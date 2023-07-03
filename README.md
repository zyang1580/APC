# APC

This is an implementation for our SIGIR2023 short paper "Prediction then Correction: An Abductive Prediction Correction Method for Sequential Recommendation" based on PyTorch (Version: 1.12).

## Instruction

- PC_amazon_cut_N_v2.py: the proposed APC method.
- main_sasrec_zy_search_amazon.py/main_sasrec_zy_amazon.py: running code for sasrec model with or wihtout grid search. We implement the grid search using "[Ray Tune](https://docs.ray.io/en/latest/tune/index.html)" (Version: 1.13). To implement SASRec, we use the [code](https://github.com/pmixer/SASRec.pytorch) suggested by the authors, and you can switch the hyperparameter ``` mode ``` $\in$ ```[forward, backward ] ```  to control its running manner, \ie reverse or not reverse.
- baselines: different versions of DTEC (user-side version)
