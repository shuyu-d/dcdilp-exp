# DCDILP: distributed learning for large-scale causal structure learning

This package implements DCDILP algorithms, which consists of the
following two consecutive steps:

    (Phase 1)     Markov blanket discovery 

    (Phase 2)     Local causal discovery, computed in parallel 

    (Phase 3)     Local graphs reconciliation by an ILP method 
            

## Requirements

- Python 3.6+
- `numpy`
- `scipy`
- `python-igraph`: Install [igraph C core](https://igraph.org/c/) and `pkg-config` first.
- `NOTEARS/utils.py` - graph simulation, data simulation, and accuracy evaluation from [Zheng et al. 2018]

#### Reference 

[1] S. Dong, M. Sebag, K. Uemura, A. Fujii, S. Chang, Y. Koyanagi, and K. Maruhashi. DCDILP: distributed learning for large-scale causal structure learning, 2024. 


## Running a demo

```bash
$ python submit_slurmjobs.py  
```


