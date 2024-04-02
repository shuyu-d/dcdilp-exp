import numpy as np
from timeit import default_timer as timer
import os, sys
import re
from scipy.sparse import csc_matrix, save_npz, load_npz

import ges
from rpy2 import robjects               # The package rpy2 is required
from rpy2.robjects import numpy2ri
from dagma.linear import DagmaLinear    # The package dagma is required


"""
DCDILP: distributed learning for large-scale causal structure learning

This function is an implementation for Phase 1 of DCDILP, which consists of the
following two consecutive steps:

    (Phase 1)     Markov blanket discovery

    (Phase 2)     Local causal discovery, computed in parallel

    (Phase 3)     Local graphs reconciliation by an ILP method


Reference

[1] S. Dong, M. Sebag, K. Uemura, A. Fujii, S. Chang, Y. Koyanagi, and K. Maruhashi. DCDILP: distributed learning for large-scale causal structure learning, 2024.
"""


""" Phase 2 solver using pcalg's GES
Installation note:
    (todo) : to add
"""
def pcalg_ges(Xobs):
    robjects.r('library("pcalg")')
    _ = robjects.r.assign("X", numpy2ri.py2rpy(Xobs))
    robjects.r('p = ges(new("GaussL0penObsScore", X), phase = c("forward", "backward", "turning") )')
    W_est = numpy2ri.rpy2py(robjects.r('as(as(p$essgraph, "graphNEL"), "matrix")'))
    return W_est


""" Phase 2 solver using DAGMA
"""
def dagma_linear(Xobs, lambda1=0.02, loss_type='l2'):
    model = DagmaLinear(loss_type='l2') # create a linear model with least squares loss
    W_est = model.fit(Xobs, lambda1=lambda1) # fit the model with L1 reg. (coeff. 0.02)
    return W_est

"""Other functions
"""
def embedding_local2global(w_loc, VAR, Si, d):
    W_glo = np.zeros([d,d])
    iloc =  Si.index(VAR)
    W_glo[VAR, Si] = w_loc[iloc,:]
    W_glo[Si, VAR] = w_loc[:,iloc]
    return W_glo


def get_MB_from_adjmatrix(Theta, VAR):
    # Input Theta is a sparse matrix
    row = (Theta[VAR,:]).toarray()
    Mi = ( (Theta[VAR,:]).toarray() != 0)
    return np.where(Mi)[-1]


def input_args(args=None):
    if args==None:
        args = {}
    for a in sys.argv:
        if "=" in a:
            p_k, p_v = a.split("=")
            p_v = p_v.split(",")
            if p_k == "ds":
                args[p_k] = [int(_) for _ in p_v]
            elif p_k in ['degs', 'ice_lambda1']:
                args[p_k] = [float(_) for _ in p_v]
            elif p_k in ['rnd', 'LAM' ]:
                args[p_k] = float(p_v[0])
            elif p_k in ['SEED', 'VAR', 'verbo']:
                args[p_k] = int(p_v[0])
            elif p_k in ['graph_type','sem_type','fdir', 'fout', 'filename', 'rowname']:
                args[p_k] = p_v[0]
            elif p_k in ['filename']:
                args[p_k] = f"{p_v[0]}.csv"
            else:
                print("Unknown parameter:", p_k)
                pass
        elif a in ['ges', 'pcalg-GES', 'temp', \
                   'DC-GES', 'DCpool-GES2', 'DC-GES2', \
                   'DC-DAGMA', 'DCpool-DAGMA']:
            args['runwho'] = [a]
        elif a == 'all':
            args['runwho'] = ['all']
    return args

if __name__ == '__main__':
    # timestr = time.strftime("%H%M%S%m%d")
    # args={'ds': [50], \
    #      'degs': [1.0, 2.0, 3.0], \
    #     'graph_type': 'ER' , 'sem_type': 'gauss', 'SEED':0,\
    #     'rnd': 10, \
    #     'runwho': [], 'toplot': 0, 'filename':'', \
    #     'LAM': 0.1, \
    #     'ipara': 0, 'fdir':'', 'fout':'', \
    #     'VAR': 0, 'verbo': 2, \
    #     }
    # args = input_args(args)
    args = input_args()

    fdir = args['fdir']
    fout = args['fout']
    filename = args['filename']

    d = args['ds'][0]
    deg = args['degs'][0]

    runwho = args['runwho']
    lambda1 = args['LAM']
    VAR = args['VAR']


    # Load data and MB results
    dir_out = '%s/%s'%(fdir,fout)

    X = np.load('%s/datamat.npy'%dir_out)
    #   PC set
    W_true = load_npz('%s/Wtrue.npz'%dir_out).toarray()
    Theta = load_npz(filename) # filename of the MB estimation data structure
    Pt = np.where(W_true[:,VAR])[0]
    Ct = np.where(W_true[VAR,:])[0]

    print('Number of variables: ', X.shape[1])
    print('Number of samples: ', X.shape[0])

    # Get the MB of variable VAR
    Mii = get_MB_from_adjmatrix(Theta, VAR) # MB of variable VAR

    # if ('GES' in args['runwho']) or ('DC-GES' in args['runwho']):
    if ('DC-GES' in args['runwho']):
        tm = timer()
        # Pick one Markov blanket
        print('Markov blanket (MB) of X_%d is: ' %VAR, Mii, ' // |MB|=%d' %len(Mii))
        tm = timer() - tm

        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        Si = list(Mii)
        print('Central variable: %d | Local nodes (Si) are: '%VAR, Si)
        if len(Si) > 1:
            t0 = timer()
            iloc = Si.index(VAR)
            Xobs = X[:,Si]
            w_ges, _ = ges.fit_bic(Xobs)
            t_ges = timer() - t0

            print('PC set: ')
            ic = np.where(w_ges[iloc,:])
            ip = np.where(w_ges[:,iloc])
            print('------>Parent nodes: ', list(np.array(Si)[ip]) )
            print('(True) Parent nodes: ', list(Pt), '\n')

            print('------>Children nodes: ', list(np.array(Si)[ic]) )
            print('(True) Children nodes: ', list(Ct) )

            # -- Report local structure learning result
            w_x = np.zeros(w_ges.shape)
            w_x[iloc,:] = w_ges[iloc,:]
            w_x[:,iloc] = w_ges[:,iloc]

            # Convert local result to the global embedding (dxd adjacency matrix)
            Wx_embed = embedding_local2global(w_x, VAR, Si, W_true.shape[0])
            # Save to npz files
            save_npz('%s/locres_VAR%d.npz'%(dir_out,VAR), csc_matrix(Wx_embed) )

    # if ('pcalg-GES' in args['runwho']) or ('DC-GES2' in args['runwho']) \
    #            or ('DCpool-GES2' in args['runwho']):
    if ('DC-GES2' in args['runwho']) or ('DCpool-GES2' in args['runwho']):
        tm = timer()
        # Pick one Markov blanket
        print('Markov blanket (MB) of X_%d is: ' %VAR, Mii, ' // |MB|=%d' %len(Mii))
        tm = timer() - tm

        # ff = '%s/%s'%(fdir, fout)
        # ff= dir_out
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        Si = list(Mii)
        print('Central variable: %d | Local nodes (Si) are: '%VAR, Si)
        if len(Si) > 1:
            t0 = timer()
            iloc = Si.index(VAR)
            Xobs = X[:,Si]
            w_ges  = pcalg_ges(Xobs)
            t_ges = timer() - t0

            print('PC set: ')
            ic = np.where(w_ges[iloc,:])
            ip = np.where(w_ges[:,iloc])
            print('------>Parent nodes: ', list(np.array(Si)[ip]) )
            print('(True) Parent nodes: ', list(Pt), '\n')

            print('------>Children nodes: ', list(np.array(Si)[ic]) )
            print('(True) Children nodes: ', list(Ct) )

            # -- Report local structure learning result
            w_x = np.zeros(w_ges.shape)
            w_x[iloc,:] = w_ges[iloc,:]
            w_x[:,iloc] = w_ges[:,iloc]

            # Conversion for the global embedding (dxd adjacency matrix) of the local result
            Wx_embed = embedding_local2global(w_x, VAR, Si, W_true.shape[0])
            # Save to npz files
            save_npz('%s/locres_VAR%d.npz'%(dir_out,VAR), csc_matrix(Wx_embed) )

    if ('DC-DAGMA' in args['runwho']) or ('DCpool-DAGMA' in args['runwho']):
        tm = timer()
        # Pick one Markov blanket
        print('Markov blanket (MB) of X_%d is: ' %VAR, Mii, ' // |MB|=%d' %len(Mii))
        tm = timer() - tm

        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        Si = list(Mii)
        print('Central variable: %d | Local nodes (Si) are: '%VAR, Si)
        if len(Si) > 1:
            t0 = timer()
            iloc = Si.index(VAR)
            Xobs = X[:,Si]
            w_loc  = dagma_linear(Xobs, lambda1=lambda1, loss_type='l2')
            t_loc = timer() - t0

            print('PC set: ')
            ic = np.where(w_loc[iloc,:])
            ip = np.where(w_loc[:,iloc])
            print('------>Parent nodes: ', list(np.array(Si)[ip]) )
            print('(True) Parent nodes: ', list(Pt), '\n')

            print('------>Children nodes: ', list(np.array(Si)[ic]) )
            print('(True) Children nodes: ', list(Ct) )

            # -- Report local structure learning result
            w_x = np.zeros(w_loc.shape)
            w_x[iloc,:] = w_loc[iloc,:]
            w_x[:,iloc] = w_loc[:,iloc]

            # Conversion for the global embedding (dxd adjacency matrix) of the local result
            Wx_embed = embedding_local2global(w_x, VAR, Si, W_true.shape[0])
            # Save to npz files
            save_npz('%s/locres_VAR%d.npz'%(dir_out,VAR), csc_matrix(Wx_embed) )

