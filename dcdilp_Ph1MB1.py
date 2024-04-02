import numpy as np
from timeit import default_timer as timer
import os, sys, csv
import pandas as pd
from scipy.sparse import csc_matrix, save_npz

from utils import utils
from utils.gen_settings import gen_data_sem_original

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

def ice_sparse_empirical(X, lams = np.linspace(4e-2,1e-1,10)):
    def _comp_ice_stats(prec, cov):
        error_emp = np.linalg.norm(emp_cov - cov, ord="fro")
        nnz = (prec!=0).sum()
        return error_emp, nnz
    def _criterion_trace(prec):
        prec_ = prec + np.diag(9e-1*np.diag(prec))
        f = (prec * emp_cov).sum() - np.log(np.linalg.det(prec_))
        # f = (prec * emp_cov).sum()
        return f
    def _selection(err_emps, fs, nnzs):
        N = len(err_emps) //2
        c1 = np.argsort(err_emps.values)[:N]
        c2 = np.argsort(fs.values)[:N]
        # Find the most sparse one among c2
        nnzvals = nnzs.values[c2]
        isel = np.argsort(nnzs.values[c2])[N//2]
        sel = c2[isel]
        return c1,c2, nnzvals, sel

    def _comp_sparse_ic(emp_cov, lambda_1):
        # Sparsify empirical precision matrix
        prec_ = np.linalg.inv(emp_cov)
        prec_off = prec_.copy()
        prec_off = prec_off - np.diag(np.diag(prec_off))
        cmax = max(abs(prec_off.ravel()))
        #
        prec_off[abs(prec_off) < lambda_1 * cmax] = 0
        #
        prec_sp = prec_off + np.diag(np.diag(prec_))
        cov_o = np.linalg.inv(prec_sp)
        return prec_sp, cov_o
    n_samples, d = X.shape
    X = X - np.mean(X, axis=0, keepdims=True)
    emp_cov = np.dot(X.T, X) / n_samples
    j = 0
    results = []
    precs =[]
    for lam in lams:
        j +=1
        name = "Sparse Empirical %d" % j
        start_time = timer()
        prec, cov = _comp_sparse_ic(emp_cov, lam)
        ctime = timer() - start_time
        err_emp, nnz = _comp_ice_stats(prec, cov)
        f = _criterion_trace(prec)
        precs.append(prec.ravel())
        results.append([name, err_emp, f, nnz, ctime, lam])

    _res=pd.DataFrame(results,
                 columns=[
                    "Estimator",
                    "Error (w Emp Cov)",
                    "Fit function",
                    "nnz (Prec)",
                    "Time",
                    "Lambda"]
                 )
    c1,c2,c3,sel = _selection(
            _res['Error (w Emp Cov)'],\
            _res['Fit function'], \
            _res['nnz (Prec)']
            )
    pd.set_option('display.max_columns', None)
    print("=======Lambda selected vs Argmin SuppDiff(wrt true Prec)=====")
    print(_res.iloc[sel])
    print("============\n")
    return precs[sel].reshape([d,d]), _res['Lambda'][sel], _res, sel, _res['Time'].values.sum()

def _get_undir_graph(Ci):
    # Encode edges (nonzeros) by -1 for
    # utils.count_accuracy to recognize undirected graphs
    C = Ci.copy()
    C -= np.diag(np.diag(C))
    C[C!=0] = -1
    return np.tril(C)

def _threshold_hard(B, tol=5e-2):
    B[abs(B)< tol] = 0
    return B
def _loss_glasso(prec, emp_cov):
    prec_ = prec + np.diag(9e-2*np.diag(prec))
    # prec_ = prec
    f = (prec * emp_cov).sum() - np.log(np.linalg.det(prec_))
    # f = (prec * emp_cov).sum()
    return f

def _comp_invcov(B, Ndiag=None):
    d = B.shape[0]
    if Ndiag == None:
        Ndiag = np.ones(d)
    return ((np.eye(d)-B) @ np.diag(Ndiag)) @ (np.eye(d)-B).T

def _MBs_fromInvCov(Theta):
    MBs = []
    for i in range(Theta.shape[0]):
        MBs.append( (Theta[i,:] != 0)  )
    return MBs

def input_args(args=None):
    if args==None:
        args = {}
    for a in sys.argv:
        if "=" in a:
            p_k, p_v = a.split("=")
            p_v = p_v.split(",")
            if p_k == "ds":
                args[p_k] = [int(_) for _ in p_v]
            elif p_k in ['degs', 'ice_lambda1', 'opts']:
                args[p_k] = [float(_) for _ in p_v]
            elif p_k in ['rnd', 'LAM' ]:
                args[p_k] = float(p_v[0])
            elif p_k in ['SEED', 'verbo']:
                args[p_k] = int(p_v[0])
            elif p_k in ['graph_type','sem_type','fdir', 'fout', 'filename', \
                            'solver_primal', 'rowname']:
                args[p_k] = p_v[0]
            elif p_k == "filename":
                args[p_k] = f"{p_v[0]}.csv"
            else:
                print("Unknown parameter:", p_k)
                pass
        elif a in ['oracle', 'ice-emp']:
            args['runwho'] = [a]
        elif a == 'all':
            args['runwho'] = ['all']
    return args

if __name__ == '__main__':
    args = input_args()

    fdir = args['fdir']
    fout = args['fout']
    filename = args['filename']
    verbo = args['verbo']

    graph_type = args['graph_type']
    sem_type = args['sem_type']
    SEED = args['SEED']
    d = args['ds'][0]
    deg = args['degs'][0]
    rnd = args['rnd']

    runwho = args['runwho']
    opts = args['opts'] # an array of numbers, default form: 0.1,0.3,10.0

    fout = args['fout']
    dir_out = '%s/%s'%(fdir,fout)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # Generate random graph and observational data
    n = int(np.ceil(rnd * d))

    W_true, X = gen_data_sem_original(d=d, deg=deg, n=n, \
                             graph_type=graph_type, \
                             sem_type=sem_type, \
                             seed=SEED) #
    print('Number of variables: ', d)
    print('Number of samples: ', n)
    print('Data matrix X size: ', X.shape)

    Theta_ev = _comp_invcov(W_true)
    Gref = _get_undir_graph(Theta_ev)
    # cov
    n_samples = X.shape[0]
    X = X - np.mean(X, axis=0, keepdims=True)
    emp_cov = np.dot(X.T, X) / n_samples

    te = 0
    if 'oracle' in args['runwho']:
        Theta = Theta_ev
    elif 'ice-emp' in args['runwho']:
        print('Running empirical ic estimation with parameters: ', opts)
        # Method ICE empirical
        ice_lam_min = opts[0]
        ice_lam_max = opts[1]
        ice_lam_n   = int(opts[2])

        t0 = timer()
        out = ice_sparse_empirical(X, lams=np.linspace(ice_lam_min, \
                                                       ice_lam_max, \
                                                       ice_lam_n))
        te = timer() - t0
        Theta = _threshold_hard(out[0], tol=1e-3)
    else:
        print("Unknown method!")
        Theta = np.zeros(W_true.shape)
    # Output MB info
    MBs = _MBs_fromInvCov(Theta)
    # Evaluate Theta
    if 'ice-emp' in args['runwho']:
        # Report result
        Theta_sym = 0.5 * (Theta + Theta.T) # optional if Theta already is symmetric
        G_ = _get_undir_graph(Theta_sym)
        acc = utils.count_accuracy(Gref, G_)
        loss = _loss_glasso(Theta_sym, emp_cov)
        dfro = np.linalg.norm(Theta_sym -Theta_ev)
        # stats
        f=[]
        f.append({'meth': 'emp', 'time': te , \
                      'lambda': out[1], 'loss': loss, \
                      'min_diag_prec': min(np.diag(Theta)), \
                       'dist': dfro, \
                       'shd': acc['shd'], \
                        'tpr': acc['tpr'], \
                        'fdr': acc['fdr'], \
                        'fpr': acc['fpr'], \
                        'nnz': acc['nnz'] \
                        })
        ff=pd.DataFrame(f, columns=f[0].keys())
        print(ff)
    else:
        ff = []
    # Output matrices
    fout2 = '%s/list_MB' % dir_out
    if not os.path.exists('%s.csv'%fout2):
        with open('%s.csv'%fout2, 'w', newline='') as csvf2:
            csvw = csv.writer(csvf2)
            for i in range(len(MBs)):
                # Pick one Markov blanket
                Mi = MBs[i]
                Mii = np.where(MBs[i])[0]
                Miib = Mii.copy()
                Miib = Miib[Miib!=i]
                if len(Miib) > 0:
                    csvw.writerow(Miib)
                else:
                    # tmp = [np.nan]
                    tmp = [-1]
                    csvw.writerow(tmp)
    save_npz('%s/Wtrue.npz'%(dir_out), csc_matrix(W_true) )
    save_npz('%s/%s'%(dir_out,filename), csc_matrix(Theta) )
    # Save data matrix to npy (Ref: https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk)
    np.save('%s/datamat.npy'%dir_out, X)

    # Print to screen (or the output txt file) the scores
    if len(ff) > 0:
        ff_ = ff.iloc[0]
        print("%d,%.3f,%.3f,%.3f,%d"%(ff_['shd'],ff_['tpr'],ff_['fdr'],ff_['fpr'],ff_['nnz']))

