import glob, itertools, re, sys, os, inspect
from scipy.sparse import  csc_matrix, save_npz, load_npz
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import scipy.linalg as slin
from timeit import default_timer as timer

from utils import utils

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


def report_solutions3(m, XC, opts, Wtrue, Bf, fdir, verbo=2):
    """ Same function as report_solutions() except that:
            * one additional output: the list of all variables of a solution
    """
    def retrieve_sols(m, d, solnum=0):
        # m.SolutionNumber = solnum # query the solnum-th alternative solution
        m.setParam("SolutionNumber", solnum) # query the solnum-th alternative solution
        # Retrieve solution
        sff = [e for e in m.getVars() if e.VarName[:2] == 'XC']
        svars = {v.VarName: v.Xn for v in m.getVars()}
        # get adjacency matrix
        Bsol = np.zeros([d,d])
        for it in sff:
            pair = [int(match) for match in re.findall(r'\d+', it.VarName)]
            Bsol[pair[0],pair[1]] = it.Xn
        # Get obj value
        objval = m.PoolObjVal
        return Bsol, sff, objval, svars
    Sols = []
    Scores = []
    if opts[1] == 0:
        # Only one solution produced,
        # output grb_sol_num0.txt, grb_solVStrue_num0.txt
        Bsol, sff, objval, svars = retrieve_sols(m, Wtrue.shape[0], solum=0) # see Manual page 759 about 'Xn'
        score = write_score_solutions(XC, Bsol, Wtrue, Bf, objval, solnum=0,  fdir_cout=fdir, verbo=verbo)
        Sols.append((Bsol, sff, svars))
        Scores.append(score)
    elif opts[1] >= 1:
        # Do the same for all other solutions as num0
        for solnum in range(opts[2]): # a preset maximal number
            Bsol, sff, objval, svars = retrieve_sols(m, Wtrue.shape[0], solnum=solnum)
            score = write_score_solutions(XC, Bsol, Wtrue, Bf, objval, solnum=solnum, fdir_cout=fdir, verbo=verbo)
            Sols.append((Bsol, sff, svars))
            Scores.append(score)
        # Make a selection and append the selection result in grbsol_scores.txt
        # --- Chantier #319 [todo]----
        df = pd.DataFrame(Scores)
        isel = df['hval'].idxmin()
        fout = '%s/grbsol_scores'%fdir # Created previously by write_score_solutions()
        with open(f'{fout}.txt', 'a') as f1:
            solnum_sel = df['solnum'].iloc[isel]
            f1.write(f"\n\n---> Selected solution is: solnum = {solnum_sel}")
    return Sols, df, isel

def write_score_solutions(XC, Bsol, Wtrue, Bf, objval, solnum=0, fdir_cout='outputs/fout', verbo=2):
    # Write grb_sol_num*.txt
    fout4 = '%s/grb_sol'%fdir_cout
    with open(f'{fout4}_num{solnum}.txt', 'w') as f1:
        for key, _ in XC.items():
            var = Bsol[key[0], key[1]]
            if var > 0:
                line = '%s: %s'%(key, var)
                f1.write(line+'\n')

    # write Wsol_num*.npz
    save_npz(f'{fdir_cout}/Wsol_num{solnum}.npz', csc_matrix(Bsol) )

    # Incrementally write in grbsol_scores.txt/csv
    acc_ = utils.count_accuracy((Wtrue.toarray())!=0, Bsol!=0)
    hval = _h(Bsol)

    sctxt = '%s/grbsol_scores.txt'%fdir_cout
    sccsv = '%s/grbsol_scores.csv'%fdir_cout
    if not os.path.exists(sctxt):
        with open(sctxt, 'w') as f1:
            f1.write(f'Solution status and accuracy: \n\n')
            f1.write("GRB.SolutionNumber: %d, Objective value: %.3f |  SHD: %d, TPR: %.3f, FDR: %.3f, FPR: %.3f, nnz: %d | hval: %.3f\n"%(solnum, objval, acc_['shd'],acc_['tpr'],acc_['fdr'],acc_['fpr'],acc_['nnz'], hval))
        with open(sccsv, 'w') as f2:
            f2.write("shd,tpr,fdr,fpr,nnz,LPobjval,hval\n")
            f2.write("%d,%.3f,%.3f,%.3f,%d,%.3f,%.3f\n"%(acc_['shd'],acc_['tpr'],acc_['fdr'],acc_['fpr'],acc_['nnz'], objval, hval))
    else:
        with open(sctxt, 'a') as f1:
            # f1.write("GRB.SolutionNumber: %d, Objective value: %.3f |  SHD: %d, TPR: %.3f, FDR: %.3f, FPR: %.3f, nnz: %d\n"%(solnum, objval, acc_['shd'],acc_['tpr'],acc_['fdr'],acc_['fpr'],acc_['nnz']))
            f1.write("GRB.SolutionNumber: %d, Objective value: %.3f |  SHD: %d, TPR: %.3f, FDR: %.3f, FPR: %.3f, nnz: %d | hval: %.3f\n"%(solnum, objval, acc_['shd'],acc_['tpr'],acc_['fdr'],acc_['fpr'],acc_['nnz'], hval))
        with open(sccsv, 'a') as f2:
            # f2.write("%d,%.3f,%.3f,%.3f,%d,%.3f\n"%(acc_['shd'],acc_['tpr'],acc_['fdr'],acc_['fpr'],acc_['nnz'], objval))
            f2.write("%d,%.3f,%.3f,%.3f,%d,%.3f,%.3f\n"%(acc_['shd'],acc_['tpr'],acc_['fdr'],acc_['fpr'],acc_['nnz'], objval, hval))
    # # Write grb_sol_VStrue_num*.txt
    # if verbo > 1:
    #     filepath = f'{fout4}_VStrue_num{solnum}.txt'
    #     # modify or create new func that uses Bsol
    #     write_table_sol_v_true3(XC, Bsol, Bf, Wtrue, filepath, acc=acc_)
    return {'solnum':solnum, 'objval':objval, 'shd':acc_['shd'], 'tpr':acc_['tpr'], 'fdr':acc_['fdr'], 'fpr':acc_['fpr'], 'nnz':acc_['nnz'], 'hval':hval}


def _load_MBs_fromfile(dir_out):
    with open('%s/list_MB.csv'%dir_out, 'r') as ff:
        fc = ff.readlines()
    mbs = []
    for ll in fc:
        ll = ll.strip()
        # Convert the string to a list of integers
        mb= list(map(int, ll.split(',')))
        if mb[0] < 0:
            mb = []
        mbs.append( mb  )
    return mbs

def _addVar_XCXS2(MBs):
    i = -1
    # C, S variables
    # exclusiveness constraint
    lines = []
    for mb in MBs:
        i += 1
        if len(mb) > 0:
            for j in mb:
                if j != i:
                    sa = 'XC[(%d, %d)] = m.addVar(vtype=GRB.BINARY, name="XC_%d_%d")' %(i, j, i, j)
                    sb = 'XC[(%d, %d)] = m.addVar(vtype=GRB.BINARY, name="XC_%d_%d")' %(j, i, j, i)
                    constr_cc = 'm.addConstr(XC[(%d,%d)] + XC[(%d,%d)] <= 1, "CC_%d_%d")' % (i,j, j,i, i,j)
                    if i < j:
                        ss = 'XS[(%d, %d)] = m.addVar(vtype=GRB.BINARY, name="XS_%d_%d")' %(i, j, i, j)
                        constr_cs = 'm.addConstr(XC[(%d,%d)] + XC[(%d,%d)] + XS[(%d, %d)]>= 1, "CS_%d_%d")' % (i,j, j,i, i,j, i,j)

                    else:
                        ss = 'XS[(%d, %d)] = m.addVar(vtype=GRB.BINARY, name="XS_%d_%d")' %(j, i, j, i)
                        constr_cs = 'm.addConstr(XC[(%d,%d)] + XC[(%d,%d)] + XS[(%d, %d)]>= 1, "CS_%d_%d")' % (i,j, j,i, j,i, j,i)
                    lines.append(sa)
                    lines.append(sb)
                    lines.append(ss)
                    lines.append(constr_cc)
                    lines.append(constr_cs)
    return lines

def _addVar_XV2(dir_out, Bf):
    # this function is based on _addVar_XV() with changelog:
    #   - input arg Bf: from the existing function for producing the naive merge Bf
    ifiles = glob.glob("%s/locres*.npz" %dir_out)
    res = []
    Xs = []
    Wtrue = load_npz('%s/Wtrue.npz'%dir_out)
    # Bf = np.zeros(Wtrue.shape)
    # for f in ifiles:
    #     ss = load_npz('%s'%f )
    #     res.append( ss )
    #     tmp = re.findall(r'\d+', f)
    #     Xs.append(int(tmp[-1]))
    #     Bf += ss.toarray()
    # # initialize lines
    d = Bf.shape[0]
    lines = []
    # V variable
    for k in range(d):
        pp = np.where(Bf[:,k]!=0)[0]
        pairs_nested = [(i, j) for i in pp for j in pp if i < j]
        for (i,j) in pairs_nested:
            v = 'XV[(%d, %d, %d)] = m.addVar(vtype=GRB.BINARY, name="XV_%d_%d_%d")' %(i, j, k, i,j,k)
            lines.append(v)
            # add constraints on XV and XC
            vconstr_1 = 'm.addConstr( XC[(%d, %d)] >=  XV[(%d, %d, %d )], "CC_inf1_%d_%d_%d")' %(i,k, i,j,k, i,j,k)
            vconstr_2 = 'm.addConstr( XC[(%d, %d)] >=  XV[(%d, %d, %d )], "CC_inf2_%d_%d_%d")' %(j,k, i,j,k, i,j,k)
            vconstr_3 = 'm.addConstr( XC[(%d, %d)] + XC[(%d,%d)] - 1 <= XV[( %d, %d, %d)], "CC_sup_%d_%d_%d")' % (i,k, j,k, i,j,k, i,j,k)
            lines.append(vconstr_1)
            lines.append(vconstr_2)
            lines.append(vconstr_3)
    return lines


def _addConstr_XV_XS(m, XV, XS):
    # Add constraints on XV and XS
    # a/ V(i,j,k) <= S(i,j)
    # b/ S(i,) <= sum_k V(i,j,k)
    keys = [key for (key,_) in XV.items()]
    for (key,_) in XS.items():
        i = key[0]
        j = key[1]
        m_triplets = [t for t in keys if t[:2] == (i,j,) ]
        m.addConstrs( (XS[(i, j)] >=  XV[(i, j, k )] for (i,j,k) in m_triplets), "CC_infvs_%d_%d"%(i,j))
        expr_ = gp.quicksum(XV[(i,j,k)] for (i,j,k) in m_triplets)
        m.addLConstr( XS[(i,j)], GRB.LESS_EQUAL, expr_, name="CC_supvs_%d_%d"%(i,j))
    return m

def _addConstr_boundedDegree2(m, XC, K):
    expr_all = gp.quicksum( XC[(key[0], key[1])] for (key,_) in XC.items())
    m.addLConstr( expr_all, GRB.LESS_EQUAL, K, name="DegAverage_sup")
    return m

def _loadlocalres_constants(XC, dir_out, choice=1, verbo=2):
    # ff is the local results folder
    ifiles = glob.glob("%s/locres*.npz" %dir_out)
    res = []
    Xs = []
    Wtrue = load_npz('%s/Wtrue.npz'%dir_out)
    Bf = np.zeros(Wtrue.shape)
    for f in ifiles:
        ss = load_npz('%s'%f )
        res.append( ss )
        tmp = re.findall(r'\d+', f)
        Xs.append(int(tmp[-1]))
        Bf += ss.toarray()
    # initialize lines
    d = Bf.shape[0]
    lines = []
    # Constant variables and constraints
    for key, _ in XC.items():
        i = key[0]
        j = key[1]
        if Bf[i, j] != 0:
            if choice == 1:
                val = 1
            else:
                val = abs(Bf[i,j])
        else:
            val = 0
        ya = "YC[(%d, %d)] = m.addVar(lb=%.2f, ub=%.2f, vtype=GRB.CONTINUOUS, name='YC_%d_%d')" % (i,j, val, val, i,j)
        yb1 = "YC[(%d, %d)].setAttr('Start', %.2f)" % (i,j, val)
        yb2 = "YC[(%d, %d)].setAttr('LB', %.2f)" % (i,j, val )
        yb3 = "YC[(%d, %d)].setAttr('UB', %.2f)" % (i,j, val )
        lines.append(ya)
        lines.append(yb1)
        lines.append(yb2)
        lines.append(yb3)
    # show Bf accuracy
    if verbo>=1:
        acc_ = utils.count_accuracy((Wtrue.toarray())!=0, Bf!=0)
        print('====> Accuracy of the raw concatenation of locres: ', acc_, '\n')
    else:
        acc_ = []
    # # NOTE (#1122): show location of merge conflicts and their types
    # if verbo>1:
    #     _print_info_mergec(Bf)
    return lines, res, Xs, Bf, Wtrue, acc_
#
def _loadlocalres_constants3(XC, dir_out, verbo=2):
    ifiles = glob.glob("%s/locres*.npz" %dir_out)
    res = []
    Xs = []
    Wtrue = load_npz('%s/Wtrue.npz'%dir_out)
    Bf = np.zeros(Wtrue.shape)
    for f in ifiles:
        ss = load_npz('%s'%f )
        res.append( ss )
        tmp = re.findall(r'\d+', f)
        Xs.append(int(tmp[-1]))
        Bf += ss.toarray()
    Bf_ = _preliminary_merge(Bf)
    # initialize lines
    d = Bf.shape[0]
    lines = []
    # Constant variables and constraints
    for key, _ in XC.items():
        i = key[0]
        j = key[1]
        if Bf_[i, j] != 0:
            val = abs(Bf_[i,j])
        else:
            val = 0
        ya = "YC[(%d, %d)] = m.addVar(lb=%.2f, ub=%.2f, vtype=GRB.CONTINUOUS, name='YC_%d_%d')" % (i,j, val, val, i,j)
        yb1 = "YC[(%d, %d)].setAttr('Start', %.2f)" % (i,j, val)
        yb2 = "YC[(%d, %d)].setAttr('LB', %.2f)" % (i,j, val )
        yb3 = "YC[(%d, %d)].setAttr('UB', %.2f)" % (i,j, val )
        lines.append(ya)
        lines.append(yb1)
        lines.append(yb2)
        lines.append(yb3)
    # show Bf accuracy
    if verbo>=1:
        acc_ = utils.count_accuracy((Wtrue.toarray())!=0, Bf!=0)
        print('====> Accuracy of the raw concatenation of locres: ', acc_, '\n')
    else:
        acc_ = [ ]
    return lines, res, Xs, Bf, Wtrue, Bf_, acc_

# NOTE (#1129) : for LP*4
def _loadlocalres_constants4(XC, dir_out, verbo=2):
    ifiles = glob.glob("%s/locres*.npz" %dir_out)
    res = []
    Xs = []
    Wtrue = load_npz('%s/Wtrue.npz'%dir_out)
    Bf = np.zeros(Wtrue.shape)
    for f in ifiles:
        # ---Update [#319]: use absolute values from the begining
        # ss = load_npz('%s'%f ) # ss is a scipy sparse matrix
        ss = abs( load_npz('%s'%f ) )
        # --------
        res.append( ss )
        tmp = re.findall(r'\d+', f)
        Xs.append(int(tmp[-1]))
        Bf += ss.toarray()
    # initialize lines
    d = Bf.shape[0]
    lines = []
    # Constant variables and constraints
    for key, _ in XC.items():
        i = key[0]
        j = key[1]
        minfo = _get_mergeinfo3(Bf, i, j)
        if minfo['code'] == 0:
            val = float(Bf[i,j]!=0)
        elif minfo['code'] in [2, 3]:
            val = 0.5 * float(Bf[i,j]!=0)
        elif minfo['code'] == 1:
            val = (2/3) * float(Bf[i,j]!=0)
        ya = "YC[(%d, %d)] = m.addVar(lb=%.2f, ub=%.2f, vtype=GRB.CONTINUOUS, name='YC_%d_%d')" % (i,j, val, val, i,j)
        yb1 = "YC[(%d, %d)].setAttr('Start', %.2f)" % (i,j, val)
        yb2 = "YC[(%d, %d)].setAttr('LB', %.2f)" % (i,j, val )
        yb3 = "YC[(%d, %d)].setAttr('UB', %.2f)" % (i,j, val )
        lines.append(ya)
        lines.append(yb1)
        lines.append(yb2)
        lines.append(yb3)
    # show Bf accuracy
    if verbo>=1:
        acc_ = utils.count_accuracy((Wtrue.toarray())!=0, Bf!=0)
        print('====> Accuracy of the raw concatenation of locres: ', acc_, '\n')
    else:
        acc_ = []
    return lines, res, Xs, Bf, Wtrue, acc_

# NOTE (#319) : for LP*6
# The function below is based on _loadlocalres_constants4()
#   To be used when hatB has weighted and signed edges
#   Turn all weighted and signed edges into 1-0 valued edges
def _loadlocalres_constants6(XC, dir_out, verbo=2):
    ifiles = glob.glob("%s/locres*.npz" %dir_out)
    res = []
    Xs = []
    Wtrue = load_npz('%s/Wtrue.npz'%dir_out)
    Bf = np.zeros(Wtrue.shape)
    for f in ifiles:
        # ---Update [#319]: use absolute values from the begining
        # ss = load_npz('%s'%f ) # ss is a scipy sparse matrix
        ss = load_npz('%s'%f )
        ss[ss!=0] = 1
        # --------
        res.append( ss )
        tmp = re.findall(r'\d+', f)
        Xs.append(int(tmp[-1]))
        Bf += ss.toarray()
    # initialize lines
    d = Bf.shape[0]
    lines = []
    # Constant variables and constraints
    for key, _ in XC.items():
        i = key[0]
        j = key[1]
        minfo = _get_mergeinfo3(Bf, i, j)
        if minfo['code'] == 0:
            val = float(Bf[i,j]!=0)
        elif minfo['code'] in [2, 3]:
            val = 0.5 * float(Bf[i,j]!=0)
        elif minfo['code'] == 1:
            val = (2/3) * float(Bf[i,j]!=0)
        else:
            print('Code not defined')
            val = 0
        ya = 'YC[(%d, %d)] = m.addVar(lb=%.2f, ub=%.2f, vtype=GRB.CONTINUOUS, name="YC_%d_%d")' % (i,j, val, val, i,j)
        yb1 = 'YC[(%d, %d)].setAttr("Start", %.2f)' % (i,j, val)
        yb2 = 'YC[(%d, %d)].setAttr("LB", %.2f)' % (i,j, val )
        yb3 = 'YC[(%d, %d)].setAttr("UB", %.2f)' % (i,j, val )
        lines.append(ya)
        lines.append(yb1)
        lines.append(yb2)
        lines.append(yb3)
    # show Bf accuracy
    if verbo>=1:
        acc_ = utils.count_accuracy((Wtrue.toarray())!=0, Bf!=0)
        print('====> Accuracy of the raw concatenation of locres: ', acc_, '\n')
    else:
        acc_ = []
    return lines, res, Xs, Bf, Wtrue, acc_


def _get_mergeinfo3(Bf, i, j):
    # Type 1 (dir/undir)
    if (Bf[i,j] == 2 and Bf[j,i]==1) or (Bf[i,j] == 1 and Bf[j,i]==2):
        minfo={'code':1, 'name':'Type 1 (Dir/Undir)'}
    # Type 2 (acute)
    elif (Bf[i,j] == 1 and Bf[j,i]==1):
        minfo={'code': 2, 'name': 'Type 2 (Acute)'}
    # Type 3 (addition)
    elif (Bf[i,j] == 1 and Bf[j,i]==0) or (Bf[i,j] == 0 and Bf[j,i]==1):
        minfo={'code': 3, 'name': 'Type 3 (Addition)'}
    else:
        minfo={'code': 0, 'name': ' '}
    return minfo


def _preliminary_merge(Bf):
    IJ_nnz = np.where(Bf)
    # Type 1 (dir/undir) conflicts
    #   find all pairs such that Bf[i,j] = 2 and Bf[j,i] = 1
    pairs_ = [(i,j) for (i,j) in zip(IJ_nnz[0], IJ_nnz[1]) \
                if (Bf[i,j] == 2 and Bf[j,i]==1) or \
                   (Bf[i,j] == 1 and Bf[j,i]==2)  ]
    for (i,j) in pairs_:
        Bf[i,j] = 2* int( Bf[i,j] > Bf[j,i] )
        Bf[j,i] = 2* int( Bf[j,i] > Bf[i,j] )
    # Type 3 (addition) conflicts
    #  Nothing to reconcile on these pairs in Bf
    return Bf

def _classify_mergeconf(Bi, Bj, Xs, i, j):
    types = {'1': 'ADDITION', '2': 'ACUTE', '3': 'CPDAG/UNDIR', '0':'Bug!', '4': 'Bug!'}
    # Valid types are indexed by {1,2,3}. '0' and '4' should not appear for a conflicting pair
    bits =  np.array([ Bi[Xs[i],Xs[j]], \
                Bj[Xs[i],Xs[j]],\
                Bi[Xs[j],Xs[i]],\
                Bj[Xs[j],Xs[i]] ])
    ll = sum(bits!=0)
    print('-----Nb of nonzeros among the 4 entries: ', ll)
    lb = types['%d'%ll]
    print('------Conflict type: ', lb)
    return lb

def _setObjective(model, XC, YC):
    Z = {}
    ZA = {}
    for (key, var) in XC.items():
            i = key[0]
            j = key[1]
            Z[(i, j)] = model.addVar(vtype=GRB.BINARY, name="Z_%d_%d"%(i,j) )
            ZA[(i, j)] = model.addVar(vtype=GRB.BINARY, name="ZA_%d_%d"%(i,j) )
            model.addConstr( Z[(i,j)] == XC[(i,j)] - YC[(i,j)], "ZC_%d_%d"%(i,j) )
            model.addConstr( ZA[(i,j)] == gp.abs_(Z[(i,j)]), "ZAC_%d_%d"%(i,j) )
    objective_expr = gp.quicksum(ZA[(key[0],key[1])] for (key,_) in XC.items())
    model.setObjective(objective_expr, sense=gp.GRB.MINIMIZE)
    return model, objective_expr

def _setObjective_XdotY(model, XC, YC):
    objective_expr = gp.quicksum(XC[(key[0],key[1])]*YC[(key[0], key[1])] for (key,_) in XC.items())
    model.setObjective(objective_expr, sense=gp.GRB.MAXIMIZE)
    return model, objective_expr

def _setObjective_XdotY_signed(model, XC, YC):
    objective_expr = gp.quicksum((XC[(key[0],key[1])]*YC[(key[0], key[1])] - 0.5 * (XC[(key[0],key[1])] + YC[(key[0], key[1])])) for (key,_) in XC.items())
    model.setObjective(objective_expr, sense=gp.GRB.MAXIMIZE)
    return model, objective_expr

# Define the callback function
def my_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        # Extract the solution
        x = model.cbGetSolution(model._vars)

        # Access solution information
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        # Add your custom logic here
        print(f"My callback: New solution found. Objective value: {objective_value}")

def _h(W):
    """Evaluate value and gradient of acyclicity constraint."""
    W[W!=0] = 1
    E = slin.expm(W * W)  # (Zheng et al. 2018)
    return np.trace(E) - W.shape[0]


def input_args(args=None):
    if args==None:
        args = {}
    for a in sys.argv:
        if "=" in a:
            p_k, p_v = a.split("=")
            p_v = p_v.split(",")
            if p_k in ['ds', 'opts']:
                args[p_k] = [int(_) for _ in p_v]
            elif p_k in ['degs', 'ice_lambda1']:
                args[p_k] = [float(_) for _ in p_v]
            elif p_k in ['rnd', 'LAM' ]:
                args[p_k] = float(p_v[0])
            elif p_k in ['SEED','VAR', 'verbo']:
                args[p_k] = int(p_v[0])
            elif p_k in ['graph_type','sem_type','fdir', 'fout', 'filename', \
                            'solver_primal', 'rowname']:
                args[p_k] = p_v[0]
            elif p_k == "filename":
                args[p_k] = f"{p_v[0]}.csv"
            else:
                print("Unknown parameter:", p_k)
                pass
        elif a in ['ILP', 'LP', 'Concat']:
            args['runwho'] = [a]
        elif a == 'all':
            args['runwho'] = ['all']
    return args


if __name__ == '__main__':

    args = input_args()

    ff = args['fdir']
    fdir_locres = '%s/%s' %(ff, args['fout'])

    # d = args['ds'][0]
    # deg = args['degs'][0]
    # rnd = args['rnd']

    verbo = args['verbo']

    SEED = args['SEED']
    fdir_cout = '%s/conqres_seed%d' %(ff, SEED)
    if not os.path.exists(fdir_cout):
        os.makedirs(fdir_cout)

    Wtrue = load_npz('%s/Wtrue.npz'%fdir_locres)

    # ----Entry for different methods
    if 'Concat' in args['runwho'] or 'all' in args['runwho']:
        print(f"=====Starting {args['runwho']}=====")
        ifiles = glob.glob("%s/locres*.npz" %fdir_locres)
        res = []
        Xs = []
        # Wtrue = load_npz('%s/Wtrue.npz'%fdir_locres)
        Bf = np.zeros(Wtrue.shape)
        for f in ifiles:
            ss = load_npz('%s'%f )
            res.append( ss )
            tmp = re.findall(r'\d+', f)
            Xs.append(int(tmp[-1]))
            Bf += ss.toarray()
        acc_ = utils.count_accuracy((Wtrue.toarray())!=0, Bf!=0)
        print('\n=====> Accuracy of the raw concatenation: ', acc_)
        # print scores in csv format
        hval = _h(Bf)
        print("%d,%.3f,%.3f,%.3f,%d,%.3f"%(acc_['shd'],acc_['tpr'],acc_['fdr'],acc_['fpr'],acc_['nnz'], hval))

    if 'ILP' in args['runwho'] or 'LP' in args['runwho'] or 'all' in args['runwho']:
        print(f"=====Starting {args['runwho']} with options: {args['opts']}")
        ti = timer()
        opts = args['opts'] # add or not new constraint, Y choice, add or not 3-cycle constraint
        # Create a new model
        m = gp.Model("mip1")
        XV = {}
        XC = {}
        XS = {}
        YC = {}

        # fout1 = '%s/list_XC_YC_XS'%fdir_cout
        # fout2 = '%s/list_XV'%fdir_cout
        # fout3 = '%s/list_YCC'%fdir_cout
        # fout4 = '%s/list_XV_XS'%fdir_cout
        # fout5 = '%s/list_setObjective'%fdir_cout

        MBs = _load_MBs_fromfile(fdir_locres)
        # Create variables and add constraints
        lines_xcxs = _addVar_XCXS2(MBs)
        linesxc = lines_xcxs
        line0 = ['%% 2-cycle exclusiveness: C(i,j) + C(j,i) <=1',\
                 '%% Markov blanket property: C(i,j) + C(j,i) + S(i,j) >= 1']

        for line in linesxc:
            exec(line)

        # Read local results
        choice = opts[0]
        if choice in [1, 2]:
            lines_y, locres, Xs, Bf, Wtrue, acc_concat = _loadlocalres_constants(XC, fdir_locres, choice=choice, verbo=verbo)
        elif choice == 3:
            lines_y, locres, Xs, Bf, Wtrue, Bf_, acc_concat = _loadlocalres_constants3(XC, fdir_locres, verbo=verbo)
            # print scores
            acc3_ = utils.count_accuracy((Wtrue.toarray())!=0, Bf_!=0)
            print('\n=====> Accuracy of the output of Alg3: ', acc_)
            print("%d,%.3f,%.3f,%.3f,%d"%(acc3_['shd'],acc3_['tpr'],acc3_['fdr'],acc3_['fpr'],acc3_['nnz']))
        elif choice == 4:
            lines_y, locres, Xs, Bf, Wtrue, acc_concat = _loadlocalres_constants4(XC, fdir_locres, verbo=verbo)
        elif choice == 6:
            lines_y, locres, Xs, Bf, Wtrue, acc_concat = _loadlocalres_constants6(XC, fdir_locres, verbo=verbo)
        else:
            raise ValueError(f"The option {choice} is not available.")

        for line in lines_y:
            exec(line)
        # if verbo > 1:
        #     with open('%s.txt'%fout3, 'w') as f1:
        #         for line in lines_y:
        #             f1.write(line)
        #             f1.write('\n')

        # # Record XC-XS to text file
        # if verbo > 1:
        #     with open('%s.txt'%fout1, 'w') as f1:
        #         for line in line0: # Commentaries
        #             f1.write(line)
        #             f1.write('\n')
        #         for line in linesxc: # AddVar and AddConstr commands
        #             f1.write(line)
        #             f1.write('\n')

        # Execute XV related commands
        # XV variables and constraints
        # lines_xv = _addVar_XV(fdir_locres)
        Bf -= np.diag(np.diag(Bf)) # Ensure that Bf does not have self-loops
        lines_xv = _addVar_XV2(fdir_locres, Bf)
        line0 = ['%% Constraints between C (causal pairs) and V (v-structures)',\
                     '%% C(i,k) >= V(i,j,k), C(j,k) >= V(i,j,k)', \
                     '%% C(i,k) + C(j,k) <= V(i,j,k) + 1']
        lines2ex = lines_xv
        for line in lines2ex:
            exec(line)
        # if verbo > 1:
        #     with open('%s.txt'%fout2, 'w') as f1:
        #         for line in line0:
        #             f1.write(line)
        #             f1.write('\n')
        #         for line in lines2ex:
        #             f1.write(line)
        #             f1.write('\n')


        # XV-XS constraints
        m = _addConstr_XV_XS(m, XV, XS) #
        line0 = ['%% Constraints between S (spouse decision var) and V (v-structures)\n',\
                     '%% S(i,j) >= V(i,j,k) for any k\n', \
                     '%% S(i,j) <= sum_k V(i,j,k)\n']
        # if verbo > 1:
        #     with open('%s.txt'%fout4, 'w') as f1:
        #         for line in line0:
        #             f1.write(line)
        #         func_source = inspect.getsource(_addConstr_XV_XS)
        #         f1.write(func_source)

        # Set objective
        m, obj = _setObjective_XdotY_signed(m, XC, YC)
        line0 = ['%% Objective function: inner product between C and the constant Y (concatenation of local results)\n',\
                '%% f(C) = <C, Y> = sum_{i,j} C(i,j) * Y(i,j), where \n', \
                '%% C and Y coefficients are transformed to signed symbols, ie, from {0,1} to {-1, 1} \n']

        # if verbo > 1:
        #     with open('%s.txt'%fout5, 'w') as f1:
        #         for line in line0:
        #             f1.write(line)
        #         func_source = inspect.getsource(_setObjective_XdotY_signed)
        #         f1.write(func_source)

        # set up PoolSerachMode [4,2,10] or [4,2,10,k]
        if len(opts) > 1:
            m.setParam('PoolSearchMode', opts[1])  # 2 corresponds to GRB.PoolSearchMode.GUIDED
            if opts[1] >= 1 and len(opts)>2:
                m.setParam('PoolSolutions', min(20, opts[2]) )
            # (Optional for XC) New constraint on the node degrees
            # This option is only enabled when opts has a fifth value
            if len(opts) > 3: # opts is [4,2,10] or [4,2,10,k]
                # k = opts[3] # an integer bounding the average degree
                # m = _addConstr_boundedDegree2(m, XC, opts[5])
                m = _addConstr_boundedDegree2(m, XC, opts[3])

        # Set up callback
        m._vars = m.getVars()
        # m.Params.DualReductions = 0

        # Optimize
        m.optimize(my_callback)

        # Report solutions
        status = m.status
        ti = timer() - ti

        if status == gp.GRB.INFEASIBLE or m.SolCount == 0:
            print("No feasible solution found.")
            print('\n=====> Accuracy of the GRB solution: [None]')
            # print scores in csv format
            print("inf,inf,inf,inf,inf")
            print("inf,inf,inf,inf,inf,inf,%.3f"%ti)
        else:
            # # NOTE (#1130): merge into the report solutions function
            # Sols = report_solutions(m, XC, opts, Wtrue, Bf, fdir_cout, verbo=verbo)
            # Sols, diffs = report_solutions2(m, XC, opts, Wtrue, Bf, fdir_cout, verbo=verbo)
            # sol = Sols[0]
            # Bgp = sol[0]
            # NOTE (#330): retain only [Sols, isel]
            Sols, df_scores, isel = report_solutions3(m, XC, opts, Wtrue, Bf, fdir_cout, verbo=verbo)
            solSel = Sols[isel]
            Bgp = solSel[0]

            # acc_ = utils.count_accuracy((Wtrue.toarray())!=0, Bgp!=0)
            acc_ = df_scores.iloc[isel].to_dict()
            print('\n=====> Accuracy of the GRB solution: ', acc_)
            print('\n')

            # [1231] Compute exp trace function value
            # hval = _h(Bgp)
            hval = acc_['hval']

            # Print scores in csv format
            print("%d,%.3f,%.3f,%.3f,%d"%(acc_concat['shd'],acc_concat['tpr'],acc_concat['fdr'],acc_concat['fpr'],acc_concat['nnz']))
            print("%d,%.3f,%.3f,%.3f,%d,%.3f,%.3f"%(acc_['shd'],acc_['tpr'],acc_['fdr'],acc_['fpr'],acc_['nnz'], hval, ti))
            # if verbo >= 1:
            #     # Save gp solution as npz matrix
            #     fgp = '%s/Wsol.npz'%fdir_cout
            #     if not os.path.exists(fgp):
            #         save_npz(fgp, csc_matrix(Bgp) )


