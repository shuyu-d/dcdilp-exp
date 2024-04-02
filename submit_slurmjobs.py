import time, subprocess, os, re
import glob
import numpy as np
import pandas as pd
str = time.strftime("%H%M%S%m%d")

def extract_numbers(input_string):
    return [int(match.group()) for match in re.finditer(r'\d+', input_string)]

# Test DCDILP-GES
exp_dcges=False
if exp_dcges:
    FDIR='outputs/dcilp_%s'%str
    os.makedirs(FDIR)

    gt='ER'
    verbo1=2
    verbo2=2
    tid=0
    jobs = []
    for st in ['gauss', 'gumbel', 'uniform']:
        for rnd in [10 ]:
            for degs in [2 ]:
                for ds in [100 ]: #[100, 1600]:
                    tid+=1
                    # define the number of parallel workers
                    nt = min(ds*2, 400)
                    for j in range(3):
                        # Maximal values in the search of lambda1 in MB estimation (phase 1)
                        if ds < 800:
                            lam1, lam2 = 0.3, 0.5
                        else:
                            lam1, lam2 = 0.1, 0.2
                        # run methods
                        mid=1
                        if len(jobs) <  1:
                            cmd = f"sbatch --ntasks={nt} run_dcdilp.slurm {str} {tid} {ds} {degs} {rnd} {gt} {st} {mid} {ds} {j} {FDIR} {verbo1} {verbo2} DCpool-GES2 'nan' ILP '6,2,10' ice-emp '0.05,{lam2},20'"
                            jobnum = subprocess.check_output(cmd, shell= True)
                            # >>> jobnum
                            # >>> b'Submitted batch job 396932\n'
                            jobs.append(extract_numbers(jobnum.decode())[0])
                        else:
                            jobid = jobs[-1]
                            cmd = f"sbatch --dependency=afterany:{jobid} --ntasks={nt} run_dcdilp.slurm {str} {tid} {ds} {degs} {rnd} {gt} {st} {mid} {ds} {j} {FDIR} {verbo1} {verbo2} DCpool-GES2 'nan' ILP '6,2,10' ice-emp '0.05,{lam2},20'"
                            jobnum = subprocess.check_output(cmd, shell= True)
                            jobs.append(extract_numbers(jobnum.decode())[0])
                        # Next meth/job 2
                        mid+=1
                        jobid = jobs[-1]
                        cmd = f"sbatch --dependency=afterany:{jobid} --ntasks={nt} run_dcdilp.slurm {str} {tid} {ds} {degs} {rnd} {gt} {st} {mid} {ds} {j} {FDIR} {verbo1} {verbo2} DCpool-GES2 'nan' ILP '6,2,10' ice-emp '0.05,{lam1},20'"
                        jobnum = subprocess.check_output(cmd, shell= True)
                        jobs.append(extract_numbers(jobnum.decode())[0])


# Test DCDILP-DAGMA
exp1_dcdagma=True
if exp1_dcdagma:
    FDIR='outputs/dcilp_%s'%str
    os.makedirs(FDIR)

    gt='ER'
    # st='gauss'
    verbo1=2
    verbo2=2

    tid=0
    jobs = []
    for st in ['gauss']:
        for rnd in [10, 50]:
            for degs in [2]:
                for ds in [50, 1600]:
                    tid+=1
                    # define the number of parallel workers
                    nt = min(ds*2, 400)
                    for j in range(3):
                        # Maximal values in the search of lambda1 in MB estimation (phase 1)
                        if ds < 800:
                            lam1, lam2 = 0.3, 0.5
                        else:
                            lam1, lam2 = 0.1, 0.2
                        # run methods
                        mid=1
                        if len(jobs) <  1:
                            cmd = f"sbatch --ntasks={nt} run_dcdilp.slurm {str} {tid} {ds} {degs} {rnd} {gt} {st} {mid} {ds} {j} {FDIR} {verbo1} {verbo2} DCpool-DAGMA 0.05 ILP '6,2,10' oracle '0,0'"
                            jobnum = subprocess.check_output(cmd, shell= True)
                            # >>> jobnum
                            # >>> b'Submitted batch job 396932\n'
                            jobs.append(extract_numbers(jobnum.decode())[0])
                        else:
                            jobid = jobs[-1]
                            cmd = f"sbatch --dependency=afterany:{jobid} --ntasks={nt} run_dcdilp.slurm {str} {tid} {ds} {degs} {rnd} {gt} {st} {mid} {ds} {j} {FDIR} {verbo1} {verbo2} DCpool-DAGMA 0.05 ILP '6,2,10' oracle '0,0'"
                            jobnum = subprocess.check_output(cmd, shell= True)
                            jobs.append(extract_numbers(jobnum.decode())[0])
                        # Next method/job
                        mid+=1
                        jobid = jobs[-1]
                        cmd = f"sbatch --dependency=afterany:{jobid} --ntasks={nt} run_dcdilp.slurm {str} {tid} {ds} {degs} {rnd} {gt} {st} {mid} {ds} {j} {FDIR} {verbo1} {verbo2} DCpool-DAGMA 0.06 ILP '6,2,10' ice-emp '0.05,{lam2},20'"
                        jobnum = subprocess.check_output(cmd, shell= True)
                        jobs.append(extract_numbers(jobnum.decode())[0])
                        # Next method/job
                        mid+=1
                        jobid = jobs[-1]
                        cmd = f"sbatch --dependency=afterany:{jobid} --ntasks={nt} run_dcdilp.slurm {str} {tid} {ds} {degs} {rnd} {gt} {st} {mid} {ds} {j} {FDIR} {verbo1} {verbo2} DCpool-DAGMA 0.08 ILP '6,2,10' ice-emp '0.05,{lam1},20'"
                        jobnum = subprocess.check_output(cmd, shell= True)
                        jobs.append(extract_numbers(jobnum.decode())[0])



