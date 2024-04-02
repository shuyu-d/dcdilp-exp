#!/bin/bash
#SBATCH -J Job_dcdilp
#SBATCH -p all
#SBATCH -A causal_structure_discovery
#SBATCH -t 23:50:00
#SBATCH -o ./%N.%x.out
#SBATCH -e ./%N.%x.err

# Déplacement sur le répertoire de travail

# Chargement des modules python
source ~/anaconda3/bin/activate py38-clone

# README
# This is an implementation of D&C algorithms for solving one single instance of the
# causal discovery problem.  
# 
# This script is based on run_dcalg.slurm with the following changelog: 
#  - run_dcges_options()    A new module, based on run_dncgrb_options(), to use
#                              dcgrb12_simple.py instead of dncgrb_10.py for
#                              the conquer step  


# MODULE FUNCTION of DC-GES algorithms 
# DC-GES    uses GES1 (slower implementation) 
# DC-GES2   uses pcalg's GES (faster implementation) 
function run_dcges2_options {
    str=$1
    tid=$2
    ds=$3
    degs=$4
    rnd=$5
    gt=$6
    st=$7
    mid=$8
    dv=$9
    j=${10}
    FDIR=${11}
    verbo1=${12}
    verbo2=${13}
    runwho=${14}
    opts=${15}
    crunwho=${16}
    copts=${17}
    mbfile=${18}
    mrunwho=${19}
    mb_time=${20}
    FDIRt=$FDIR/tid$tid/meth$mid 
    mkdir -p $FDIRt
    mkdir -p $FDIRt/locres_seed$j 
    echo "$FDIRt"
    SECONDS=0
    # TODO: modify the following print msg 
    echo "Method $runwho: (new constraint, Y choice, 3-cycle constraints) = '$copts'" >> "$FDIRt"/runlog_seed$j.txt 
    echo "Starting local tasks" >> "$FDIRt"/runlog_seed$j.txt 
    for ((i=0; i<$dv; i++)); do
        # NOTE (dev): replaced dc_local2.py by dcdilp_phase2.py 
        # srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 2 python dc_local2.py $runwho filename=$mbfile ds=$ds graph_type=$gt sem_type=$st degs=$degs rnd=$rnd VAR=$i SEED=$j fdir=$FDIRt fout=locres_seed$j verbo=$verbo1 >> $FDIRt/locres_seed$j/screenlog_var"$i".txt & 
        srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 2 python dcdilp_phase2.py $runwho filename=$mbfile ds=$ds graph_type=$gt sem_type=$st degs=$degs rnd=$rnd VAR=$i SEED=$j fdir=$FDIRt fout=locres_seed$j verbo=$verbo1 >> $FDIRt/locres_seed$j/screenlog_var"$i".txt & 
    done;
    wait
    tj=$SECONDS
    echo "SEED ($j): All local tasks finished" >> "$FDIRt"/runlog_seed$j.txt 
    echo "---------Elapsed Time: $tj seconds" >> "$FDIRt"/runlog_seed$j.txt 
    wait
    # 
    # NOTE (dev): replaced dcgrb12_simple.py by dcdilp_ILP1.py 
    # srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 1  python dcgrb12_simple.py $crunwho opts=$copts fdir=$FDIRt fout=locres_seed$j SEED=$j verbo=$verbo2 >> $FDIRt/runlog_seed$j.txt 
    srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 1  python dcdilp_ILP1.py $crunwho opts=$copts fdir=$FDIRt fout=locres_seed$j SEED=$j verbo=$verbo2 >> $FDIRt/runlog_seed$j.txt 
    wait 
    tj2=$SECONDS
    # Get scores from the last two lines of runlog_seed*.txt
    scores_final=$(tail -n 1 "$FDIRt/runlog_seed$j.txt")
    scores_concat=$(tail -n 2 "$FDIRt/runlog_seed$j.txt" | head -n 1)
    echo "Conquer step finished" >> "$FDIRt"/runlog_seed$j.txt 
    echo "Elapsed Time: $tj2 seconds" >> "$FDIRt"/runlog_seed$j.txt 
    algcode="$mid"LP_"$(echo "$opts" | tr ',' '-')" 
    # Res of the DC result  
    echo "tid,d,deg,rnd,graph_type,sem_type,mid,dv,SEED,MBinfo,alg,algspec,shd,tpr,fdr,fpr,nnz,hval,time,time_mb,time_locres" > $FDIR/res_job$SLURM_JOB_ID.txt
    echo -n "$tid,$ds,$degs,$rnd,$gt,$st,$mid,$dv,$j,$mrunwho,$runwho,$algcode," >> $FDIR/res_job$SLURM_JOB_ID.txt
    # Running time of phase 3 is now the last entry of scores_final 
    echo "$scores_final,$mb_time,$tj" >> $FDIR/res_job$SLURM_JOB_ID.txt 
    # Res of Concat 
    echo "tid,d,deg,rnd,graph_type,sem_type,mid,dv,SEED,MBinfo,alg,algspec,shd,tpr,fdr,fpr,nnz,hval,time_mb,time_locres,time" > $FDIR/resConcat_job$SLURM_JOB_ID.txt
    echo -n "$tid,$ds,$degs,$rnd,$gt,$st,$mid,$dv,$j,$mrunwho,DC*Concat,na," >> $FDIR/resConcat_job$SLURM_JOB_ID.txt
    echo "$scores_concat,$mb_time,$tj,$tj" >> $FDIR/resConcat_job$SLURM_JOB_ID.txt 
}



# DC-DAGMA and more  
# NOTE: the input argument opts is used for 'LAM=$opts' 
function run_dcmore_options {
    str=$1
    tid=$2
    ds=$3
    degs=$4
    rnd=$5
    gt=$6
    st=$7
    mid=$8
    dv=$9
    j=${10}
    FDIR=${11}
    verbo1=${12}
    verbo2=${13}
    runwho=${14}
    opts=${15}
    crunwho=${16}
    copts=${17}
    mbfile=${18}
    mrunwho=${19}
    mb_time=${20}
    FDIRt=$FDIR/tid$tid/meth$mid 
    mkdir -p $FDIRt
    mkdir -p $FDIRt/locres_seed$j 
    echo "$FDIRt"
    SECONDS=0
    # 
    echo "Method $runwho: (lambda1 = $opts)" >> "$FDIRt"/runlog_seed$j.txt 
    echo "Starting local tasks" >> "$FDIRt"/runlog_seed$j.txt 
    for ((i=0; i<$dv; i++)); do
        #
        # srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 2  python dc_local2.py $runwho filename=$mbfile ds=$ds graph_type=$gt sem_type=$st degs=$degs rnd=$rnd VAR=$i LAM=$opts SEED=$j fdir=$FDIRt fout=locres_seed$j verbo=$verbo1 > $FDIRt/locres_seed$j/screenlog_var"$i".txt & 
        srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 2  python dcdilp_phase2.py $runwho filename=$mbfile ds=$ds graph_type=$gt sem_type=$st degs=$degs rnd=$rnd VAR=$i LAM=$opts SEED=$j fdir=$FDIRt fout=locres_seed$j verbo=$verbo1 > $FDIRt/locres_seed$j/screenlog_var"$i".txt & 
    done;
    wait
    tj=$SECONDS
    echo "SEED ($j): All local tasks finished" >> "$FDIRt"/runlog_seed$j.txt 
    echo "---------Elapsed Time: $tj seconds" >> "$FDIRt"/runlog_seed$j.txt 
    # Conquer phase using dncgrb_10 (non-pool version)  
    wait
    # 
    # srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 1 python dcgrb12_simple.py $crunwho opts=$copts fdir=$FDIRt fout=locres_seed$j SEED=$j verbo=$verbo2 >> $FDIRt/runlog_seed$j.txt 
    srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 1 python dcdilp_ILP1.py $crunwho opts=$copts fdir=$FDIRt fout=locres_seed$j SEED=$j verbo=$verbo2 >> $FDIRt/runlog_seed$j.txt 
    wait 
    tj2=$SECONDS
    # Get scores from the last line of runlog_seed*.txt
    scores_final=$(tail -n 1 "$FDIRt/runlog_seed$j.txt")
    scores_concat=$(tail -n 2 "$FDIRt/runlog_seed$j.txt" | head -n 1)
    echo "Conquer step finished" >> "$FDIRt"/runlog_seed$j.txt 
    echo "Elapsed Time: $tj2 seconds" >> "$FDIRt"/runlog_seed$j.txt 
    # 
    algcode="$mid"LP_"$(echo "$opts" | tr ',' '-')" 
    # Res of the DC result  
    # 
    echo "tid,d,deg,rnd,graph_type,sem_type,mid,dv,SEED,MBinfo,alg,algspec,shd,tpr,fdr,fpr,nnz,hval,time,time_mb,time_locres" > $FDIR/res_job$SLURM_JOB_ID.txt
    echo -n "$tid,$ds,$degs,$rnd,$gt,$st,$mid,$dv,$j,$mrunwho,$runwho,$algcode," >> $FDIR/res_job$SLURM_JOB_ID.txt
    # Running time of phase 3 is now the last entry of scores_final 
    echo "$scores_final,$mb_time,$tj" >> $FDIR/res_job$SLURM_JOB_ID.txt 
    # Res of Concat 
    echo "tid,d,deg,rnd,graph_type,sem_type,mid,dv,SEED,MBinfo,alg,algspec,shd,tpr,fdr,fpr,nnz,hval,time_mb,time_locres,time" > $FDIR/resConcat_job$SLURM_JOB_ID.txt
    echo -n "$tid,$ds,$degs,$rnd,$gt,$st,$mid,$dv,$j,$mrunwho,DC*Concat,na," >> $FDIR/resConcat_job$SLURM_JOB_ID.txt
    echo "$scores_concat,$mb_time,$tj,$tj" >> $FDIR/resConcat_job$SLURM_JOB_ID.txt 
}

# DC-DAGMA and more using (LP15, etc) 
# The module below is based on sl_dncgrb_2.sh/run_dncgrb_pool2 with changelog: 
#   - Adaptation for standardized input/output for cddc experiments 
function run_dcdilp_pool2 {
    str=$1
    tid=$2
    ds=$3
    degs=$4
    rnd=$5
    gt=$6
    st=$7
    mid=$8
    dv=$9
    j=${10}
    FDIR=${11}
    verbo1=${12} 
    verbo2=${13} 
    runwho=${14}
    opts=${15}
    crunwho=${16}
    copts=${17}
    mbfile=${18}
    mrunwho=${19}
    mb_time=${20}
    FDIRt=$FDIR/tid$tid/meth$mid 
    mkdir -p $FDIRt
    mkdir -p $FDIRt/locres_seed$j 
    echo "$FDIRt"
    SECONDS=0
    echo "Method $runwho (DCDILP $opts)" >> "$FDIRt"/runlog_seed$j.txt 
    echo "Starting local tasks" >> "$FDIRt"/runlog_seed$j.txt 
    for ((i=0; i<$dv; i++)); do
        # srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 2 python dc_local2.py $runwho filename=$mbfile ds=$ds graph_type=$gt sem_type=$st degs=$degs rnd=$rnd VAR=$i LAM=$opts SEED=$j fdir=$FDIRt fout=locres_seed$j verbo=$verbo1 > $FDIRt/locres_seed$j/screenlog_var"$i".txt & 
        srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 2 python dcdilp_phase2.py $runwho filename=$mbfile ds=$ds graph_type=$gt sem_type=$st degs=$degs rnd=$rnd VAR=$i LAM=$opts SEED=$j fdir=$FDIRt fout=locres_seed$j verbo=$verbo1 > $FDIRt/locres_seed$j/screenlog_var"$i".txt & 
    done;
    wait
    tj=$SECONDS
    echo "SEED ($j): All local tasks finished" >> "$FDIRt"/runlog_seed$j.txt 
    echo "---------Elapsed Time: $tj seconds" >> "$FDIRt"/runlog_seed$j.txt 
    wait
    #   IDEA: encode poolsearch options in $opts: currently, it is 1,4,0 for LP14. 
    #          We can extend it as opts=1,4,0,2,20 where the last two numbers are for the custom values of PoolSearchMode and PoolSolutions
    # NOTE: replace dcgrb13_newio.py by dcdilp_ILP2.py 
    # srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 1 python dcgrb13_newio.py $crunwho opts=$copts fdir=$FDIRt fout=locres_seed$j SEED=$j verbo=$verbo2 >> $FDIRt/runlog_seed$j.txt 
    srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 1 python dcdilp_ILP2.py $crunwho opts=$copts fdir=$FDIRt fout=locres_seed$j SEED=$j verbo=$verbo2 >> $FDIRt/runlog_seed$j.txt 
    wait 
    tj2=$SECONDS
    # Get scores from the last line of runlog_seed*.txt
    scores_final=$(tail -n 1 "$FDIRt/runlog_seed$j.txt")
    scores_concat=$(tail -n 2 "$FDIRt/runlog_seed$j.txt" | head -n 1)
    echo "Conquer step finished" >> "$FDIRt"/runlog_seed$j.txt 
    echo "Elapsed Time: $tj2 seconds" >> "$FDIRt"/runlog_seed$j.txt 
    # 
    algcode="$mid"LP_"$(echo "$opts" | tr ',' '-')" 
    # Res of the DC result  
    echo "tid,d,deg,rnd,graph_type,sem_type,mid,dv,SEED,MBinfo,alg,algspec,shd,tpr,fdr,fpr,nnz,hval,time,time_mb,time_locres" > $FDIR/res_job$SLURM_JOB_ID.txt
    echo -n "$tid,$ds,$degs,$rnd,$gt,$st,$mid,$dv,$j,$mrunwho,$runwho,$algcode," >> $FDIR/res_job$SLURM_JOB_ID.txt
    # Running time of phase 3 is now the last entry of scores_final 
    echo "$scores_final,$mb_time,$tj" >> $FDIR/res_job$SLURM_JOB_ID.txt 
    # Res of Concat 
    echo "tid,d,deg,rnd,graph_type,sem_type,mid,dv,SEED,MBinfo,alg,algspec,shd,tpr,fdr,fpr,nnz,hval,time_mb,time_locres,time" > $FDIR/resConcat_job$SLURM_JOB_ID.txt
    echo -n "$tid,$ds,$degs,$rnd,$gt,$st,$mid,$dv,$j,$mrunwho,DC*Concat,na," >> $FDIR/resConcat_job$SLURM_JOB_ID.txt
    echo "$scores_concat,$mb_time,$tj,$tj" >> $FDIR/resConcat_job$SLURM_JOB_ID.txt 
}


function run_MB_discovery {
    str=$1
    tid=$2
    ds=$3
    degs=$4
    rnd=$5
    gt=$6
    st=$7
    mid=$8
    dv=$9
    j=${10}
    FDIR=${11}
    verbo1=${12}
    verbo2=${13}
    runwho=${14}
    opts=${15}
    FDIRt=$FDIR/tid$tid/meth$mid 
    mkdir -p $FDIRt
    echo "$FDIRt"
    SECONDS=0
    # 
    echo "Method $runwho: (lambda1 = $opts)" >> "$FDIRt"/mb_runlog_seed$j.txt 
    # NOTE: replaced dc_mbdiscovery.py by dcdilp_Ph1MB1.py 
    # srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 2 python dc_mbdiscovery.py $runwho opts=$opts ds=$ds graph_type=$gt sem_type=$st degs=$degs rnd=$rnd SEED=$j fdir=$FDIRt fout=locres_seed$j filename="MBs.npz" verbo=$verbo1 >> $FDIRt/mb_runlog_seed$j.txt 
    srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task 2 python dcdilp_Ph1MB1.py $runwho opts=$opts ds=$ds graph_type=$gt sem_type=$st degs=$degs rnd=$rnd SEED=$j fdir=$FDIRt fout=locres_seed$j filename="MBs.npz" verbo=$verbo1 >> $FDIRt/mb_runlog_seed$j.txt 
    tj=$SECONDS
    scores=$(tail -n 1 "$FDIRt/mb_runlog_seed$j.txt")
    echo "---------Elapsed Time: $tj seconds" >> "$FDIRt"/mb_runlog_seed$j.txt 
    # print to external txt file
    echo "tid,d,deg,rnd,graph_type,sem_type,mid,dv,SEED,MBinfo,alg,algspec,shd,tpr,fdr,fpr,nnz,time_mb" > $FDIR/resMB_job$SLURM_JOB_ID.txt
    echo -n "$tid,$ds,$degs,$rnd,$gt,$st,$mid,$dv,$j,$mrunwho,DC,NA," >> $FDIR/resMB_job$SLURM_JOB_ID.txt
    if [ $mrunwho = "oracle" ] ; then
        echo ",,,,,$tj" >> $FDIR/resMB_job$SLURM_JOB_ID.txt 
    else
        echo "$scores,$tj" >> $FDIR/resMB_job$SLURM_JOB_ID.txt 
    fi
    # 
    echo "$FDIRt/locres_seed$j/MBs.npz"
    echo "$tj"
}


# MAIN 
str=$1 
tid=$2
ds=$3
degs=$4
rnd=$5
gt=$6
st=$7
mid=$8
dv=$9
j=${10}
FDIR=${11}
verbo1=${12}
verbo2=${13}
runwho=${14}    # local cd method
opts=${15}
crunwho=${16}   # merge algorithm
copts=${17}
mrunwho=${18}   # mb discovery method
mopts=${19}

# Phase 1: Run Markov blanket discovery 
out=($(run_MB_discovery $str $tid $ds $degs $rnd $gt $st $mid $dv $j $FDIR 2 2 $mrunwho $mopts)) 
wait


# Phase 2 -- Phase 3: Run DC algorithm 
mbfile=${out[1]}
mb_time=${out[2]}

if [ $runwho = "DC-GES" ] ; then
    # ----GES
    out=($(run_dcges_options $str $tid $ds $degs $rnd $gt $st $mid $dv $j $FDIR $verbo1 $verbo2 $crunwho $copts $mbfile $mrunwho $mb_time))
    wait
fi

if [ $runwho = "DC-GES2" ] ; then
    # ----GES
    out=($(run_dcges2_options $str $tid $ds $degs $rnd $gt $st $mid $dv $j $FDIR $verbo1 $verbo2 $runwho $opts $crunwho $copts $mbfile $mrunwho $mb_time))
    wait
fi
if [[ "$runwho" =~ ^(DC-DAGMA|DC-BLAH|DC-BLA)$ ]]; then
    out=($(run_dcmore_options $str $tid $ds $degs $rnd $gt $st $mid $dv $j $FDIR $verbo1 $verbo2 $runwho $opts $crunwho $copts $mbfile $mrunwho $mb_time))
    wait
fi
if [[ "$runwho" =~ ^(DCpool-DAGMA|DCpool-GES2|DCpool-BLA)$ ]]; then
    out=($(run_dcdilp_pool2    $str $tid $ds $degs $rnd $gt $st $mid $dv $j $FDIR $verbo1 $verbo2 $runwho $opts $crunwho $copts $mbfile $mrunwho $mb_time))
    wait
fi


# 3: Utility: 
echo $SLURM_JOB_ID >> $FDIR/tid$tid/JobIds.txt 

# Remove datamat.npy 
rm $FDIR/tid$tid/meth$mid/locres_seed$j/datamat.npy  
if [ $verbo1 -lt 2 ] ; then
    rm $FDIR/tid$tid/meth$mid/locres_seed$j/*.npz  
fi


