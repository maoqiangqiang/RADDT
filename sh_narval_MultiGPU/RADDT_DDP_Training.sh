#!/bin/bash
#SBATCH --account=def-xxxxxx
#SBATCH --nodes=2   
#SBATCH --gpus-per-node=a100:4            
#SBATCH --ntasks-per-node=4 
#SBATCH --cpus-per-task=1 
#SBATCH --array=23
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=log/a_RADDT_DDP_%j_%N.out      


# go to the directory where the job will run
# cd $SLURM_SUBMIT_DIR
module load StdEnv/2020 python/3.9.6
source /home/xxxxxx/projects/def-xxxxxx/xxxxxx/py396Env/bin/activate


export NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((12345 + $SLURM_JOB_ID % 10000))

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "Rank $SLURM_PROCID starting on $SLURMD_NODENAME (MASTER_ADDR: $MASTER_ADDR, PORT: $MASTER_PORT)"


srun python test/test_RADDT_DDP.py --init_method tcp://$MASTER_ADDR:$MASTER_PORT --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) --dataNumStart $((${SLURM_ARRAY_TASK_ID})) --dataNumEnd $((${SLURM_ARRAY_TASK_ID})) --runsNumStart 1 --runsNumEnd 1 --treeDepth 12 --epochNum 3000 --device "cuda" --startNum 10 --numScale 5 --csvOutputFlag 1 > log/RADDT_DDP_Data$((${SLURM_ARRAY_TASK_ID}))_$((${SLURM_ARRAY_TASK_ID}))_Run1_1_D12_Epoch3000_nScale{10}_cuda_{$SLURM_JOBID}.log 







