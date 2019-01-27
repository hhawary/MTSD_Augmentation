set -x
set -e


# example1: ./main.sh MTSD 1 1 66 4400
# example2: ./main.sh GTSDB 1 1 44 4400

if [ "$#" -ne 5 ]; then
	echo "$#"
	echo "USAGE: $0 \${DATASET} \${EXP} \${PHASE} \${NUM_CLS} \${NUM_SAMPLES}"
	exit 1
fi


DATASET=$1  
EXP=$2
PHASE=$3
NUM_CLS=$4
NUM_SAMPLES=$5


LOG_DIR="./results/${DATASET}_result${EXP}"

LOG_FILE="${LOG_DIR}/log_PHASE_${PHASE}_`date +'%Y-%m-%d_%H-%M-%S'`.txt"

if [ ! -d "$LOG_DIR" ]; then
	mkdir ${LOG_DIR}
fi

exec &> >(tee -a "$LOG_FILE")
echo "Logging output to $LOG_FILE"

echo "DATASET=$DATASET \nEXP=$EXP \nPHASE=$PHASE \nNUM_CLS=$NUM_CLS \nNUM_SAMPLES=$NUM_SAMPLES"



time python main.py ${DATASET} ${EXP} ${PHASE} ${NUM_CLS} ${NUM_SAMPLES}

echo "FINISHED..." 


#| tee -a ./results/${DATASET}_result${EXP}/log${PHASE}.log
