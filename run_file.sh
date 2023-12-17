TASK='self_supervised'
INPUT_SIZE=40
NUM_LAYERS=2
DROPOUT=0.2
BIDIRECTIONAL='True'
MODE='binary'
HIDDEN_SIZE=16
KERNEL_SIZE=3
NUM_STATES=9
RESOLUTION=2
BATCH_SIZE=5 ## was 35
TOTAL_EPOCHS=50
ALPHA=0.85 ## have tried 0.2 ,0,7, 0.85
AVG='True'
LAMBDA=0.0005 ## used to be 0.2
T=100 ## have tried 10, 100, 1000
K=$BATCH_SIZE
VERBOSE='False'
if [$TASK == 'self_supervised']
then
	python3 main.py --task $TASK --input_size $INPUT_SIZE --num_layers $NUM_LAYERS --dropout $DROPOUT --bidirectional $BIDIRECTIONAL --mode $MODE --hidden_size $HIDDEN_SIZE --kernel_size $KERNEL_SIZE --num_states $NUM_STATES --resolution $RESOLUTION --batch_size $BATCH_SIZE --total_epochs $TOTAL_EPOCHS --alpha $ALPHA --avg $AVG --lambda $LAMBDA --t $T --k $K --verbose $VERBOSE
fi