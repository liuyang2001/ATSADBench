#!/bin/bash

REPEATS=1

PARAM_SETS=(
    "--seq_len 30 --pred_len 1 --sample_p 0.2 --sparse_th 0.002 --n_block 5 --ff_dim 1024 --number=0"
    "--seq_len 70 --pred_len 1 --sample_p 0.2 --sparse_th 0.008 --n_block 6 --ff_dim 1024 --number=2"
    "--seq_len 5 --pred_len 1 --sample_p 0.2 --sparse_th 0.008 --n_block 6 --ff_dim 2048 --number=3"
    "--seq_len 30 --pred_len 1 --sample_p 0.2 --sparse_th 0.005 --n_block 3 --ff_dim 1024 --number=4"
)

TASK_NAME_SETS=(
    "my/M-IL-FVA M-IL-FVA"
    "my/M-IL-CDA M-IL-CDA"
    "my/M-IL-TVDA M-IL-TVDA"
    "my/M-OL-FVA M-OL-FVA"
    "my/M-OL-CDA M-OL-CDA"
    "my/M-OL-TVDA M-OL-TVDA"
    "my/U-FVA U-FVA"
    "my/U-CDA U-CDA"
    "my/U-TVDA U-TVDA"
)

COMMON_ARGS="--dropout 0 --learning_rate 0.0001 --seed 42 --target None --data_path dataset"
#!/bin/bash

mkdir -p GCAD/my_predictions
mkdir -p GCAD/my_outputs
mkdir -p GCAD/my_models
mkdir -p GCAD/my_final_result

for TASK_PAIR in "${TASK_NAME_SETS[@]}"; do
    NAME=$(echo "$TASK_PAIR" | awk '{print $1}')
    TASK_NAME=$(echo "$TASK_PAIR" | awk '{print $2}')
    echo "name=$NAME, task_name=$TASK_NAME"

    for ((param_idx=1; param_idx<=${#PARAM_SETS[@]}; param_idx++)); do
        PARAMS="${PARAM_SETS[$((param_idx-1))]}"
        echo "$param_idx"

        
        OUTPUT_FILE="GCAD/my_outputs/${TASK_NAME}_output${param_idx}.txt"
        > "$OUTPUT_FILE"

        PRED_FILE_PREFIX="GCAD/my_predictions/${TASK_NAME}_predictions${param_idx}"  
        echo "$PRED_FILE_PREFIX"
        echo " $OUTPUT_FILE"
        
        python GCAD/main.py $COMMON_ARGS $PARAMS --name "$NAME" --task_name "$TASK_NAME" --pred_file "$PRED_FILE_PREFIX" --output_file "$OUTPUT_FILE"

        python GCAD/average.py "$OUTPUT_FILE"
    done

    FILE_PATHS=""
    for ((param_idx=1; param_idx<=${#PARAM_SETS[@]}; param_idx++)); do
        FILE_PATHS="$FILE_PATHS GCAD/my_outputs/${TASK_NAME}_output${param_idx}.txt"
    done
    FINAL_OUTPUT="GCAD/my_final_result/${TASK_NAME}_final_metrics.txt"
    python GCAD/five_average.py $FILE_PATHS "$FINAL_OUTPUT"
done