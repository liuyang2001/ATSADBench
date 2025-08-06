#!/bin/bash

# run_tasks.bash

set -e

TASKS=("M-IL-FVA" "M-IL-CDA" "M-IL-TVDA" "M-OL-FVA" "M-OL-CDA" "M-OL-TVDA" "U-FVA" "U-CDA" "U-TVDA")
UNI_WINDOW_SIZES=(500)
MULTI_WINDOW_SIZES=(10)

# "generate" to obtain the LLM response
# "process" to obtain the final results and F1-score
STAGES=("generate" "process")

for task in "${TASKS[@]}"; do
    mkdir -p "${task}_logs" || { echo "Failed to create directory ${task}_logs"; exit 1; }
done

# Modify variables in config.py.
modify_config() {
    local window_size=$1
    local step_size=$2
    local stage=$3
    # Use sed to modify WINDOW_SIZE, STEP_SIZE, and stage in config.py
    sed -i'' -E "s/WINDOW_SIZE = [0-9]+/WINDOW_SIZE = $window_size/" config.py || { echo "Failed to modify WINDOW_SIZE in config.py"; exit 1; }
    sed -i'' -E "s/STEP_SIZE = [0-9]+/STEP_SIZE = $step_size/" config.py || { echo "Failed to modify STEP_SIZE in config.py"; exit 1; }
    sed -i'' -E "s/stage = \".*\"/stage = \"$stage\"/" config.py || { echo "Failed to modify stage in config.py"; exit 1; }
}

for task in "${TASKS[@]}"; do
    # Determine whether the task is univariate or multivariate
    if [[ $task == Uni-* ]]; then
        window_sizes=("${UNI_WINDOW_SIZES[@]}")
    else
        window_sizes=("${MULTI_WINDOW_SIZES[@]}")
    fi

    # Iterate over WINDOW_SIZE
    for window_size in "${window_sizes[@]}"; do
        # Calculate STEP_SIZE
        if [[ $task == Uni-* ]]; then
            step_size=$((window_size / 5))
        else
            step_size=$window_size
        fi

        # Iterate over stage
        for stage in "${STAGES[@]}"; do
            echo "Parameters for run:"
            echo "  Task: $task"
            echo "  WINDOW_SIZE: $window_size"
            echo "  STEP_SIZE: $step_size"
            echo "  stage: $stage"
            
            # Modify config.py
            modify_config $window_size $step_size $stage
            
            # Determine the log file name; add the _process suffix if the stage is "process"
            if [[ $stage == "process" ]]; then
                log_file="${task}_logs/${task}_qwen3-235b-a22b_${window_size}_process.log"
            else
                log_file="${task}_logs/${task}_qwen3-235b-a22b_${window_size}.log"
            fi
            
            echo "Running task: $task, WINDOW_SIZE: $window_size, STEP_SIZE: $step_size, stage: $stage, Log: $log_file"
            
            # Run the Python script and redirect output to the log file
            python -u main_all_bingfa.py --task "$task" --model qwen3-235b-a22b > "$log_file" 2>&1 || { echo "Python script failed for task $task, WINDOW_SIZE $window_size, stage $stage"; exit 1; }
            
            echo "Completed: $task, WINDOW_SIZE: $window_size, stage: $stage, Log: $log_file"
        done
    done
done

echo "All tasks completed!"