#!/bin/bash

mkdir -p sub_adjacent_transformer/my_predictions
mkdir -p sub_adjacent_transformer/my_outputs
mkdir -p sub_adjacent_transformer/my_models
mkdir -p sub_adjacent_transformer/my_final_result
mkdir -p sub_adjacent_transformer/my_tmp
mkdir -p sub_adjacent_transformer/my_logs  

datasets=(
    "M-IL-FVA" "M-IL-CDA" "M-IL-TVDA"
    "M-OL-FVA" "M-OL-CDA" "M-OL-TVDA"
    "U-CDA" "U-FVA" "U-TVDA"
)

total_experiments=0
for dataset in "${datasets[@]}"; do

    output_files=()
        
    if [[ $dataset == M-* ]]; then
        input_c=27
        output_c=27
    else
        input_c=1
        output_c=1
    fi

    output_file="sub_adjacent_transformer/aaa_my_outputs/${dataset}_output1.txt"
    output_files+=("$output_file")

    for times in {0..9}; do
        echo "Running experiment: dataset=$dataset, times=$times"

        echo "Training..."
        python sub_adjacent_transformer/main.py \
            --anormly_ratio 0.5 \
            --num_epochs 100 \
            --batch_size 256 \
            --mode train \
            --dataset "$dataset" \
            --data_path "dataset" \
            --input_c "$input_c" \
            --output_c "$output_c" \
            --times "$times" \
            --config_idx 1 \
            --model_save_path "sub_adjacent_transformer/my_models"


        echo "Testing..."
        python sub_adjacent_transformer/main.py \
            --anormly_ratio 0.5 \
            --num_epochs 100 \
            --batch_size 256 \
            --mode test \
            --dataset "$dataset" \
            --data_path "dataset" \
            --input_c "$input_c" \
            --output_c "$output_c" \
            --pretrained_model 20 \
            --times "$times" \
            --config_idx 1

        total_experiments=$((total_experiments + 1))
        echo "Completed $total_experiments / 90 experiments"
    done

    echo "Calculating average for $output_file"
    python average.py "$output_file"
    echo "Average metrics appended to $output_file"
    final_output_file="sub_adjacent_transformer/aaa_my_final_result/${dataset}_final_metrics.txt"
    echo "Calculating final average for dataset $dataset"
    python five_average.py \
        "${output_files[0]}" \
        "${output_files[1]}" \
        "${output_files[2]}" \
        "${output_files[3]}" \
        "${output_files[4]}" \
        "$final_output_file"
    echo "Final average metrics written to $final_output_file"
done

echo "All 90 experiments completed!"