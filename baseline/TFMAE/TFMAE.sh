#!/bin/bash

mkdir -p TFMAE/my_logs
mkdir -p TFMAE/my_predictions
mkdir -p TFMAE/my_outputs
mkdir -p TFMAE/my_models
mkdir -p TFMAE/my_final_result

datasets=(
    "M-IL-FVA" "M-IL-CDA" "M-IL-TVDA" "M-OL-FVA" "M-OL-CDA" "M-OL-TVDA"
    "U-CDA" "U-FVA" "U-TVDA"
)

total_experiments=0
for dataset in "${datasets[@]}"; do
    output_files=()

    for config_id in {1..5}; do
        config_file="TFMAE/configs/config${config_id}.conf"
        output_file="TFMAE/aaa_my_outputs/${dataset}_output${config_id}.txt"
        output_files+=("$output_file")

        for times in {0..9}; do
            echo "Running experiment: dataset=$dataset, config_id=$config_id, times=$times"
            python TFMAE/main.py --config "$config_file" --times $times --dataset "$dataset"
            total_experiments=$((total_experiments + 1))
            echo "Completed $total_experiments"
        done

        echo "Calculating average for $output_file"
        python TFMAE/average.py "$output_file"
        echo "Average metrics appended to $output_file"
    done

    final_output_file="TFMAE/aaa_my_final_result/${dataset}_final_metrics.txt"
    echo "Calculating final average for dataset $dataset"
    python TFMAE/five_average.py "${output_files[0]}" "${output_files[1]}" "${output_files[2]}" "${output_files[3]}" "${output_files[4]}" "$final_output_file"
    echo "Final average metrics written to $final_output_file"
done

echo "All experiments completed!"