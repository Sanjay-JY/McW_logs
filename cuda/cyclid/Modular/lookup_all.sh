#!/bin/bash

# Specify the output file
output_file="lookup_gpu.txt"

> "$output_file"  # Clear the output file

# Run 100 iterations
for i in {1..100}; do
  # Capture individual results
  aout_result=$(./lookupall | tail -n 2 | head -n 1)

  # Combine and write raw results
  echo "$aout_result" >> "$output_file"
done

# Calculate averages after all results are written
awk '{ sum1 += $1; count++ } END { print "Average:", sum1/count,"ms"}' "$output_file" >> "$output_file"

echo "Results stored in $output_file"