#!/bin/bash

# Specify the output file
output_file="results.txt"

> "$output_file"  # Clear the output file

# Run 100 iterations
for i in {1..100}; do
  # Capture individual results
  aout_result=$(./a.out | tail -n 1)
  add_result=$(./add | tail -n 1)

  # Combine and write raw results
  echo "$aout_result $add_result" >> "$output_file"
done

# Calculate averages after all results are written
awk '{ sum1 += $1; sum2 += $2; count++ } END { print "Average:", sum1/count, sum2/count }' "$output_file" >> "$output_file"

echo "Results stored in $output_file"
