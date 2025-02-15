#!/bin/bash

# Specify the output file
<<<<<<< HEAD
output_file="realworld.txt"
=======
output_file="modthreads_alllags.txt"
>>>>>>> dbcd26cf1c03fa63e3b35a016d198ab11ca5e773

> "$output_file"  # Clear the output file

# Run 100 iterations
for i in {1..100}; do
  # Capture individual results
<<<<<<< HEAD
  aout_result=$(./realworld | tail -n 2 | head -n 1)
=======
  aout_result=$(./a.out | tail -n 2 | head -n 1)
>>>>>>> dbcd26cf1c03fa63e3b35a016d198ab11ca5e773

  # Combine and write raw results
  echo "$aout_result" >> "$output_file"
done

# Calculate averages after all results are written
awk '{ sum1 += $1; count++ } END { print "Average:", sum1/count,"ms"}' "$output_file" >> "$output_file"

echo "Results stored in $output_file"