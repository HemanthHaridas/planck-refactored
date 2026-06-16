#!/bin/bash

for file in *.hfinp
do
    echo "Running $file"

    logfile="${file%.hfinp}.restart.log"

    ../../../build/hartree-fock "$file" > "$logfile" 2>&1

    # Check convergence
    if grep -q "\[INF\] CASSCF : *Converged\." "$logfile"; then
        # Extract energy (4th column from matching line)
        energy=$(grep "CASSCF Total Energy" "$logfile" | awk '{print $4}')

        echo "✔ $file converged | Energy: $energy"
    else
        echo "✘ $file did NOT converge"
    fi
done
