#!/bin/bash

export OMP_NUM_THREADS=4

format_decimal() {
    awk '
    {
        split($0, a, ".")
        whole = a[1]
        frac  = a[2]

        out = ""
        for (i = 1; i <= length(frac); i++) {
            out = out substr(frac, i, 1)
            if (i % 3 == 0 && i < length(frac)) {
                out = out " "
            }
        }

        if (frac != "")
            printf "%s.%s\n", whole, out
        else
            printf "%s\n", whole
    }'
}

{
    printf "FILENAME\tELECTRONIC_ENERGY\tTOTAL_ENERGY\tERI_ENGINE\tSYMMETRY\tCASSCF_ENERGY\n"

    for file in *.hfinp
    do
        ../../../build/hartree-fock "$file" > "${file%.hfinp}.log" 2>&1

        ELECT=$(grep 'Electronic Energy' "${file%.hfinp}.log" | awk '{print $3}')
        TOTAL=$(grep 'Total Energy' "${file%.hfinp}.log" | grep -v 'CASSCF' | awk '{print $3}')
	
	# check if casscf converged
    	if grep -q "\[INF\] CASSCF : *Converged\." "${file%.hfinp}.log"; then
		CASSCF=$(grep 'CASSCF Total Energy' "${file%.hfinp}.log" | awk '{print $4}')
	fi

        ENGINE=$(grep 'engine' "$file" | awk '{print $2}')
        SYMM=$(grep 'use_symm' "$file" | awk '{print $2}')

        ELECT_FMT=$(printf '%s\n' "$ELECT" | format_decimal)
        TOTAL_FMT=$(printf '%s\n' "$TOTAL" | format_decimal)
	    CASSCF_FMT=$(printf '%s\n' "$CASSCF" | format_decimal)

        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$file" "$ELECT_FMT" "$TOTAL_FMT" "$ENGINE" "$SYMM" "$CASSCF_FMT"
    done
} | column -s $'\t' -t
