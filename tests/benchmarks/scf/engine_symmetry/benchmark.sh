#!/bin/bash

set -euo pipefail

lookup_pyscf_energy() {
    case "$1" in
        ethylene_casscf_321g_*)
            printf '%s\n' "-77.4195705536"
            ;;
        ethylene_cas44_sto3g_sa2_*)
            printf '%s\n' "-76.8522465545"
            ;;
        *)
            printf 'No PySCF RHF reference configured for %s\n' "$1" >&2
            return 1
            ;;
    esac
}

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
    printf "FILENAME\tELECTRONIC_ENERGY\tTOTAL_ENERGY\tERI_ENGINE\tSYMMETRY\tPYSCF_ENERGY\n"

    for file in *.hfinp
    do
        ../../../build/hartree-fock "$file" > "${file%.hfinp}.log"

        ELECT=$(grep 'Electronic Energy' "${file%.hfinp}.log" | awk '{print $3}')
        TOTAL=$(grep 'Total Energy' "${file%.hfinp}.log" | awk '{print $3}')

        ENGINE=$(grep 'engine' "$file" | awk '{print $2}')
        SYMM=$(grep 'use_symm' "$file" | awk '{print $2}')
        PYSCF=$(lookup_pyscf_energy "${file%.hfinp}")

        ELECT_FMT=$(printf '%s\n' "$ELECT" | format_decimal)
        TOTAL_FMT=$(printf '%s\n' "$TOTAL" | format_decimal)
        PYSCF_FMT=$(printf '%s\n' "$PYSCF" | format_decimal)

        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$file" "$ELECT_FMT" "$TOTAL_FMT" "$ENGINE" "$SYMM" "$PYSCF_FMT"
    done
} | column -s $'\t' -t
