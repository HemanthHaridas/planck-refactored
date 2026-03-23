# Helper script called by the fetch-basis-sets target.
# Variables passed via -D:
#   BSE_NAME  — basis set name as listed in basis-sets/basis
#   BSE_OUT   — absolute path to the output file

execute_process(
    COMMAND bse get-basis ${BSE_NAME} gaussian94
    OUTPUT_FILE  ${BSE_OUT}
    ERROR_VARIABLE bse_err
    RESULT_VARIABLE bse_result
)

if(bse_result)
    file(REMOVE ${BSE_OUT})
    message(FATAL_ERROR "bse get-basis ${BSE_NAME} failed:\n${bse_err}")
endif()
