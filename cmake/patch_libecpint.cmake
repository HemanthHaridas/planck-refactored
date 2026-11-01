# libecpint's code generator writes the type-2 angular prefactors into the
# generated Q*.cpp kernels at default ostream precision (6 digits), e.g. 157.914
# for 16 pi^2 = 157.91367041742973. Every off-centre semilocal integral is then
# off by a constant ~2e-6 relative (docs/DEF2_ECP_SCOPE.md step 4). Write them at
# full double precision instead. Run from the source dir by ExternalProject.
set(file src/generate.cpp)
file(READ ${file} text)
set(old "std::ofstream outfile(ofname);")
set(new "std::ofstream outfile(ofname);\n  outfile.precision(17);")
string(FIND "${text}" "outfile.precision(17)" done)
if(done EQUAL -1)
    string(FIND "${text}" "${old}" at)
    if(at EQUAL -1)
        message(FATAL_ERROR "patch_libecpint: '${old}' not found in ${file}")
    endif()
    string(REPLACE "${old}" "${new}" text "${text}")
    file(WRITE ${file} "${text}")
endif()
