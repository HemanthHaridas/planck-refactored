#include "require_boys_library.h"

#include <iostream>
#include <string>

#include "lookup/loader/license_path.h"
#include "lookup/loader/orchestrator.h"

bool require_boys_library()
{
    const std::string blob_url_template =
        std::string("https://boys-blob-worker.boys-core.workers.dev/{version}/planck_boys-") +
        HartreeFock::Loader::get_platform_string() + ".enc";
    auto res = HartreeFock::Loader::initialize_boys_library(
        HartreeFock::Loader::get_license_path(), blob_url_template);
    if (!res)
    {
        std::cerr << "FAIL: License check failed: " << res.error() << '\n';
        return false;
    }
    return true;
}
