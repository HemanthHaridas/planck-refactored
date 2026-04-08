#ifndef HF_BASIS_DEF_H
#define HF_BASIS_DEF_H

#include <string>
#include <cstdlib>

inline std::string get_basis_path()
{
    // Check environment variable override
    const char *env_path = std::getenv("BASIS_PATH");
    if (env_path && *env_path)
    {
        return std::string(env_path);
    }
    // Fallback to compiled-in install path
    return "/Users/hemanthharidas/Desktop/codes/planck-refactored/install/share/basis-sets";
}

#endif // !HF_BASIS_DEF_H
