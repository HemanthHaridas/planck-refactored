#ifndef HF_NUMERIC_H
#define HF_NUMERIC_H

#include <array>
#include <cstddef>

template<std::size_t N>
constexpr std::array<double, N> generate_factorials()
{
    std::array<double, N> f{};

    f[0] = 1.0;

    for (std::size_t i = 1; i < N; ++i)
        f[i] = i * f[i - 1];

    return f;
}

inline constexpr auto factorials = generate_factorials<20>();

#endif // !HF_NUMERIC_H
