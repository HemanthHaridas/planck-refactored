---
name: Error Handling Pattern
description: std::expected<T, std::string> used throughout — no exceptions in hot path
type: gotcha
priority: medium
include_in_claude: true
tags: [error-handling, expected, gotcha, c++23]
---

# Error Handling Pattern

## Convention

All public functions return `std::expected<T, std::string>`. The error type is always `std::string` (a human-readable message).

```cpp
// Return type pattern
std::expected<DataSCF, std::string> scf::run(Calculator& calc);

// Propagating errors
auto result = scf::run(calc);
if (!result) return std::unexpected(result.error());
DataSCF data = *result;

// Monadic chaining (C++23)
auto data = scf::run(calc)
    .and_then([](DataSCF d) { return mp2::compute(d); })
    .transform([](MP2Result r) { return r.energy; });
```

## Why No Exceptions

The hot path (ERI loops, SCF iterations) must not throw. Exceptions add overhead and obscure control flow. `std::expected` makes failure handling visible at every call site and composable via `.and_then` / `.transform`.

## When Adding New Functions

Any new function that can fail (file I/O, allocation, convergence failure, invalid input) should return `std::expected<T, std::string>`. Do not return bool + out-param or throw. Do not use `std::optional` for errors (it loses the message).

## Logging vs Returning Errors

Use the logging system (`src/io/logging.h`) for diagnostic output at the call site. Return the error string upward via `std::unexpected(...)` for the caller to handle. The top-level driver (`src/driver.cpp`) catches the final error and prints it to stderr.
