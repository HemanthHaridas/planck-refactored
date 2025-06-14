#include <iostream>

#include "planck_input.hpp"

int main(int argc, const char *argv[])
{
    try
    {
        if (argc < 2)
        {
            throw Planck::Exceptions::IOException("No input file was found");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    std::string input_ = argv[1];
    try
    {
        Planck::IO::InputReader _input(input_);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}
