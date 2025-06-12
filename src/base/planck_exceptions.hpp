#pragma once

#include <exception>
#include <string>

namespace Planck::Exceptions
{
    enum class ExceptionTypes
    {
        IO,
        Geom,
        SCF,
        Opt
    };

    // Base class for handling excpetions
    // Takes a string as the error message and an error category
    class BaseException : public std::exception
    {
    private:
        std::string _error;        // error message
        ExceptionTypes _exception; // error category

    public:
        BaseException() = default; // default constructor

        // explicit constuctor to construct the error message
        explicit BaseException(const std::string &message, ExceptionTypes exception) : _error(std::move(message)), _exception(exception) {}

        // override to return the error message as a c-type string
        const char *what() const noexcept override { return _error.c_str(); }
    };

    // class to handle exceptions in IO routines
    class IOException : public BaseException
    {
    public:
        explicit IOException(const std::string &message) noexcept : BaseException("IOError : " + message, ExceptionTypes::IO) {}
    };

    // class to handle excpetions in geometry
    class GeomException : public BaseException
    {
    public:
        explicit GeomException(const std::string &message) noexcept : BaseException("GeomError : " + message, ExceptionTypes::Geom) {}
    };

    // class to handle exceptions in SCF routines
    class SCFException : public BaseException
    {
    public:
        explicit SCFException(const std::string &message) noexcept : BaseException("SCFError : " + message, ExceptionTypes::SCF) {}
    };

    // class to handle exceptions in Optimization routines
    class OptException : public BaseException
    {
    public:
        explicit OptException(const std::string &message) noexcept : BaseException("OptError : " + message, ExceptionTypes::Opt) {}
    };

};