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

    class BaseException : public std::exception
    {
    private:
        std::string _error;
        ExceptionTypes _exception;

    public:
        explicit BaseException(const std::string &message, ExceptionTypes exception) : _error(std::move(message)), _exception(exception) {}
        const char *what() const noexcept override { return _error.c_str(); }
    };

    class IOException : public BaseException
    {
    public:
        explicit IOException(const std::string &message) noexcept : BaseException("IOError : " + message, ExceptionTypes::IO) {}
    };

    class GeomException : public BaseException
    {
    public:
        explicit GeomException(const std::string &message) noexcept : BaseException("GeomError : " + message, ExceptionTypes::Geom) {}
    };

    class SCFException : public BaseException
    {
    public:
        explicit SCFException(const std::string &message) noexcept : BaseException("SCFError : " + message, ExceptionTypes::SCF) {}
    };

    class OptException : public BaseException
    {
    public:
        explicit OptException(const std::string &message) noexcept : BaseException("OptError : " + message, ExceptionTypes::Opt) {}
    };

};