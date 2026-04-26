from .parser import ParsedRun, parse_log


def create_app(*args, **kwargs):
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["ParsedRun", "parse_log", "create_app"]
