#!/usr/bin/env python

import os
import sys
import re
from setuptools import setup

# Utility functions so that we can populate the package description and the
# version number automatically.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def find_version(fname):
    version_file = read(fname)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# copied from https://github.com/samuelcolvin/pydantic/blob/master/setup.py
if os.name == 'nt':
    from setuptools.command import build_ext

    def get_export_symbols(self, ext):
        """
        Slightly modified from:
        https://github.com/python/cpython/blob/8849e5962ba481d5d414b3467a256aba2134b4da\
        /Lib/distutils/command/build_ext.py#L686-L703
        """
        # Patch from: https://bugs.python.org/issue35893
        parts = ext.name.split('.')
        if parts[-1] == '__init__':
            suffix = parts[-2]
        else:
            suffix = parts[-1]

        # from here on unchanged
        try:
            # Unicode module name support as defined in PEP-489
            # https://www.python.org/dev/peps/pep-0489/#export-hook-name
            suffix.encode('ascii')
        except UnicodeEncodeError:
            suffix = 'U' + suffix.encode('punycode').replace(b'-', b'_').decode('ascii')

        initfunc_name = 'PyInit_' + suffix
        if initfunc_name not in ext.export_symbols:
            ext.export_symbols.append(initfunc_name)
        return ext.export_symbols

    build_ext.build_ext.get_export_symbols = get_export_symbols


ext_modules = None
if not any(arg in sys.argv for arg in ['clean', 'check']) and 'SKIP_CYTHON' not in os.environ:
    try:
        from Cython.Build import cythonize
    except ImportError:
        pass
    else:
        # For cython test coverage install with `make build-trace`
        compiler_directives = {}
        if 'CYTHON_TRACE' in sys.argv:
            compiler_directives['linetrace'] = True
        # Set CFLAG to all optimizations (-O3)
        # Any additional CFLAGS will be appended. Only the last optimization flag will have effect
        os.environ['CFLAGS'] = '-O3 ' + os.environ.get('CFLAGS', '')
        ext_modules = cythonize(
            'clorm/**/*.py',
            exclude=['clorm/orm/lark_fact_parser.py', 'clorm/orm/types.py'],
            nthreads=int(os.getenv('CYTHON_NTHREADS', 0)),
            language_level=3,
            compiler_directives=compiler_directives,
        )


setup(
    name="Clorm",
    version=find_version("clorm/__init__.py"),
    author="David Rajaratnam",
    author_email="daver@gemarex.com.au",
    description="Clingo ORM (CLORM) provides a ORM interface for interacting with the Clingo Answer Set Programming (ASP) solver",
    license="MIT",
    url="https://github.com/potassco/clorm",
    packages=["clorm","clorm.orm","clorm.util","clorm.lib"],
    install_requires=['clingo'] if sys.version_info >= (3, 8) else ['clingo', 'typing_extensions'],
    long_description=read("README.rst"),
    ext_modules=ext_modules,
)
