'''A plugin replacement for the ``clingo`` modules that wraps the key clingo
objects to integrate Clorm ``Predicate`` and ``FactBase`` interfaces. See the
`Clingo API <https://potassco.org/clingo/python-api/current/clingo.html#Model>`_
for more details.

'''




# I want to replace the original clingo - re-exporting everything in clingo
# except replacing the class overides with my version: _class_overides = [
# 'Control', 'Model', 'SolveHandle' ]. The following seems to work but I'm not
# sure if this is bad.

# ------------------------------------------------------------------------------
# Reference to the original clingo objects so that when we replace them our
# references point to the originals.
# ------------------------------------------------------------------------------
from .orm import *
import clingo as oclingo

if oclingo.__version__ >= "5.5.0":
    from clingo.ast import parse_string
else:
    from clingo import parse_program  # type: ignore

from clingo import *
from ._clingo import *

__all__ = list([k for k in oclingo.__dict__.keys() if k[0] != '_'])

__version__ = oclingo.__version__

