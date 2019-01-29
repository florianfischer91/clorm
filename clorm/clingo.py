'''
Provides a wrapper around the key clingo objects to integrate Predicate and
FactBase interfaces.
'''

import io
import sys
import functools
from .orm import *
#import clorm as orm

# I want to replace the original clingo - re-exporting everything in clingo
# except replacing the class overides with my version: _class_overides = [
# 'Control', 'Model', 'SolveHandle' ]. The following seems to work but I'm not
# sure if this is bad.

#------------------------------------------------------------------------------
# Reference to the original clingo objects so that when we replace them our
# references point to the originals.
#------------------------------------------------------------------------------
import clingo as oclingo
OModel=oclingo.Model
OSolveHandle=oclingo.SolveHandle
OControl=oclingo.Control

from clingo import *
__all__ = list([ k for k in oclingo.__dict__.keys() if k[0] != '_'])
__version__ = oclingo.__version__

# ------------------------------------------------------------------------------
# Wrap clingo.Model and override some functions
# ------------------------------------------------------------------------------
def _model_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return fn(self._model, *args, **kwargs)
    return wrapper

class _ModelMetaClass(type):
    def __new__(meta, name, bases, dct):
        overrides=["contains"]
        for key,value in OModel.__dict__.items():
            if key not in overrides and callable(value):
                dct[key]=_model_wrapper(value)
        return super(_ModelMetaClass, meta).__new__(meta, name, bases, dct)

class Model(object, metaclass=_ModelMetaClass):
    '''Provides access to a model during a solve call.

    Objects mustn't be created manually. Instead they are returned by
    ``clorm.clingo.Control.solve`` callbacks.

    Behaves like ``clingo.Model`` but offers better integration with clorm facts
    and fact bases.

    '''

    def __init__(self, model):
        self._model = model

    #------------------------------------------------------------------------------
    # A new function to return a list of facts - similar to symbols
    #------------------------------------------------------------------------------

    def facts(self, factbase, atoms=False, terms=False, shown=False):
        '''Returns a FactBase containing the facts in the model that unify with the
        FactBase class.

        A wrapper around ``clingo.Model.symbols`` to return a FactBase.

        '''
        return factbase(
            symbols=self._model.symbols(atoms=atoms,terms=terms,shown=shown),
            delayed_init=True)

    #------------------------------------------------------------------------------
    # Overide contains
    #------------------------------------------------------------------------------

    def contains(self, fact):
        '''Return whether the fact or symbol is contained in the model. Extends
        ``clingo.Model.contains`` to allow for a clorm facts as well as a
        clingo symbols.

        '''
        if isinstance(fact, NonLogicalSymbol):
            return self._model.contains(fact.raw)
        return self._model.contains(fact)


# ------------------------------------------------------------------------------
# Wrap clingo.SolveHandle and override some functions
# ------------------------------------------------------------------------------
def _solvehandle_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return fn(self._handle, *args, **kwargs)
    return wrapper

class _SolveHandleMetaClass(type):
    def __new__(meta, name, bases, dct):
        overrides=["__init__", "__iter__", "__next__"]
        for key,value in OSolveHandle.__dict__.items():
            if key not in overrides and callable(value):
                dct[key]=_model_wrapper(value)
        return super(_SolveHandleMetaClass, meta).__new__(meta, name, bases, dct)

class SolveHandle(object, metaclass=_SolveHandleMetaClass):
    '''Handle for solve calls.

    Objects mustn't be created manually. Instead they are returned by
    ``clorm.clingo.Control.solve``.

    Behaves like ``clingo.SolveHandle`` but iterates over ``clorm.clingoModel``
    objects.

    '''

    def __init__(self, handle):
        self._handle = handle

    def __iter__(self):
        for m in self._handle.__iter__():
            yield Model(m)

    def __next__(self):
        m = self._handle.__next__()
        return Model(m)

# ------------------------------------------------------------------------------
# Wrap clingo.Control and override some functions
# ------------------------------------------------------------------------------
def _control_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return fn(self._ctrl, *args, **kwargs)
    return wrapper

class _ControlMetaClass(type):
    def __new__(meta, name, bases, dct):
        overrides=["__init__", "__new__", "assign_external", "release_external", "solve"]
        for key,value in OControl.__dict__.items():
            if key not in overrides and callable(value):
                dct[key]=_control_wrapper(value)
        return super(_ControlMetaClass, meta).__new__(meta, name, bases, dct)



class Control(object, metaclass=_ControlMetaClass):
    '''Control object for the grounding/solving process.

    Behaves like ``clingo.Control`` but with modifications to deal with ClORM
    facts and fact bases.

    '''

    def __init__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 1 and "control_" in kwargs:
            self._ctrl = kwargs["control_"]
        else:
            self._ctrl = OControl(*args, **kwargs)

    # A new function to add facts from a factbase or a list of facts
    def add_facts(self, facts):
        '''Add facts to the control object. Note: facts must be added before grounding.

        Args:
          facts: can be a ``clorm.FactBase`` or a list of ``clorm.Predicate`` instances.
        '''

        # Facts can be a FactBase or a list of facts
        if isinstance(facts, FactBase):
            asp_str = facts.asp_str()
        else:
            out = io.StringIO()
            for f in facts:
                print("{}.".format(f), file=out)
            asp_str = out.getvalue()
            out.close()

        # Parse and add the facts
        with self._ctrl.builder() as b:
            oclingo.parse_program(asp_str, lambda stmt: b.add(stmt))

    # Overide assign_external to deal with NonLogicalSymbol object and a Clingo Symbol
    def assign_external(self, fact, truth):
        '''Assign a truth value to an external atom (represented as a function symbol
        or program literam or a clorm fact.

        This function extends ``clingo.Control.release_external``.

        '''
        if isinstance(fact, NonLogicalSymbol):
            self._ctrl.assign_external(fact.raw, truth)
        else:
            self._ctrl.assign_external(fact, truth)

    # Overide release_external to deal with NonLogicalSymbol object and a Clingo Symbol
    def release_external(ctrl, fact):
        '''Release an external atom represented by the given symbol, program literal, or
        clorm fact.

        This function extends ``clingo.Control.release_external``.

        '''
        if isinstance(fact, NonLogicalSymbol):
            self._ctrl.release_external(fact.raw)
        else:
            self._ctrl.release_external(fact)

    #---------------------------------------------------------------------------
    # Overide solve and if necessary replace on_model with a wrapper that
    # returns a clorm.Model object. Also because of the issue with using the
    # keyword "async" as a parameter in Python 3.7 (which means that newer
    # clingo version use "async_") we use a more complicated way to determine
    # the function parameters.
    #---------------------------------------------------------------------------
    def solve(self, **kwargs):
        '''Run the clingo solver.

        This function extends ``clingo.Control.solve`` to take assumptions that
        are facts or a fact base, and return clorm.clingo.SolveHandle and
        clorm.clingo.Model objects.

        '''

        # validargs stores the valid arguments and their default values
        validargs = { "assumptions": [], "on_model" : None,
                        "on_finish": None, "yield_" : False }

        # Use "async" or "async_" depending on the python or clingo version
        if sys.version_info >= (3,7) or oclingo.__version__ > '5.3.1':
            validargs["async_"] = False
        else:
            validargs["async"] = False

        # validate the arguments and assign any missing default values
        keys = set(kwargs.keys())
        validkeys = set(validargs.keys())
        if not keys.issubset(validkeys):
            diff = keys - validkeys
            msg = "solve() got an unexpected keyword argument '{}'".format(next(iter(diff)))
            raise TypeError(msg)
        for k,v in validargs.items():
            if k not in kwargs: kwargs[k]=v

        # generate a new assumptions list if necesary
        assumptions = kwargs["assumptions"]
        if isinstance(assumptions, FactBase):
            kwargs["assumptions"] = [f.raw for f in assumptions.facts()]
        else:
            kwargs["assumptions"] = [ (f.raw if isinstance(f, NonLogicalSymbol) \
                                       else f, b) for f,b in assumptions ]

        # generate a new on_model function if necessary
        on_model=kwargs["on_model"]
        @functools.wraps(on_model)
        def on_model_wrapper(model):
            return on_model(Model(model))
        if on_model: kwargs["on_model"] =  on_model_wrapper

        result = self._ctrl.solve(**kwargs)
        if kwargs["yield_"]:
            return SolveHandle(result)
        else:
            return result


#------------------------------------------------------------------------------
# Modify the original clingo docstrings.
#------------------------------------------------------------------------------

if oclingo.Model.__doc__ != "Used by autodoc_mock_imports.":
    Control.__doc__ += oclingo.Control.__doc__
    Control.assign_external.__doc__ += oclingo.Control.assign_external.__doc__
    Control.release_external.__doc__ += oclingo.Control.release_external.__doc__
    Control.solve.__doc__ += oclingo.Control.solve.__doc__

    Model.__doc__ += oclingo.Model.__doc__
    Model.contains.__doc__ += oclingo.Model.contains.__doc__

    SolveHandle.__doc__ += oclingo.SolveHandle.__doc__
else:
    Model.__doc__ += "\n" + \
        "    For more details see the Clingo API for " + \
        '''`Model <https://potassco.org/clingo/python-api/current/clingo.html#Model>`_'''
    SolveHandle.__doc__ += "\n" + \
        "    For more details see the Clingo API for " + \
        '''`SolveHandle <https://potassco.org/clingo/python-api/current/clingo.html#SolveHandle>`_'''
    Control.__doc__ += "\n" + \
        "    For more details see the Clingo API for " + \
        '''`Control <https://potassco.org/clingo/python-api/current/clingo.html#Control>`_'''


#print("MODEL DOCS:\n\n{}\n\n".format(Model.__doc__))
#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------
if __name__ == "__main__":
    raise RuntimeError('Cannot run modules')
