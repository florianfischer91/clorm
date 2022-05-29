# -----------------------------------------------------------------------------
# Clorm ORM FactBase query implementation. It provides the rich query API.
# ------------------------------------------------------------------------------

import abc
import collections
import enum
import inspect
import io
import itertools
import operator
import sys
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterable,
                    Iterator, List, Optional, Tuple, Type, Union)

from .core import (Comparator, Predicate, PredicatePath, QCondition, and_,
                   falseall, hashable_path, kwargs_check_keys, notcontains,
                   or_, path, trueall)
from .factcontainers import FactIndex, FactMap, FactSet
from .abstract_query import Query

__all__ = [
    'Query',
    'Placeholder',
    'desc',
    'asc',
    'ph_',
    'ph1_',
    'ph2_',
    'ph3_',
    'ph4_',
    'func',
    'fixed_join_order',
    'basic_join_order',
    'oppref_join_order',
    ]

#------------------------------------------------------------------------------
# Global
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Defining and manipulating conditional elements
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Placeholder allows for variable substituion of a query. Placeholder is
# an abstract class that exposes no API other than its existence.
# ------------------------------------------------------------------------------
class Placeholder(abc.ABC):

    r"""An abstract class for defining parameterised queries.

    Currently, Clorm supports 4 placeholders: ph1\_, ph2\_, ph3\_, ph4\_. These
    correspond to the positional arguments of the query execute function call.

    """
    @abc.abstractmethod
    def __eq__(self, other):
        ...

    @abc.abstractmethod
    def __hash__(self):
        ...


class NamedPlaceholder(Placeholder):

    # Only keyword arguments are allowd. Note: None could be a legitimate value
    # so cannot use it to test for default
    def __init__(self, *, name, **kwargs):
        self._name = name
        # Check for unexpected arguments
        badkeys = kwargs_check_keys(set(["default"]), set(kwargs.keys()))
        if badkeys:
            mstr = "Named placeholder unexpected keyword arguments: "
            raise TypeError(f"{mstr}{','.join(sorted(badkeys))}")

        # Set the keyword argument
        if "default" in kwargs:
            self._default = (True, kwargs["default"])
#        elif len(args) > 1
        else: self._default = (False,None)

    @property
    def name(self):
        return self._name
    @property
    def has_default(self):
        return self._default[0]
    @property
    def default(self):
        return self._default[1]
    def __str__(self):
        tmpstr = "" if not self._default[0] else f",{self._default[1]}"
        return f"ph_(\"{self._name}\"{tmpstr})"
    def __repr__(self):
        return self.__str__()
    def __eq__(self, other):
        if not isinstance(other, NamedPlaceholder):
            return NotImplemented
        return (self.name, self._default) == (other.name, other._default )
    def __hash__(self):
        return hash(self._name)

class PositionalPlaceholder(Placeholder):
    def __init__(self, *, posn):
        self._posn = posn
    @property
    def posn(self):
        return self._posn
    def __str__(self):
        return f"ph{self._posn+1}_"
    def __repr__(self):
        return self.__str__()
    def __eq__(self, other):
        if not isinstance(other, PositionalPlaceholder):
            return NotImplemented
        return self._posn == other._posn
    def __hash__(self):
        return hash(self._posn)

def ph_(value, *args, **kwargs):
    ''' A function for building new placeholders, either named or positional.'''

    badkeys = kwargs_check_keys(set(["default"]), set(kwargs.keys()))
    if badkeys:
        mstr = "ph_() unexpected keyword arguments: "
        raise TypeError(f"{mstr}{','.join(sorted(badkeys))}")

    # Check the match between positional and keyword arguments
    if "default" in kwargs and len(args) > 0:
        raise TypeError("ph_() got multiple values for argument 'default'")
    if len(args) > 1:
        raise TypeError(("ph_() takes from 0 to 2 positional"
                         f"arguments but {len(args)+1} given"))

    # Set the default argument
    if "default" in kwargs:
        default = (True, kwargs["default"])
    elif len(args) > 0:
        default = (True, args[0])
    else:
        default = (False,None)

    try:
        idx = int(value)
    except ValueError:
        # It's a named placeholder
        nkargs = { "name" : value }
        if default[0]:
            nkargs["default"] = default[1]
        return NamedPlaceholder(**nkargs)

    # Its a positional placeholder
    if default[0]:
        raise TypeError("Positional placeholders don't support default values")
    idx -= 1
    if idx < 0:
        raise ValueError(f"Index {idx+1} is not a positional argument")
    return PositionalPlaceholder(posn=idx)

#------------------------------------------------------------------------------
# Some pre-defined positional placeholders
#------------------------------------------------------------------------------

ph1_ = PositionalPlaceholder(posn=0)
ph2_ = PositionalPlaceholder(posn=1)
ph3_ = PositionalPlaceholder(posn=2)
ph4_ = PositionalPlaceholder(posn=3)


# ------------------------------------------------------------------------------
# API function to build a functor wrapper object as part of a specifying a where
# clause or an output statement.
# ------------------------------------------------------------------------------

FuncInputSpec = collections.namedtuple('FuncInputSpec', 'paths functor')
def func(paths, func_):
    '''Wrap a boolean functor with predicate paths for use as a query condition'''
    return FuncInputSpec(paths,func_)
    return FunctionComparator.from_specification(paths,func_)

# ------------------------------------------------------------------------------
# QCondition objects are generated by the Clorm API. But we want to treat the
# different components differently depending on whether they are a comparison
# (either using standard operators or using a functor), or a boolean, or a join
# condition.
#
# So we create separate classes for the different components.  Note: these
# object are treated as immutable. So when manipulating such objects the
# following functions make modified copies of the objects rather than modifying
# the objects themselves. So if a function doesn't need to modify an object at
# all it simply returns the object itself.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Support function returns True if all the elements of the list are root paths
# ------------------------------------------------------------------------------

def is_root_paths(paths):
    return all(p.meta.is_root for p in map(path, paths))

# ------------------------------------------------------------------------------
# Support function to make sure we have a list of paths
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# support function - give an iterable that may include a predicatepath return a
# list copy where any predicatepaths are replaced with hashable paths. Without
# this comparison operators will fail.
# ------------------------------------------------------------------------------

def _hashables(seq):
    f = lambda x : x.meta.hashable if isinstance(x,PredicatePath) else x
    return map(f,seq)

def get_value_for_placeholder(ph, ctx, *args, **kwargs):
    if not isinstance(ph, Placeholder):
        return ph
    if isinstance(ph,PositionalPlaceholder):
        if ph.posn < len(args):
            return args[ph.posn]
        raise ValueError((f"Missing positional placeholder argument '{ph}' "
                            f"when grounding '{ctx}' with positional arguments: {args}"))
    if isinstance(ph, NamedPlaceholder):
        v = kwargs.get(ph.name,None)
        if v:
            return v
        if ph.has_default:
            return ph.default
        raise ValueError((f"Missing named placeholder argument '{ph}' "
                            f"when grounding '{ctx}' with arguments: {kwargs}"))

    raise NotImplementedError(f"case for type {ph.__class__} of {ph} not implemented")


# ------------------------------------------------------------------------------
# MembershipSeq is used for holding a reference to some form of sequence that is
# used as part of a query membership "in_" (or "notin_") condition. When a query
# is executed the sequence is turned into a set which is then used for
# membership comparisons. So any update of the reference sequence after the
# query is declared but before the query is executed will affect the execution
# of the query. It also also for the sequence to be specified as a sub-query.
# ------------------------------------------------------------------------------

class MembershipSeq:

    def __init__(self, src):
        self._src = src

    def fixed(self):
        if isinstance(self._src, Placeholder):
            raise ValueError(f"Cannot fix unground sequence specification : {self}")
        if isinstance(self._src, Query):
            return frozenset(self._src.all())
        return frozenset(self._src)

    def ground(self, *args: Any, **kwargs: Any) -> "MembershipSeq":
        if isinstance(self._src, Query):
            where = self._src.qspec.where
            if where is None:
                return self
            return MembershipSeq(self._src.bind(*args,**kwargs))

        value = get_value_for_placeholder(self._src, self, *args, **kwargs)
        return self if value is self._src else MembershipSeq(value)

    @property
    def placeholders(self):
        if isinstance(self._src, Placeholder):
            return set([self._src])
        if not isinstance(self._src, Query):
            return set()
        where = self._src.qspec.where
        return where.placeholders if where else set()

    def __eq__(self,other):
        if not isinstance(other, MembershipSeq):
            return NotImplemented
        return self._src is other._src

    def __hash__(self):
        return hash(id(self._src))

    def __str__(self):
        return f"MS:{self._src.__str__()}"

    def __repr__(self):
        return self._src.__repr__()


# ------------------------------------------------------------------------------
# functions to validate a QCondition objects for a standard comparator objects
# from QCondition objects for either a join or where condition. Returns a pair
# consisting of the operator and validated and normalised arguments. The
# function will raise exceptions if there are any problems.
# ------------------------------------------------------------------------------

def _normalise_op_args(arg):
    p=path(arg,exception=False)
    return arg if p is None else p

def _is_static_op_arg(arg):
    return not isinstance(arg,PredicatePath)

def where_comparison_op(qcond: QCondition) -> Tuple[Any, Any]:
    newargs = [_normalise_op_args(a) for a in qcond.args]

    if all(map(_is_static_op_arg, newargs)):
        raise ValueError(("Invalid comparison of only static inputs "
                          "(at least one argument must reference a "
                          f"a component of a fact): {qcond}"))

    spec = StandardComparator.operators.get(qcond.operator,None)
    if spec is None:
        raise TypeError(("Internal bug: cannot create StandardComparator() with "
                         f"non-comparison operator '{qcond.operator}' "))
    if not spec.where and spec.join:
        raise ValueError((f"Invalid 'where' comparison operator '{qcond.operator}' is only "
                          "valid for a join specification"))
    return (qcond.operator,newargs)

def where_membership_op(qcond: QCondition) -> Tuple[Any, Any]:
    pth = _normalise_op_args(qcond.args[1])
    if not isinstance(pth,PredicatePath):
        raise ValueError((f"Invalid 'where' condition '{qcond}': missing path in "
                          "membership declaration"))
    seq = qcond.args[0]
    if isinstance(seq,PredicatePath):
        raise ValueError((f"Invalid 'where' condition '{qcond}': invalid sequence in "
                          "membership declaration"))

    return (qcond.operator, [MembershipSeq(seq),pth])

def join_comparison_op(qcond: QCondition) -> Tuple[Any, Tuple[Any, ...]]:
    if qcond.operator is falseall:
        raise ValueError("Internal bug: cannot use falseall operator in QCondition")

    paths = list(filter(lambda x: isinstance(x,PredicatePath), qcond.args))
    hashable_paths = set(map(hashable_path, paths))
    roots = set(map(lambda x: hashable_path(path(x).meta.root), hashable_paths))
    if len(roots) != 2:
        raise ValueError((f"Invalid join expression '{qcond}'. A join expression must join "
                          "paths with two distinct predicate roots"))

    if qcond.operator is trueall:
        if hashable_paths != roots:
            raise ValueError((f"Cross-product expression '{qcond}' must contain only "
                              "root paths"))
    return (qcond.operator,qcond.args)


# ------------------------------------------------------------------------------
# keyable functions are operator specific functions to extract keyable/indexable
# information form StandardComparator instances. This is then used to give keyed
# lookups on a FactIndex. If the function returns None then the comparator
# cannot be used to key on the given list of indexes.
# ------------------------------------------------------------------------------

def comparison_op_keyable(sc, indexes):
    indexes = {hashable_path(p) for p in indexes}
    swapop = {
        operator.eq : operator.eq,  operator.ne : operator.ne,
        operator.lt : operator.gt,  operator.gt : operator.lt,
        operator.le : operator.ge,  operator.ge : operator.le,
        trueall : trueall,  falseall : falseall }

    def hp(a):
        try:
            return hashable_path(a)
        except TypeError:
            return a

    a0 = hp(sc.args[0])
    a1 = hp(sc.args[1])
    if isinstance(a0,PredicatePath.Hashable) and a0 in indexes:
        return (a0, sc.operator, sc.args[1])
    if isinstance(a1,PredicatePath.Hashable) and a1 in indexes:
        return (a1, swapop[sc.operator], sc.args[0])
    return None

def membership_op_keyable(sc,indexes):
    indexes = {hashable_path(p) for p in indexes}
    hpa1 = hashable_path(sc.args[1])
    return (hpa1, sc.operator, sc.args[0]) if hpa1 in indexes else None



# ------------------------------------------------------------------------------
# Comparator for the standard operators
#
# Implementation detail: with the use of membership operators
# ('operator.contains' and 'notcontains') we want to pass an arbitrary sequence
# to the object. This object may be mutable and will therefore not be
# hashable. To support hashing and adding comparators to a set, the hash of the
# comparator only takes into account predicate paths. The main thing is that
# equality works properly. While this is a bit of a hack I think it is ok
# provided StandardComparator is used only in this specific way within clorm and
# is not exposed outside clorm.
# ------------------------------------------------------------------------------

class StandardComparator(Comparator):
    class Preference(enum.IntEnum):
        LOW= 0
        MEDIUM= 1
        HIGH=2

    OpSpec = collections.namedtuple('OpSpec','pref join where negop swapop keyable form')
    operators = {
        operator.eq : OpSpec(pref=Preference.HIGH,
                             join=join_comparison_op,
                             where=where_comparison_op,
                             negop=operator.ne, swapop=operator.eq,
                             keyable=comparison_op_keyable,
                             form=QCondition.Form.INFIX),
        operator.ne : OpSpec(pref=Preference.LOW,
                             join=join_comparison_op,
                             where=where_comparison_op,
                             negop=operator.eq, swapop=operator.ne,
                             keyable=comparison_op_keyable,
                             form=QCondition.Form.INFIX),
        operator.lt : OpSpec(pref=Preference.MEDIUM,
                             join=join_comparison_op,
                             where=where_comparison_op,
                             negop=operator.ge, swapop=operator.gt,
                             keyable=comparison_op_keyable,
                             form=QCondition.Form.INFIX),
        operator.le : OpSpec(pref=Preference.MEDIUM,
                             join=join_comparison_op,
                             where=where_comparison_op,
                             negop=operator.gt, swapop=operator.ge,
                             keyable=comparison_op_keyable,
                             form=QCondition.Form.INFIX),
        operator.gt : OpSpec(pref=Preference.MEDIUM,
                             join=join_comparison_op,
                             where=where_comparison_op,
                             negop=operator.le, swapop=operator.lt,
                             keyable=comparison_op_keyable,
                             form=QCondition.Form.INFIX),
        operator.ge : OpSpec(pref=Preference.MEDIUM,
                             join=join_comparison_op,
                             where=where_comparison_op,
                             negop=operator.lt, swapop=operator.le,
                             keyable=comparison_op_keyable,
                             form=QCondition.Form.INFIX),
        trueall     : OpSpec(pref=Preference.LOW,
                             join=join_comparison_op,
                             where=None,
                             negop=falseall, swapop=trueall,
                             keyable=comparison_op_keyable,
                             form=QCondition.Form.FUNCTIONAL),
        falseall    : OpSpec(pref=Preference.HIGH,
                             join=join_comparison_op,
                             where=None,
                             negop=trueall, swapop=falseall,
                             keyable=comparison_op_keyable,
                             form=QCondition.Form.FUNCTIONAL),
        operator.contains : OpSpec(pref=Preference.HIGH,
                                   join=None,
                                   where=where_membership_op,
                                   negop=notcontains, swapop=None,
                                   keyable=membership_op_keyable,
                                   form=QCondition.Form.INFIX),
        notcontains       : OpSpec(pref=Preference.LOW,
                                   join=None,
                                   where=where_membership_op,
                                   negop=operator.contains, swapop=None,
                                   keyable=membership_op_keyable,
                                   form=QCondition.Form.INFIX)}

    def __init__(self,operator_,args):
        spec = StandardComparator.operators.get(operator_,None)
        if spec is None:
            raise TypeError(("Internal bug: cannot create StandardComparator() with "
                             f"non-comparison operator '{operator_}' "))
        self._operator = operator_
        self._args = tuple(args)
        self._hashableargs =tuple((hashable_path(a) if isinstance(a,PredicatePath) \
                                    else a for a in self._args))
        self._paths=tuple(filter(lambda x : isinstance(x,PredicatePath),self._args))

        tmppaths = set([])
        tmproots = set([])
        for a in self._args:
            if isinstance(a,PredicatePath):
                tmppaths.add(hashable_path(a))
                tmproots.add(hashable_path(a.meta.root))
        self._paths=tuple(map(path, tmppaths))
        self._roots=tuple(map(path, tmproots))

    # -------------------------------------------------------------------------
    # non-ABC functions
    # -------------------------------------------------------------------------

    @classmethod
    def from_where_qcondition(cls, qcond: QCondition) -> "StandardComparator":
        if not isinstance(qcond, QCondition):
            raise TypeError(("Internal bug: trying to make StandardComparator() "
                             f"from non QCondition object: {qcond}"))

        spec = StandardComparator.operators.get(qcond.operator,None)
        if spec is None:
            raise TypeError(("Internal bug: cannot create StandardComparator() with "
                             f"non-comparison operator '{qcond.operator}' "))
        if not spec.where and spec.join:
            raise ValueError((f"Invalid 'where' comparison operator '{qcond.operator}' is only "
                              "valid for a join specification"))
        if not spec.where:
            raise ValueError((f"Invalid 'where' comparison operator '{qcond.operator}'"))
        op, newargs = spec.where(qcond)
        return cls(op,newargs)

    @classmethod
    def from_join_qcondition(cls,qcond: QCondition) -> "StandardComparator":
        if not isinstance(qcond, QCondition):
            raise TypeError(("Internal bug: trying to make Join() "
                             f"from non QCondition object: {qcond}"))

        spec = StandardComparator.operators.get(qcond.operator,None)
        if spec is None:
            raise TypeError(("Internal bug: cannot create StandardComparator() with "
                             f"non-comparison operator '{qcond.operator}' "))

        if not spec.join and spec.where:
            raise ValueError((f"Invalid 'join' comparison operator '{qcond.operator}' is only "
                              "valid for a join specification"))
        if not spec.join:
            raise ValueError(f"Invalid 'join' comparison operator '{qcond.operator}'")

        op, newargs = spec.join(qcond)
        return cls(op,newargs)


    # -------------------------------------------------------------------------
    # Implement ABC functions
    # -------------------------------------------------------------------------

    def fixed(self) -> "StandardComparator":
        gself = self.ground()
        if gself.operator not in [ operator.contains, notcontains]:
            return gself
        if isinstance(gself.args[0],frozenset):
            return gself
        if not isinstance(gself.args[0],MembershipSeq):
            raise ValueError(f"Internal error: unexpected sequence type object: '{gself.args[0]}'")
        return StandardComparator(self.operator, [gself.args[0].fixed(),gself.args[1]])

    def ground(self,*args,**kwargs):
        if self._operator not in [ operator.contains, notcontains] or \
           not isinstance(self._args[0],MembershipSeq):
            def wrap(ph):
                return get_value_for_placeholder(ph, self, *args, **kwargs)
            newargs = tuple(map(wrap,self._args))
        else:
            newargs = tuple((self._args[0].ground(*args,**kwargs), self._args[1]))

        if _hashables(newargs) == _hashables(self._args):
            return self
        return StandardComparator(self._operator, newargs)

    def negate(self):
        spec = StandardComparator.operators[self._operator]
        return StandardComparator(spec.negop, self._args)

    def dealias(self):
        def getdealiased(arg):
            if isinstance(arg,PredicatePath):
                return arg.meta.dealiased
            return arg
        newargs = tuple(map(getdealiased, self._args))
        if _hashables(newargs) == _hashables(self._args):
            return self
        return StandardComparator(self._operator, newargs)

    def swap(self):
        spec = StandardComparator.operators[self._operator]
        if not spec.swapop:
            raise ValueError((f"Internal bug: comparator '{self._operator}' doesn't support "
                              "the swap operation"))
        return StandardComparator(spec.swapop, reversed(self._args))

    def keyable(self, indexes: Dict) -> Optional[Tuple[Any, Any, Any]]:
        spec = StandardComparator.operators[self._operator]
        return spec.keyable(self, indexes)

    @property
    def paths(self):
        return self._paths

    @property
    def placeholders(self):
        tmp = set(filter(lambda x : isinstance(x,Placeholder), self._args))
        if self._operator not in [ operator.contains, notcontains]:
            return tmp
        if not isinstance(self._args[0], MembershipSeq):
            return tmp
        tmp.update(self._args[0].placeholders)
        return tmp

    @property
    def preference(self):
        pref = StandardComparator.operators[self._operator].pref
        if pref is None:
            raise ValueError(f"Operator '{self._operator}' does not have a join preference")
        return pref

    @property
    def form(self):
        return StandardComparator.operators[self._operator].form

    @property
    def operator(self):
        return self._operator

    @property
    def args(self):
        return self._args

    @property
    def roots(self):
        return self._roots

    @property
    def executable(self):
        for arg in self._args:
            if isinstance(arg,PositionalPlaceholder):
                return False
            if isinstance(arg,NamedPlaceholder) and not arg.has_default:
                return False
        return True

    def make_callable(self, root_signature):
        for arg in self._args:
            if isinstance(arg,Placeholder):
                raise TypeError(("Internal bug: cannot make a non-ground "
                                 f"comparator callable: {self}"))
        sig = make_input_alignment_functor(root_signature, self._args)
        return ComparisonCallable(self._operator,sig)

    def __eq__(self,other):
        def getval(val):
            if isinstance(val,PredicatePath):
                return val.meta.hashable
            return val

        if not isinstance(other, StandardComparator):
            return NotImplemented
        if self._operator != other._operator:
            return False
        for a,b in zip(self._args,other._args):
            if getval(a) != getval(b):
                return False
        return True

    def __hash__(self):
        return hash((self._operator,) + self._hashableargs)

    def __str__(self):
        # For convenience just return a QCondition string
        return str(QCondition(self._operator, *self._args))

    def __repr__(self):
        return self.__str__()

# ------------------------------------------------------------------------------
# Comparator for arbitrary functions. From the API generated with func()
# The constructor takes a reference to the function and a path signature.
# ------------------------------------------------------------------------------

class FunctionComparator(Comparator):
    def __init__(self,func_,path_signature,negative=False,assignment=None):
        self._func = func_
        self._funcsig = collections.OrderedDict()
        self._pathsig = tuple(map(hashable_path, path_signature))
        self._negative = negative
        self._assignment = None if assignment is None else dict(assignment)
        self._placeholders = set()  # Create matching named placeholders

        # Calculate the root paths
        tmproots = set([])
        tmppaths = set([])
        for a in self._pathsig:
            if isinstance(a,PredicatePath.Hashable):
                tmppaths.add(a)
                tmproots.add(hashable_path(path(a).meta.root))
        self._paths=tuple(map(path, tmppaths))
        self._roots=tuple(map(path, tmproots))

        # The function signature must be compatible with the path signature
        funcsig = inspect.signature(func_)
        if len(funcsig.parameters) < len(self._pathsig):
            raise ValueError((f"More paths specified in the path signature '{self._pathsig}' "
                              f"than there are in the function signature '{funcsig}'"))

        # Track the parameters that are not part of the path signature but are
        # part of the function signature. This determines if the
        # FunctionComparator is "ground".
        for i,(k,v) in enumerate(funcsig.parameters.items()):
            if i >= len(self._pathsig):
                self._funcsig[k]=v
                if assignment and k in assignment:
                    continue
                if v.default == inspect.Parameter.empty:
                    ph = NamedPlaceholder(name=k)
                else:
                    ph = NamedPlaceholder(name=k,default=v.default)
                self._placeholders.add(ph)

        # Check the path signature
        if not self._pathsig:
            raise ValueError("Invalid empty path signature")

        if any(not isinstance(pp, PredicatePath.Hashable) for pp in self._pathsig):
            raise TypeError(("The boolean functor call signature must "
                            "consist of predicate paths"))

        # if there is an assigned ordereddict then check the keys
        self._check_assignment()

        # Used for the hash function
        self._assignment_tuple = tuple(self._assignment.items()) if self._assignment else ()


    #-------------------------------------------------------------------------
    # Internal function to check the assigment. Will set default values if the
    # assigment is non-None.
    # -------------------------------------------------------------------------
    def _check_assignment(self):
        if self._assignment is None:
            return
        assignmentkeys=set(list(self._assignment.keys()))
        funcsigkeys=set(list(self._funcsig.keys()))
        tmp = assignmentkeys-funcsigkeys
        if tmp:
            raise ValueError(("FunctionComparator is being given "
                             "an assignment for unrecognised function "
                             f"parameters '{tmp}'"))

        unassigned = funcsigkeys-assignmentkeys
        if not unassigned:
            return

        # There are unassigned so check if there are default values
        tmp = set()
        for name in unassigned:
            default = self._funcsig[name].default
            if default == inspect.Parameter.empty:
                tmp.add(name)
            else:
                self._assignment[name] = default

        if tmp:
            raise ValueError(f"Missing functor parameters for '{tmp}'")

    @classmethod
    def from_specification(cls,paths,func_):
        return cls(func_, list(map(path, paths)))

    # -------------------------------------------------------------------------
    # ABC functions
    # -------------------------------------------------------------------------
    @property
    def form(self):
        return QCondition.Form.FUNCTIONAL

    @property
    def paths(self):
        return self._paths

    @property
    def placeholders(self):
        return set(self._placeholders)

    @property
    def roots(self):
        return self._roots

    @property
    def executable(self):
        return all(ph.has_default for ph in self._placeholders)

    def negate(self):
        neg = not self._negative
        return FunctionComparator(self._func,self._pathsig,neg,
                                          assignment=self._assignment)

    def dealias(self):
        newpathsig = [path(hp).meta.dealiased for hp in self._pathsig]
        if _hashables(newpathsig) == _hashables(self._pathsig):
            return self
        return FunctionComparator(self._func, newpathsig, self._negative,
                                          assignment=self._assignment)

    def fixed(self):
        return self.ground()

    def ground(self, *args, **kwargs):
        if self._assignment is not None:
            return self
        assignment = {}
        # Assign any positional arguments first then add the keyword arguments
        # and make sure there is no repeats. Finally, assign any placeholders
        # with defaults. Note: funcsig is an orderedDict
        for idx,(k,_) in enumerate(self._funcsig.items()):
            if idx >= len(args):
                break
            assignment[k] = args[idx]
        for k,v in kwargs.items():
            if k in assignment:
                raise ValueError(("Both positional and keyword values given "
                                  f"for the argument '{k}'"))
            assignment[k] = v
        for ph in self._placeholders:
            if isinstance(ph, NamedPlaceholder) and ph.name not in assignment:
                if ph.has_default:
                    assignment[ph.name] = ph.default
                else:
                    raise ValueError((f"Missing named placeholder argument '{ph.name}' "
                                      f"when grounding '{self}' with arguments: {kwargs}"))

        return FunctionComparator(self._func,self._pathsig,
                                  self._negative,assignment)

    def make_callable(self, root_signature):
        if self._assignment is None:
            raise RuntimeError(("Internal bug: make_callable called on a "
                                f"ungrounded object: {self}"))

        # from the function signature and the assignment generate the fixed
        # values for the non-path items
        funcsigparam = [ self._assignment[k] for k,_ in self._funcsig.items() ]
        outputsig = tuple(list(self._pathsig) + funcsigparam)
        alignfunc = make_input_alignment_functor(root_signature,outputsig)
        op = self._func if not self._negative else lambda *args : not self._func(*args)
        return ComparisonCallable(op,alignfunc)

    def __eq__(self, other):
        if not isinstance(other, FunctionComparator):
            return NotImplemented
        if self._func != other._func:
            return False
        if self._pathsig != other._pathsig:
            return False
        if self._negative != other._negative:
            return False
        if self._assignment != other._assignment:
            return False
        return True

    def __hash__(self):
        return hash((self._func,) + self._pathsig + self._assignment_tuple)

    def __str__(self):
        assignstr = f": {self._assignment}" if self._assignment else ""
        funcstr = f"func({self._pathsig}{assignstr}, {self._func})"
        return f"not_({funcstr})" if self._negative else funcstr

    def __repr__(self):
        return self.__str__()

# ------------------------------------------------------------------------------
# Comparators (Standard and Function) have a comparison function and input of
# some form; eg "F.anum == 3" has operator.eq_ and input (F.anum,3) where F.anum
# is a path and will be replaced by some fact sub-field value.
#
# We need to extract the field input from facts and then call the comparison
# function with the appropriate input. But it is not straight forward to get the
# field input. If the query search is on a single predicate type then the input
# will be a singleton tuple. However, if there is a join in the query there will
# be multiple elements to the tuple. Furthermore, the order of facts will be
# determined by the query optimiser as it may be more efficient to join X with Y
# rather than Y with X.
#
# With this complication we need a way to remap a search input fact-tuple into
# the expected form for each query condition component.
#
# make_input_alignment_functor() returns a function that takes a tuple
# of facts as given by the input signature and returns a tuple of values as
# given by the output signature.
# ------------------------------------------------------------------------------
def make_input_alignment_functor(input_root_signature, output_signature):

    # Input signature are paths that must correspond to predicate types
    def validate_input_signature():
        if not input_root_signature:
            raise TypeError("Empty input predicate path signature")
        inputs=[]
        try:
            for p in input_root_signature:
                pp = path(p)
                if not pp.meta.is_root:
                    raise ValueError(f"path '{pp}' is not a predicate root")
                inputs.append(pp)
        except Exception as e:
            raise TypeError((f"Invalid input predicate path signature {input_root_signature}: "
                             f"{e}")) from None
        return inputs

    # Output signature are field paths or statics (but not placeholders)
    def validate_output_signature():
        if not output_signature:
            raise TypeError("Empty output path signature")
        outputs=[]
        for a in output_signature:
            p = path(a,exception=False)
            outputs.append(p if p else a)
            if p:
                continue
            if isinstance(a, Placeholder):
                raise TypeError((f"Output signature '{output_signature}' contains a placeholder "
                                 f"'{a}'"))
        return outputs

    insig = validate_input_signature()
    outsig = validate_output_signature()

    # build a list of lambdas one for each output item that chooses the
    # appropriate item from the input.
    pp2idx = { hashable_path(pp) : idx for idx,pp in enumerate(insig) }
    getters = []
    for out in outsig:
        if isinstance(out,PredicatePath):
            idx = pp2idx.get(hashable_path(out.meta.root),None)
            if idx is None:
                raise TypeError((f"Invalid signature match between {input_root_signature} "
                                 f"and {output_signature}: missing input predicate path for {out}"))
            ag=out.meta.attrgetter
            getters.append(lambda facts, ag=ag, idx=idx: ag(facts[idx]))
        else:
            getters.append(lambda facts, out=out: out)

    getters = tuple(getters)

    # Create the getter
    def func_(facts):
        try:
            return tuple(getter(facts) for getter in getters)
        except IndexError:
            raise TypeError(("Invalid input to getter function: expecting "
                             f"a tuple with {len(insig)} elements and got a tuple with "
                             f"{len(facts)}")) from None
        except TypeError as e:
            raise TypeError(f"Invalid input to getter function: {e}") from None
        except AttributeError as e:
            raise TypeError(f"Invalid input to getter function: {e}") from None
    return func_


# ------------------------------------------------------------------------------
# ComparisonCallable is a functional object that wraps a comparison operator and
# ensures the comparison operator gets the correct input. The input to a
# ComparisonCallable is a tuple of facts (the form of which is determined by a
# signature) and returns whether the facts satisfy some condition.
# ------------------------------------------------------------------------------

class ComparisonCallable:
    def __init__(self, operator_, getter_map):
        self._operator = operator_
        self._getter_map = getter_map

    def __call__(self, facts):
        args = self._getter_map(facts)
        return self._operator(*args)


# ------------------------------------------------------------------------------
# 'Where' query clauses handling.
#
# The goal is to turn the where clause into a CNF clausal normal form. So
# functions to validate the 'where' clause and then turn it into NNF, then CNF,
# and then a pure clausal form.
# ------------------------------------------------------------------------------

g_bool_operators = {
    operator.and_ : True, operator.or_ : True, operator.not_ : True }

def is_boolean_qcondition(cond):
#    if isinstance(cond, FunctionComparator): return False
    if isinstance(cond, FuncInputSpec):
        return False
    if not isinstance(cond, QCondition):
        return False
    v = g_bool_operators.get(cond.operator,False)
    return v

def is_comparison_qcondition(cond):
    if not isinstance(cond, QCondition):
        return False
    spec = StandardComparator.operators.get(cond.operator,None)
    return bool(spec)

# ------------------------------------------------------------------------------
# Validates and turns non-boolean QCondition objects into the appropriate
# comparator (functors are wrapped in a FunctionComparator object - and
# FuncInputSpec are also turned into FunctionComparator objects).  Also
# simplifies any static conditions (conditions that can be evaluated without a
# fact) which are replaced with their a boolean evaluation.
#
# The where expression is validated with respect to a sequence of predicate root
# paths that indicate the valid predicates (and aliases) that are being
# reference in the query.
# ------------------------------------------------------------------------------

def validate_where_expression(qcond: Union[QCondition, FuncInputSpec], roots=None):
    roots = roots if roots else []
    # Make sure we have a set of hashable paths
    try:
        roots = set(map(hashable_path, roots))
    except Exception as e:
        raise ValueError(f"Invalid predicate paths signature {roots}: {e}") from None
    for pp in roots:
        if not pp.path.meta.is_root:
            raise ValueError((f"Invalid roots element {pp} does not refer to "
                              "the root of a predicate path "))

    # Check that the path is a sub-path of one of the roots
    def check_path(path_):
        if hashable_path(path_.meta.root) not in roots:
            raise ValueError((f"Invalid 'where' expression '{qcond}' contains a path "
                              f"'{path_}' that is not a sub-path of one of the "
                              f"roots '{roots}'"))

    # Check if a condition is static - to be called after validating the
    # sub-parts of the conidition.
    def is_static_condition(cond):
        if isinstance(cond,(Comparator, QCondition)):
            return False
        if callable(cond):
            raise TypeError(("Internal bug: invalid static test "
                             f"with callable: {cond}"))
        return True

    # Check callable - construct a FunctionComparator
    def validate_callable(func_):
        if len(roots) != 1:
            raise ValueError((f"Incompatible usage between raw functor {func_} and "
                              f"non-singleton predicates {roots}"))
        return FunctionComparator.from_specification(roots,func_)

    # Check boolean condition - simplifying if it is a static condition
    def validate_bool_condition(bcond: StandardComparator):
        if bcond.operator is operator.not_:
            newsubcond = validate_condition(bcond.args[0])
            if is_static_condition(newsubcond):
                return bcond.operator(newsubcond)
            if newsubcond == bcond.args[0]:
                return bcond
            return QCondition(bcond.operator,newsubcond)
        newargs = list(map(validate_condition, bcond.args))
        first, second = newargs
        if all(is_static_condition(arg) for arg in newargs):
            return bcond.operator(first,second)
        if bcond.operator is operator.and_:
            if is_static_condition(first):
                return False if not first else second
            if is_static_condition(second):
                return False if not second else first
        if bcond.operator is operator.or_:
            if is_static_condition(first):
                return True if first else second
            if is_static_condition(second):
                return True if second else first
        if bcond.args == newargs:
            return bcond
        return QCondition(bcond.operator,*newargs)

    # Check comparison condition - at least one argument must be a predicate path
    def validate_comp_condition(ccond):
        for a in ccond.args:
            if isinstance(a,PredicatePath):
                check_path(a)
        return StandardComparator.from_where_qcondition(ccond)

    # Validate a condition
    def validate_condition(cond):
        if isinstance(cond,Placeholder):
            raise ValueError((f"Invalid 'where' condition '{cond}' in query '{qcond}': a "
                              "placeholder must be part of a comparison condition"))
        if isinstance(cond,PredicatePath):
            raise ValueError((f"Invalid 'where' condition '{cond}' in query '{qcond}': a "
                              "reference to fact (or fact field) must be part of "
                              "a comparison condition"))

        if callable(cond):
            return validate_callable(cond)
        if isinstance(cond,FuncInputSpec):
            return FunctionComparator.from_specification(cond.paths,cond.functor)
        if isinstance(cond,Comparator):
            return cond
        if is_boolean_qcondition(cond):
            return validate_bool_condition(cond)
        if is_comparison_qcondition(cond):
            return validate_comp_condition(cond)
        return bool(cond)

#    where = validate_condition(qcond)
#    hroots = set([ hashable_path(r) for r in where.roots])
    return  validate_condition(qcond)

# ------------------------------------------------------------------------------
# negate a query condition and push the negation into the leaf nodes - note:
# input must have already been validated. Because we can negate all comparison
# conditions we therefore end up with no explicitly negated boolean conditions.
# ------------------------------------------------------------------------------
def negate_where_expression(qcond):

    # Note: for the not operator negate twice to force negation inward
    def negate_bool_condition(bcond: StandardComparator) -> Union[QCondition, Comparator]:
        if bcond.operator is operator.not_:
            return negate_condition(negate_condition(bcond.args[0]))
        negated = map(negate_condition, bcond.args)
        if bcond.operator is operator.and_:
            return or_(*negated)
        if bcond.operator is operator.or_:
            return and_(*negated)
        raise TypeError((f"Internal bug: unknown boolean operator '{bcond.operator}' "
                         f"in query condition '{qcond}'"))

    # Negate the condition
    def negate_condition(cond: Any) -> Union[QCondition, Comparator]:
        if isinstance(cond, Comparator):
            return cond.negate()
        if not is_boolean_qcondition(cond):
            raise TypeError((f"Internal bug: unexpected non-boolean condition '{cond}' "
                             f"in query '{qcond}'"))
        return negate_bool_condition(cond)

    return negate_condition(qcond)

# ------------------------------------------------------------------------------
# Convert the where expression to negation normal form by pushing any negations
# inwards. Because we can negate all comparators we therefore end up with no
# explicit negated boolean conditions. Note: input must have been validated
# ------------------------------------------------------------------------------
def where_expression_to_nnf(where):
    # Because negate_where_expression pushes negation inward, so negating twice
    # produces a formula in NNF
    return negate_where_expression(negate_where_expression(where))

# ------------------------------------------------------------------------------
# Convert the query condition to conjunctive normal form. Because we can negate
# all comparison conditions we therefore end up with no explicit negated boolean
# conditions.  Note: input must have been validated
# ------------------------------------------------------------------------------
def where_expression_to_cnf(qcond):

    def dist_if_or_over_and(bcond):
        if bcond.operator is not operator.or_:
            return bcond
        if isinstance(bcond.args[0],QCondition):
            if bcond.args[0].operator is operator.and_:
                x = bcond.args[0].args[0]
                y = bcond.args[0].args[1]
                return and_(or_(x,bcond.args[1]),or_(y,bcond.args[1]))
        if isinstance(bcond.args[1],QCondition):
            if bcond.args[1].operator is operator.and_:
                x = bcond.args[1].args[0]
                y = bcond.args[1].args[1]
                return and_(or_(bcond.args[0],x),or_(bcond.args[0],y))
        return bcond

    def bool_condition_to_cnf(bcond):
        oldbcond = bcond
        while True:
            bcond = dist_if_or_over_and(oldbcond)
            if bcond is oldbcond:
                break
            oldbcond = bcond
        arg0 = condition_to_cnf(bcond.args[0])
        arg1 = condition_to_cnf(bcond.args[1])
        if arg0 is bcond.args[0] and arg1 is bcond.args[1]:
            return bcond
        return QCondition(bcond.operator,arg0,arg1)

    def condition_to_cnf(cond):
        if isinstance(cond, Comparator):
            return cond
        if not is_boolean_qcondition(cond):
            raise TypeError((f"Internal bug: unexpected non-boolean condition '{cond}' "
                             f"in 'where' expression '{qcond}'"))
        if cond.operator is operator.not_:
            cond = where_expression_to_nnf(cond)
            if isinstance(cond,Comparator):
                return cond
        return bool_condition_to_cnf(cond)
    return condition_to_cnf(qcond)

# ------------------------------------------------------------------------------
# A Clause is a list of comparisons that should be interpreted as a disjunction.
# ------------------------------------------------------------------------------

class Clause:

    def __init__(self, comparators: Iterable[StandardComparator]) -> None:
        if not comparators:
            raise ValueError("Empty list of comparison expressions")
        non_comp = next((comp for comp in comparators if not isinstance(comp, Comparator)), None)
        if non_comp:
            raise ValueError(f"Internal bug: only Comparator objects allowed: {non_comp} ")
        self._comparators = tuple(comparators)

        tmppaths = set([])
        tmproots = set([])
        for comp in self._comparators:
            for p in comp.paths:
                tmppaths.add(hashable_path(p))
            for r in comp.roots:
                tmproots.add(hashable_path(r))
        self._paths=tuple(map(path, tmppaths))
        self._roots=tuple(map(path, tmproots))


    def make_callable(self, root_signature):
        callables = [ c.make_callable(root_signature) for c in self._comparators]
        callables = tuple(callables)
        return lambda facts: any(c(facts) for c in callables)

    @property
    def paths(self):
        return self._paths

    @property
    def placeholders(self):
        return set(itertools.chain.from_iterable(
            [p.placeholders for p in self._comparators]))

    @property
    def roots(self):
        return self._roots

    @property
    def executable(self):
        return all(c.executable for c in self._comparators)

    def dealias(self):
        newcomps = [ c.dealias() for c in self._comparators]
        return self if newcomps == self._comparators else Clause(newcomps)

    def fixed(self):
        newcomps = tuple((c.fixed() for c in self._comparators))
        return self if self._comparators == newcomps else Clause(newcomps)

    def ground(self,*args, **kwargs):
        newcomps = tuple((comp.ground(*args,**kwargs) for comp in self._comparators))
        return self if newcomps == self._comparators else Clause(newcomps)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._comparators == other._comparators

    def __len__(self):
        return len(self._comparators)

    def __getitem__(self, idx):
        return self._comparators[idx]

    def __iter__(self):
        return iter(self._comparators)

    def __hash__(self):
        return hash(self._comparators)

    def __str__(self):
        return f"[ {' | '.join([str(c) for c in self._comparators])} ]"

    def __repr__(self):
        return self.__str__()


# ------------------------------------------------------------------------------
# A group of clauses. This should be interpreted as a conjunction of clauses.
# We want to maintain multiple blocks where each block is identified by a single
# predicate/alias root or by a catch all where there is more than one. The idea
# is that the query will consists of multiple queries for each predicate/alias
# type. Then the joins are performed and finally the joined tuples are filtered
# by the multi-root clause block.
# ------------------------------------------------------------------------------

class ClauseBlock:

    def __init__(self, clauses: Optional[Iterable[Clause]]=None) -> None:
        self._clauses = tuple(clauses if clauses else [])
        if not clauses:
            raise ValueError("Empty list of clauses")

        tmppaths = set([])
        tmproots = set([])
        non_clause = next((clause for clause in self._clauses if not isinstance(clause, Clause)), None)
        if non_clause:
            raise ValueError(("A ClauseBlock must consist of a list of "
                            f"Clause elements. '{non_clause}' is of type "
                            f"'{type(non_clause)}'"))
        for clause in self._clauses:
            tmppaths.update(map(hashable_path, clause.paths))
            tmproots.update(map(hashable_path, clause.roots))
        self._paths=tuple(map(path, tmppaths))
        self._roots=tuple(map(path, tmproots))

    @property
    def paths(self):
        return self._paths

    @property
    def placeholders(self):
        return set(itertools.chain.from_iterable(
            [c.placeholders for c in self._clauses]))

    @property
    def roots(self):
        return self._roots

    @property
    def clauses(self):
        return self._clauses

    @property
    def executable(self):
        return all(cl.executable for cl in self._clauses)

    def fixed(self):
        newclauses = tuple((cl.fixed() for cl in self._clauses))
        return self if self._clauses == newclauses else ClauseBlock(newclauses)

    def ground(self,*args, **kwargs):
        newclauses = [ clause.ground(*args,**kwargs) for clause in self._clauses]
        return self if self._clauses == newclauses else ClauseBlock(newclauses)

    def dealias(self):
        newclauses = [ c.dealias() for c in self._clauses]
        return self if self._clauses == newclauses else ClauseBlock(newclauses)

    def make_callable(self, root_signature):
        callables = tuple((c.make_callable(root_signature) for c in self._clauses))
        return lambda facts: all(c(facts) for c in callables)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return ClauseBlock(self._clauses + other._clauses)

    def __radd__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return ClauseBlock(other._clauses + self._clauses)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._clauses == other._clauses

    def __len__(self):
        return len(self._clauses)

    def __getitem__(self, idx):
        return self._clauses[idx]

    def __iter__(self):
        return iter(self._clauses)

    def __hash__(self):
        return hash(self._clauses)

    def __str__(self):
        return f"( {' & '.join([str(c) for c in self._clauses])} )"

    def __repr__(self):
        return self.__str__()

# ------------------------------------------------------------------------------
# Normalise takes a formula and turns it into a clausal CNF (a list of
# disjunctive clauses). Note: input must have been validated.
# ------------------------------------------------------------------------------
def normalise_where_expression(qcond):
    NEWCL = "new_clause"
    stack=[NEWCL]

    def is_leaf(arg):
        return isinstance(arg, Comparator)

    def stack_add(cond):
        if is_leaf(cond):
            stack.append(cond)
        else:
            for arg in cond.args:
                if cond.operator is operator.and_:
                    stack.append(NEWCL)
                stack_add(arg)
                if cond.operator is operator.and_:
                    stack.append(NEWCL)

    def build_clauses():
        clauses = []
        tmp = []
        for a in stack:
            if a == NEWCL:
                if tmp:
                    clauses.append(Clause(tmp))
                tmp = []
            elif a != NEWCL:
                tmp.append(a)
        if tmp:
            clauses.append(Clause(tmp))
        return clauses

    stack.append(NEWCL)
    stack_add(where_expression_to_cnf(qcond))
    return (ClauseBlock(build_clauses()))


# ------------------------------------------------------------------------------
#  process_where takes a where expression from the user select statement as well
#  as a list of roots; validates the where statement (ensuring that it only
#  refers to paths derived from one of the roots) and turns it into CNF in the
#  form of a clauseblock.
#  ------------------------------------------------------------------------------

def process_where(expression, roots=None):
    roots = roots if roots else []

    if not (isinstance(expression,(QCondition, Comparator)) or \
        callable(expression) and not isinstance(expression,PredicatePath)):
        raise TypeError(f"'{expression}' is not a valid query 'where' expression")

    where = validate_where_expression(expression,roots)
    return normalise_where_expression(where)

# ------------------------------------------------------------------------------
# Given a list of clauses breaks the clauses up into two pairs. The first
# contains a list of clausal blocks consisting of blocks for clauses that refer
# to only one root path. The second is a catch all block containing clauses that
# references multiple roots. The second can also be None. This is used to break
# up the query into separate parts for the different join components
# ------------------------------------------------------------------------------
def partition_clauses(clauses=None):
    clauses = clauses if clauses else []
    catchall = []
    root2clauses = {}
    # Separate the clauses
    for clause in clauses:
        roots = list(map(hashable_path, clause.roots))
        if len(roots) == 1:
            root2clauses.setdefault(roots[0],[]).append(clause)
        else:
            catchall.append(clause)

    # Generate clause blocks
    clauseblocks = [ClauseBlock(clauses)  for clauses in root2clauses.values()]

    return (clauseblocks, ClauseBlock(catchall) if catchall else None)

# ------------------------------------------------------------------------------
# To support database-like inner joins.  Join conditions are made from
# QCondition objects with the standard comparison operators
# ------------------------------------------------------------------------------

def is_join_qcondition(cond):
    if not isinstance(cond, QCondition):
        return False
    spec = StandardComparator.operators.get(cond.operator, None)
    return spec.join if spec is not None else False

# ------------------------------------------------------------------------------
# validate join expression. Turns QCondition objects into Join objects Note:
# joinall (the trueall operator) are used for ensure a connected graph but are
# then removed as they don't add anything.
# ------------------------------------------------------------------------------

def validate_join_expression(qconds, roots):
    jroots = set()    # The set of all roots in the join clauses
    joins = []        # The list of joins
    edges = {}       # Edges to ensure a fully connected graph

    def add_join(join):
        nonlocal edges, joins, jroots

        joins.append(join)
        jr = {hashable_path(r) for r in join.roots}
        jroots.update(jr)
        if len(jr) != 2:
            raise ValueError(("Internal bug: join specification should have "
                              f"exactly two root paths: '{jr}'"))
        x,y = jr
        edges.setdefault(x,[]).append(y)
        edges.setdefault(y,[]).append(x)
        remain = jr - broots
        if remain:
            raise ValueError((f"Join specification '{jr}' contains unmatched "
                              f"root paths '{remain}'"))

    # Check that the join graph is connected by counting the visited nodes from
    # some starting point.
    def is_connected():
        nonlocal edges, joins, jroots
        visited = set()
        def visit(r):
            visited.add(r)
            for c in edges[r]:
                if c in visited:
                    continue
                visit(c)

        for start in jroots:
            visit(start)
            break
        return visited == jroots

    # Make sure we have a set of hashable paths
    try:
        broots = set(map(hashable_path, roots))
    except Exception as e:
        raise ValueError(f"Invalid predicate paths signature {roots}: {e}") from None
    if not broots:
        raise ValueError(("Specification of join without root paths doesn't make sense"))
    for p in broots:
        if not p.path.meta.is_root:
            raise ValueError(f"Invalid field specification {p} does not refer"
                              "to the root of a predicate path ")

    for qcond in qconds:
        if not is_join_qcondition(qcond):
            if not isinstance(qcond,QCondition):
                raise ValueError(f"Invalid join element '{qcond}': expecting a "
                                  "comparison specifying the join between two fields")

            raise ValueError(f"Invalid join operator '{qcond.operator}' in {qcond}")

        add_join(StandardComparator.from_join_qcondition(qcond))

    # Check that we have all roots in the join matches the base roots
    if jroots != broots:
        raise ValueError(f"Invalid join specification: missing joins for '{broots-jroots}'")
    if not is_connected():
        raise ValueError(f"Invalid join specification: contains un-joined components '{qconds}'")

    # Now that we've validated the graph can remove all the pure
    # cross-product/join-all joins.
    return list(filter(lambda x: x.operator is not trueall, joins))


# ------------------------------------------------------------------------------
#  process_join takes a join expression (a list of join statements) from the
#  user select statement as well as a list of roots; validates the join
#  statements (ensuring that they only refers to paths derived from one of the
#  roots) and turns it into a list of validated StandardComparators that have
#  paths as both arguments
#  ------------------------------------------------------------------------------

def process_join(join_expression: Iterable[Any], roots: Optional[Iterable]=None) -> List:
    j = next((j for j in join_expression if not isinstance(j, QCondition)), None)
    if j:
        raise TypeError(f"'{j}' ({type(j)}) is not a valid 'join' in '{join_expression}'")

    return validate_join_expression(join_expression,roots if roots else [])


#------------------------------------------------------------------------------
# Specification of an ordering over a field of a predicate/complex-term
#------------------------------------------------------------------------------
class OrderBy:
    def __init__(self, path_, asc_):
        self._path = path_
        self._asc = asc_

    @property
    def path(self):
        return self._path

    @property
    def asc(self):
        return self._asc

    def dealias(self):
        dealiased = path(self._path).meta.dealiased
        if hashable_path(self._path) == hashable_path(dealiased):
            return self
        return OrderBy(dealiased,self._asc)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if hashable_path(self._path) != hashable_path(other._path):
            return False
        return self._asc == other._asc

    def __hash__(self):
        return hash((hashable_path(self._path),self._asc))

    def __str__(self):
        template = "asc({0})" if self._asc else "desc({0})"
        return template.format(self._path)

    def __repr__(self):
        return self.__str__()

#------------------------------------------------------------------------------
# Helper functions to return a OrderBy in descending and ascending order. Input
# is a PredicatePath. The ascending order function is provided for completeness
# since the order_by parameter will treat a path as ascending order by default.
# ------------------------------------------------------------------------------
def desc(pth):
    return OrderBy(path(pth),asc_=False)
def asc(pth):
    return OrderBy(path(pth),asc_=True)

# ------------------------------------------------------------------------------
# OrderByBlock groups together an ordering of OrderBy statements
# ------------------------------------------------------------------------------

class OrderByBlock:
    def __init__(self,orderbys: Optional[Iterable[OrderBy]]=None) -> None:
        self._orderbys = tuple(orderbys if orderbys else [])
        self._paths = tuple((path(ob.path) for ob in self._orderbys))
#        if not orderbys:
#            raise ValueError("Empty list of order_by statements")

    @property
    def paths(self):
        return self._paths

    @property
    def roots(self):
        return {hashable_path(p.meta.root) for p in self._paths}

    def dealias(self):
        neworderbys = tuple((ob.dealias() for ob in self._orderbys))
        return self if self._orderbys == neworderbys else OrderByBlock(neworderbys)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._orderbys == other._orderbys
        if isinstance(other, tuple):
            return self._orderbys == other
        if isinstance(other, list):
            return self._orderbys == tuple(other)
        return NotImplemented

    def __len__(self):
        return len(self._orderbys)

    def __getitem__(self, idx):
        return self._orderbys[idx]

    def __iter__(self):
        return iter(self._orderbys)

    def __hash__(self):
        return hash(self._orderbys)

    def __bool__(self):
        return bool(self._orderbys)

    def __str__(self):
        return f"[{','.join([str(ob) for ob in self._orderbys])}]"

    def __repr__(self):
        return self.__str__()


# ------------------------------------------------------------------------------
# Validate the order_by expression - returns an OrderByBlock
# ------------------------------------------------------------------------------

def validate_orderby_expression(
    orderby_expressions: Iterable,
    roots: Optional[Iterable]=None
) -> OrderByBlock:
    roots = roots if roots else []
    if not is_root_paths(roots):
        raise ValueError(f"roots='{roots}' must contain only root paths")
    hroots = {hashable_path(rp) for rp in roots}

    path_ordering = []
    # If only a PredicatePath is specified assume ascending order
    for exp in orderby_expressions:
        if isinstance(exp, OrderBy):
            path_ordering.append(exp)
        elif isinstance(exp, PredicatePath):
            path_ordering.append(asc(exp))
        elif inspect.isclass(exp) and issubclass(exp, Predicate):
            path_ordering.append(asc(path(exp)))
        else: raise ValueError(f"Invalid 'order_by' expression: {exp}")
    obb = OrderByBlock(path_ordering)

    if  not obb.roots.issubset(hroots):
        raise ValueError(f"Invalid 'order_by' expression '{obb}' refers to root paths that "
                    f"are not in '{hroots}'")
    return obb

# ------------------------------------------------------------------------------
# Return an OrderByBlock corresponding to the validated order by expression
# ------------------------------------------------------------------------------

def process_orderby(orderby_expressions, roots=None):
    return validate_orderby_expression(orderby_expressions,roots)

# ------------------------------------------------------------------------------
# Return an OrderByBlock for an ordered flag
# ------------------------------------------------------------------------------

def process_ordered(roots):
    ordering=[asc(r) for r in roots]
    return OrderByBlock(ordering)

# ------------------------------------------------------------------------------
# make_prejoin_pair(indexed_paths, clauses)
#
# Given a set of indexed paths and a set of clauses that refer to a single root
# try to extract a preferred clause that can be used for indexing.
#
# - indexed_paths - a list of paths for which there is a factindex
# - clauses - a clause block that can only refer to a single root
# ------------------------------------------------------------------------------
def make_prejoin_pair(indexed_paths, clauseblock):
    def preference(cl):
        c = min(cl, key=lambda c: c.preference)
        return c.preference

    def is_candidate_sc(indexes, sc):
        if len(sc.paths) != 1:
            return False
        return hashable_path(sc.paths[0].meta.dealiased) in indexes

    def is_candidate(indexes, cl):
        return all(isinstance(c, StandardComparator) and is_candidate_sc(indexes, c) for c in cl)

    if not clauseblock:
        return (None,None)

    tmp = {hashable_path(p.meta.dealiased) for p in clauseblock.paths}
    indexes = set(filter(lambda x: x in tmp,
                         [hashable_path(p) for p in indexed_paths]))

    # Search for a candidate to use with a fact index
    candidates = []
    rest = []
    for cl in clauseblock:
        if is_candidate(indexes, cl):
            candidates.append(cl)
        else:
            rest.append(cl)
    if not candidates:
        return (None, clauseblock)

    # order the candidates by their comparator preference and take the first
    candidates.sort(key=preference, reverse=True)
    rest.extend(candidates[1:])
    cb = ClauseBlock(rest) if rest else None
    return (candidates[0],cb)

# ------------------------------------------------------------------------------
# make_join_pair(joins, clauseblock)
# - a list of join StandardComparators
# - an existing clauseblock (or None)
# - a list of orderby statements
#
# Takes a list of joins and picks the best one for indexing (based on their
# operator preference and the orderby statements). Returns a pair that is the
# chosen join and the rest of the joins added to the input clauseblock.
# ------------------------------------------------------------------------------
def make_join_pair(
    joins: List[StandardComparator],
    clauseblock: Optional[ClauseBlock],
    orderbys: Optional[OrderByBlock]=None
) -> Tuple[Optional[StandardComparator], Optional[ClauseBlock]]:
    opaths={hashable_path(ob.path) for ob in orderbys} if orderbys else set([])
    def num(sc):
        return len(opaths & {hashable_path(p) for p in sc.paths})

    if not joins:
        return (None,clauseblock)
    joins = sorted(joins, key=lambda x : (x.preference, num(x)), reverse=True)
    joinsc, *remainder = joins
    if remainder:
        block = ClauseBlock([Clause([sc]) for sc in remainder])
        if clauseblock:
            return (joinsc, clauseblock + block)
        return (joinsc, block)

    return (joinsc, clauseblock if clauseblock else None)

# ------------------------------------------------------------------------------
# JoinQueryPlan support functions. The JQP is a part of the query plan that
# describes the plan to execute a single link in a join.
# ------------------------------------------------------------------------------

# Check that the formula only refers to paths with the allowable roots
def _check_roots(allowable_roots, formula):
    if not formula:
        return True
    allowable_hroots = set(map(hashable_path, allowable_roots))
    hroots = set(map(hashable_path, formula.roots))
    return hroots.issubset(allowable_hroots)

# Align the arguments in a standard comparator so that the first argument is a
# path whose root is the given root
def _align_sc_path(root, sc):
    hroot = hashable_path(root)
    if not sc:
        return None
    if isinstance(sc.args[0], PredicatePath) and \
       hashable_path(sc.args[0].meta.root) == hroot:
        return sc
    sc = sc.swap()
    if not isinstance(sc.args[0], PredicatePath) or \
       hashable_path(sc.args[0].meta.root) != hroot:
        raise ValueError((f"Cannot align key comparator '{root}' with root '{sc}' since "
                           "it doesn't reference the root"))
    return sc

# Extract the placeholders
def _extract_placeholders(elements):
    return set().union(*[f.placeholders for f in elements if f])

# ------------------------------------------------------------------------------
# JoinQueryPlan class is a single join within a broader QueryPlan
# ------------------------------------------------------------------------------

class JoinQueryPlan:
    '''Input:
       - input_signature tuple,
       - root,
       - indexes associated with the underlying fact type
       - a prejoin clause for indexed quering (or None),
       - a prejoin clauseblock (or None),
       - a prejoin orderbyblock (or None)
       - a join standard comparator (or None),
       - a postjoin clauseblock (or None)
       - a postjoin orderbyblock (or None)
    '''
    def __init__(self,input_signature, root, indexes,
                 prejoincl, prejoincb, prejoinobb,
                 joinsc, postjoincb, postjoinobb):
        if not indexes:
            indexes = []
        self._insig = tuple(map(path, input_signature))
        self._root = path(root)
        self._predicate = self._root.meta.predicate
        self._indexes = tuple((p for p in indexes \
                               if path(p).meta.predicate == self._predicate))
        self._joinsc = _align_sc_path(self._root, joinsc)
        self._postjoincb = postjoincb
        self._postjoinobb = postjoinobb
        self._prejoincl = prejoincl
        self._prejoincb = prejoincb
        self._prejoinobb = prejoinobb

        # Check that the input signature and root are valid root paths
        if not self._root.meta.is_root:
            raise ValueError(f"Internal bug: '{self._root}' is not a root path")
        for p in self._insig:
            if not p.meta.is_root:
                raise ValueError((f"Internal bug: '{p}' in input signature is not a "
                                  "root path"))

        # The prejoin parts must only refer to the dealised root
        if not _check_roots([self._root.meta.dealiased], prejoincl):
            raise ValueError((f"Pre-join comparator '{prejoincl}' refers to "
                              f"non '{self._root}' paths"))
        if not _check_roots([self._root.meta.dealiased], prejoincb):
            raise ValueError((f"Pre-join clause block '{prejoincb}' refers to "
                              f"non '{self._root}' paths"))
        if not _check_roots([self._root.meta.dealiased], prejoinobb):
            raise ValueError((f"Pre-join clause block '{prejoinobb}' refers to "
                              f"non '{self._root}' paths"))

        # The joinsc cannot have placeholders
        if self._joinsc and self._joinsc.placeholders:
            raise ValueError(("A comparator with a placeholder is not valid as "
                              f"a join specificaton: {joinsc}"))

        # The joinsc, postjoincb, and postjoinobb must refer only to the insig + root
        allroots = list(self._insig) + [self._root]
        if not _check_roots(allroots, joinsc):
            raise ValueError((f"Join comparator '{joinsc}' refers to "
                              f"non '{allroots}' paths"))
        if not _check_roots(allroots, postjoincb):
            raise ValueError((f"Post-join clause block '{postjoincb}' refers to "
                              f"non '{allroots}' paths"))

        if not _check_roots(allroots, postjoinobb):
            raise ValueError((f"Post-join order by block '{postjoinobb}' refers to "
                              f"non '{allroots}' paths"))

        self._placeholders = _extract_placeholders(
            [self._postjoincb, self._prejoincl, self._prejoincb])


    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------
    @classmethod
    def from_specification(cls, indexes, input_signature,
                           root, joins=None, clauses=None,
                           orderbys=None):

        input_signature = list(map(path, input_signature))
        root = path(root)

        rootcbs, catchall = partition_clauses(clauses)
        if not rootcbs:
            (prejoincl,prejoincb) = (None,None)
        elif len(rootcbs) == 1:
            (prejoincl,prejoincb) = make_prejoin_pair(indexes, rootcbs[0])
            prejoincl = prejoincl.dealias() if prejoincl else None
            prejoincb = prejoincb.dealias() if prejoincb else None

        else:
            raise ValueError(("Internal bug: unexpected multiple single root "
                              f"clauses '{rootcbs}' when we expected only "
                              f"clauses for root {root}"))

        (joinsc,postjoincb) = make_join_pair(joins, catchall)

        prejoinobb = None
        postjoinobb = None
        if orderbys:
            orderbys = OrderByBlock(orderbys)
            hroots = [ hashable_path(r) for r in orderbys.roots ]
            postjoinobb = OrderByBlock(orderbys)

# BUG: NEED TO RETHINK BELOW - THE FOLLOWING ONLY WORKS IN SPECIAL CASES

#       if len(hroots) > 1 or hroots[0] !=
#            hashable_path(root): postjoinobb = OrderByBlock(orderbys) else:
#            prejoinobb = orderbys.dealias()

        return cls(input_signature,root, indexes,
                   prejoincl,prejoincb,prejoinobb,
                   joinsc,postjoincb,postjoinobb)


    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------
    @property
    def input_signature(self):
        return self._insig

    @property
    def root(self):
        return self._root

    @property
    def indexes(self):
        return self._indexes

    @property
    def prejoin_key_clause(self) -> Optional[Clause]:
        return self._prejoincl

    @property
    def join_key(self) -> Optional[StandardComparator]:
        return self._joinsc

    @property
    def prejoin_clauses(self) -> Optional[ClauseBlock]:
        return self._prejoincb

    @property
    def prejoin_orderbys(self) -> Optional[OrderByBlock]:
        return self._prejoinobb

    @property
    def postjoin_clauses(self):
        return self._postjoincb

    @property
    def postjoin_orderbys(self):
        return self._postjoinobb

    @property
    def placeholders(self):
        return self._placeholders

    @property
    def executable(self):
        if self._prejoincl and not self._prejoincl.executable:
            return False
        if self._prejoincb and not self._prejoincb.executable:
            return False
        if self._postjoincb and not self._postjoincb.executable:
            return False
        return True

    def ground(self,*args,**kwargs):
        gprejoincl  = self._prejoincl.ground(*args,**kwargs)  if self._prejoincl else None
        gprejoincb  = self._prejoincb.ground(*args,**kwargs)  if self._prejoincb else None
        gpostjoincb = self._postjoincb.ground(*args,**kwargs) if self._postjoincb else None

        if gprejoincl == self._prejoincl and gprejoincb == self._prejoincb and \
           gpostjoincb == self._postjoincb: return self
        return JoinQueryPlan(self._insig,self._root, self._indexes,
                             gprejoincl,gprejoincb,self._prejoinobb,
                             self._joinsc,gpostjoincb,self._postjoinobb)

    def print(self,file=sys.stdout,pre=""):
        print(f"{pre}QuerySubPlan:", file=file)
        print(f"{pre}\tInput Signature: {self._insig}", file=file)
        print(f"{pre}\tRoot path: {self._root}", file=file)
        print(f"{pre}\tIndexes: {self._indexes}", file=file)
        print(f"{pre}\tPrejoin keyed search: {self._prejoincl}", file=file)
        print(f"{pre}\tPrejoin filter clauses: {self._prejoincb}", file=file)
        print(f"{pre}\tPrejoin order_by: {self._prejoinobb}", file=file)
        print(f"{pre}\tJoin key: {self._joinsc}", file=file)
        print(f"{pre}\tPost join clauses: {self._postjoincb}", file=file)
        print(f"{pre}\tPost join order_by: {self._postjoinobb}", file=file)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        self_tuple = (self._insig, self._root.meta.hashable, self._prejoincl, self._prejoincb,
                      self._prejoinobb, self._joinsc, self._postjoincb)
        other_tuple = (other._insig, other._root.meta.hashable, other._prejoincl, other._prejoincb,
                       other._prejoinobb, other._joinsc, other._postjoincb)
        return self_tuple == other_tuple

    def __str__(self):
        with io.StringIO() as out:
            self.print(out)
            result=out.getvalue()
        return result

    def __repr__(self):
        return self.__str__()


# ------------------------------------------------------------------------------
# QueryPlan is a complete plan for a query. It consists of a sequence of
# JoinQueryPlan objects that represent increasing joins in the query.
# ------------------------------------------------------------------------------

class QueryPlan:
    def __init__(self, subplans: Iterable[JoinQueryPlan]) -> None:
        if not subplans:
            raise ValueError("An empty QueryPlan is not valid")
        sig = []
        for jqp in subplans:
            insig = [hashable_path(rp) for rp in jqp.input_signature]
            if sig != insig:
                raise ValueError(("Invalid 'input_signature' for JoinQueryPlan. "
                                  f"Got '{insig}' but expecting '{sig}'"))
            sig.append(hashable_path(jqp.root))
        self._jqps = tuple(subplans)

    @property
    def placeholders(self):
        return set(itertools.chain.from_iterable(
            [jqp.placeholders for jqp in self._jqps]))

    @property
    def output_signature(self):
        jqp = self._jqps[-1]
        return tuple(jqp.input_signature + (jqp.root,))

    @property
    def executable(self):
        return all(jqp.executable for jqp in self._jqps)

    def ground(self,*args,**kwargs):
        newqpjs = [qpj.ground(*args,**kwargs) for qpj in self._jqps]
        return self if tuple(newqpjs) == self._jqps else QueryPlan(newqpjs)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._jqps == other._jqps

    def __len__(self):
        return len(self._jqps)

    def __getitem__(self, idx):
        return self._jqps[idx]

    def __iter__(self):
        return iter(self._jqps)

    def print(self,file=sys.stdout,pre=""):
        print("------------------------------------------------------",file=file)
        for qpj in self._jqps:
            qpj.print(file,pre)
        print("------------------------------------------------------",file=file)

    def __str__(self):
        with io.StringIO() as out:
            self.print(out)
            result=out.getvalue()
        return result

    def __repr__(self):
        return self.__str__()


# ------------------------------------------------------------------------------
# Sort the orderby statements into partitions based on the root_join_order,
# where an orderby statement cannot appear at an index before its root
# node. Note: there will be exactly the same number of partitions as the number
# of roots.
# ------------------------------------------------------------------------------
def make_orderby_partitions(root_join_order,orderbys=None):
    if not orderbys:
        return [OrderByBlock([]) for _ in root_join_order]

    visited=set({})
    orderbys=list(orderbys)

    # For a list of orderby statements return the largest pure subsequence that
    # only refers to the visited root nodes
    def visitedorderbys(visited, obs):
        out = []
        count = 0
        for ob in obs:
            if hashable_path(ob.path.meta.root) in visited:
                count += 1
                out.append(ob)
            else: break
        while count > 0:
            obs.pop(0)
            count -= 1
        return out

    partitions = []
    for root in root_join_order:
        visited.add(hashable_path(root))
        part = visitedorderbys(visited, orderbys)
        partitions.append(OrderByBlock(part))

    return partitions


# ------------------------------------------------------------------------------
# Remove the gaps between partitions by moving partitions down
# ------------------------------------------------------------------------------
def remove_orderby_gaps(partitions):
    def gap(partitions):
        startgap = -1
        for idx,obs in enumerate(partitions):
            if not obs:
                continue
            if startgap == -1 or (startgap != -1 and startgap == idx-1):
                startgap = idx
            elif startgap != -1:
                return (startgap,idx)
        return (-1,-1)

    # Remove any gaps by merging
    while True:
        startgap, endgap = gap(partitions)
        if startgap == -1:
            break
        partitions[endgap-1] = partitions[startgap]
        partitions[startgap] = OrderByBlock([])
    return partitions


# ------------------------------------------------------------------------------
# After the first orderby partition all subsequent partitions can only refer to their
# own root. So start from the back of the list and move up till we find a
# non-root partition then pull everything else down into this partition.
# ------------------------------------------------------------------------------

def merge_orderby_partitions(root_join_order, partitions):
    partitions = list(partitions)
    root_join_order = [ hashable_path(r) for r in root_join_order ]

    # Find the last (from the end) non-root partition
    nridx = 0
    for idx, part in reversed(list(enumerate(partitions))):
        if part:
            hroots = [ hashable_path(r) for r in part.roots ]
            if len(hroots) > 1 or hroots[0] != root_join_order[idx]:
                nridx = idx
                break
    if nridx == 0:
        return partitions

    # Now merge all other partitions from 0 to nridx-1 into nridx
    bigone = []
    tozero=[]
    for idx,part in (enumerate(partitions)):
        if idx > nridx:
            break
        if part:
            bigone = list(bigone) + list(part)
            tozero.append(idx)
            break
    if not bigone:
        return partitions
    for idx in tozero:
        partitions[idx] = OrderByBlock([])
    partitions[nridx] = OrderByBlock(bigone + list(partitions[nridx]))
    return partitions

# ------------------------------------------------------------------------------
# guaranteed to return a list the same size as the root_join_order.  The ideal
# case is that the orderbys are in the same order as the root_join_order.  BUG
# NOTE: This partitioning scheme is flawed. It only works in a special case
# (when the join clause matches the ordering statement). See below for temporary
# fix.
# ------------------------------------------------------------------------------

def partition_orderbys(root_join_order, orderbys=None):
    partitions = make_orderby_partitions(root_join_order,orderbys)
    partitions = remove_orderby_gaps(partitions)
    partitions = merge_orderby_partitions(root_join_order,partitions)
    return partitions


# ------------------------------------------------------------------------------
# Because of the logical bug generating valid sorts (what I was doing previously
# only works for a special case), a temporary solution is to merge all
# partitions into the lowest root with an orderby statement.
# ------------------------------------------------------------------------------

def partition_orderbys_simple(root_join_order, orderbys=None):
    partitions = [OrderByBlock([])]*len(root_join_order)

    if not orderbys:
        return partitions
    visited = {hashable_path(ob.path.meta.root) for ob in orderbys}

    # Loop through the root_join_order until all orderby statements have been
    # visited.
    for i,root in enumerate(root_join_order):
        rp = hashable_path(root)
        visited.discard(rp)
        if not visited:
            partitions[i] = OrderByBlock(orderbys)
            return partitions
    raise RuntimeError("Shouldn't reach here")


#------------------------------------------------------------------------------
# QuerySpec stores all the parameters needed to generate a query plan in one
# data-structure
# ------------------------------------------------------------------------------

class QuerySpec:
    allowed = [ "roots", "join", "where", "order_by", "ordered",
                "group_by", "tuple", "distinct", "bind", "select",
                "heuristic", "joh" ]

    if TYPE_CHECKING:
        roots: Tuple[Type[Predicate], ...] = tuple([])
        join: List[StandardComparator] = []
        where: Optional[ClauseBlock] = None
        order_by: Optional[OrderByBlock] = None
        group_by: Optional[OrderByBlock] = None
        bind: bool = False
        tuple: bool = False
        distinct: bool = False
        heuristic: bool = False
        ordered: bool = False
        select: Optional[Tuple[Type[Predicate], ...]] = None
        @staticmethod
        def joh(index_paths: Iterable[PredicatePath.Hashable], qspec: "QuerySpec") -> List[PredicatePath]:
            ...

    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            if k not in QuerySpec.allowed:
                raise ValueError(f"Trying to set unknown parameter '{k}'")
            if v is None:
                raise ValueError((f"Error for QuerySpec parameter '{k}': 'None' "
                                  "values are not allowed"))
        self._params = dict(kwargs)

    # Return a new QuerySpec with added parameters
    def newp(self, **kwargs):
        if not kwargs:
            return self
        nparams = dict(self._params)
        for k,v in kwargs.items():
            if v is None:
                raise ValueError(f"Cannot specify empty '{v}'")
            if k in self._params:
                raise ValueError(f"Cannot specify '{k}' multiple times")
            nparams[k] = v
        return QuerySpec(**nparams)

    # Return a new QuerySpec with modified parameters
    def modp(self, **kwargs):
        if not kwargs:
            return self
        nparams = dict(self._params)
        for k,v in kwargs.items():
            if v is None:
                raise ValueError("Cannot specify empty '{v}'")
            nparams[k] = v
        return QuerySpec(**nparams)

    # Return a new QuerySpec with specified parameters deleted
    def delp(self, keys=None):
        if not keys:
            return self
        nparams = dict(self._params)
        for k in keys:
            nparams.pop(k,None)
        return QuerySpec(**nparams)

    # Return the value of a parameter - behaves slightly differently to simply
    # specify the parameter as an attribute because you can return a default
    # value if the parameter is not set.
    def getp(self,name,default=None):
        return self._params.get(name,default)


    def bindp(self, *args, **kwargs):
        where = self.where
        if where is None:
            raise ValueError("'where' must be specified before binding placeholders")
        np = {}
        pp = {}
        for p in where.placeholders:
            if isinstance(p, NamedPlaceholder):
                np[p.name] = p
            elif isinstance(p, PositionalPlaceholder):
                pp[p.posn] = p
        for idx, v in enumerate(args):
            if idx not in pp:
                raise ValueError((f"Trying to bind value '{v}' to positional "
                                  f"argument '{idx}'  but there is no corresponding "
                                  f"positional placeholder in where clause '{where}'"))
        for k,v in kwargs.items():
            if k not in np:
                raise ValueError((f"Trying to bind value '{v}' to named "
                                  f"argument '{k}' but there is no corresponding "
                                  f"named placeholder in where clause '{where}'"))
        nwhere = where.ground(*args, **kwargs)
        return self.modp(where=nwhere,bind=True)

    def fill_defaults(self):
        toadd = dict(self._params)
        for n in [ "roots","join","where","order_by" ]:
            v = self._params.get(n,None)
            if v is None:
                toadd[n]=[]
        toadd["group_by"] = self._params.get("group_by",[])
        toadd["bind"] = self._params.get("bind",{})
        toadd["tuple"] = self._params.get("tuple",False)
        toadd["distinct"] = self._params.get("distinct",False)
        toadd["heuristic"] = self._params.get("heuristic",False)
        toadd["joh"] = self._params.get("joh",oppref_join_order)

        # Note: No default values for "select" so calling its attribute will
        # return None

        return QuerySpec(**toadd) if toadd else self

    def __getattr__(self, item):
        if item not in QuerySpec.allowed:
            raise ValueError(f"Trying to get the value of unknown parameter '{item}'")
        return self._params.get(item,None)

    def __str__(self):
        return str(self._params)

    def __repr__(self):
        return repr(self._params)


# ------------------------------------------------------------------------------
# Takes a list of paths that have an index, then based on a
# list of root paths and a query specification, builds the queryplan.
# ------------------------------------------------------------------------------

def make_query_plan_preordered_roots(
    indexed_paths: Iterable[PredicatePath.Hashable],
    root_join_order: List[PredicatePath],
    qspec: QuerySpec
) -> QueryPlan:

    joinset= set(qspec.join if qspec.join else [])
    clauseset = set(qspec.where if qspec.where else [])
    orderbys = list(qspec.order_by) if qspec.order_by else []
    visited=set([])

    if not root_join_order:
        raise ValueError("Cannot make query plan with empty root join order")

#    orderbygroups = partition_orderbys(root_join_order, orderbys)
    orderbygroups = partition_orderbys_simple(root_join_order, orderbys)

    # For a set of visited root paths and a set of comparator
    # statements return the subset of join statements that only reference paths
    # that have been visited.  Removes these joins from the original set.
    def visitedsubset(visited, inset):
        outlist=[]
        for comp in inset:
            if visited.issuperset([hashable_path(r) for r in comp.roots]):
                outlist.append(comp)
        for comp in outlist:
            inset.remove(comp)
        return outlist


    # Generate a list of JoinQueryPlan consisting of a root path and join
    # comparator and clauses that only reference previous plans in the list.
    output=[]
    for idx,(root,rorderbys) in enumerate(zip(root_join_order,orderbygroups)):
        if rorderbys:
            rorderbys = OrderByBlock(rorderbys)
        visited.add(hashable_path(root))
        rpjoins = visitedsubset(visited, joinset)
        rpclauses = visitedsubset(visited, clauseset)
        rpclauses = ClauseBlock(rpclauses) if rpclauses else None
        joinsc, rpclauses = make_join_pair(rpjoins, rpclauses,rorderbys)
        if not rpclauses:
            rpclauses = []
        rpjoins = [joinsc] if joinsc else []

        output.append(JoinQueryPlan.from_specification(indexed_paths,
                                                       root_join_order[:idx],
                                                       root,rpjoins,rpclauses,
                                                       rorderbys))
    return QueryPlan(output)

# ------------------------------------------------------------------------------
# Join-order heuristics. The heuristic is a function that takes a set of
# indexes, and a query specification with a set of roots and join/where/order_by
# expressions. It then returns an ordering over the roots that are used to
# determine how the joins are built. To interpret the returned list of root
# paths: the first element will be the outer loop query and the last will be the
# inner loop query.
#
# Providing two fixed heuristics: 1) fixed_join_order is a heuristic generator
# and the user specifies the exact ordering, 2) basic_join_order simply retains
# the ordering given as part of the query specification.
#
# The default heuristic, oppref_join_order, is a operator preference heuristic.
# The idea is to assign a preference value to each join expression based on the
# number of join expressions connected with a root path and the operator
# preference. The higher the value the further it is to the outer loop. The
# intuition is that the joins reduce the number of tuples, so by assigning the
# joins early you generate the fewest tuples. Note: not sure about my intuitions
# here. Need to look more closely at the mysql discussion on query execution.
# ------------------------------------------------------------------------------

def fixed_join_order(*roots):
    def validate(r):
        r=path(r)
        if not r.meta.is_root:
            raise ValueError((f"Bad query roots specification '{roots}': '{r}' is not "
                             "a root path"))
        return r

    if not roots:
        raise ValueError("Missing query roots specification: cannot create "
                         "a fixed join order heuristic from an empty list")
    paths = [validate(r) for r in roots]
    hashables = { hashable_path(r) for r in roots }

    def fixed_join_order_heuristic(indexed_paths, qspec):
        hps = {hashable_path(r) for r in qspec.roots}
        if hps != set(hashables):
            raise ValueError(("Mis-matched query roots: fixed join order "
                              f"heuristic '{roots}' must contain exactly the "
                              f"roots '{qspec.roots}"))
        return list(paths)
    return fixed_join_order_heuristic

def basic_join_order(indexed_paths, qspec):
    return [path(r) for r in qspec.roots]


def oppref_join_order(indexed_paths, qspec):
    roots= qspec.roots
    joins= qspec.join

    root2val = { hashable_path(rp) : 0 for rp in roots }
    for join in joins:
        for rp in join.roots:
            hrp = hashable_path(rp)
            _ = root2val.setdefault(hrp, 0)
            root2val[hrp] += join.preference
    return [path(hrp) for hrp in \
            sorted(root2val.keys(), key = lambda k : root2val[k], reverse=True)]


# ------------------------------------------------------------------------------
# Take a join order heuristic, a list of joins, and a list of clause blocks and
# and generates a query.
# ------------------------------------------------------------------------------

def make_query_plan(
    indexed_paths: Iterable[PredicatePath.Hashable],
    qspec: QuerySpec
) -> QueryPlan:
    qspec = qspec.fill_defaults()
    root_order = qspec.joh(indexed_paths, qspec)
    return make_query_plan_preordered_roots(indexed_paths, root_order, qspec)


#------------------------------------------------------------------------------
# Implementing Queries - taking a QuerySpec, QueryPlan, and a FactMap and
# generating an actual query.
# ------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Creates a mechanism for sorting using the order_by statements within queries.
#
# Works by creating a list of pairs consisting of a keyfunction and reverse
# flag, corresponding to the orderbyblocks in reverse order. A list can then by
# sorted by successively applying each sort function. Stable sort guarantees
# that the the result is a multi-criteria sort.
#------------------------------------------------------------------------------
class InQuerySorter:
    def __init__(self, orderbyblock, insig=None):
        if insig is None and len(orderbyblock.roots) > 1:
            raise ValueError(("Cannot create an InQuerySorter with no input "
                              "signature and an OrderByBlock with multiple "
                              f"roots '{orderbyblock}'"))
        if insig is not None and not insig:
            raise ValueError("Cannot create an InQuerySorter with an empty signature")
        if not insig:
            insig=()

        # Create the list of (keyfunction,reverse flag) pairs then reverse it.
        self._sorter = []
        rp2idx = { hashable_path(rp) : idx for idx,rp in enumerate(insig) }
        for ob in orderbyblock:
            kf = ob.path.meta.attrgetter
            if insig:
                idx = rp2idx[hashable_path(ob.path.meta.root)]
                ig=operator.itemgetter(idx)
                kf = lambda f, kf=kf,ig=ig : kf(ig(f))
            self._sorter.append((kf,not ob.asc))
        self._sorter = tuple(reversed(self._sorter))

    # List in-place sorting
    def listsort(self, inlist):
        for kf, reverse in self._sorter:
            inlist.sort(key=kf,reverse=reverse)

    # Sort an iterable input and return an output list
    def sorted(self, input_):
        outlist = list(input_)
        for kf, reverse in self._sorter:
            outlist.sort(key=kf,reverse=reverse)
        return outlist

# ------------------------------------------------------------------------------
# prejoin query is the querying of the underlying factset or factindex
# - factsets - a dictionary mapping a predicate to a factset
# - factindexes - a dictionary mapping a hashable_path to a factindex
# ------------------------------------------------------------------------------

def make_first_prejoin_query(jqp, factsets, factindexes):
    factset = factsets.get(jqp.root.meta.predicate, FactSet())

    pjk = jqp.prejoin_key_clause
    prejcb = jqp.prejoin_clauses

    # If there is a prejoin key clause then every comparator within it must
    # refer to exactly one index
    if pjk:
        factindex_sc = []

        for sc in pjk:
            keyable = sc.keyable(factindexes)
            if keyable is None:
                raise ValueError((f"Internal error: prejoin key clause '{pjk}' "
                                  f"is invalid for JoinQueryPlan {jqp}"))
            kpath,op,key = keyable
            factindex_sc.append((factindexes[kpath],op,key))

    def unsorted_query():
        if prejcb:
            cc = prejcb.make_callable([jqp.root.meta.dealiased])
        else:
            cc = lambda _ : True

        gen = factset if not pjk else (f for fi,op,key in factindex_sc for f in fi.find(op,key))

        yield from ((f,) for f in gen if cc((f,)))

    return unsorted_query

# ------------------------------------------------------------------------------
#
# - factsets - a dictionary mapping a predicate to a factset
# - factindexes - a dictionary mapping a hashable_path to a factindex
# ------------------------------------------------------------------------------

def make_first_join_query(
    jqp: JoinQueryPlan,
    factsets: Dict[Type[Predicate], FactSet],
    factindexes: Dict[PredicatePath.Hashable, FactIndex]
) -> Callable[[], Iterator[Any]]:

    if jqp.input_signature:
        raise ValueError(("A first JoinQueryPlan must have an empty input "
                          f"signature but '{jqp.input_signature}' found"))
    if jqp.prejoin_orderbys and jqp.postjoin_orderbys:
        raise ValueError(("Internal error: it doesn't make sense to have both "
                          "a prejoin and join orderby sets for the first sub-query"))

    base_query=make_first_prejoin_query(jqp,factsets, factindexes)
    iqs=None
    if jqp.prejoin_orderbys:
        iqs = InQuerySorter(jqp.prejoin_orderbys,(jqp.root,))
    elif jqp.postjoin_orderbys:
        iqs = InQuerySorter(jqp.postjoin_orderbys,(jqp.root,))

    return (lambda: iqs.sorted(base_query())) if iqs else base_query

# ------------------------------------------------------------------------------
# Returns a function that takes no arguments and returns a populated data
# source.  The data source can be either a FactIndex, a FactSet, or a list.  In
# the simplest case this function simply passes through a reference to the
# underlying factset or factindex object. If it is a list then either the order
# doesn't matter or it is sorted by the prejoin_orderbys sort order.
#
# NOTE: We don't use this for the first JoinQueryPlan as that is handled as a
# special case.
# ------------------------------------------------------------------------------

def make_prejoin_query_source(
    jqp: JoinQueryPlan,
    factsets: Dict[Type[Predicate], FactSet],
    factindexes: Dict[PredicatePath.Hashable, FactIndex]
) -> Callable[[], Union[FactIndex, FactSet, List[Any]]]:
    pjk  = jqp.prejoin_key_clause
    pjc  = jqp.prejoin_clauses
    jk   = jqp.join_key
    predicate = jqp.root.meta.predicate

    # If there is a prejoin key clause then every comparator within it must
    # refer to exactly one index
    factindex_sc = []
    scs = pjk.__iter__() if pjk else []
    for sc in scs:
        keyable = sc.keyable(factindexes)
        if keyable is None:
            raise ValueError((f"Internal error: prejoin key clause '{pjk}' "
                                f"is invalid for JoinQueryPlan {jqp}"))
        factindex_sc.append((factindexes[keyable[0]],keyable[1],keyable[2]))

    pjc_check = None
    if pjc:
        dealiased = pjc.dealias()
        if len(dealiased.roots) != 1 and dealiased.roots[0].meta.predicate != predicate:
            raise ValueError((f"Internal error: prejoin clauses '{dealiased}' is invalid "
                              f"for JoinQueryPlan {jqp}"))
        pjc_check = dealiased.make_callable([dealiased.roots[0]])

    if jk and jk.args[0].meta.predicate != predicate:
        raise ValueError((f"Internal error: join key '{jk}' is invalid "
                              f"for JoinQueryPlan {jqp}"))

    return QuerySource(factindexes, factindex_sc, factsets.get(predicate, FactSet()),
                       jk, pjc_check, jqp.prejoin_orderbys)


class QuerySource:
    def __init__(
        self,
        factindexes: Dict[PredicatePath.Hashable, FactIndex],
        factindex_sc: List[Tuple[FactIndex, Any, Any]],
        factset: FactSet,
        jk: Optional[StandardComparator],
        pjc_check: Optional[Callable[..., bool]],
        pjob: Optional[OrderByBlock]
    ) -> None:
        self._factindexes = factindexes
        self._factindex_sc = factindex_sc
        self._factset= factset
        self._jk = jk
        self._pjc_check = pjc_check
        self._pjiqs = InQuerySorter(pjob) if pjob else None
        self._pjob = pjob

    # A prejoin_key_clause query uses the factindex
    def query_pjk(self):
        yield from (f for fi,op,key in self._factindex_sc for f in fi.find(op,key))

    # prejoin_clauses query uses the prejoin_key_clause query or the underlying factset
    def query_pjc(self, check):
        iter_ = self.query_pjk() if self._factindex_sc else self._factset
        yield from (f for f in iter_ if check((f,)))

    def execute_jk(self):
        assert self._jk
        jk_key_path = hashable_path(self._jk.args[0].meta.dealiased)
        if self._pjc_check:
            iter_ = self.query_pjc(self._pjc_check)
        elif self._factindex_sc:
            iter_ = self.query_pjk()
        else:
            hpath = hashable_path(jk_key_path)
            if hpath in self._factindexes:
                return self._factindexes[hpath]
            iter_ = self._factset

        fi = FactIndex(path(jk_key_path))
        for f in iter_:
            fi.add(f)
        return fi

    # If there is either a pjk or pjc then we need to create a temporary source
    # (using a FactIndex if there is a join key or a list otherwise). If there
    # is no pjk or pjc but there is a key then use an existing FactIndex if
    # there is one or create it.
    def __call__(self):
        if self._jk:
            return self.execute_jk()

        source = None
        if not self._pjc_check and not self._factindex_sc and not self._pjob:
            return self._factset
        if self._pjc_check:
            source = list(self.query_pjc(self._pjc_check))
        elif self._factindex_sc:
            source = list(self.query_pjk())
        if source and not self._pjob:
            return source

        if not source and self._pjob and len(self._pjob) == 1:
            pjo = self._pjob[0]
            hpath = hashable_path(pjo.path)
            if hpath in self._factindexes:
                fi = self._factindexes[hpath]
                return fi if pjo.asc else list(reversed(fi))

        if source is None:
            source = self._factset

        # If there is only one sort order use attrgetter
        return self._pjiqs.sorted(source)


# ------------------------------------------------------------------------------
#
# - factsets - a dictionary mapping a predicate to a factset
# - factindexes - a dictionary mapping a hashable_path to a factindex
# ------------------------------------------------------------------------------

def make_chained_join_query(
    jqp: JoinQueryPlan,
    inquery: Callable[[], Generator[Any, None, None]],
    factsets: Dict[Type[Predicate], FactSet],
    factindexes: Dict[PredicatePath.Hashable, FactIndex]
) -> Callable[[], Iterator[Any]]:

    if not jqp.input_signature:
        raise ValueError(("A non-first JoinQueryPlan must have a non-empty input "
                          f"signature but '{jqp.input_signature}' found"))

    prej_order = jqp.prejoin_orderbys
    jk   = jqp.join_key
    jc   = jqp.postjoin_clauses
    postj_order  = jqp.postjoin_orderbys

    prej_iqs = None
    if jk and prej_order:
        prej_iqs = InQuerySorter(prej_order)

    # query_source is a function that returns a FactSet, FactIndex, or list
    query_source = make_prejoin_query_source(jqp, factsets, factindexes)

    # Setup any join clauses
    if jc:
        jc_check = jc.make_callable(list(jqp.input_signature) + [jqp.root])
    else:
        jc_check = lambda _: True

    def query_jk():
        assert jk
        operator_ = jk.operator
        align_query_input = make_input_alignment_functor(
            jqp.input_signature,(jk.args[1],))
        fi = query_source()
        assert isinstance(fi, FactIndex)
        for intuple in inquery():
            v, = align_query_input(intuple)
            result = list(fi.find(operator_,v))
            if prej_order:
                assert prej_iqs
                prej_iqs.listsort(result)
            for f in result:
                out = tuple(intuple + (f,))
                if jc_check(out):
                    yield out

    def query_no_jk():
        source = query_source()
        for intuple in inquery():
            for f in source:
                out = tuple(intuple + (f,))
                if jc_check(out):
                    yield out


    unsorted_query=query_jk if jk else query_no_jk
    if not postj_order:
        return unsorted_query

    jiqs = InQuerySorter(postj_order,list(jqp.input_signature) + [jqp.root])
    def sorted_query():
        return iter(jiqs.sorted(unsorted_query()))

    return sorted_query

#------------------------------------------------------------------------------
# Makes a query given a ground QueryPlan and the underlying data. The returned
# query object is a Python generator function that takes no arguments.
# ------------------------------------------------------------------------------

def make_query(
    qp: QueryPlan,
    factsets: Dict[Type[Predicate], FactSet],
    factindexes: Dict[Any, Any]
) -> Callable[[], Iterator[Any]]:
    if qp.placeholders:
        raise ValueError(("Cannot execute an ungrounded query. Missing values "
                          f"for placeholders: {', '.join([str(p) for p in qp.placeholders])}"))
    first, *remainder = qp
    query =  make_first_join_query(first,factsets,factindexes)

    for jqp in remainder:
        query = make_chained_join_query(jqp,query,factsets,factindexes)
    return query


#------------------------------------------------------------------------------
# QueryOutput allows you to output the results of a Select query it different
# ways.
# ------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Given an input tuple of facts generate the appropriate output. Depending on
# the output signature what we want to generate this can be a simple of a
# complex operation. If it is just predicate paths or static values then a
# simple outputter is ok, but if it has a function then a complex one is needed.
# ------------------------------------------------------------------------------

def make_outputter(insig,outsig):

    def make_simple_outputter():
        af=make_input_alignment_functor(insig, outsig)
        return lambda intuple, af=af: af(intuple)

    def make_complex_outputter():
        metasig = []
        for out in outsig:
            if isinstance(out,PredicatePath) or \
                 (inspect.isclass(out) and issubclass(out,Predicate)):
                tmp = make_input_alignment_functor(insig, (path(out),))
                metasig.append(lambda x,af=tmp: af(x)[0])
            elif isinstance(out,FuncInputSpec):
                tmp=make_input_alignment_functor(insig, out.paths)
                metasig.append(lambda x,af=tmp,f=out.functor: f(*af(x)))
            elif callable(out):
                metasig.append(lambda x,f=out: f(*x))
            else:
                metasig.append(lambda x, out=out: out)

        maf=tuple(metasig)
        return lambda intuple, maf=maf: tuple(af(intuple) for af in maf)

    needcomplex=False
    for out in outsig:
        if isinstance(out,PredicatePath) or \
           (inspect.isclass(out) and issubclass(out,Predicate)):
            continue
        if isinstance(out,FuncInputSpec) or callable(out):
            needcomplex=True
            break

    return make_complex_outputter() if needcomplex else make_simple_outputter()


#------------------------------------------------------------------------------
# QueryExecutor - actually executes the query and does the appropriate action
# (eg., displaying to the user or deleting from the factbase)
# ------------------------------------------------------------------------------

class QueryExecutor:
    _qplan: QueryPlan
    _outputter: Any
    _unwrap: bool
    _distinct: bool
    #--------------------------------------------------------------------------
    # factmaps - dictionary mapping predicates to FactMap.
    # roots - the roots
    # qspec - dictionary containing the specification of the query and output
    #--------------------------------------------------------------------------
    def __init__(self, factmaps: Dict[Type[Predicate], FactMap], qspec: QuerySpec) -> None:
        self._factmaps = factmaps
        self._qspec = qspec.fill_defaults()


    #--------------------------------------------------------------------------
    # Support function
    #--------------------------------------------------------------------------
    @classmethod
    def get_factmap_data(
        cls,
        factmaps: Dict[Type[Predicate], FactMap],
        qspec: QuerySpec
    ) -> Tuple[Dict[Type[Predicate], FactSet], Dict[PredicatePath.Hashable, FactIndex]]:
        roots = qspec.roots
        ptypes = { path(r).meta.predicate for r in roots}
        factsets = {}
        factindexes = {}
        for ptype in ptypes:
            fm =factmaps[ptype]
            factsets[ptype] = fm.factset
            for hpth, fi in fm.path2factindex.items():
                factindexes[hpth] = fi
        return (factsets,factindexes)

    # --------------------------------------------------------------------------
    # Internal support function
    # --------------------------------------------------------------------------
    def _make_plan_and_query(self):
        where = self._qspec.where
        if where and not where.executable:
            placeholders = where.placeholders
            phstr=",".join(f"'{ph}'" for ph in placeholders)
            raise ValueError((f"Placeholders {phstr} must be bound to values before "
                              "executing the query"))
        qspec = self._qspec

        # FIXUP: This is hacky - if there is a group_by clause replace the
        # order_by list with the group_by list and later when sorting the for
        # each group the order_by list will be used.

        if where:
            qspec = qspec.modp(where=where.fixed())

        if qspec.group_by:
            qspec = qspec.modp(order_by=self._qspec.group_by)
            qspec = qspec.delp(["group_by"])
        elif qspec.ordered:
            qspec = qspec.modp(order_by=process_ordered(qspec.roots))

        (factsets,factindexes) = \
            QueryExecutor.get_factmap_data(self._factmaps, qspec)
        qplan = make_query_plan(factindexes.keys(), qspec)
#        qplan = qplan.ground()
        query = make_query(qplan,factsets,factindexes)
        return (qplan,query)


    # --------------------------------------------------------------------------
    # Internal function generator for returning all results
    # --------------------------------------------------------------------------
    def _all(self, query: Callable[[], Iterator[Any]]) -> Any:
        cache = set()
        for input_ in query():
            output = self._outputter(input_)
            if self._unwrap:
                output = output[0]
            if self._distinct:
                if output not in cache:
                    cache.add(output)
                    yield output
            else:
                yield output

    # --------------------------------------------------------------------------
    # Internal function generator for returning all grouped results
    # --------------------------------------------------------------------------
    def _group_by_all(self, query: Callable[[], Iterator[Any]]) -> Any:
        iqs=None
        def groupiter(group):
            cache = set()
            if iqs:
                group=iqs.sorted(group)
            for input_ in group:
                output = self._outputter(input_)
                if self._unwrap:
                    output = output[0]
                if self._distinct:
                    if output not in cache:
                        cache.add(output)
                        yield output
                else:
                    yield output

        qspec = self._qspec
        if qspec.ordered:
            qspec = qspec.modp(order_by=process_ordered(qspec.roots))

        if qspec.order_by:
            iqs=InQuerySorter(OrderByBlock(qspec.order_by),
                              self._qplan.output_signature)
        unwrapkey = len(qspec.group_by) == 1 and not qspec.tuple

        group_by_keyfunc = make_input_alignment_functor(
            self._qplan.output_signature, qspec.group_by.paths)
        for k,g in itertools.groupby(query(), group_by_keyfunc):
            if unwrapkey:
                yield k[0], groupiter(g)
            else:
                yield k, groupiter(g)


    #--------------------------------------------------------------------------
    # Function to return a generator of the query output
    # --------------------------------------------------------------------------

    def all(self) -> Generator[Any, None, None]:
        if self._qspec.distinct and not self._qspec.select:
            raise ValueError("'distinct' flag requires a 'select' projection")

        (self._qplan, query) = self._make_plan_and_query()

        outsig = self._qspec.select
        if outsig is None or not outsig:
            outsig = self._qspec.roots

        self._outputter = make_outputter(self._qplan.output_signature, outsig)
        self._unwrap = not self._qspec.tuple and len(outsig) == 1
        self._distinct = self._qspec.distinct

        if len(self._qspec.group_by) > 0:
            return self._group_by_all(query)
        return self._all(query)

    # --------------------------------------------------------------------------
    # Delete a selection of facts. Maintains a set for each predicate type
    # and adds the selected fact to that set. The delete the facts in each set.
    # --------------------------------------------------------------------------

    def delete(self):
        incompatible = ["group_by", "distinct", "tuple"]
        for incomp in incompatible:
            if getattr(self._qspec, incomp):
                raise ValueError(f"'{incomp}' is incompatible with 'delete'")

        (self._qplan, query) = self._make_plan_and_query()

        selection = self._qspec.select
        roots = [hashable_path(p) for p in self._qspec.roots]
        if selection:
            subroots = { hashable_path(p) for p in selection }
        else:
            subroots = set(roots)

        if not subroots.issubset(set(roots)):
            raise ValueError((f"For a 'delete' query the selected items '{selection}' "
                              f"must be a subset of the query roots '{roots}'"))

        # Special case for deleting all facts of a predicate
        if (len(roots) == 1 and len(subroots) == 1 and
            not self._qspec.where and not self._qspec.join):
            fm = self._factmaps[path(roots[0]).meta.predicate]
            count = len(fm)
            fm.clear()
            return count

        # Find the roots to delete and generate a set of actions that are
        # executed to add to a delete set
        deletesets = {}
        for r in subroots:
            pr = path(r)
            deletesets[pr.meta.predicate] = set()

        actions = []
        for out in self._qplan.output_signature:
            hout = hashable_path(out)
            if hout in subroots:
                ds = deletesets[out.meta.predicate]
                actions.append(lambda x, ds=ds: ds.add(x))
            else:
                actions.append(lambda x : None)

        # Running the query adds the facts to the appropriate delete set
        for input_ in query():
            for fact, action in zip(input_,actions):
                action(fact)

        # Delete the facts
        count = 0
        for pt,ds in deletesets.items():
            count += len(ds)
            fm = self._factmaps[pt]
            for f in ds:
                fm.remove(f)
        return count

#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------
if __name__ == "__main__":
    raise RuntimeError('Cannot run modules')
