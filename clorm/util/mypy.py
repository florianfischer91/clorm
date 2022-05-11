from pickle import FALSE
from pkgutil import get_data
from typing import Callable, List, Optional, Type as TypingType, Union, get_args
from attr import attributes

from mypy.plugin import Plugin, FunctionSigContext, FunctionContext, ClassDefContext, MethodSigContext, MethodContext, SemanticAnalyzerPluginInterface, CheckerPluginInterface
from mypy.typeops import try_getting_str_literals, try_getting_int_literals_from_type
from mypy.options import Options
from mypy.nodes import (AssignmentStmt, OverloadedFuncDef, IntExpr, ClassDef, TypeInfo, Argument, FuncDef, Var, ARG_POS,ARG_NAMED_OPT, ARG_NAMED, TypeInfo, Block, PassStmt, SymbolTable, SymbolTableNode, MDEF,
Decorator, NameExpr, IndexExpr )
from mypy.types import Overloaded, TypeVarType, LiteralType, Instance, Type, NoneTyp, AnyType, TypeType, TypeOfAny, CallableType, FunctionLike, ProperType, get_proper_type
from mypy.typevars import fill_typevars
from mypy.semanal_shared import set_callable_name
from mypy.version import __version__ as mypy_version
from mypy.plugins.dataclasses import DataclassTransformer, DataclassAttribute
from mypy.util import get_unique_redefinition_name

BUILTINS_NAME = 'builtins' if float(mypy_version) >= 0.930 else '__builtins__'
CLORM_PREDICATE = "clorm.orm.core.Predicate"

def plugin(version: str) -> 'TypingType[Plugin]':
    """
    `version` is the mypy version string
    We might want to use this to print a warning if the mypy version being used is
    newer, or especially older, than we expect (or need).
    """
    return ClormPlugin

class ClormPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        super().__init__(options)

    def get_base_class_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo):
            if any(base.fullname == "clorm.orm.core.Predicate" for base in sym.node.mro):
                # print(sym)
                return self._clorm_predicate_class_maker_callback
        return None

    def _clorm_predicate_class_maker_callback(self, ctx: ClassDefContext) -> None:
        transformer = ClormPredicateTransformer(ctx) #, self.plugin_config)
        transformer.transform()

    def get_method_signature_hook(self, fullname: str
                                  ) -> Optional[Callable[[MethodSigContext], FunctionLike]]:
        print("methodsig: " + fullname)
        fqn = self.lookup_fully_qualified(fullname)
        if fqn is not None and fqn.node is not None:
            # print(fullname)
            # print(fqn.node)
            # print(fqn.node.name)
            
            # if fullname == CLORM_PREDICATE + ".__iter__":
            #     # print(fullname)
            #     return predicate_iter_callback
            # return super().get_method_signature_hook(fullname)
            pass
        return None

    def get_function_signature_hook(self, fullname: str) -> Optional[Callable[[FunctionSigContext], FunctionLike]]:
        print("functionsig: " + fullname)
        return functionsig_callback
        return super().get_function_signature_hook(fullname)


    def get_method_hook(self, fullname: str
                        ) -> Optional[Callable[[MethodContext], Type]]:
        # if fullname == "clorm.orm.core.Predicate" + ".__iter__":
        #     print(fullname)
        #     return array_iter_callback
        print("get_method_hook: " + fullname)
        return method_callback
        # # if fullname in (CLORM_PREDICATE + ".__getitem__", CLORM_PREDICATE + ".__setitem__"):
        # if "__getitem__" in fullname:
        #     # print(fullname)
        #     return predicate_getitem_callback
        # # if fullname in (CLORM_PREDICATE + ".__iter__"):
        # # if fullname in (CLORM_PREDICATE + ".__iter__"):
        # #     # print(fullname)
        # #     return predicate_iter_callback
        # if "__iter__" in fullname:
        #     return method_callback
        return super().get_method_hook(fullname)
        pass

    def get_function_hook(
        self,
        fullname: str,
    ) -> Optional[Callable[[FunctionContext], Type]]:
        sym = self.lookup_fully_qualified(fullname)
        print(fullname)
        if sym and sym.node:
            print("get_function_hook: " +sym.node.name)
        return function_callback

def functionsig_callback(ctx: FunctionSigContext) -> FunctionLike:
    print("fsc: " + str(ctx.context.line))
    
    return ctx.default_signature

def method_callback(ctx: MethodContext) -> Type:
    # print(ctx)
    print(f"method_callback: {ctx.context.line}")
    tp = get_proper_type(ctx.type)
    if isinstance(tp, Instance):
        print(tp.type.fullname)
    else:
        print(tp.line)
    print(ctx)
    # print(ctx.type.line)
    
    return ctx.default_return_type

def function_callback(ctx: FunctionContext) -> Type:
    print(ctx)
    print("fb")
    print(ctx.context.line)
    return ctx.default_return_type

def predicate_iter_callback(ctx: MethodContext) -> Type:
    print(ctx.context.end_line)
    # Overloaded()
    # print(f"__iter__: {ctx.line}")
    # print(ctx.type)
    # ptype = get_proper_type(ctx.type)
    # print(f"{ctx.args=}")
    # if isinstance(ptype, Instance):
    #     print(ptype)
    #     print(ptype.last_known_value)

    # print(ctx.context)

    # print(ctx.context)
    return ctx.default_return_type

def predicate_getitem_callback(ctx: MethodContext) -> Type:
    print("getitem_callback")
    arg = ctx.args[0][0]
    # print(f"__getitem__: {arg.line=}")
    # print(arg)
    tp = get_proper_type(ctx.type)
    if isinstance(tp, Instance):
        print(tp.type.fullname)
        # assert tp.type.fullname == CLORM_PREDICATE
        print(f"__getitem__: {arg.line=}")
        print(arg)
    # if isinstance(arg, NameExpr):
    # else:
    #     print(arg)
    # print(ctx.callee_arg_names)
    # print(ctx)
    # # print(str(ctx.type) == "clorm.orm.core.Predicate") 
    # keys = try_getting_str_literals(ctx.args[0][0], ctx.arg_types[0][0])
    # # index = try_getting_int_literals_from_type(ctx.arg_types[0][0])
    # fact: Instance
    # if isinstance(ctx.type, Instance):
    #     fact  =ctx.type
    #     print(f"{fact.type_ref=}")
    # print(ctx.context.__class__)
    # if isinstance(ctx.context, IndexExpr):
    #     print(ctx.context.index)
    # # indexExpr: IndexExpr = ctx.context
    # # print(index)
    return ctx.default_return_type

def array_getitem_callback(ctx: MethodContext) -> Type:
    """Callback to provide an accurate return type for ctypes.Array.__getitem__."""
    print(ctx)
    print(ctx.args[0][0])
    et = _get_array_element_type(ctx.type)
    if et is not None:
        unboxed = _autounboxed_cdata(et)
        assert len(ctx.arg_types) == 1, \
            'The stub of ctypes.Array.__getitem__ should have exactly one parameter'
        assert len(ctx.arg_types[0]) == 1, \
            "ctypes.Array.__getitem__'s parameter should not be variadic"
        index_type = get_proper_type(ctx.arg_types[0][0])
        if isinstance(index_type, Instance):
            if index_type.type.has_base('builtins.int'):
                return unboxed
            elif index_type.type.has_base('builtins.slice'):
                return ctx.api.named_generic_type('builtins.list', [unboxed])
    return ctx.default_return_type

def array_iter_callback(ctx: MethodContext) -> Type:
    """Callback to provide an accurate return type for ctypes.Array.__iter__."""
    print(ctx)
    et = _get_array_element_type(ctx.type)
    if et is not None:
        unboxed = _autounboxed_cdata(et)
        # print(f"{unboxed=}")
        return ctx.api.named_generic_type('typing.Iterator', [unboxed])
    return ctx.default_return_type 

def _get_array_element_type(tp: Type) -> Optional[ProperType]:
    """Get the element type of the Array type tp, or None if not specified."""
    tp = get_proper_type(tp)
    if isinstance(tp, Instance):
        print(f"{tp=}")
        print(f"{tp.args=}")
        return tp
        # assert tp.type.fullname == 'ctypes.Array'
        # if len(tp.args) == 1:
        #     return get_proper_type(tp.args[0])
    return None

def _autounboxed_cdata(tp: Type) -> ProperType:
    """Get the auto-unboxed version of a CData type, if applicable.
    For *direct* _SimpleCData subclasses, the only type argument of _SimpleCData in the bases list
    is returned.
    For all other CData types, including indirect _SimpleCData subclasses, tp is returned as-is.
    """
    tp = get_proper_type(tp)

    # if isinstance(tp, UnionType):
    #     return make_simplified_union([_autounboxed_cdata(t) for t in tp.items])
    if isinstance(tp, Instance):
        for base in tp.type.bases:
            if base.type.fullname == 'ctypes._SimpleCData':
                # If tp has _SimpleCData as a direct base class,
                # the auto-unboxed type is the single type argument of the _SimpleCData type.
                assert len(base.args) == 1
                return get_proper_type(base.args[0])
    # If tp is not a concrete type, or if there is no _SimpleCData in the bases,
    # the type is not auto-unboxed.
    return tp

class ClormPredicateTransformer:
    def __init__(self, ctx: ClassDefContext) -> None:
        self._ctx = ctx
        self.dt_transformer = DataclassTransformer(ctx)

    def transform(self) -> None:
        ctx = self._ctx
        info = self._ctx.cls.info
        print("TCD")
        # if not ctx.api.final_iteration:
        #     ctx.api.defer()
        # for stmt in ctx.cls.info.defn.defs.body:
        #     if isinstance(stmt, AssignmentStmt):
        #         print(stmt.lvalues)
        print(ctx.cls.info.names)
        self.collect_attributes()
        self.attributes = self.dt_transformer.collect_attributes()
        # for attr in self.attributes:
        #     print(attr.serialize())
        self.add_initializer()
        # self.add_getitem()

    def collect_attributes(self):
        ctx = self._ctx
        cls = self._ctx.cls
        # for stmt in cls.defs.body:
        #     print(stmt)

    def add_initializer(self):
        ctx = self._ctx
        info = ctx.cls.info
        # init_argument = []
        # print(info)
        # cls = ctx.cls
        # for info in cls.info.mro[1:]:
        #     # if not isinstance
        #     print(info)
        # print(info.names)
        init_argument = [attr.to_argument() for attr in self.attributes]
        # for key, type_ in info.names.items():
        #     # print(key, type_)
        # # init_argument = [Argument(Var("a"),info["a"].type,None, ARG_POS)]
        #     init_argument.append(Argument(Var(key),info[key].type,None, ARG_POS))
        # init_argument.append(Argument(Var("sign"),,None, ARG_NAMED))
        # add_method(ctx, "__init__", init_argument, NoneTyp())
        init_argument.append(Argument(Var("sign"),ctx.api.builtin_type(f"{BUILTINS_NAME}.bool"),None, ARG_NAMED_OPT))
        # print(init_argument)
        add_method_to_class(ctx.api, ctx.cls, "__init__", init_argument, NoneTyp())

    def add_getitem(self):
        ctx = self._ctx
        info = ctx.cls.info
        # init_argument = []
        # types = list(info.names.values())
        # for key, type_ in info.names.items():
        #     print(key, type_)
        #     pass
        # init_argument = [Argument(Var("a"),info["a"].type,None, ARG_POS)]
            # return_type = Argument(Var(key),info[key].type,None, ARG_POS)
            # init_argument.append(Argument(Var("sign"),,None, ARG_NAMED))
        # arg = [Argument(Var("idx"),ctx.api.named_type(f"{BUILTINS_NAME}.int"),None,ARG_POS)]
        i=0
        for attr in self.attributes:
            arg = [Argument(Var("idx"),LiteralType(i, ctx.api.named_type(f"{BUILTINS_NAME}.int")),None,ARG_POS)]
            # print(t.type)
            i += 1
            # print(arg[0])
            # ret_type = t.type if t.type else AnyType(TypeOfAny.explicit)
            add_method_to_class(ctx.api, ctx.cls, "__getitem__",arg, attr.type, is_overload=True)
        
        arg = [Argument(Var("idx"),ctx.api.named_type(f"{BUILTINS_NAME}.int"),None,ARG_POS)]
        add_method_to_class(ctx.api, ctx.cls, "__getitem__",arg, AnyType(TypeOfAny.explicit), is_overload=True)
        add_method_to_class(ctx.api, ctx.cls, "__getitem__",arg, AnyType(TypeOfAny.explicit))
        print(info.defn.defs.body)
        pass

# def add_method(ctx: ClassDefContext, name: str, args: List[Argument], return_type: Type) -> None:
#     info = ctx.cls.info
#     if name in info.names:
#         sym = info.names[name]
#         if sym.plugin_generated and isinstance(sym.node, FuncDef):
#             ctx.cls.defs.body.remove(sym.node)
#     first = [Argument(Var('self'), fill_typevars(info), None, ARG_POS)]
#     args = first + args

#     arg_types, arg_names, arg_kinds = [], [], []
#     for arg in args:
#         assert arg.type_annotation, 'All arguments must be fully typed.'
#         arg_types.append(arg.type_annotation)
#         arg_names.append(arg.variable.name)
#         arg_kinds.append(arg.kind)

#     function_type = ctx.api.named_type(f'{BUILTINS_NAME}.function')
#     # function_type = ctx.api.named_type(f'Clorm.function')
#     signature = CallableType(arg_types, arg_kinds, arg_names, return_type, function_type)
#     # if tvar_def:
#     #     signature.variables = [tvar_def]

#     func = FuncDef(name, args, Block([PassStmt()]))
#     func.info = info
#     func.type = set_callable_name(signature, func)
#     func.is_class = False # is_classmethod
#     # func.is_static = is_staticmethod
#     func._fullname = info.name +"."+name # get_fullname(info) + '.' + name
#     func.line = info.line

#     # NOTE: we would like the plugin generated node to dominate, but we still
#     # need to keep any existing definitions so they get semantically analyzed.
#     # if name in info.names:
#     #     # Get a nice unique name instead.
#     #     r_name = get_unique_redefinition_name(name, info.names)
#     #     info.names[r_name] = info.names[name]

#     if False: #is_classmethod:  # or is_staticmethod:
#         func.is_decorated = True
#         v = Var(name, func.type)
#         v.info = info
#         v._fullname = func._fullname
#         # if is_classmethod:
#         v.is_classmethod = True
#         dec = Decorator(func, [NameExpr('classmethod')], v)
#         # else:
#         #     v.is_staticmethod = True
#         #     dec = Decorator(func, [NameExpr('staticmethod')], v)

#         dec.line = info.line
#         sym = SymbolTableNode(MDEF, dec)
#     else:
#         sym = SymbolTableNode(MDEF, func)
#     sym.plugin_generated = True

#     info.names[name] = sym
#     info.defn.defs.body.append(func)


def add_method_to_class(
        api: Union[SemanticAnalyzerPluginInterface, CheckerPluginInterface],
        cls: ClassDef,
        name: str,
        args: List[Argument],
        return_type: Type,
        self_type: Optional[Type] = None,
        tvar_def: Optional[TypeVarType] = None,
        is_overload: bool = False,
) -> None:
    """Adds a new method to a class definition."""
    info = cls.info

    # First remove any previously generated methods with the same name
    # to avoid clashes and problems in the semantic analyzer.
    if name in info.names:
        sym = info.names[name]
        if sym.plugin_generated and isinstance(sym.node, FuncDef):
            cls.defs.body.remove(sym.node)

    self_type = self_type or fill_typevars(info)
    if isinstance(api, SemanticAnalyzerPluginInterface):
        function_type = api.named_type('builtins.function')
    else:
        function_type = api.named_generic_type('builtins.function', [])

    args = [Argument(Var('self'), self_type, None, ARG_POS)] + args
    arg_types, arg_names, arg_kinds = [], [], []
    for arg in args:
        assert arg.type_annotation, 'All arguments must be fully typed.'
        arg_types.append(arg.type_annotation)
        arg_names.append(arg.variable.name)
        arg_kinds.append(arg.kind)

    signature = CallableType(arg_types, arg_kinds, arg_names, return_type, function_type)
    if tvar_def:
        signature.variables = [tvar_def]

    func = FuncDef(name, args, Block([PassStmt()]))
    func.info = info
    func.type = set_callable_name(signature, func)
    func._fullname = info.fullname + '.' + name
    func.line = info.line

    # NOTE: we would like the plugin generated node to dominate, but we still
    # need to keep any existing definitions so they get semantically analyzed.
    if name in info.names:
        # Get a nice unique name instead.
        r_name = get_unique_redefinition_name(name, info.names)
        info.names[r_name] = info.names[name]

    # func.is_overload = is_overload
    if is_overload:
        func.is_decorated = True
        func.is_overload = True
        v = Var(name, func.type)
        v.info = info
        v._fullname = func._fullname
        dec = Decorator(func, [NameExpr("overload")], v)
        dec.line = info.line

        sym = SymbolTableNode(MDEF, dec, plugin_generated=True)
    else:
        sym = SymbolTableNode(MDEF, func, plugin_generated=True)

    # print(func)
    info.names[name] = sym
    info.defn.defs.body.append(func)