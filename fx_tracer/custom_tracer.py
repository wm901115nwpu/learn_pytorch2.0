import functools
import inspect
from copy import deepcopy
import sys
import types
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


import torch
import torch.nn as nn
from log import MMLogger

try:
    from torch._C import ScriptObject  # type: ignore[attr-defined]
    from torch.ao.quantization.quantize_fx import QuantizationTracer
    from torch.fx import Graph, GraphModule, Tracer
    from torch.fx._symbolic_trace import (_autowrap_check, _find_proxy,
                                          _patch_wrapped_functions, _Patcher)
    from torch.fx.proxy import Proxy
except ImportError:
    from utils import get_placeholder
    ScriptObject = get_placeholder('torch>=1.13')
    QuantizationTracer = get_placeholder('torch>=1.13')
    GraphModule = get_placeholder('torch>=1.13')
    Tracer = get_placeholder('torch>=1.13')
    Graph = get_placeholder('torch>=1.13')
    _autowrap_check = get_placeholder('torch>=1.13')
    _patch_wrapped_functions = get_placeholder('torch>=1.13')
    _Patcher = get_placeholder('torch>=1.13')
    Proxy = get_placeholder('torch>=1.13')

from utils import import_modules_from_strings

_orig_module_call: Callable = nn.Module.__call__
_orig_module_getattr: Callable = nn.Module.__getattr__

sys.setrecursionlimit(int(pow(2, 20)))

logger = MMLogger.get_current_instance()

def auto_wrap(patcher, owner):
    auto_wrapper = AutoWrapper(patcher)
    auto_wrapper.wrap(None, '', owner)


class AutoWrapper:

    def __init__(self, patcher) -> None:
        self.patcher: _Patcher = patcher

    # wrap

    def wrap(self, owner, name, val):

        def is_method(val):
            return (inspect.ismethod(val) or inspect.isfunction(val)
                    or isinstance(val, types.BuiltinFunctionType)
                    or isinstance(val, staticmethod)
                    or isinstance(val, classmethod))

        if owner is None and isinstance(val, dict):
            self.wrap_frame(owner, name, val)
        else:
            # class
            if inspect.isclass(val):
                self.wrap_class(owner, name, val)
            # method
            elif inspect.isclass(owner) and is_method(val):
                self.wrap_method(owner, name, val)
            # function
            elif inspect.isfunction(val) or isinstance(
                    val, types.BuiltinFunctionType):
                self.wrap_function(owner, name, val)
            # package
            elif isinstance(val, ModuleType):
                self.wrap_module(owner, name, val)
            # instance
            elif isinstance(val, object):
                self.wrap_class(None, '', type(val))
            # else
            else:
                logger.debug(f'unsupported type to wrap: {name}/{type(val)}')

    def wrap_frame(self, owner, name: str, val: dict):
        assert isinstance(val, dict)

        if self.patcher.visit_once(val):
            frame_name = val['__name__'] if '__name__' in val else ''
            logger.debug(f'wrap a frame {frame_name}')
            for key in val:
                self.wrap(val, key, val[key])

    def wrap_module(self, owner, name, val):
        if self.visit_once(val):
            if val in [torch]:
                logger.debug(f'wrap a module {owner[name]}')
                self.wrap(None, '', val.__dict__)

    def wrap_class(self, owner, name, val):
        assert inspect.isclass(val)
        if issubclass(val, nn.Module):
            if self.visit_once(val):
                logger.debug(f'wrap a class {val}')
                for key in val.__dict__:
                    key: str
                    if not (key.startswith('__')):
                        self.wrap(val, key, val.__dict__[key])

    def wrap_function(self, owner, name, val):
        if self.visit_once(val):
            self.patcher.patch(owner, name, self.func_wapper(val))
            logger.debug(f'wrap a function {name}')

    def wrap_method(self, owner, name, val):
        assert inspect.isclass(owner)
        if self.visit_once(val):
            try:
                if isinstance(val, staticmethod):
                    pass
                    logger.debug(f'wrap a staticmethod {name} (unimplement)')
                elif isinstance(val, classmethod):
                    pass
                    logger.debug(f'wrap a classmethod {name} (unimplement)')
                else:
                    self.patcher.patch_method(owner, name,
                                              self.method_wrapper(val))
                    logger.debug(f'wrap an instance method {name}')
            except Exception:
                self.patcher.patches_made.pop()

    # wrapper
    def func_wapper(self, orig_fn):

        @functools.wraps(orig_fn)
        def wrapped(*args, **kwargs):
            """Given an closed-over ``orig_function`` to invoke, search the
            args and kwargs for a Proxy object.

            If there is one, emit a ``call_function`` node to preserve the call
            to this leaf function directly. Otherwise, just return the results
            of this function call, as this function is not being traced.
            """
            _autowrap_check(self.patcher, getattr(orig_fn, '__globals__', {}),
                            set())
            try:
                end = orig_fn(*args, **kwargs)
                return end
            except Exception:
                logger.debug(f'auto wrap {orig_fn}')
                proxy = _find_proxy(args, kwargs)
                if proxy is not None:
                    return_proxy = proxy.tracer.create_proxy(
                        'call_function', orig_fn, args, kwargs)
                    return_proxy.node.meta['is_wrapped'] = True
                    return return_proxy
                else:
                    return orig_fn(*args, **kwargs)

        return wrapped

    def method_wrapper(self, orig_fn):

        @functools.wraps(orig_fn)
        def wrapped(*args, **kwargs):
            """Given an closed-over ``orig_function`` to invoke, search the
            args and kwargs for a Proxy object.

            If there is one, emit a ``call_function`` node to preserve the call
            to this leaf function directly. Otherwise, just return the results
            of this function call, as this function is not being traced.
            """
            _autowrap_check(self.patcher, getattr(orig_fn, '__globals__', {}),
                            set())
            # logger.debug(f'call method {orig_fn}')
            try:
                end = orig_fn(*args, **kwargs)
                return end
            except Exception:
                logger.debug(f'auto wrap {orig_fn}')
                proxy: Proxy = _find_proxy(args, kwargs)
                if proxy is not None:
                    return_proxy = proxy.tracer.create_proxy(
                        'call_method', orig_fn.__name__, args, kwargs)
                    return_proxy.node.meta['is_wrapped'] = True
                    return return_proxy
                else:
                    return orig_fn(*args, **kwargs)

        return wrapped

    # others
    def visit_once(self, obj):
        return self.patcher.visit_once(obj)

    def is_visited(self, obj):
        id_ = id(obj)
        return id_ in self.patcher.visited

class UntracedMethodRegistry:
    """A `Descriptor` class which records untraced methods. Thus, when the
    class is traced with CustomTracer, the decorated method will be as a leaf
    node, not be nested traced.

    Example:
        >>> # `imported_cls` is the owner of the untraced method;
        >>> # `method_str` is the name of the untraced method.
        >>> method_registry = UntracedMethodRegistry(method)
        >>> method_registry.__set_name__(imported_cls, method_str)

    Args:
        method (FunctionType): Function to be registered.
    """
    method_dict: Dict = dict()
    tracer = None

    def __init__(self, method: FunctionType):
        self.method = method
        self.owner = None

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name
        wrapped = self.method_wrapper()
        self.method_dict[name] = dict(mod=self.owner, wrapped=wrapped)

    def method_wrapper(self):

        @functools.wraps(self.method)
        def wrapped_method(mod, *args, **kwargs):

            def method(*args, **kwargs):
                return self.method(mod, *args, **kwargs)

            return self.tracer.call_method(mod, self.name, method, args,
                                           kwargs)

        return wrapped_method


def _prepare_module_dict(model: torch.nn.Module, fx_graph):
    """If there is a class method that can not be traced by the symbolic
    tracer, a ``call_method`` ``Node`` will be inserted into the ``Graph`` in
    ``CustomTracer``.

    Example:
        >>> class Model:
        ...     def __init__(self):
        ...         self.head = ClsHead()
        ...
        >>> class ClsHead(nn.Module):
        ...     def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        ...         return feats[-1]
        ...
        ...     def loss(self, feats: Tuple[torch.Tensor],
        ...              data_samples: List[ClsDataSample], **kwargs) -> dict:
        ...         cls_score = self(feats)
        ...         # The part can not be traced by torch.fx
        ...         losses = self._get_loss(cls_score, data_samples, **kwargs)
        ...         return losses
        ...
        ...     def _get_loss(self, cls_score: torch.Tensor,
        ...                   data_samples: List[ClsDataSample], **kwargs):
        ...         if 'score' in data_samples[0].gt_label:
        ...             xxx
        ...         else:
        ...             xxx
        ...         losses = xxx
        ...         return losses

    As the ``_get_loss`` can not be traced by torch.fx, ``Toy._get_loss`` need
    to be added to ``skipped_methods`` in ``CustomTracer``. Hence the code
    above will product the following Graph::

    .. code-block:: text
        ... ...
        %head : [#users=1] = get_attr[target=head]
        %_get_loss : [#users=1] = call_method[target=_get_loss](args = (%head, %head_fc, %data_samples), kwargs = {})  # noqa: E501
        return _get_loss

    Hence, the head module in the ``GraphModule`` and that in the original
    model are the same one (refer to https://github.com/pytorch/pytorch/blob/master/torch/fx/graph_module.py#L346).  # noqa: E501
    So changes made to the graph module (in ``prepare()``) will also modify
    the original model.

    Args:
        model (torch.nn.Module): Module or function to be
            traced and converted into a Graph representation.
        fx_graph (torch.fx.Graph): The fx Graph traced by fx tracer. It
            contains the nodes this GraphModule should use for code generation.
    """

    def _get_attrs(target, attrs):
        attrs = attrs.split('.')
        for att in attrs:
            target = getattr(target, att)
        return target

    module_dict = dict()
    special_nodes = []

    for node in fx_graph.nodes:
        if node.op == 'get_attr':
            attr = _get_attrs(model, node.target)
            if isinstance(attr, nn.Module):
                module_dict[node.target] = nn.Module()
                special_nodes.append(node)
            # module_dict[node.target] = _get_attrs(model, node.target)
        elif node.op == 'call_method':
            for special_node in special_nodes:
                if special_node in node.args or \
                        special_node in node.kwargs.values():
                    origin_module = getattr(model, special_node.target)
                    setattr(module_dict[special_node.target], node.target,
                            getattr(origin_module, node.target))

    return module_dict


def duplicate_reused_nodes(graph: Graph, modules: Dict[str, Any] = {}):
    """Deepcopy the shared modules (e.g. shared detection head in RetinaNet) to
    make sure modules can be fused correctly.

    Modified from https://github.com/ModelTC/MQBench/blob/main/mqbench/prepare_by_platform.py  # noqa: E501
    """
    _dup_prefix = '_dup'
    target_dict = dict()
    dup_modules = dict()
    for node in graph.nodes:
        if node.op == 'call_module':
            if node.target not in target_dict:
                target_dict[node.target] = [node]
            else:
                target_dict[node.target].append(node)
    for key in target_dict:
        if len(target_dict[key]) > 1:
            for idx, node in enumerate(target_dict[key]):
                if idx == 0:
                    continue
                module = deepcopy(modules[node.target])
                node.target += _dup_prefix + str(idx)
                dup_modules[node.target] = module
    graph.lint()
    return graph, dup_modules


def build_graphmodule(model: torch.nn.Module,
                      fx_graph,
                      name: str = 'GraphModule'):
    """To build GraphModule with the generated graph by CustomTracer. The
    implement of skipping methods in CustomTracer will cause the confliction of
    that a node is both a leaf node and non-leaf node, which will lead that the
    modification to the ``graph`` also change the original ``forward``.

    Args:
        model (torch.nn.Module): Module or function to be
            traced and converted into a Graph representation.
        fx_graph (torch.fx.Graph): The fx Graph traced by fx tracer. It
            contains the nodes this GraphModule should use for code generation.
        name (str): The name of generated GraphModule.

    Returns:
        GraphModule: GraphModule is an nn.Module generated from an fx.Graph.
        Graphmodule has a ``graph`` attribute, as well as ``code`` and
        ``forward`` attributes generated from that ``graph``.

    .. warning::
        When ``graph`` is reassigned, ``code`` and ``forward`` will be
        automatically regenerated. However, if you edit the contents of the
        ``graph`` without reassigning the ``graph`` attribute itself, you must
        call ``recompile()`` to update the generated code.
    """
    modules = dict(model.named_modules())
    module_dict = _prepare_module_dict(model, fx_graph)
    fx_graph, duplicated_modules = duplicate_reused_nodes(fx_graph, modules)
    modules.update(module_dict)
    modules.update(duplicated_modules)
    return GraphModule(modules, fx_graph, name)


class CustomTracer(QuantizationTracer):
    """Custom tracer based on QuantizationTracer of pytorch. It can not only
    skip some modules and classes while tracing, but also skip some methods
    untraced by torch.fx.Tracer.

    Args:
        skipped_methods (List[str], optional): Methods to be skipped while
            tracing. Defaults to None.
        skipped_module_names (List[str], optional): Modules to be skipped
            while tracing. Defaults to None.
        skipped_module_classes (List[Callable], optional): Class to be skipped
            while tracing. Defaults to None.
    """

    def __init__(self,
                 skipped_methods: List[str] = [],
                 skipped_module_names: List[str] = [],
                 skipped_module_classes: List[Callable] = [],
                 *args,
                 **kwargs):
        super(CustomTracer, self).__init__(skipped_module_names,
                                           skipped_module_classes)
        UntracedMethodRegistry.tracer = self  # type: ignore
        self.skipped_methods = skipped_methods
        if self.skipped_methods:
            self.register_skipped_methods()

    @staticmethod
    def _check_valid_source(source):
        """Check if the source's format is valid."""
        if not isinstance(source, str):
            raise TypeError(f'source should be a str '
                            f'instance, but got {type(source)}')

        assert len(source.split('.')) > 1, \
            'source must have at least one `.`'

    def register_skipped_methods(self):
        """Register skipped methods to UntracedMethodRegistry.method_dict."""
        if not isinstance(self.skipped_methods, list):
            self.skipped_methods = [self.skipped_methods]
        for s_method in self.skipped_methods:
            self._check_valid_source(s_method)
            mod_str = '.'.join(s_method.split('.')[:-2])
            cls_str = s_method.split('.')[-2]
            method_str = s_method.split('.')[-1]

            try:
                mod = import_modules_from_strings(mod_str)
            except ImportError:
                raise ImportError(f'{mod_str} is not imported correctly.')

            imported_cls: type = getattr(mod, cls_str)
            if not isinstance(imported_cls, type):
                raise TypeError(f'{cls_str} should be a type '
                                f'instance, but got {type(imported_cls)}')
            assert hasattr(imported_cls, method_str), \
                   f'{method_str} is not in {mod_str}.'

            method = getattr(imported_cls, method_str)

            method_registry = UntracedMethodRegistry(method)
            method_registry.__set_name__(imported_cls, method_str)

    def call_method(self, m: torch.nn.Module, name: str, method: Callable,
                    args: Tuple, kwargs: Dict):
        """Method that specifies the behavior of this ``Tracer`` when it
        encounters a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf
        module via ``is_leaf_module``. If it is, emit a ``call_module``
        node referring to ``m`` in the ``Graph``. Otherwise, call the
        ``Module`` normally, tracing through the operations in its ``forward``
        function.

        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.

        Args:
            m (torch.nn.Module): The module for which a call is being emitted
            name (str): The name of proxy to be created.
            method (Callable): The method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Return:

            The return value from the Module call. In the case that a
            ``call_module`` node was emitted, this is a ``Proxy`` value.
            Otherwise, it is whatever value was returned from the ``Module``
            invocation.
        """
        # module_qualified_name = self.path_of_module(m)
        if not self.is_skipped_method(m):
            return method(*args, **kwargs)
        args_l = list(args)
        args_l.insert(0, m)
        args = tuple(args_l)
        return self.create_proxy('call_method', name, args, kwargs)

    def trace(self,
              root: Union[torch.nn.Module, Callable[..., Any]],
              concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        """Trace ``root`` and return the corresponding FX ``Graph``
        representation. ``root`` can either be an ``nn.Module`` instance or a
        Python callable. Note that after this call, ``self.root`` may be
        different from the ``root`` passed in here. For example, when a free
        function is passed to ``trace()``, we will create an ``nn.Module``
        instance to use as the root and add embedded constants to.

        Args:
            root (Union[Module, Callable]): Either a ``Module`` or a function
                to be traced through. Backwards-compatibility for this
                parameter is guaranteed.
            concrete_args (Optional[Dict[str, any]]): Concrete arguments that
                should not be treated as Proxies. This parameter is
                experimental and its backwards-compatibility is *NOT*
                guaranteed.

        Returns:
            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        if isinstance(root, torch.nn.Module):
            self.root = root
            
            assert hasattr(type(root), self.traced_func_name), (
                f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"  # noqa
            )  # noqa
            
            fn = type(root).forward
            self.submodule_paths: Optional[Dict[torch.nn.Module, str]] = {
                mod: name
                for name, mod in root.named_modules()
            }
        else:
            self.root = nn.Module()
            fn = root

        tracer_cls: Optional[Type['Tracer']] = getattr(self, '__class__', None)
        self.graph = Graph(tracer_cls=tracer_cls)

        # When we encounter a Tensor value that's not a parameter, we look if
        # it is some other attribute on the model. Construct a dict mapping
        # Tensor values to the qualified name here for efficiency. This is
        # used downstream in create_arg
        self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

        def collect_tensor_attrs(m: nn.Module, prefix_atoms: List[str]):
            for k, v in m.__dict__.items():
                if isinstance(v, (torch.Tensor, ScriptObject)):
                    self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
            for k, v in m.named_children():
                collect_tensor_attrs(v, prefix_atoms + [k])

        collect_tensor_attrs(self.root, [])

        assert isinstance(fn, FunctionType)

        fn_globals = fn.__globals__  # run before it gets patched
        fn, args = self.create_args_for_root(fn, isinstance(root, nn.Module),
                                             concrete_args)

        # Reduce number of get_attr calls
        parameter_proxy_cache: Dict[str, Proxy] = {}

        # Method dispatch on parameters is not recorded unless it's directly
        # used. Thus, we need to insert a proxy when __getattr__ requests a
        # parameter.
        @functools.wraps(_orig_module_getattr)
        def module_getattr_wrapper(mod, attr):
            attr_val = _orig_module_getattr(mod, attr)
             ########################################################################
            if digit_version(torch.__version__) >= digit_version('1.13.0'):
                return self.getattr(attr, attr_val, parameter_proxy_cache)
            else:
                return self._module_getattr(attr, attr_val,
                                            parameter_proxy_cache)
            ########################################################################

        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):

            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)

            _autowrap_check(
                patcher,
                getattr(getattr(mod, 'forward', mod), '__globals__', {}),
                self._autowrap_function_ids)
            ########################################################################
            auto_wrap(patcher, mod)
            ########################################################################
            return self.call_module(mod, forward, args, kwargs)

        with _Patcher() as patcher:
            # allow duplicate patches to support the case of nested calls
            patcher.patch_method(
                nn.Module,
                '__getattr__',
                module_getattr_wrapper,
                deduplicate=False)
            patcher.patch_method(
                nn.Module, '__call__', module_call_wrapper, deduplicate=False)

            for name, value in UntracedMethodRegistry.method_dict.items():
                wrapped = value['wrapped']
                patcher.patch_method(
                    value['mod'], name, wrapped, deduplicate=False)

            _patch_wrapped_functions(patcher)
            ########################################################################
            patcher.visit_once(globals())
            auto_wrap(patcher, self.root)
            ########################################################################
            _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
            for module in self._autowrap_search:
                _autowrap_check(patcher, module.__dict__,
                                self._autowrap_function_ids)
            self.create_node(
                'output',
                'output', (self.create_arg(fn(*args)), ), {},
                type_expr=fn.__annotations__.get('return', None))

        self.submodule_paths = None

        return self.graph
    
    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:

        try:
            return super().call_module(m, forward, args, kwargs)
        except Exception:
            module_qualified_name = self.path_of_module(m)
            return self.create_proxy('call_module', module_qualified_name,
                                     args, kwargs)

    def create_arg(self, a: Any) -> 'Argument':
        try:
            arg = super().create_arg(a)
            return arg
        except Exception:
            return a

    def is_skipped_method(self, m: torch.nn.Module):
        """Judge if ``m`` is registered skipped method."""
        mods = tuple(value['mod']
                     for value in UntracedMethodRegistry.method_dict.values())
        custom = isinstance(m, mods)
        return custom

    def is_leaf_module(self, m: torch.nn.Module,
                       module_qualified_name: str) -> bool:
        """A method to specify whether a given ``nn.Module`` is a "leaf"
        module. Leaf modules are the atomic units that appear in the IR,
        referenced by ``call_module`` calls. By default, Modules in the PyTorch
        standard library namespace (torch.nn) are leaf modules. All other
        modules are traced through and their constituent ops are recorded,
        unless specified otherwise via this parameter.

        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module.
                For example, if you have a module hierarchy where submodule
                ``foo`` contains submodule ``bar``, which contains submodule
                ``baz``, that module will appear with the qualified name
                ``foo.bar.baz`` here.
        """
        leaf = super().is_leaf_module(m, module_qualified_name)
        return leaf


def custom_symbolic_trace(
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None) -> GraphModule:
    """Modified `symbolic_trace` function in pytorch. Given an ``nn.Module`` or
    function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    Args:
        root (torch.nn.Module): Module or function to be
            traced and converted into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially
            specialized.

    Returns:
        GraphModule: a Module created from the recorded operations from
        ``root``.
    """
    tracer = CustomTracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(
        root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)
