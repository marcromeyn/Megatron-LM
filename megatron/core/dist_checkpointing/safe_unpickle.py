# safe_unpickle.py

import pickle
from typing import Any
import io
from contextlib import contextmanager
import sys


# Prevent subclassing to ensure security restrictions cannot be bypassed
class FinalType(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.__init_subclass__ = lambda: TypeError(f"Cannot subclass {cls.__name__}")


class MegatronUnpickler(pickle.Unpickler, metaclass=FinalType):
    _allowed_modules = {
        "builtins": frozenset(
            {
                "list",
                "dict",
                "tuple",
                "int",
                "float",
                "bool",
                "str",
                "set",
                "frozenset",
                "bytearray",
                "memoryview",
            }
        ),
        "collections": frozenset({"OrderedDict", "defaultdict"}),
        "torch": frozenset(
            {
                "Tensor",
                "nn.Parameter",
                "device",
                "dtype",
                "Size",
                "SymInt",
                "SymFloat",
                "SymBool",
            }
        ),
        "torch._utils": frozenset(
            {"_rebuild_tensor_v2", "_rebuild_parameter", "_rebuild_cuda_tensor"}
        ),
        "torch.storage": frozenset({"TypedStorage", "UntypedStorage"}),
        "torch.distributed.checkpoint.metadata": frozenset(
            {
                "Metadata",
                "TensorStorageMetadata",
                "BytesStorageMetadata",
                "ChunkStorageMetadata",
                "TensorProperties",
                "StorageMeta",
            }
        ),
        "torch.distributed.checkpoint.planner": frozenset(
            {
                "SavePlan",
                "LoadPlan",
                "WriteItem",
                "ReadItem",
                "WriteItemType",
                "LoadItemType",
                "TensorWriteData",
            }
        ),
        "megatron.core.dist_checkpointing.mapping": frozenset(
            {"ShardedTensor", "ShardedObject", "ShardedStateDict", "StateDict"}
        ),
    }

    def find_class(self, module: str, name: str) -> Any:
        if module in self._allowed_modules and name in self._allowed_modules[module]:
            if module == "builtins":
                return getattr(__builtins__, name)
            else:
                mod = __import__(module, fromlist=[name])
                return getattr(mod, name)
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")


class RestrictedPickleModule:
    """Wrapper for torch.load to use MegatronUnpickler."""

    def load(self, file, **kwargs):
        return MegatronUnpickler(file).load()

    def loads(self, s):
        return MegatronUnpickler(io.BytesIO(s)).load()


def safe_pickle():
    """Context manager to temporarily set pickle.Unpickler to MegatronUnpickler."""

    @contextmanager
    def _safe_pickle():
        original_unpickler = pickle.Unpickler
        pickle.Unpickler = MegatronUnpickler
        try:
            yield
        finally:
            if pickle.Unpickler is not MegatronUnpickler:
                raise RuntimeError("Security violation: pickle.Unpickler was altered")
            pickle.Unpickler = original_unpickler

    return _safe_pickle()


def verify_integrity():
    if not isinstance(pickle.Unpickler, type):
        raise RuntimeError("pickle.Unpickler has been tampered with")
    # Optionally, check MegatronUnpickler itself hasn't been altered
    if MegatronUnpickler.find_class.__qualname__ != "MegatronUnpickler.find_class":
        raise RuntimeError("MegatronUnpickler has been tampered with")


verify_integrity()

# Define public API
__all__ = ["MegatronUnpickler", "RestrictedPickleModule", "safe_pickle"]

# Make attributes read-only using properties
for attr in __all__:
    globals()[attr] = property(lambda self, a=attr: globals()[a])


# Prevent attribute modification or addition
def __setattr__(name, value):
    raise AttributeError("Cannot modify or add attributes to this module")


this_module = sys.modules[__name__]
this_module.__setattr__ = __setattr__
