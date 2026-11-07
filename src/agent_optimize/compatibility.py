"""Signatures of the actual Python/numeric/compiler/CPU environment."""

import platform
import struct
import sys


KERNEL_OPTIONS = {"fastmath": False, "boundscheck": True, "cache": False}


def environment_signature():
    import awkward
    import numpy
    import numba
    import llvmlite
    from llvmlite import binding

    return {
        "python": platform.python_version(), "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "awkward": awkward.__version__, "numpy": numpy.__version__,
        "numpy_error_policy": numpy.geterr(),
        "numba": numba.__version__, "llvmlite": llvmlite.__version__,
        "system": platform.system(), "release": platform.release(),
        "machine": platform.machine(), "bits": struct.calcsize("P") * 8,
        "cpu_name": str(binding.get_host_cpu_name()),
        "cpu_features": binding.get_host_cpu_features().flatten(),
        "numba_cpu_name": numba.config.CPU_NAME,
        "numba_cpu_features": numba.config.CPU_FEATURES,
        "numba_disable_jit": bool(numba.config.DISABLE_JIT),
        "kernel_options": KERNEL_OPTIONS,
    }
