import os

import jax

# Persistent compilation cache: compiled programs are stored on disk and reused across runs and across identical
# env/policy instances (their methods are jitted with `self` static, so each new instance is otherwise recompiled).
# Most of the compilation time on GPU is autotuning, which the cache also saves. The cache key includes the program,
# the JAX version, the XLA flags and the device, so stale entries are never used.
# - Directory: $JAX_COMPILATION_CACHE_DIR if set, otherwise ~/.cache/socialjym/jax_compilation_cache.
# - Disable: SOCIALJYM_COMPILATION_CACHE=0.
if os.environ.get("SOCIALJYM_COMPILATION_CACHE", "1") != "0" and jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        os.path.join(os.path.expanduser("~"), ".cache", "socialjym", "jax_compilation_cache"),
    )
