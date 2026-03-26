"""
Tqdm patching for ClearVoice progress display.

Forza tqdm a mostrar progreso incluso cuando stdout no es TTY.
Se debe importar ANTES de clearvoice.

Usage:
    from speechlib.enhance_audio_patch import patch_tqdm
    patch_tqdm()

    from clearvoice import ClearVoice
    # Ahora tqdm mostrará progreso garantizado
"""

import sys
import typing as tp
from functools import partialmethod


class _DummyTqdm:
    """Dummy tqdm que simplemente itera sin mostrar nada."""

    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable
        self.n = 0
        self.total = kwargs.get("total", 0)

    def __iter__(self):
        if self.iterable:
            for item in self.iterable:
                yield item
        else:
            for i in range(self.total):
                yield i

    def update(self, n=1):
        self.n += n

    def set_description(self, desc=None):
        pass

    def set_postfix(self, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self


def patch_tqdm(force_display: bool = True) -> None:
    """
    Parchea tqdm para forzar display de progreso.

    Args:
        force_display: Si True, tqdm siempre muestra progreso aunque no sea TTY.
                      Si False, usa comportamiento por defecto de tqdm.
    """
    try:
        from tqdm import tqdm
        import tqdm as tqdm_module

        if force_display:
            original_init = tqdm.__init__
            original_trange = tqdm_module.trange

            def _patched_init(self, iterable=None, total=None, **kwargs):
                kwargs.setdefault("disable", False)
                kwargs.setdefault("leave", True)
                return original_init(self, iterable, total, **kwargs)

            def _patched_trange(*args, **kwargs):
                kwargs.setdefault("disable", False)
                return original_trange(*args, **kwargs)

            tqdm.__init__ = _patched_init
            tqdm_module.trange = _patched_trange

        print("[speechlib] tqdm patched for progress display")

    except ImportError:
        print("[speechlib] tqdm not found, skipping patch")


def tqdm_always_enabled() -> tp.ContextManager[None]:
    """
    Context manager que parcha tqdm temporalmente.

    Usage:
        with tqdm_always_enabled():
            from clearvoice import ClearVoice
            # tqdm mostrará progreso aquí
    """
    try:
        from tqdm import tqdm as original_tqdm
        import tqdm as tqdm_module

        original_init = original_tqdm.__init__
        original_trange = tqdm_module.trange

        def _patched_init(self, iterable=None, total=None, **kwargs):
            kwargs.setdefault("disable", False)
            kwargs.setdefault("leave", True)
            return original_init(self, iterable, total, **kwargs)

        def _patched_trange(*args, **kwargs):
            kwargs.setdefault("disable", False)
            return original_trange(*args, **kwargs)

        original_tqdm.__init__ = _patched_init
        tqdm_module.trange = _patched_trange

        yield

        original_tqdm.__init__ = original_init
        tqdm_module.trange = original_trange

    except ImportError:
        yield
