"""Compatibilidad de numba-cuda 0.30.4 con este entorno (numpy 2.5 + PyTorch cu124).

1. numpy >= 2.5 eliminó np.row_stack (alias de np.vstack), que numba-cuda todavía
   registra. Sin el alias, la primera compilación falla con
   AttributeError: module 'numpy' has no attribute 'row_stack'.
   Importar este módulo antes que numba.cuda lo restaura; no cambia ningún resultado.

2. Conflicto de nvJitLink: PyTorch 2.6+cu124 carga su propia librería
   nvJitLink (CUDA 12.4) al importarse, y Windows entrega esa misma copia a Numba,
   que genera código con NVVM de CUDA 12.9 y necesita la versión 12.9. El síntoma es
   nvJitLinkError: "may need newer version of nvJitLink library". Basta con IMPORTAR
   torch para provocarlo. Solución: preparar_enlazador() compila un kernel mínimo
   para que Numba cargue primero su nvJitLink 12.9; PyTorch, importado después,
   recibe esa versión (más nueva y compatible) y funciona con normalidad.
"""
import sys

import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

_listo = False


def preparar_enlazador():
    """Carga el nvJitLink de Numba. Debe llamarse ANTES de importar torch."""
    global _listo
    if _listo:
        return
    if "torch" in sys.modules:
        raise RuntimeError("Importe src.kernels_numba (o llame a numba_compat.preparar_enlazador()) "
                           "ANTES de importar torch: PyTorch carga un nvJitLink más antiguo que el de Numba.")
    from numba import cuda

    @cuda.jit
    def _nada(a):
        a[0] = 1

    _nada[1, 1](cuda.device_array(1, dtype=np.float32))
    cuda.synchronize()
    _listo = True
