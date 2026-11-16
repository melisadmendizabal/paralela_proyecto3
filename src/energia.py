"""Desactiva el "power throttling" (EcoQoS) de Windows 11 para el proceso actual.

Problema encontrado en el Paso 5: Windows clasifica como trabajo de fondo los
procesos sin ventana en primer plano y, tras unos segundos de carga, los envía
a núcleos de eficiencia a baja frecuencia. En este equipo eso hizo que el MISMO
código (un bucle de cálculo puro) corriera ~11 veces más lento a partir de la
cuarta ronda, lo que invalida cualquier medición.

La API documentada SetProcessInformation(ProcessPowerThrottling) permite que
un proceso pida no ser limitado. Se llama al inicio de todos los programas de
medición (secuencial y paralelo, y en cada proceso trabajador), de modo que
ambas versiones se miden en las mismas condiciones.
"""
import sys

_PROCESS_POWER_THROTTLING = 4          # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_EXECUTION_SPEED = 0x1                 # PROCESS_POWER_THROTTLING_EXECUTION_SPEED


def desactivar_ecoqos():
    """Devuelve True si se desactivó, False si falló, None si no es Windows."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Estado(ctypes.Structure):    # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    # ControlMask = EXECUTION_SPEED y StateMask = 0  ->  "no limitar la velocidad"
    estado = Estado(1, _EXECUTION_SPEED, 0)
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
                                          ctypes.byref(estado), ctypes.sizeof(estado)))
