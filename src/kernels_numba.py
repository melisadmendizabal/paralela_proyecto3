"""PASO 5 - Kernels propios con Numba CUDA (variante T4) y fase F5 con ellos.

Implementa el diseño de la sección 5.3 del informe para la propagación hacia
adelante de la CNN 2 (inferencia), con todos los tensores en formato NCHW:

  conv3x3_relu   un hilo por valor de salida; bloque (B, B); grid
                 (⌈W/B⌉, ⌈H/B⌉, N·C_out); tesela (B+2)×(B+2) de la entrada y los
                 9·C_in pesos del filtro en memoria compartida; sesgo + ReLU fusionados.
  conv_global    misma asignación de hilos, pero leyendo todo de memoria global
                 (sirve para medir el efecto de la memoria compartida).
  maxpool2       un hilo por valor de salida, grid de 1 dimensión, bloques de 256.
  densa          multiplicación de matrices por teselas T×T en memoria compartida
                 (salida = entrada · Wᵀ + sesgo, ReLU opcional).

Los tamaños de la memoria compartida deben ser constantes al compilar, por eso
cada kernel se genera para un tamaño de bloque/tesela concreto (8, 16 o 32).
"""
import math

import numpy as np

from . import numba_compat  # debe importarse antes que numba.cuda
from numba import cuda, float32

# Este módulo debe importarse ANTES que torch (ver src/numba_compat.py)
numba_compat.preparar_enlazador()

MAX_PESOS = 9 * 64          # C_in máximo de la CNN 2 es 64 -> 576 pesos por filtro
HILOS_POOL = 256
_CONV, _CONV_GLOBAL, _DENSA = {}, {}, {}


# ============================================================================
# Fábricas de kernels
# ============================================================================
def kernel_conv(B, max_reg=None):
    """Convolución 3×3 (padding 1) + sesgo + ReLU con memoria compartida, bloque B×B.

    max_reg limita los registros por hilo (el compilador guarda el resto en memoria
    local). Sin límite el kernel usa 72 registros: con B=32 (1,024 hilos) se
    necesitarían 73,728 registros por bloque y el SM solo tiene 65,536.
    """
    if (B, max_reg) in _CONV:
        return _CONV[(B, max_reg)]
    T = B + 2                  # tesela con halo de 1 píxel por lado
    NH = B * B

    @cuda.jit(max_registers=max_reg)
    def conv3x3_relu(x, w, b, y, relu):
        tesela = cuda.shared.array((T, T), float32)
        pesos = cuda.shared.array(MAX_PESOS, float32)
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        c_in, alto, ancho = x.shape[1], x.shape[2], x.shape[3]
        c_out = y.shape[1]
        n, co = bz // c_out, bz % c_out
        ox, oy = bx * B + tx, by * B + ty
        tid = ty * B + tx

        # Los 9·C_in pesos del filtro co, cargados entre todos los hilos del bloque
        for k in range(tid, c_in * 9, NH):
            pesos[k] = w[co, k // 9, (k % 9) // 3, k % 3]

        acc = float32(0.0)
        for ci in range(c_in):
            cuda.syncthreads()           # nadie sigue leyendo la tesela anterior
            # Carga cooperativa de la tesela (B+2)×(B+2); fuera de la imagen vale 0 (padding)
            for k in range(tid, T * T, NH):
                fy, fx = k // T, k % T
                gy, gx = by * B - 1 + fy, bx * B - 1 + fx
                v = float32(0.0)
                if gy >= 0 and gy < alto and gx >= 0 and gx < ancho:
                    v = x[n, ci, gy, gx]
                tesela[fy, fx] = v
            cuda.syncthreads()           # tesela completa antes de usarla
            if oy < alto and ox < ancho:
                base = ci * 9
                for dy in range(3):
                    for dx in range(3):
                        acc += tesela[ty + dy, tx + dx] * pesos[base + dy * 3 + dx]
        if oy < alto and ox < ancho:
            acc += b[co]
            if relu and acc < 0:
                acc = float32(0.0)
            y[n, co, oy, ox] = acc

    _CONV[(B, max_reg)] = conv3x3_relu
    return conv3x3_relu


def kernel_conv_multi(B=16, CO=8, CI=4):
    """Versión 2: cada hilo calcula CO filtros de salida para su píxel.

    Por cada carga de CI canales de entrada en memoria compartida, cada hilo hace
    9·CI·CO multiplicaciones (con B=16, CO=8, CI=4: 288) en lugar de 9. Los pesos
    de los CO filtros del bloque se guardan en memoria compartida (CO·9·C_in floats,
    hasta 18 KB) y todos los hilos de un warp leen la misma dirección (difusión).
    grid = (⌈W/B⌉, ⌈H/B⌉, N·C_out/CO).
    """
    clave = ("multi", B, CO, CI)
    if clave in _CONV:
        return _CONV[clave]
    T = B + 2
    NH = B * B
    MAXW = CO * MAX_PESOS

    @cuda.jit
    def conv_multi(x, w, b, y, relu):
        tesela = cuda.shared.array((CI, T, T), float32)
        pesos = cuda.shared.array(MAXW, float32)
        acc = cuda.local.array(CO, float32)
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
        c_in, alto, ancho = x.shape[1], x.shape[2], x.shape[3]
        grupos = y.shape[1] // CO
        n, co0 = bz // grupos, (bz % grupos) * CO
        ox, oy = bx * B + tx, by * B + ty
        tid = ty * B + tx
        for k in range(tid, CO * c_in * 9, NH):          # pesos de los CO filtros del bloque
            j, r = k // (c_in * 9), k % (c_in * 9)
            pesos[k] = w[co0 + j, r // 9, (r % 9) // 3, r % 3]
        for j in range(CO):
            acc[j] = float32(0.0)
        for c0 in range(0, c_in, CI):
            cuda.syncthreads()
            for k in range(tid, CI * T * T, NH):            # CI canales de entrada a la vez
                c, r = k // (T * T), k % (T * T)
                fy, fx = r // T, r % T
                gy, gx = by * B - 1 + fy, bx * B - 1 + fx
                v = float32(0.0)
                if c0 + c < c_in and gy >= 0 and gy < alto and gx >= 0 and gx < ancho:
                    v = x[n, c0 + c, gy, gx]
                tesela[c, fy, fx] = v
            cuda.syncthreads()
            if oy < alto and ox < ancho:
                for c in range(CI):
                    if c0 + c < c_in:
                        base = (c0 + c) * 9
                        for dy in range(3):
                            for dx in range(3):
                                v = tesela[c, ty + dy, tx + dx]
                                for j in range(CO):
                                    acc[j] += v * pesos[j * c_in * 9 + base + dy * 3 + dx]
        if oy < alto and ox < ancho:
            for j in range(CO):
                s = acc[j] + b[co0 + j]
                if relu and s < 0:
                    s = float32(0.0)
                y[n, co0 + j, oy, ox] = s

    _CONV[clave] = conv_multi
    return conv_multi


def kernel_conv_global(B):
    """Misma asignación de hilos que kernel_conv, sin memoria compartida."""
    if B in _CONV_GLOBAL:
        return _CONV_GLOBAL[B]

    @cuda.jit
    def conv_global(x, w, b, y, relu):
        c_in, alto, ancho = x.shape[1], x.shape[2], x.shape[3]
        c_out = y.shape[1]
        n, co = cuda.blockIdx.z // c_out, cuda.blockIdx.z % c_out
        ox = cuda.blockIdx.x * B + cuda.threadIdx.x
        oy = cuda.blockIdx.y * B + cuda.threadIdx.y
        if oy < alto and ox < ancho:
            acc = float32(0.0)
            for ci in range(c_in):
                for dy in range(3):
                    gy = oy + dy - 1
                    if gy >= 0 and gy < alto:
                        for dx in range(3):
                            gx = ox + dx - 1
                            if gx >= 0 and gx < ancho:
                                acc += x[n, ci, gy, gx] * w[co, ci, dy, dx]
            acc += b[co]
            if relu and acc < 0:
                acc = float32(0.0)
            y[n, co, oy, ox] = acc

    _CONV_GLOBAL[B] = conv_global
    return conv_global


@cuda.jit
def maxpool2(x, y):
    """Max-pooling 2×2 con paso 2. Un hilo por valor de salida (grid de 1 dimensión)."""
    i = cuda.grid(1)
    n_, c_, ho_, wo_ = y.shape
    if i < n_ * c_ * ho_ * wo_:
        wo = i % wo_
        t = i // wo_
        ho = t % ho_
        t = t // ho_
        c = t % c_
        n = t // c_
        y0, x0 = 2 * ho, 2 * wo
        m = max(max(x[n, c, y0, x0], x[n, c, y0, x0 + 1]), max(x[n, c, y0 + 1, x0], x[n, c, y0 + 1, x0 + 1]))
        y[n, c, ho, wo] = m


def kernel_densa(TS):
    """salida[M×N] = a[M×K] · w[N×K]ᵀ + b (+ ReLU), con teselas TS×TS en memoria compartida."""
    if TS in _DENSA:
        return _DENSA[TS]

    TS1 = TS + 1               # +1 columna: evita conflictos de banco (constante al compilar)

    @cuda.jit
    def densa(a, w, b, salida, relu):
        sa = cuda.shared.array((TS, TS), float32)
        sw = cuda.shared.array((TS, TS1), float32)
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        fila = cuda.blockIdx.y * TS + ty               # imagen del lote
        col = cuda.blockIdx.x * TS + tx                # neurona de salida
        m, k_total = a.shape
        n_out = w.shape[0]
        acc = float32(0.0)
        for t in range((k_total + TS - 1) // TS):
            k = t * TS + tx
            sa[ty, tx] = a[fila, k] if (fila < m and k < k_total) else float32(0.0)
            j = cuda.blockIdx.x * TS + ty                # fila de W que carga este hilo
            sw[ty, tx] = w[j, k] if (j < n_out and k < k_total) else float32(0.0)
            cuda.syncthreads()
            for kk in range(TS):
                acc += sa[ty, kk] * sw[tx, kk]
            cuda.syncthreads()
        if fila < m and col < n_out:
            acc += b[col]
            if relu and acc < 0:
                acc = float32(0.0)
            salida[fila, col] = acc

    _DENSA[TS] = densa
    return densa


# ============================================================================
# Lanzadores (configuración de grid y bloque)
# ============================================================================
def lanzar_conv(x, w, b, y, B=16, compartida=True, relu=True, max_reg=None, multi=0):
    """multi > 0: versión 2 con `multi` filtros de salida por hilo."""
    n, c_out, alto, ancho = y.shape
    if multi:
        grid = (math.ceil(ancho / B), math.ceil(alto / B), n * c_out // multi)
        kernel_conv_multi(B, multi)[grid, (B, B)](x, w, b, y, relu)
        return grid
    grid = (math.ceil(ancho / B), math.ceil(alto / B), n * c_out)
    k = kernel_conv(B, max_reg) if compartida else kernel_conv_global(B)
    k[grid, (B, B)](x, w, b, y, relu)
    return grid


def lanzar_pool(x, y):
    total = y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3]
    grid = math.ceil(total / HILOS_POOL)
    maxpool2[grid, HILOS_POOL](x, y)
    return grid


def lanzar_densa(a, w, b, salida, TS=16, relu=False):
    m, n_out = salida.shape
    grid = (math.ceil(n_out / TS), math.ceil(m / TS))
    kernel_densa(TS)[grid, (TS, TS)](a, w, b, salida, relu)
    return grid


# ============================================================================
# CNN 2 completa (inferencia) con los kernels propios
# ============================================================================
# Índices de las capas en el state_dict de src/modelo.py (CNN2)
_CONVS = [("features.0", 3, 32), ("features.2", 32, 32), ("features.5", 32, 64),
          ("features.7", 64, 64), ("features.10", 64, 128)]
_POOL_DESPUES = {1, 3, 4}            # max-pooling después de las convoluciones 2, 4 y 5
_DENSAS = [("classifier.1", True), ("classifier.4", False)]


class CNN2Numba:
    """Propagación hacia adelante de la CNN 2 usando solo kernels propios."""

    def __init__(self, state_dict, B=16, TS=16, compartida=True, lote_max=256, multi=8):
        def a_dev(t):
            return cuda.to_device(np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32))
        self.B, self.TS, self.compartida, self.multi = B, TS, compartida, multi
        self.convs = [(a_dev(state_dict[f"{k}.weight"]), a_dev(state_dict[f"{k}.bias"])) for k, _, _ in _CONVS]
        self.densas = [(a_dev(state_dict[f"{k}.weight"]), a_dev(state_dict[f"{k}.bias"]), r) for k, r in _DENSAS]
        # Buffers de activaciones reservados una sola vez para el lote máximo
        L = lote_max
        formas = [(L, 32, 64, 64), (L, 32, 64, 64), (L, 32, 32, 32), (L, 64, 32, 32), (L, 64, 32, 32),
                  (L, 64, 16, 16), (L, 128, 16, 16), (L, 128, 8, 8), (L, 256), (L, 29)]
        self.buf = [cuda.device_array(f, dtype=np.float32) for f in formas]
        self.lote_max = L

    def adelante(self, x):
        """x: arreglo de dispositivo (n, 3, 64, 64) float32 contiguo. Devuelve logits (n, 29) en la GPU."""
        n = x.shape[0]
        b = [v[:n] for v in self.buf]
        c1, c2, p1, c3, c4, p2, c5, p3, d1, d2 = b
        (w1, b1), (w2, b2), (w3, b3), (w4, b4), (w5, b5) = self.convs
        lanzar_conv(x, w1, b1, c1, self.B, self.compartida, multi=self.multi)
        lanzar_conv(c1, w2, b2, c2, self.B, self.compartida, multi=self.multi)
        lanzar_pool(c2, p1)
        lanzar_conv(p1, w3, b3, c3, self.B, self.compartida, multi=self.multi)
        lanzar_conv(c3, w4, b4, c4, self.B, self.compartida, multi=self.multi)
        lanzar_pool(c4, p2)
        lanzar_conv(p2, w5, b5, c5, self.B, self.compartida, multi=self.multi)
        lanzar_pool(c5, p3)
        plano = p3.reshape((n, 128 * 8 * 8))          # Flatten en orden C, H, W (igual que PyTorch)
        (wd1, bd1, r1), (wd2, bd2, r2) = self.densas
        lanzar_densa(plano, wd1, bd1, d1, self.TS, relu=r1)   # Dropout no actúa en inferencia
        lanzar_densa(d1, wd2, bd2, d2, self.TS, relu=r2)
        return d2

    def predecir(self, X):
        """X: arreglo de dispositivo (N, 3, 64, 64). Devuelve (logits host, predicciones host)."""
        salidas = []
        for i in range(0, X.shape[0], self.lote_max):
            salidas.append(self.adelante(X[i:i + self.lote_max]).copy_to_host())
        logits = np.concatenate(salidas)
        return logits, logits.argmax(axis=1)


def registros_por_hilo(kernel):
    """Registros por hilo de un kernel ya compilado (para calcular la ocupación real)."""
    try:
        r = kernel.get_regs_per_thread()
        return max(r.values()) if isinstance(r, dict) else int(r)
    except Exception:
        return None


def memoria_compartida(kernel):
    try:
        r = kernel.get_shared_mem_per_block()
        return max(r.values()) if isinstance(r, dict) else int(r)
    except Exception:
        return None
