"""Arquitecturas del Lab 3 portadas de Keras a PyTorch.

Se migró a PyTorch porque TensorFlow >= 2.11 no usa la GPU en Windows nativo.
Para mantener la equivalencia con el modelo original:
  - Conv2D 3x3 con padding "same"  -> nn.Conv2d(kernel_size=3, padding=1)
  - inicialización glorot_uniform  -> nn.init.xavier_uniform_, sesgos en cero
  - softmax + sparse categorical CE -> nn.CrossEntropyLoss sobre logits
El número de parámetros coincide exactamente con el reportado por Keras.
"""
import torch
from torch import nn

NUM_CLASES = 29


def _bloque_conv(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU(inplace=True)]


class CNN2(nn.Module):
    """CNN 2 (profunda): mejor modelo del Lab 3. 2,244,285 parámetros."""

    def __init__(self, num_clases=NUM_CLASES, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            *_bloque_conv(3, 32), *_bloque_conv(32, 32), nn.MaxPool2d(2),    # 64 -> 32
            *_bloque_conv(32, 64), *_bloque_conv(64, 64), nn.MaxPool2d(2),   # 32 -> 16
            *_bloque_conv(64, 128), nn.MaxPool2d(2),                         # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_clases),
        )
        _inicializar_como_keras(self)

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN1(nn.Module):
    """CNN 1 (base). 2,120,413 parámetros."""

    def __init__(self, num_clases=NUM_CLASES, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            *_bloque_conv(3, 32), nn.MaxPool2d(2),
            *_bloque_conv(32, 64), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 64, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_clases),
        )
        _inicializar_como_keras(self)

    def forward(self, x):
        return self.classifier(self.features(x))


class RedSimpleFC(nn.Module):
    """Red totalmente conectada. 6,430,749 parámetros."""

    def __init__(self, num_clases=NUM_CLASES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 3, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_clases),
        )
        _inicializar_como_keras(self)

    def forward(self, x):
        return self.net(x)


def _inicializar_como_keras(modelo):
    for m in modelo.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


MODELOS = {"cnn2": CNN2, "cnn1": CNN1, "fc": RedSimpleFC}


def construir_modelo(nombre="cnn2", **kwargs):
    return MODELOS[nombre](**kwargs)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


if __name__ == "__main__":
    for nombre, clase in MODELOS.items():
        print(f"{nombre}: {contar_parametros(clase()):,} parámetros")
    x = torch.zeros(2, 3, 64, 64)
    print("salida cnn2:", tuple(CNN2()(x).shape))
