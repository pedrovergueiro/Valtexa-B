"""Dispersão de esporos pelo vento: advecção-difusão-deposição no dossel, dia a dia.

O núcleo de contato do gêmeo (:mod:`app.gemeo.epidemia`) diz "folha encosta em folha": decai
igual para os dois lados da fila e não sabe de vento. Esporo de ferrugem viaja no ar, e o ar tem
direção. O modelo físico conceitual é a equação de advecção-difusão da concentração ``C`` de
esporos no ar do dossel:

    ∂C/∂t + u · ∇C = D ∇²C - λ C + S(x, y, t)

``u`` é o vento no dossel, ``D`` a difusão efetiva (turbulência), ``λ`` a perda (deposição nas
folhas e no chão, morte por UV) e ``S`` a fonte (plantas esporulando). Com passo diário, cada
dia é tratado em **regime permanente** (o esporo atravessa o talhão em minutos; o vento médio do
dia é o que importa para a soma do dia), e a solução para uma fonte pontual é a função de Green

    G(r) = exp(u · r / 2D) . K₀(κ |r|) / (2π D),   κ = √(|u|²/4D² + λ/D)

— ``K₀`` é a função de Bessel modificada de segunda espécie. Ela conserva a massa
(``λ ∫ G dA = 1``: o que sai da fonte deposita em algum lugar) e, sem vento, é isotrópica com
alcance ``√(D/λ)``; com vento, desloca e alonga a nuvem para onde o ar vai. A concentração do
talhão é a convolução das fontes com ``G`` (FFT numa grade, :class:`Grade`).

No gêmeo, a pressão de infecção de cada planta passa a ser a mistura

    pressão = (1 - ω) . Σ_j w_contato I_j  +  ω . Σ_j w_vento I_j

com ``w_vento`` escalado para ter a **mesma massa** do núcleo de contato no talhão (o ``β`` ajustado
continua significando o mesmo) — o vento muda **para onde** vai a pressão, não quanto. Dia sem
vento medido usa só o contato.

**O que não é medição.** A forma da equação é física; os números (``D``, ``λ``, a fração do
vento de 10 m que chega ao dossel, ``ω``) são **hipótese de projeto** em faixas plausíveis para
esporo de ferrugem em lavoura de café, não medidos. Para um patógeno específico faltam a
produção, a sobrevivência, a deposição e a infecção medidas; a calibração honesta pede
armadilha de esporos e anemômetro no talhão (``training/pinn_dispersao.py`` mostra como os
parâmetros são recuperados de medições, em dado sintético). Por isso a dispersão é **opcional e
desligada por padrão** no gêmeo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.gemeo.clima import DiaClimatico


@dataclass(frozen=True)
class ParametrosDispersao:
    difusao_m2_s: float = 0.5
    """``D`` em calmaria (turbulência do próprio dossel). Hipótese."""
    difusao_por_vento_m: float = 0.5
    """Quanto a difusão cresce com o vento: ``D = D₀ + k . |u_dossel|`` (m). Hipótese."""
    deposicao_1_s: float = 0.05
    """``λ``: com ``D₀`` dá alcance em calmaria ``√(D/λ)`` ≈ 3 m. Hipótese."""
    fracao_do_vento_no_dossel: float = 0.3
    """O vento de 10 m (estação, Open-Meteo) cai dentro da copa; fração que transporta."""
    fracao_aerea: float = 0.5
    """``ω``: parte da pressão que vem pelo ar (o resto, contato e respingo). Hipótese."""
    celula_m: float = 1.0

    def __post_init__(self) -> None:
        for nome in ("difusao_m2_s", "deposicao_1_s", "celula_m"):
            if not getattr(self, nome) > 0:
                raise ValueError(f"{nome} precisa ser positivo.")
        if not 0.0 <= self.fracao_aerea <= 1.0 or not 0 <= self.fracao_do_vento_no_dossel <= 1:
            raise ValueError("frações em [0, 1].")
        if self.difusao_por_vento_m < 0:
            raise ValueError("difusao_por_vento_m negativa.")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------------------------
# A física
# ---------------------------------------------------------------------------------------------


def bessel_k0(x: np.ndarray) -> np.ndarray:
    """``K₀(x)`` para ``x > 0`` (Abramowitz & Stegun 9.8.5 e 9.8.6; erro relativo < 1e-7)."""
    x = np.asarray(x, dtype=np.float64)
    saida = np.empty_like(x)
    pequeno = x <= 2.0
    t = x[pequeno]
    tt = (t / 3.75) ** 2
    i0 = 1.0 + tt * (3.5156229 + tt * (3.0899424 + tt * (1.2067492 + tt * (
        0.2659732 + tt * (0.0360768 + tt * 0.0045813)))))
    y = t * t / 4.0
    saida[pequeno] = (-np.log(t / 2.0) * i0 + (-0.57721566 + y * (0.42278420 + y * (
        0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.0000074)))))))
    g = x[~pequeno]
    y = 2.0 / g
    saida[~pequeno] = (np.exp(-g) / np.sqrt(g)) * (1.25331414 + y * (-0.07832358 + y * (
        0.02189568 + y * (-0.01062446 + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))))))
    return saida


def green(dx: np.ndarray, dy: np.ndarray, ux: float, uy: float, difusao: float, deposicao: float,
          r_minimo: float) -> np.ndarray:
    """Concentração permanente por unidade de fonte, a ``(dx, dy)`` da fonte."""
    r = np.maximum(np.hypot(dx, dy), r_minimo)
    kappa = math.sqrt((ux * ux + uy * uy) / (4.0 * difusao * difusao) + deposicao / difusao)
    # exp(u.r/2D) . K0(κ r): para r grande o produto é pequeno, mas cada fator pode estourar
    # sozinho; junta os expoentes (K0(z) ~ e^-z √(π/2z)).
    expo = (ux * dx + uy * dy) / (2.0 * difusao)
    z = kappa * r
    valor = np.where(z > 2.0,
                     np.exp(expo - z) * bessel_k0(np.maximum(z, 2.0)) * np.exp(
                         np.maximum(z, 2.0)),
                     np.exp(expo) * bessel_k0(np.minimum(z, 2.0)))
    return valor / (2.0 * math.pi * difusao)


def vento_no_dossel(dia: DiaClimatico, p: ParametrosDispersao, azimute_rua_graus: float
                    ) -> tuple[float, float, float]:
    """``(u ao longo da rua, u atravessando as ruas, D)`` do dia, no referencial do talhão.

    As ruas seguem o azimute (graus a partir do norte, horário); a rua 0 fica à esquerda de
    quem olha no sentido do azimute e a numeração cresce para a direita (a convenção das ruas
    ``G`` do SaaS).
    """
    az = math.radians(azimute_rua_graus)
    leste, norte = dia.vento_u_ms * p.fracao_do_vento_no_dossel, (
        dia.vento_v_ms * p.fracao_do_vento_no_dossel)
    ao_longo = leste * math.sin(az) + norte * math.cos(az)
    atravessando = leste * math.cos(az) - norte * math.sin(az)
    difusao = p.difusao_m2_s + p.difusao_por_vento_m * (
        dia.vento_ms * p.fracao_do_vento_no_dossel)
    return ao_longo, atravessando, difusao


# ---------------------------------------------------------------------------------------------
# A grade (convolução por FFT)
# ---------------------------------------------------------------------------------------------


@dataclass
class Grade:
    """As plantas numa grade regular; a concentração é ``fontes ⊛ G`` por FFT."""

    x: np.ndarray
    """``(N,)`` metros ao longo da rua."""
    y: np.ndarray
    """``(N,)`` metros atravessando (rua x espaçamento)."""
    celula_m: float
    _ix: np.ndarray = field(init=False, repr=False)
    _iy: np.ndarray = field(init=False, repr=False)
    _forma: tuple[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        x0, y0 = (self.x.min(), self.y.min()) if len(self.x) else (0.0, 0.0)
        self._ix = np.rint((self.x - x0) / self.celula_m).astype(np.int64)
        self._iy = np.rint((self.y - y0) / self.celula_m).astype(np.int64)
        self._forma = (int(self._iy.max()) + 1 if len(self.x) else 1,
                       int(self._ix.max()) + 1 if len(self.x) else 1)

    def concentracao(self, fonte: np.ndarray, ux: float, uy: float, difusao: float,
                     deposicao: float) -> np.ndarray:
        """``(N,)``: a concentração em cada planta, sem a contribuição da própria."""
        ny, nx = self._forma
        h = self.celula_m
        campo = np.zeros((ny, nx))
        np.add.at(campo, (self._iy, self._ix), fonte)
        # O núcleo em todos os deslocamentos possíveis, com o raio mínimo de meia célula
        # (a singularidade logarítmica de K0 fica na média da célula, não no centro).
        oy, ox = np.mgrid[-(ny - 1):ny, -(nx - 1):nx]
        # Fontes pontuais (uma por planta): C(x) = Σ_j q_j G(x - x_j), sem fator de área.
        nucleo = green(ox * h, oy * h, ux, uy, difusao, deposicao, h / 2.0)
        forma = (2 * ny - 1 + ny, 2 * nx - 1 + nx)
        conv = np.fft.irfft2(np.fft.rfft2(campo, forma) * np.fft.rfft2(nucleo, forma), forma)
        conv = conv[ny - 1:2 * ny - 1, nx - 1:2 * nx - 1]
        proprio = nucleo[ny - 1, nx - 1] * np.asarray(fonte, dtype=np.float64)
        return np.maximum(conv[self._iy, self._ix] - proprio, 0.0)


class NucleoComVento:
    """O núcleo do gêmeo com a parte que vai pelo ar: ``para(dia)`` dá a pressão do dia.

    Mesma interface do :class:`~app.gemeo.epidemia.Nucleo` (chamável com ``I``); o
    :func:`~app.gemeo.epidemia.passo` pede ``para(dia)`` quando existe.
    """

    def __init__(self, contato, grafo, p: ParametrosDispersao, azimute_rua_graus: float) -> None:
        self.contato = contato
        self.p = p
        self.azimute_rua_graus = float(azimute_rua_graus)
        self.grade = Grade(grafo.x_m, grafo.rua * grafo.espacamento_entre_ruas_m, p.celula_m)
        # Mesma massa que o contato, medida no próprio talhão: a resposta dos dois núcleos a
        # uma fonte uniforme, em calmaria. (O contínuo não serve: entre ruas a 3,5 m o contato
        # é quase só ao longo da fila.) A massa depositada não depende do vento (λ ∫ G = 1),
        # então a escala vale para todo dia.
        um = np.ones(len(grafo.x_m))
        calmo = p.deposicao_1_s * self.grade.concentracao(um, 0.0, 0.0, p.difusao_m2_s,
                                                          p.deposicao_1_s)
        self._massa = float(np.median(contato(um)) / max(np.median(calmo), 1e-12))
        self._cache: dict[tuple, Any] = {}

    def __call__(self, valor: np.ndarray) -> np.ndarray:
        return self.contato(valor)

    def para(self, dia: DiaClimatico):
        if not dia.tem_vento or self.p.fracao_aerea == 0.0:
            return self.contato
        ux, uy, difusao = vento_no_dossel(dia, self.p, self.azimute_rua_graus)
        omega, lam, massa = self.p.fracao_aerea, self.p.deposicao_1_s, self._massa

        def pressao(valor: np.ndarray) -> np.ndarray:
            # λ.C é o que deposita; na escala do contato (mesma massa).
            ar = lam * massa * self.grade.concentracao(valor, ux, uy, difusao, lam)
            return (1.0 - omega) * self.contato(valor) + omega * ar

        return pressao
