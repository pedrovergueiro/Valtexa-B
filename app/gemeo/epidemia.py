"""O modelo epidemiológico do gêmeo: a física do contágio na lavoura, planta a planta.

Cada planta é um nó do grafo da lavoura (:mod:`app.gemeo.grafo`) e está em um de quatro
estados; o modelo carrega a **probabilidade** de cada um (somam 1 por planta — a conservação
que o modelo nunca quebra):

* ``S`` suscetível; ``E`` infectada em período latente (pegou, mas ainda não esporula);
* ``I`` infecciosa (esporula: é fonte de inóculo, e é o que o Rover vê);
* ``R`` removida (a planta parou de ser fonte: desfolha, lesões velhas).

É a aproximação de campo médio de um SEIR individual — o padrão da epidemiologia de plantas
com dado de **incidência**, que é o que a fusão entrega: P(planta doente), não a área lesionada.
A primeira versão carregava fração de área foliar e ligava à câmera por ``1 - e^(-κ.I)``; com
sinal binário, a assimilação desse estado não tinha solução boa (ver
:func:`estado_da_observacao`) e ``κ`` era só calibre.

O passo é **diário** e cada taxa sai do tempo daquele dia (:mod:`app.gemeo.clima`):

    favorabilidade  = f_T(temperatura) . f_W(molhamento foliar) . (1 + c . chuva)
    força_i         = β . favorabilidade . Σ_j w_ij . I_j  +  fundo . favorabilidade
    S -> E          : S_i . (1 - e^(-força_i)) . (1 - proteção_i)
    E -> I          : E_i . (1 - e^(-1/latência(T)))
    I -> R          : I_i . (1 - e^(-1/infecciosidade))

* ``f_T`` é a função beta de Yan & Hunt (1999): zero abaixo de ``t_min`` e acima de ``t_max``,
  máxima em ``t_opt``;
* ``f_W`` é logística nas horas de folha molhada: o fungo precisa de água livre para germinar
  (no bicho-mineiro, que é praga, o sinal inverte: o tempo seco favorece);
* a chuva entra como dispersão por respingo (``c > 0``) ou como lavagem (``c < 0``);
* a latência encurta com a temperatura perto do ótimo e alonga no frio;
* ``w_ij = exp(-d_rua/a_rua) . exp(-d_entre/a_entre)``: o mesmo núcleo anisotrópico da
  prescrição (:mod:`app.prescricao`) — folha encosta em folha na fila, não na entrelinha;
* ``fundo`` é o inóculo que chega de fora do talhão (vento, talhão vizinho).

**O limite do campo médio.** O modelo põe em cada planta a força *esperada* de todas as
vizinhas; na lavoura a doença anda em reboleira — a vizinha que esporula tem vizinhas que já
pegaram —, e o campo médio espalha mais depressa que a epidemia de verdade: com os parâmetros
verdadeiros, previa P(esporulando) cinco vezes maior nas plantas limpas de um talhão sintético
estocástico (16/10/2026). O ajuste compensa com ``β`` menor (cerca de metade). A incidência não
linear ``(Σ w I)^q`` da literatura (Liu, Hethcote & Levin, 1987) foi testada para corrigir isso
e **recusada**: o ajuste levou ``q`` ao limite de cima e a previsão da passada seguinte piorou
(ver ``MODELS.md``).

**A observação** liga o modelo ao Rover: P(sinal) = ``a + (s - a) . I`` — ``s`` é a
sensibilidade (a planta que esporula é sinalizada com probabilidade ``s``: lesão no baixeiro,
oclusão) e ``a`` o alarme falso por planta. A posterior da fusão entra como **evidência
virtual**: a razão de verossimilhança ``odds(posterior) / odds(prior da cultura)``, a mesma que
a fusão somou dos recortes (:func:`razao_de_verossimilhanca`).

**De onde vêm os números.** As formas (beta, logística, exponencial) são da epidemiologia de
plantas. Os parâmetros biológicos de cada doença (:data:`PRESETS`) são **faixas qualitativas**
— a ferrugem do café germina com água livre e temperatura amena, a cercosporiose gosta de
calor, a phoma de frio úmido, o bicho-mineiro de calor seco — e **não** medição nossa. Os
que mais pesam (``β``, ``a_rua``, ``a_entre``, ``fundo``, ``a``, ``s``) são ajustados às
passadas do próprio talhão (:mod:`app.gemeo.ajuste`); os biológicos ficam fixos até haver
série longa para estimá-los, e devem ser revistos pelo agrônomo.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from app.gemeo.clima import DiaClimatico
from app.gemeo.grafo import GrafoLavoura, Vizinhanca

COMPARTIMENTOS = ("S", "E", "I", "R")


@dataclass(frozen=True)
class ParametrosDoenca:
    doenca: str
    t_min: float
    t_opt: float
    t_max: float
    molhamento_meio_h: float
    """Horas de folha molhada com metade do efeito máximo."""
    molhamento_inclinacao: float
    """Inclinação da logística (1/h). Negativa: o molhamento **desfavorece** (praga)."""
    efeito_chuva: float
    """Dispersão por respingo (> 0) ou lavagem (< 0), na chuva de 20 mm ou mais."""
    latencia_min_dias: float
    latencia_max_dias: float
    infecciosidade_dias: float
    beta: float
    alcance_na_rua_m: float
    alcance_entre_ruas_m: float
    fundo: float
    sensibilidade: float = 0.9
    """P(sinal) numa planta que esporula: a lesão no baixeiro ou atrás de folha escapa."""
    alarme_falso: float = 0.03
    """P(sinal) numa planta sem lesão visível (ver ``plant_false_alarm`` da calibração da
    fusão)."""
    latente_por_infeccioso: float = 1.0
    """Na condição inicial, quanto ``E`` existe para cada ``I`` (o que ainda vai aparecer)."""
    origem: str = "prior: faixa qualitativa, a calibrar com as passadas"

    def __post_init__(self) -> None:
        if not self.t_min < self.t_opt < self.t_max:
            raise ValueError(f"{self.doenca}: exige t_min < t_opt < t_max.")
        if not 0 < self.latencia_min_dias <= self.latencia_max_dias:
            raise ValueError(f"{self.doenca}: latência inválida.")
        for nome in ("infecciosidade_dias", "beta", "alcance_na_rua_m", "alcance_entre_ruas_m"):
            if not getattr(self, nome) > 0:
                raise ValueError(f"{self.doenca}: {nome} precisa ser positivo.")
        if self.fundo < 0:
            raise ValueError(f"{self.doenca}: fundo negativo.")
        if not 0 <= self.alarme_falso < self.sensibilidade <= 1:
            raise ValueError(f"{self.doenca}: exige 0 <= alarme_falso < sensibilidade <= 1.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ParametrosDoenca:
        return cls(**data)


PRESETS: dict[str, ParametrosDoenca] = {
    # ``infecciosidade_dias`` é da **planta**, não da lesão: a planta com ferrugem segue fonte
    # enquanto lesões novas substituem as que necrosam (a autoinfecção mora aqui).
    # Ferrugem (Hemileia vastatrix): germina com água livre, temperatura amena; respingo de
    # chuva espalha; lesão esporula por semanas. A doença mais estruturada do café.
    "Rust": ParametrosDoenca("Rust", t_min=15.0, t_opt=22.0, t_max=28.0,
                             molhamento_meio_h=6.0, molhamento_inclinacao=0.8,
                             efeito_chuva=0.5, latencia_min_dias=20.0, latencia_max_dias=60.0,
                             infecciosidade_dias=120.0, beta=0.08, alcance_na_rua_m=1.5,
                             alcance_entre_ruas_m=0.8, fundo=1e-4),
    # Cercosporiose (Cercospora coffeicola): calor e umidade; muito ligada a estresse e
    # nutrição — que o modelo não vê (ver MODELS.md).
    "Cercospora_Leaf_Spot": ParametrosDoenca(
        "Cercospora_Leaf_Spot", t_min=15.0, t_opt=25.0, t_max=32.0, molhamento_meio_h=8.0,
        molhamento_inclinacao=0.6, efeito_chuva=0.3, latencia_min_dias=10.0,
        latencia_max_dias=30.0, infecciosidade_dias=90.0, beta=0.05, alcance_na_rua_m=1.0,
        alcance_entre_ruas_m=0.6, fundo=2e-4),
    # Phoma: frio úmido, vento; mais em lavoura alta e exposta.
    "Phoma_Leaf_Spot": ParametrosDoenca(
        "Phoma_Leaf_Spot", t_min=10.0, t_opt=18.0, t_max=25.0, molhamento_meio_h=8.0,
        molhamento_inclinacao=0.6, efeito_chuva=0.3, latencia_min_dias=7.0,
        latencia_max_dias=20.0, infecciosidade_dias=60.0, beta=0.05, alcance_na_rua_m=1.0,
        alcance_entre_ruas_m=0.8, fundo=2e-4),
    # Bicho-mineiro (Leucoptera coffeella): praga de tempo quente e seco; chuva lava e derruba.
    # "Latência" é ovo → mina visível; "infecciosidade", a geração que sai da mina.
    "Leaf_Miner_Damage": ParametrosDoenca(
        "Leaf_Miner_Damage", t_min=15.0, t_opt=27.0, t_max=34.0, molhamento_meio_h=6.0,
        molhamento_inclinacao=-0.5, efeito_chuva=-0.5, latencia_min_dias=15.0,
        latencia_max_dias=40.0, infecciosidade_dias=45.0, beta=0.08, alcance_na_rua_m=3.0,
        alcance_entre_ruas_m=2.0, fundo=5e-4),
}


def parametros_para(doenca: str) -> ParametrosDoenca:
    if doenca not in PRESETS:
        raise KeyError(f"Sem modelo epidemiológico para {doenca!r}: "
                       f"disponíveis {', '.join(sorted(PRESETS))}.")
    return PRESETS[doenca]


# ---------------------------------------------------------------------------------------------
# Respostas ao tempo
# ---------------------------------------------------------------------------------------------


def resposta_temperatura(t: float, p: ParametrosDoenca) -> float:
    """Função beta de Yan & Hunt (1999): 0 fora de (t_min, t_max), 1 em t_opt."""
    if not p.t_min < t < p.t_max:
        return 0.0
    expoente = (p.t_opt - p.t_min) / (p.t_max - p.t_opt)
    return ((p.t_max - t) / (p.t_max - p.t_opt)) * ((t - p.t_min) / (p.t_opt - p.t_min)) ** expoente


def resposta_molhamento(horas: float, p: ParametrosDoenca) -> float:
    return 1.0 / (1.0 + math.exp(-p.molhamento_inclinacao * (horas - p.molhamento_meio_h)))


def favorabilidade(dia: DiaClimatico, p: ParametrosDoenca) -> float:
    """O quanto o dia favorece a infecção nova, de 0 a ~1,5 (com a chuva espalhando)."""
    chuva = 1.0 + p.efeito_chuva * min(max(dia.chuva_mm, 0.0), 20.0) / 20.0
    return (resposta_temperatura(dia.t_media, p) * resposta_molhamento(dia.molhamento_h, p)
            * max(0.0, chuva))


def latencia_dias(t: float, p: ParametrosDoenca) -> float:
    return p.latencia_min_dias + (p.latencia_max_dias - p.latencia_min_dias) * (
        1.0 - resposta_temperatura(t, p))


# ---------------------------------------------------------------------------------------------
# Estado e simulação
# ---------------------------------------------------------------------------------------------


@dataclass
class Estado:
    """Probabilidade de cada planta estar em cada estado (somam 1 por planta)."""

    S: np.ndarray
    E: np.ndarray
    I: np.ndarray  # noqa: E741 (o nome do compartimento)
    R: np.ndarray

    def copia(self) -> Estado:
        return Estado(self.S.copy(), self.E.copy(), self.I.copy(), self.R.copy())

    @classmethod
    def sadio(cls, n: int) -> Estado:
        z = np.zeros(n)
        return cls(np.ones(n), z.copy(), z.copy(), z.copy())


@dataclass(frozen=True)
class Tratamento:
    """Uma aplicação: em que dia, em que plantas, e o que o produto faz."""

    dia: int
    """Índice do dia na simulação (0 = o primeiro dia simulado)."""
    plantas: np.ndarray
    """Máscara booleana ``(N,)``."""
    protecao: float
    """Fração da infecção nova que o produto impede enquanto dura o residual."""
    residual_dias: int
    curativo: float = 0.0
    """Fração do latente (``E``) que o produto elimina no dia da aplicação."""


@dataclass(frozen=True)
class Nucleo:
    """O núcleo de contágio de um talhão com um par de alcances: ``nucleo(v) = Σ_j w_ij v_j``."""

    vizinhanca: Vizinhanca
    alcance_na_rua_m: float
    alcance_entre_ruas_m: float

    def __call__(self, valor: np.ndarray) -> np.ndarray:
        return self.vizinhanca.aplicar(valor, self.alcance_na_rua_m, self.alcance_entre_ruas_m)


def nucleo_de_contagio(grafo: GrafoLavoura, p: ParametrosDoenca) -> Nucleo:
    """O núcleo de contato; com ``grafo.dispersao``, também a parte que vai pelo vento."""
    contato = Nucleo(grafo.vizinhanca, p.alcance_na_rua_m, p.alcance_entre_ruas_m)
    if grafo.dispersao is None or grafo.dispersao[0].fracao_aerea == 0.0:
        return contato
    from app.gemeo.dispersao import NucleoComVento

    parametros, azimute = grafo.dispersao
    return NucleoComVento(contato, grafo, parametros, azimute)


def pressao(nucleo: Nucleo, infecciosa: np.ndarray) -> np.ndarray:
    """``Σ_j w_ij I_j`` (:class:`~app.gemeo.grafo.Vizinhanca`, exata em O(N)).

    Sem o termo da própria planta: no estado da planta, a que esporula já pegou — a
    autoinfecção está no período infeccioso longo, não numa reinfecção.
    """
    return nucleo(infecciosa)


def passo(estado: Estado, dia: DiaClimatico, nucleo: Nucleo, p: ParametrosDoenca,
          protecao: np.ndarray | None = None) -> Estado:
    """Um dia. Conserva ``S + E + I + R = 1`` em cada planta."""
    fav = favorabilidade(dia, p)
    # O núcleo com vento (:mod:`app.gemeo.dispersao`) muda de dia para dia.
    do_dia = nucleo.para(dia) if hasattr(nucleo, "para") else nucleo
    forca = p.beta * fav * pressao(do_dia, estado.I) + p.fundo * fav
    novas = estado.S * (1.0 - np.exp(-forca))
    if protecao is not None:
        novas = novas * (1.0 - protecao)
    aparecem = estado.E * (1.0 - math.exp(-1.0 / latencia_dias(dia.t_media, p)))
    saem = estado.I * (1.0 - math.exp(-1.0 / p.infecciosidade_dias))
    return Estado(estado.S - novas, estado.E + novas - aparecem, estado.I + aparecem - saem,
                  estado.R + saem)


def aplicar_tratamentos(estado: Estado, tratamentos: Sequence[Tratamento], dia: int
                        ) -> tuple[Estado, np.ndarray | None]:
    """O dia ``dia`` da simulação sob os tratamentos: o curativo age no dia da aplicação
    (tira do latente), a proteção dura o residual. Devolve o estado e a proteção do dia."""
    protecao: np.ndarray | None = None
    for t in tratamentos:
        if t.dia == dia and t.curativo > 0:
            curado = estado.E * t.curativo * t.plantas
            estado = Estado(estado.S + curado, estado.E - curado, estado.I, estado.R)
        if t.dia <= dia < t.dia + t.residual_dias:
            atual = t.protecao * t.plantas
            protecao = atual if protecao is None else np.maximum(protecao, atual)
    return estado, protecao


@dataclass
class Trajetoria:
    estados: list[Estado]
    """Um por dia, incluindo o inicial (``len = dias + 1``)."""
    favorabilidade: list[float]

    def em(self, dia: int) -> Estado:
        return self.estados[dia]


def simular(estado0: Estado, dias: Sequence[DiaClimatico], grafo: GrafoLavoura,
            p: ParametrosDoenca, tratamentos: Sequence[Tratamento] = (),
            nucleo: Nucleo | None = None) -> Trajetoria:
    nucleo = nucleo_de_contagio(grafo, p) if nucleo is None else nucleo
    estado = estado0.copia()
    estados, favs = [estado.copia()], []
    for k, dia in enumerate(dias):
        estado, protecao = aplicar_tratamentos(estado, tratamentos, k)
        estado = passo(estado, dia, nucleo, p, protecao)
        estados.append(estado.copia())
        favs.append(favorabilidade(dia, p))
    return Trajetoria(estados, favs)


FONTE_ROVER = "rover"


@dataclass(frozen=True)
class Canal:
    """Como uma fonte vê a planta: P(sinal | esporula) = ``sensibilidade``, P(sinal | não) =
    ``alarme_falso``. A do Rover é ajustada ao talhão (é a de :class:`ParametrosDoenca`); as
    outras vêm de :data:`CANAIS` (calibração com plantas rotuladas, ou suposição declarada)."""

    sensibilidade: float
    alarme_falso: float
    origem: str = "ajustado ao talhão"

    def __post_init__(self) -> None:
        if not 0 <= self.alarme_falso < self.sensibilidade <= 1:
            raise ValueError("exige 0 <= alarme_falso < sensibilidade <= 1.")


CANAIS: dict[str, Canal] = {
    # O drone vê a copa de cima: a lesão de ferrugem começa no terço inferior e médio, que a
    # copa esconde, e a câmera do Drone Action não é de inspeção. **Suposição de projeto, sem
    # medição** — vale até a calibração com plantas rotuladas (a mesma do Rover, ``fit_fusion``).
    "drone": Canal(0.45, 0.06, "suposição de projeto (copa vista de cima), a medir"),
    # O agrônomo no pé: a validação de campo (``PlantLabel`` do SaaS). Quase certo, não certo.
    "agronomo": Canal(0.97, 0.01, "validação de campo: o rótulo do agrônomo"),
}


def canal_de(p: ParametrosDoenca, fonte: str = FONTE_ROVER,
             canais: dict[str, Canal] | None = None) -> Canal:
    """O canal de uma fonte: o do Rover sai dos parâmetros ajustados; o das outras, da tabela."""
    if fonte == FONTE_ROVER:
        return Canal(p.sensibilidade, p.alarme_falso)
    tabela = CANAIS if canais is None else canais
    if fonte not in tabela:
        raise KeyError(f"Fonte de observação sem canal: {fonte!r} (conhecidas: "
                       f"{', '.join([FONTE_ROVER, *sorted(tabela)])}).")
    return tabela[fonte]


def probabilidade_de_sinal(estado: Estado, p: ParametrosDoenca,
                           canal: Canal | None = None) -> np.ndarray:
    """O que a fonte veria: P(planta sinalizada) = a + (s - a) . I (o Rover, sem ``canal``)."""
    c = canal or canal_de(p)
    return c.alarme_falso + (c.sensibilidade - c.alarme_falso) * estado.I


RAZAO_MAXIMA = 1e6
"""Teto da razão de verossimilhança de uma planta numa passada (posterior 1 exata não existe:
o teto evita que um recorte saturado decida sozinho)."""


def razao_de_verossimilhanca(posterior: np.ndarray, prior: float) -> np.ndarray:
    """``odds(posterior) / odds(prior)``: o quanto a passada pesou para "doente".

    É a evidência que a fusão somou dos recortes (``posterior = sigmoid(logit(prior) + LLR)``,
    :meth:`app.plant_fusion.PlantFusionStore.pass_observations`), devolvida à escala de razão.
    ``NaN`` (planta não vista) vira 1: nenhuma evidência.
    """
    prior = min(max(float(prior), 1e-9), 1 - 1e-9)
    y = np.clip(np.nan_to_num(posterior, nan=prior), 1e-9, 1 - 1e-9)
    razao = (y / (1 - y)) / (prior / (1 - prior))
    return np.clip(razao, 1.0 / RAZAO_MAXIMA, RAZAO_MAXIMA)


def log_verossimilhanca(sinal: np.ndarray, razao: np.ndarray) -> np.ndarray:
    """``log P(evidência | modelo)`` por planta, a menos de uma constante que só depende da
    evidência: ``log(q . Λ + 1 - q)``, com ``q`` = P(sinal) previsto e ``Λ`` a razão.

    Zero quando a passada não diz nada (``Λ = 1``); positiva quando o modelo previa o que a
    fusão viu.
    """
    return np.log1p(np.clip(sinal, 0.0, 1.0) * (razao - 1.0))


def prevalencia_empirica(razao: np.ndarray, p: ParametrosDoenca, iteracoes: int = 200,
                         canal: Canal | None = None) -> float:
    """A fração de plantas esporulando que melhor explica uma passada (EM em uma variável).

    É o prior da primeira passada (Bayes empírico): sem ele, ou se tomava cada alarme falso
    como foco (os ~3% de alarme valem 50 plantas num talhão de 1.600 pés, contra uns poucos
    focos de verdade), ou se exigia evidência que uma passada não tem.

    **Piso de uma planta** (``1/n`` das vistas): com evidência fraca e poucas plantas, o EM vai
    a zero, e de zero o Bayes não sai — três sinais seguidos no mesmo pé deixavam P(I) em
    0,002 (talhão de 15 pés sem calibração da fusão, 16/10/2026). Num talhão de milhares de
    pés o piso não pesa.
    """
    if not len(razao):
        return 0.0
    c = canal or canal_de(p)
    li = c.sensibilidade * razao + (1 - c.sensibilidade)
    lo = c.alarme_falso * razao + (1 - c.alarme_falso)
    piso = 1.0 / len(razao)
    pi = 0.01
    for _ in range(iteracoes):
        novo = float(np.mean(pi * li / (pi * li + (1 - pi) * lo)))
        if abs(novo - pi) < 1e-9:
            pi = novo
            break
        pi = novo
    return max(pi, piso)


def estado_da_observacao(razao: np.ndarray, observada: np.ndarray, p: ParametrosDoenca,
                         base: Estado | None = None, canal: Canal | None = None) -> Estado:
    """Assimila a passada: Bayes exato por planta, com a razão da fusão como evidência virtual.

    Verossimilhança da evidência em cada estado: ``s . Λ + 1 - s`` em ``I`` e
    ``a . Λ + 1 - a`` nos outros três (que o Rover não distingue); a posterior é o estado
    previsto (``base``) vezes a verossimilhança, normalizado. Planta sem sinal perde ``I`` na
    medida da evidência, sem zerar; sinal isolado numa planta que o modelo dava como limpa pesa
    como o alarme falso que provavelmente é, e repetido, acumula.

    **Por que não inverter a posterior** (a primeira versão): com estado em área foliar,
    ``I = -ln(1 - y)/κ`` sobrescrito a cada passada zerava a planta doente que não foi
    sinalizada naquela passada, criava um foco em cada alarme falso e batia num teto por mais
    tomada que a planta estivesse; o ajuste compensava com ``β`` dez vezes menor e o alcance no
    limite (talhão sintético com sinal binário, 16/10/2026).

    **Primeira passada** (sem ``base``): o prior é a prevalência do talhão
    (:func:`prevalencia_empirica`) em ``I`` e ``latente_por_infeccioso`` vezes ela em ``E``.
    Plantas não vistas ficam com o estado de ``base`` (ou com esse prior).

    **Várias fontes** (``canal``): Rover, drone, agrônomo. Dada a planta, cada fonte vê
    independente das outras; a verossimilhança de cada passada usa o canal da fonte dela, e a
    fusão é o produto — a mesma conta, passada a passada. Uma fonte de canal pior pesa menos
    sozinha, sem regra à parte.
    """
    c = canal or canal_de(p)
    vista = observada.astype(bool)
    if base is None:
        n = len(razao)
        pi = prevalencia_empirica(razao[vista], p, canal=c)
        latente = min(pi * p.latente_por_infeccioso, 1.0 - pi)
        base = Estado(np.full(n, 1.0 - pi - latente), np.full(n, latente), np.full(n, pi),
                      np.zeros(n))
    li = c.sensibilidade * razao + (1 - c.sensibilidade)
    lo = c.alarme_falso * razao + (1 - c.alarme_falso)
    z = base.I * li + (1.0 - base.I) * lo
    fi = np.where(vista, li / z, 1.0)
    fo = np.where(vista, lo / z, 1.0)
    estado = Estado(base.S * fo, base.E * fo, base.I * fi, base.R * fo)
    estado.S = np.clip(1.0 - estado.E - estado.I - estado.R, 0.0, 1.0)
    return estado


PROBABILIDADES = frozenset({"alarme_falso", "sensibilidade"})
"""Parâmetros que são probabilidades: ajustados em logit; os outros (taxas, alcances), em log."""


def para_livre(nome: str, valor: float) -> float:
    """O parâmetro no espaço sem restrição do ajuste e da incerteza (log ou logit)."""
    if nome in PROBABILIDADES:
        v = min(max(valor, 1e-9), 1 - 1e-9)
        return math.log(v / (1 - v))
    return math.log(valor)


def de_livre(nome: str, x: float) -> float:
    if nome in PROBABILIDADES:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x)


@dataclass
class PassoDoFiltro:
    """O filtro numa passada: o estado depois de assimilá-la e a previsão para a seguinte."""

    k: int
    dia: int
    """Dia (desde ``grafo.inicio``) em que a passada ``k`` foi assimilada."""
    assimilado: Estado
    previsto_esporulando: np.ndarray | None
    """``P(I)`` de cada planta no dia em que a passada ``k+1`` a viu (``None`` na última)."""
    vista_proxima: np.ndarray | None


def dia_da_ultima_assimilacao(grafo: GrafoLavoura) -> int:
    """O dia em que :func:`filtrar` assimila a última passada, sem rodá-lo (o relógio dele).

    É a data base da previsão: a mediana da passada, ou o último dia dela quando a previsão da
    passada anterior já andou até lá.
    """
    dia: int | None = None
    for k in range(grafo.t):
        d0 = grafo.dia_da_passada(k)
        if d0 < 0:
            continue
        dia = d0 if dia is None else max(dia, d0)
        if k < grafo.t - 1:
            proxima = grafo.dia_obs[k + 1]
            if (proxima >= 0).any():
                dia = max(dia, int(proxima.max()))
    if dia is None:
        raise ValueError("Nenhuma passada com planta vista.")
    return dia


def filtrar(grafo: GrafoLavoura, razao: np.ndarray, clima: Sequence[DiaClimatico],
            p: ParametrosDoenca, nucleo: Nucleo | None = None,
            canais: dict[str, Canal] | None = None) -> Iterator[PassoDoFiltro]:
    """O filtro sobre as passadas — o mesmo no ajuste, na rede e na previsão.

    Assimila a passada ``k`` no dia típico dela (a mediana) e projeta dia a dia, colhendo o
    ``I`` de cada planta no dia em que a passada ``k+1`` a viu; o relógio do estado segue do
    último dia colhido (a passada leva dias: recomeçar da mediana simularia de novo os dias
    já andados). Nenhum estado diário inteiro fica guardado. Cada passada é assimilada com o
    canal da fonte dela (``grafo.fontes``).
    """
    nucleo = nucleo_de_contagio(grafo, p) if nucleo is None else nucleo
    estado: Estado | None = None
    dia = 0
    for k in range(grafo.t):
        d0 = grafo.dia_da_passada(k)
        if d0 < 0:
            continue
        if estado is None:
            dia = d0
        while dia < d0:
            estado = passo(estado, clima[dia], nucleo, p)
            dia += 1
        estado = estado_da_observacao(razao[k], grafo.dia_obs[k] >= 0, p, estado,
                                      canal_de(p, grafo.fonte(k), canais))
        assimilado, dia_assimilado = estado, dia
        if k == grafo.t - 1:
            yield PassoDoFiltro(k, dia_assimilado, assimilado, None, None)
            return
        proxima = grafo.dia_obs[k + 1]
        vista = proxima >= 0
        if not vista.any():
            yield PassoDoFiltro(k, dia_assimilado, assimilado, None, vista)
            continue
        d1 = max(int(proxima[vista].max()), dia)
        alvo = np.where(vista, np.clip(proxima, dia, d1), -1)
        infecciosa = np.zeros(grafo.n)
        for d in range(dia, d1 + 1):
            agora = alvo == d
            infecciosa[agora] = estado.I[agora]
            if d < d1:
                estado = passo(estado, clima[d], nucleo, p)
        dia = d1
        yield PassoDoFiltro(k, dia_assimilado, assimilado, infecciosa, vista)


@dataclass
class ModeloAjustado:
    """Os parâmetros de uma doença num talhão, com a incerteza do ajuste."""

    parametros: ParametrosDoenca
    nomes_livres: tuple[str, ...] = ("beta", "alcance_na_rua_m", "alcance_entre_ruas_m",
                                     "fundo", "alarme_falso", "sensibilidade")
    media_livre: np.ndarray | None = None
    covariancia_livre: np.ndarray | None = None
    """Aproximação de Laplace no espaço livre (:func:`para_livre`): a incerteza que o conjunto
    da previsão propaga."""
    diagnostico: dict = field(default_factory=dict)

    def amostras(self, n: int, rng: np.random.Generator) -> list[ParametrosDoenca]:
        if self.media_livre is None or self.covariancia_livre is None or n <= 1:
            return [self.parametros] * max(1, n)
        saida: list[ParametrosDoenca] = []
        while len(saida) < n:
            linha = rng.multivariate_normal(self.media_livre, self.covariancia_livre)
            try:
                saida.append(replace(self.parametros, **{
                    k: de_livre(k, float(x)) for k, x in zip(self.nomes_livres, linha,
                                                             strict=True)}))
            except ValueError:  # sorteio com alarme falso >= sensibilidade: outro
                continue
        return saida

    def to_dict(self) -> dict:
        return {"parametros": self.parametros.to_dict(), "nomes_livres": list(self.nomes_livres),
                "media_livre": None if self.media_livre is None else self.media_livre.tolist(),
                "covariancia_livre": None if self.covariancia_livre is None
                else self.covariancia_livre.tolist(),
                "diagnostico": self.diagnostico}

    @classmethod
    def from_dict(cls, data: dict) -> ModeloAjustado:
        return cls(ParametrosDoenca.from_dict(data["parametros"]),
                   tuple(data["nomes_livres"]),
                   None if data.get("media_livre") is None else np.array(data["media_livre"]),
                   None if data.get("covariancia_livre") is None
                   else np.array(data["covariancia_livre"]),
                   dict(data.get("diagnostico") or {}))
