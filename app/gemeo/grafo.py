"""O grafo espaço-temporal da lavoura: plantas, vizinhança e o que o Rover viu em cada passada.

* **Nós** são as plantas da fusão (:meth:`app.plant_fusion.PlantFusionStore.pass_observations`),
  com a posição local: ``x`` ao longo da rua (o metro da navegação) e ``y`` atravessando as ruas
  (ordem natural da rua x espaçamento entre ruas).
* **Vizinhança** (:class:`Vizinhanca`): o contágio anda pelo núcleo separável
  ``exp(-|Δx|/a_rua) . exp(-|Δr| . esp/a_entre)`` até :data:`CORTE_RUAS` ruas de cada lado
  (:mod:`app.gemeo.epidemia`). A soma ``Σ_j w_ij v_j`` sai **exata em O(N)**: o núcleo
  exponencial ao longo da rua é uma soma acumulada de ida e outra de volta, em escala
  logarítmica. A primeira versão guardava as arestas até 12 m: num talhão de 40.000 pés eram
  6,7 milhões, ~215 MB, e 5,4 s por avaliação do ajuste — inviável (medido em 16/10/2026).
  Arestas, só onde fazem sentido: nos focos, entre as plantas marcadas (:func:`componentes`).
* **Tempo**: uma camada por passada fechada, com a posterior de cada doença por planta
  (``NaN`` = não vista nesta passada) e o **dia** em que o Rover passou por ela.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np

from app.plant_fusion import _row_sort_key, parse_timestamp

CORTE_RUAS = 2
"""Quantas ruas para cada lado entram como vizinhança (a 3,5 m de espaçamento, a segunda rua
pesa exp(-7/a_entre): 1,6·10⁻⁴ com o alcance do prior)."""
ALCANCE_MAXIMO_M = 3.0
"""O maior alcance de contágio que o ajuste aceita, nos dois eixos."""


@dataclass
class Vizinhanca:
    """``Σ_j w_ij v_j`` sem a própria planta, com ``w_ij = exp(-|x_i - x_j|/a_rua) .
    exp(-|r_i - r_j| . esp/a_entre)`` e ``|r_i - r_j| <= CORTE_RUAS`` — exata, em O(N).

    Para a rua de origem ``s`` e a planta ``i``:
    ``Σ_j v_j e^(-|x_i - x_j|/a) = e^(-x_i/a) Σ_{x_j <= x_i} v_j e^(x_j/a)
    + e^(x_i/a) Σ_{x_j > x_i} v_j e^(-x_j/a)``. As duas somas são acumuladas em log
    (``logaddexp.accumulate``: ``x/a`` passa de 2.000 numa rua de 500 m, e ``e^x`` estouraria)
    numa matriz rua x posição; os índices de cada planta nas ruas vizinhas são calculados uma
    vez por talhão.
    """

    ruas: int
    largura: int
    posicao: np.ndarray
    """``(N,)``: índice achatado da planta na matriz ``ruas x largura``."""
    x_matriz: np.ndarray
    """``(ruas, largura)``: posição ao longo da rua (0 no enchimento)."""
    x: np.ndarray
    ida: dict[int, np.ndarray]
    """Por deslocamento de rua: o índice achatado da última vizinha com ``x_j <= x_i`` (-1)."""
    volta: dict[int, np.ndarray]
    """Por deslocamento de rua: o da primeira com ``x_j > x_i`` (-1)."""
    espacamento_entre_ruas_m: float

    @classmethod
    def construir(cls, rua: np.ndarray, x: np.ndarray, espacamento_entre_ruas_m: float,
                  corte_ruas: int = CORTE_RUAS) -> Vizinhanca:
        n = len(rua)
        ruas = int(rua.max()) + 1 if n else 0
        ordem = {r: np.flatnonzero(rua == r) for r in range(ruas)}
        ordem = {r: idx[np.argsort(x[idx], kind="stable")] for r, idx in ordem.items()}
        largura = max((len(v) for v in ordem.values()), default=0)
        posicao = np.zeros(n, dtype=np.int64)
        x_matriz = np.zeros((ruas, max(largura, 1)))
        xs = {}
        for r, idx in ordem.items():
            posicao[idx] = r * largura + np.arange(len(idx))
            x_matriz[r, :len(idx)] = x[idx]
            xs[r] = x[idx]
        ida = {d: np.full(n, -1, dtype=np.int64) for d in range(-corte_ruas, corte_ruas + 1)}
        volta = {d: np.full(n, -1, dtype=np.int64) for d in ida}
        for r, alvos in ordem.items():
            for d in ida:
                fonte = r + d
                if fonte not in xs or not len(xs[fonte]):
                    continue
                k = np.searchsorted(xs[fonte], x[alvos], side="right")
                ida[d][alvos] = np.where(k > 0, fonte * largura + k - 1, -1)
                volta[d][alvos] = np.where(k < len(xs[fonte]), fonte * largura + k, -1)
        return cls(ruas, largura, posicao, x_matriz, x.astype(float), ida, volta,
                   float(espacamento_entre_ruas_m))

    def aplicar(self, valor: np.ndarray, alcance_na_rua_m: float,
                alcance_entre_ruas_m: float) -> np.ndarray:
        n = len(valor)
        if not n:
            return np.zeros(0)
        log_v = np.full(self.ruas * self.largura, -np.inf)
        with np.errstate(divide="ignore"):
            log_v[self.posicao] = np.log(np.maximum(valor, 0.0))
        log_v = log_v.reshape(self.ruas, self.largura)
        xa = self.x_matriz[:, :self.largura] / alcance_na_rua_m
        ida = np.logaddexp.accumulate(log_v + xa, axis=1).ravel()
        volta = np.logaddexp.accumulate((log_v - xa)[:, ::-1], axis=1)[:, ::-1].ravel()
        xi = self.x / alcance_na_rua_m
        total = np.zeros(n)
        for d, idx in self.ida.items():
            peso = math.exp(-abs(d) * self.espacamento_entre_ruas_m / alcance_entre_ruas_m)
            ok = idx >= 0
            parte = np.zeros(n)
            parte[ok] = np.exp(ida[idx[ok]] - xi[ok])
            ok = self.volta[d] >= 0
            parte[ok] += np.exp(volta[self.volta[d][ok]] + xi[ok])
            total += peso * parte
        # A ida da própria rua inclui a própria planta (peso 1): fora.
        return np.maximum(total - valor, 0.0)


@dataclass
class GrafoLavoura:
    field_id: str
    crop: str
    doencas: list[str]
    plant_ids: list[str]
    ruas: list[str]
    """Rótulo de cada rua, na ordem natural (``G2`` antes de ``G10``)."""
    rua: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    vizinhanca: Vizinhanca
    passadas: list[str]
    inicio: date
    """Dia 0 da simulação: o dia da primeira passada."""
    dia_obs: np.ndarray
    """``(T, N)``: dia (desde ``inicio``) em que a planta foi vista na passada; -1 = não vista."""
    posterior: dict[str, np.ndarray] = field(default_factory=dict)
    """Por doença, ``(T, N)``, ``NaN`` onde não vista."""
    n_recortes: np.ndarray | None = None
    prior_fusao: dict[str, float] = field(default_factory=dict)
    """O prior de cada doença com que a fusão fez a posterior (para desfazê-lo)."""
    espacamento_entre_ruas_m: float = 3.5
    espacamento_plantas_m: float = 0.7
    fontes: list[str] = field(default_factory=list)
    """A fonte de cada passada (``rover``, ``drone``, ``agronomo``); vazia = todas do Rover."""
    dispersao: Any = None
    """``(ParametrosDispersao, azimute da rua em graus)`` para o contágio pelo vento
    (:mod:`app.gemeo.dispersao`); ``None`` = só o núcleo de contato (o padrão)."""

    def fonte(self, k: int) -> str:
        return self.fontes[k] if k < len(self.fontes) else "rover"

    @property
    def n(self) -> int:
        return len(self.plant_ids)

    @property
    def t(self) -> int:
        return len(self.passadas)

    def dia_da_passada(self, k: int) -> int:
        """O dia típico da passada (mediana das plantas vistas nela)."""
        dias = self.dia_obs[k][self.dia_obs[k] >= 0]
        return int(np.median(dias)) if len(dias) else -1

    def razao(self, doenca: str) -> np.ndarray:
        """``(T, N)``: a razão de verossimilhança de cada passada em cada planta (1 = não vista).

        Sem o prior da fusão (observações de fora dela), a posterior é tomada como feita sobre
        prior 0,5 — a razão é então a própria odds.
        """
        from app.gemeo.epidemia import razao_de_verossimilhanca

        return razao_de_verossimilhanca(self.posterior[doenca], self.prior_fusao.get(doenca, 0.5))

    def resumo(self) -> dict[str, Any]:
        return {"field_id": self.field_id, "crop": self.crop, "plantas": self.n,
                "ruas": len(self.ruas),
                "passadas": [{"pass_id": p, "dia": self.dia_da_passada(k),
                              "fonte": self.fonte(k),
                              "plantas_vistas": int((self.dia_obs[k] >= 0).sum())}
                             for k, p in enumerate(self.passadas)],
                "inicio": self.inicio.isoformat(), "doencas": self.doencas,
                "espacamento_entre_ruas_m": self.espacamento_entre_ruas_m,
                "espacamento_plantas_m": self.espacamento_plantas_m,
                "nucleo": f"exp(-dx/a_rua)·exp(-dr·esp/a_entre), ±{CORTE_RUAS} ruas, sem corte "
                          "na rua"}


def arestas(rua: np.ndarray, x: np.ndarray, espacamento_entre_ruas_m: float,
            corte_rua_m: float, corte_ruas: int = CORTE_RUAS
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """As arestas dirigidas (as duas direções), sem laço: ``origem, destino, d_rua, d_entre``.

    Por par de ruas, uma busca binária na rua vizinha ordenada: O(N · vizinhas), sem matriz.
    """
    origens, destinos, longos, atraves = [], [], [], []
    por_rua = {r: np.flatnonzero(rua == r) for r in np.unique(rua)}
    ordem = {r: idx[np.argsort(x[idx], kind="stable")] for r, idx in por_rua.items()}
    for r, idx in por_rua.items():
        for dr in range(-corte_ruas, corte_ruas + 1):
            alvo = ordem.get(r + dr)
            if alvo is None:
                continue
            xs = x[alvo]
            lo = np.searchsorted(xs, x[idx] - corte_rua_m, side="left")
            hi = np.searchsorted(xs, x[idx] + corte_rua_m, side="right")
            contagem = hi - lo
            if not contagem.sum():
                continue
            fonte = np.repeat(idx, contagem)
            base = np.repeat(lo - np.cumsum(contagem) + contagem, contagem)
            posicao = np.arange(contagem.sum()) + base
            vizinha = alvo[posicao]
            manter = vizinha != fonte
            origens.append(vizinha[manter])
            destinos.append(fonte[manter])
            longos.append(np.abs(x[vizinha[manter]] - x[fonte[manter]]))
            atraves.append(np.full(int(manter.sum()), abs(dr) * espacamento_entre_ruas_m))
    if not origens:
        vazio = np.zeros(0)
        return vazio.astype(np.int64), vazio.astype(np.int64), vazio, vazio
    return (np.concatenate(origens).astype(np.int64), np.concatenate(destinos).astype(np.int64),
            np.concatenate(longos), np.concatenate(atraves))


def juntar_externas(observacoes: dict[str, Any], externas: list[dict[str, Any]]
                    ) -> tuple[dict[str, Any], dict[str, int]]:
    """As passadas da fusão do Rover mais as de outras fontes, em ordem de tempo.

    Cada externa: ``{"fonte", "pass_id", "time", "doenca": {plant_id: llr}}`` — a LLR (log da
    razão de verossimilhança) que a outra fonte mediu na planta, por doença. Vira posterior com
    o prior da cultura (o mesmo da fusão), e o grafo devolve a razão (:meth:`GrafoLavoura.razao`):
    a evidência chega intacta. Planta que a fusão não conhece fica de fora (contada).
    """
    priors = observacoes.get("priors") or {}
    por_id = {p["plant_id"]: p for p in observacoes["plants"]}
    plantas = {pid: {**p, "observations": dict(p["observations"])} for pid, p in por_id.items()}
    passes = [dict(p) for p in observacoes["passes"]]
    desconhecidas = 0
    for ext in externas:
        pid_passada = f"{ext['fonte']}:{ext['pass_id']}"
        passes.append({"pass_id": pid_passada, "fonte": ext["fonte"],
                       "first_time": ext["time"], "last_time": ext["time"]})
        por_planta: dict[str, dict[str, float]] = {}
        for doenca, llrs in ext["evidencia"].items():
            prior = min(max(float(priors.get(doenca, 0.5)), 1e-9), 1 - 1e-9)
            base = math.log(prior / (1 - prior))
            for pid, llr in llrs.items():
                if pid not in plantas:
                    desconhecidas += 1
                    continue
                por_planta.setdefault(pid, {})[doenca] = 1.0 / (1.0 + math.exp(-(base + llr)))
        for pid, post in por_planta.items():
            plantas[pid]["observations"][pid_passada] = {"time": ext["time"], "n_tiles": 1,
                                                         "posterior": post}
    def instante(passada: dict[str, Any]) -> datetime:
        t = passada.get("first_time")
        quando = parse_timestamp(t) if t else None
        return quando or datetime.min.replace(tzinfo=UTC)

    passes.sort(key=instante)
    return ({**observacoes, "passes": passes, "plants": list(plantas.values())},
            {"passadas_externas": len(externas), "plantas_desconhecidas": desconhecidas})


def construir(observacoes: dict[str, Any], *, corte_ruas: int = CORTE_RUAS) -> GrafoLavoura:
    """O grafo a partir de ``PlantFusionStore.pass_observations(field_id)``."""
    plantas = observacoes["plants"]
    passadas = [p["pass_id"] for p in observacoes["passes"]]
    if not plantas:
        raise ValueError("Talhão sem plantas: o gêmeo precisa de ao menos uma passada.")
    if not passadas:
        raise ValueError("Talhão sem passada fechada: feche a passada da missão antes.")
    ruas = sorted({p["row_id"] for p in plantas}, key=_row_sort_key)
    indice = {r: i for i, r in enumerate(ruas)}
    rua = np.array([indice[p["row_id"]] for p in plantas], dtype=np.int64)
    x = np.array([float(p["x_along_m"]) for p in plantas])
    entre = float(observacoes.get("row_spacing_m") or 3.5)
    vizinhanca = Vizinhanca.construir(rua, x, entre, corte_ruas)

    tempos: list[list[datetime | None]] = []
    for pid in passadas:
        tempos.append([parse_timestamp(p["observations"][pid]["time"])
                       if pid in p["observations"] and p["observations"][pid]["time"]
                       else None for p in plantas])
    todas = [t for linha in tempos for t in linha if t is not None]
    if not todas:
        raise ValueError("As passadas não têm horário: o gêmeo precisa do tempo de cada uma.")
    inicio = min(todas).date()
    dia_obs = np.array([[-1 if t is None else (t.date() - inicio).days for t in linha]
                        for linha in tempos], dtype=np.int64)
    doencas = [d for d in observacoes["diseases"] if d != "Healthy"]
    posterior = {}
    for doenca in doencas:
        m = np.full((len(passadas), len(plantas)), np.nan)
        for k, pid in enumerate(passadas):
            for i, p in enumerate(plantas):
                obs = p["observations"].get(pid)
                if obs is not None:
                    m[k, i] = float(obs["posterior"].get(doenca, np.nan))
        posterior[doenca] = m
    recortes = np.array([[float(p["observations"][pid]["n_tiles"])
                          if pid in p["observations"] else 0.0 for p in plantas]
                         for pid in passadas])
    lat = np.array([np.nan if p.get("lat") is None else float(p["lat"]) for p in plantas])
    lon = np.array([np.nan if p.get("lon") is None else float(p["lon"]) for p in plantas])
    return GrafoLavoura(
        field_id=observacoes["field_id"], crop=observacoes["crop"], doencas=doencas,
        plant_ids=[p["plant_id"] for p in plantas], ruas=ruas, rua=rua, x_m=x,
        y_m=rua.astype(float) * entre, lat=lat, lon=lon, vizinhanca=vizinhanca,
        passadas=passadas, inicio=inicio, dia_obs=dia_obs, posterior=posterior,
        n_recortes=recortes, espacamento_entre_ruas_m=entre,
        prior_fusao={d: float(v) for d, v in (observacoes.get("priors") or {}).items()
                     if d in doencas},
        fontes=[str(p.get("fonte") or "rover") for p in observacoes["passes"]],
        espacamento_plantas_m=float(observacoes.get("plant_spacing_m") or 0.7))


def componentes(grafo: GrafoLavoura, marcadas: np.ndarray, distancia_max_m: float
                ) -> list[np.ndarray]:
    """Focos: grupos de plantas marcadas ligadas por arestas mais curtas que ``distancia_max_m``.

    União-busca sobre as arestas **entre as marcadas** (montadas aqui, com o corte do foco): um
    foco é a reboleira que o agrônomo vê no talhão.
    """
    marcadas_idx = np.flatnonzero(marcadas)
    if not len(marcadas_idx):
        return []
    corte_ruas = max(0, int(distancia_max_m // grafo.espacamento_entre_ruas_m))
    o, d, dl, de = arestas(grafo.rua[marcadas_idx], grafo.x_m[marcadas_idx],
                           grafo.espacamento_entre_ruas_m, distancia_max_m, corte_ruas)
    origem, destino = marcadas_idx[o], marcadas_idx[d]
    pai = np.arange(grafo.n)

    def raiz(i: int) -> int:
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = int(pai[i])
        return i

    perto = np.hypot(dl, de) <= distancia_max_m
    for a, b in zip(origem[perto], destino[perto], strict=True):
        ra, rb = raiz(int(a)), raiz(int(b))
        if ra != rb:
            pai[rb] = ra
    grupos: dict[int, list[int]] = {}
    for i in marcadas_idx:
        grupos.setdefault(raiz(int(i)), []).append(int(i))
    return sorted((np.array(g) for g in grupos.values()), key=len, reverse=True)
