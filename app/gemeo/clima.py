"""O tempo do talhão, dia a dia: o que dirige o modelo epidemiológico.

Três fontes, nesta ordem de preferência para cada dia:

1. **estação da fazenda** (``POST /fields/{id}/clima``, registros horários): temperatura,
   umidade relativa, chuva e, se o sensor existir, molhamento foliar medido;
2. **Open-Meteo** (com ``CLIMA_OPEN_METEO=true``): histórico (reanálise ERA5) e previsão horária
   até 16 dias, pela posição do talhão, sem chave;
3. **climatologia**, além do horizonte da previsão: a média do mesmo período do ano na série
   que já existe (ou dos últimos 30 dias, sem ano anterior). Cada dia diz de onde veio
   (``fonte``) — uma previsão de 60 dias é, depois do 16º, uma estimativa de clima, e o
   relatório tem de dizer isso.

**Vento** (opcional: estação com anemômetro, ou o Open-Meteo a 10 m). Por dia, o vetor
médio **para onde** o ar vai (``vento_u_ms`` para leste, ``vento_v_ms`` para norte) e a
velocidade média escalar (``vento_ms``, que mede a turbulência). É o que transporta os esporos
na dispersão (:mod:`app.gemeo.dispersao`); dia sem vento medido fica sem ele, e o modelo de
contato continua sozinho.

**Molhamento foliar.** O fungo precisa de água livre na folha. Sem sensor, as horas de folha
molhada são estimadas pelo método mais simples e mais usado: hora com umidade relativa ≥ 90%
ou com chuva. Superestima em noite de orvalho curto e subestima em folha que seca devagar; a
série medida, quando existe, sempre vence (``molhamento_medido``).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

UR_MOLHADA = 90.0
"""Umidade relativa (%) a partir da qual a hora conta como de folha molhada (sem sensor)."""

CHUVA_MOLHA_MM = 0.1

FONTES = ("estacao", "open-meteo:historico", "open-meteo:previsao", "climatologia")


@dataclass(frozen=True)
class DiaClimatico:
    data: date
    t_media: float
    t_min: float
    t_max: float
    ur_media: float
    chuva_mm: float
    molhamento_h: float
    molhamento_medido: bool = False
    fonte: str = "estacao"
    vento_ms: float | None = None
    """Velocidade média do vento no dia (m/s)."""
    vento_u_ms: float | None = None
    """Componente média para leste do vetor do vento (m/s; para onde o ar vai)."""
    vento_v_ms: float | None = None
    """Componente média para norte."""

    @property
    def tem_vento(self) -> bool:
        return None not in (self.vento_ms, self.vento_u_ms, self.vento_v_ms)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data"] = self.data.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DiaClimatico:
        valores = dict(data)
        valores["data"] = date.fromisoformat(str(valores["data"]))
        return cls(**valores)


def _hora(registro: Mapping[str, Any]) -> datetime:
    texto = str(registro["time"]).replace("Z", "+00:00")
    return datetime.fromisoformat(texto)


def dias_de_registros_horarios(registros: Iterable[Mapping[str, Any]],
                               fonte: str = "estacao") -> list[DiaClimatico]:
    """Registros horários → dias. Cada registro: ``time``, ``temperatura_c``, ``ur``,
    ``chuva_mm`` e, opcionais, ``molhamento`` (fração da hora com folha molhada, 0 a 1),
    ``vento_ms`` e ``vento_dir_graus`` (de onde o vento **vem**, convenção meteorológica).

    Dia com menos de 18 horas registradas fica de fora: média de meio dia engana o modelo.
    """
    por_dia: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
    for r in registros:
        por_dia[_hora(r).date()].append(r)
    dias = []
    for d in sorted(por_dia):
        horas = por_dia[d]
        if len(horas) < 18:
            continue
        temps = [float(h["temperatura_c"]) for h in horas]
        urs = [float(h.get("ur", 0.0)) for h in horas]
        chuvas = [float(h.get("chuva_mm", 0.0) or 0.0) for h in horas]
        medido = all(h.get("molhamento") is not None for h in horas)
        if medido:
            molhado = sum(min(max(float(h["molhamento"]), 0.0), 1.0) for h in horas)
        else:
            molhado = sum(1.0 for u, c in zip(urs, chuvas, strict=True)
                          if u >= UR_MOLHADA or c >= CHUVA_MOLHA_MM)
        escala = 24.0 / len(horas)
        vento = _vento_do_dia(horas)
        dias.append(DiaClimatico(d, sum(temps) / len(temps), min(temps), max(temps),
                                 sum(urs) / len(urs), sum(chuvas) * escala,
                                 min(24.0, molhado * escala), medido, fonte, *vento))
    return dias


def _vento_do_dia(horas: Sequence[Mapping[str, Any]]
                  ) -> tuple[float | None, float | None, float | None]:
    """Velocidade média e vetor médio para onde o ar vai; ``None`` se faltar meio dia."""
    pares = [(float(h["vento_ms"]), float(h["vento_dir_graus"])) for h in horas
             if h.get("vento_ms") is not None and h.get("vento_dir_graus") is not None]
    if len(pares) < len(horas) / 2:
        return None, None, None
    # A direção é de onde vem: o ar vai no sentido oposto.
    u = [-v * math.sin(math.radians(d)) for v, d in pares]
    w = [-v * math.cos(math.radians(d)) for v, d in pares]
    return (round(sum(v for v, _ in pares) / len(pares), 3), round(sum(u) / len(u), 3),
            round(sum(w) / len(w), 3))


def _distancia_no_ano(a: date, b: date) -> int:
    """Dias entre as duas datas no calendário anual, dando a volta no ano (29/02 incluso)."""
    d = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(d, 365 - d)


@dataclass
class SerieClimatica:
    lat: float | None
    lon: float | None
    dias: dict[date, DiaClimatico]

    @classmethod
    def vazia(cls, lat: float | None = None, lon: float | None = None) -> SerieClimatica:
        return cls(lat, lon, {})

    def junta(self, novos: Iterable[DiaClimatico]) -> SerieClimatica:
        """Acrescenta dias; a estação vence o Open-Meteo, e o observado vence a previsão."""
        prioridade = {f: i for i, f in enumerate(FONTES)}
        for d in novos:
            atual = self.dias.get(d.data)
            if atual is None or prioridade[d.fonte] <= prioridade[atual.fonte]:
                self.dias[d.data] = d
        return self

    def periodo(self, inicio: date, fim: date) -> list[DiaClimatico]:
        """Um dia por data, de ``inicio`` a ``fim`` inclusive; falta vira climatologia."""
        saida, d = [], inicio
        while d <= fim:
            saida.append(self.dias.get(d) or self.climatologia(d))
            d += timedelta(days=1)
        return saida

    def climatologia(self, d: date) -> DiaClimatico:
        """Média do mesmo período do ano (±7 dias, outros anos); sem isso, últimos 30 dias."""
        reais = [x for x in self.dias.values() if x.fonte != "climatologia"]
        if not reais:
            raise ValueError(f"Sem clima para {d.isoformat()}: mande a série da estação "
                             "ou ligue CLIMA_OPEN_METEO.")
        mesmo_periodo = [x for x in reais if x.data.year != d.year
                         and _distancia_no_ano(x.data, d) <= 7]
        base = mesmo_periodo or sorted(reais, key=lambda x: x.data)[-30:]

        def media(attr: str) -> float:
            return sum(getattr(x, attr) for x in base) / len(base)

        com_vento = [x for x in base if x.tem_vento]

        def media_vento(attr: str) -> float | None:
            if len(com_vento) < len(base) / 2:
                return None
            return sum(getattr(x, attr) for x in com_vento) / len(com_vento)

        return DiaClimatico(d, media("t_media"), media("t_min"), media("t_max"),
                            media("ur_media"), media("chuva_mm"), media("molhamento_h"),
                            False, "climatologia", media_vento("vento_ms"),
                            media_vento("vento_u_ms"), media_vento("vento_v_ms"))

    def to_dict(self) -> dict[str, Any]:
        return {"lat": self.lat, "lon": self.lon,
                "dias": [self.dias[d].to_dict() for d in sorted(self.dias)]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SerieClimatica:
        dias = [DiaClimatico.from_dict(x) for x in data.get("dias", [])]
        return cls(data.get("lat"), data.get("lon"), {d.data: d for d in dias})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporario = path.with_suffix(".tmp")
        temporario.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        temporario.replace(path)

    @classmethod
    def load(cls, path: Path) -> SerieClimatica:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------------------------

PREVISAO_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICO_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIAVEIS = ("temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,"
             "wind_direction_10m")

Buscador = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def _buscar_http(url: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
    import httpx

    resposta = httpx.get(url, params=dict(params), timeout=20.0)
    resposta.raise_for_status()
    return resposta.json()


def _horarios(corpo: Mapping[str, Any]) -> list[dict[str, Any]]:
    h = corpo["hourly"]
    n = len(h["time"])
    vel = h.get("wind_speed_10m") or [None] * n
    dire = h.get("wind_direction_10m") or [None] * n
    return [{"time": t, "temperatura_c": temp, "ur": ur, "chuva_mm": chuva,
             "vento_ms": v, "vento_dir_graus": dd}
            for t, temp, ur, chuva, v, dd in zip(h["time"], h["temperature_2m"],
                                                 h["relative_humidity_2m"], h["precipitation"],
                                                 vel, dire, strict=True)
            if temp is not None and ur is not None]


def open_meteo(lat: float, lon: float, *, inicio: date, hoje: date, dias_previsao: int = 16,
               buscar: Buscador | None = None) -> list[DiaClimatico]:
    """Histórico de ``inicio`` até ontem e a previsão de hoje a ``hoje + dias_previsao``."""
    buscar = buscar or _buscar_http
    dias: list[DiaClimatico] = []
    ontem = hoje - timedelta(days=1)
    if inicio <= ontem:
        corpo = buscar(HISTORICO_URL, {"latitude": lat, "longitude": lon,
                                       "start_date": inicio.isoformat(),
                                       "end_date": ontem.isoformat(), "hourly": VARIAVEIS,
                                       "wind_speed_unit": "ms", "timezone": "auto"})
        dias += dias_de_registros_horarios(_horarios(corpo), "open-meteo:historico")
    corpo = buscar(PREVISAO_URL, {"latitude": lat, "longitude": lon, "hourly": VARIAVEIS,
                                  "past_days": 2, "forecast_days": min(16, dias_previsao),
                                  "wind_speed_unit": "ms", "timezone": "auto"})
    previsao = dias_de_registros_horarios(_horarios(corpo), "open-meteo:previsao")
    vistos = {d.data for d in dias}
    return dias + [d for d in previsao if d.data not in vistos]


def centro(lats: Sequence[float | None], lons: Sequence[float | None]
           ) -> tuple[float | None, float | None]:
    pares = [(a, b) for a, b in zip(lats, lons, strict=True) if a is not None and b is not None]
    if not pares:
        return None, None
    return (sum(a for a, _ in pares) / len(pares), sum(b for _, b in pares) / len(pares))
