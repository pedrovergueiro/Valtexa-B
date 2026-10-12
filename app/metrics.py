"""Métricas da IA no formato do Prometheus (brief, seções 34 e 35).

Duas famílias, no mesmo ``GET /metrics`` (protegido pelo ``SERVICE_TOKEN``, como
toda rota da IA — o Prometheus manda o mesmo ``Authorization: Bearer``):

* **HTTP** — requisições, duração e erros 5xx por **molde** de rota
  (``/fields/{field_id}/incidence``, nunca o caminho com o id). Medido por um
  *middleware* ASGI, então vale também para as rotas de terreno e de talhão sem
  tocar nelas;
* **inferência** — por modelo: imagens classificadas, chamadas, erros por tipo,
  latência por imagem e a **distribuição das classes previstas**. Medido em
  :meth:`app.model.CornDiseaseModel.predict_images`, a porta por onde passam a
  previsão avulsa, o lote, o rover e a fusão por planta.

O formato segue o mesmo contrato do ``saas/valtexa/observability/metricas.py``
(texto 0.0.4). É uma cópia pequena de propósito: SaaS e IA são serviços separados
(brief, seção 4) e nenhum importa o outro.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Awaitable, Callable, Iterable, MutableMapping, Sequence
from time import perf_counter
from typing import Any

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
LIMITES_HTTP_S = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
LIMITES_INFERENCIA_S = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
SEM_ROTA = "nao_mapeada"


def _escapa(valor: str) -> str:
    return str(valor).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _numero(valor: float) -> str:
    if math.isinf(valor):
        return "+Inf" if valor > 0 else "-Inf"
    if float(valor).is_integer() and abs(valor) < 1e15:
        return str(int(valor))
    return repr(float(valor))


def _rotulos(nomes: Sequence[str], valores: Sequence[str], extra: str = "") -> str:
    pares = [f'{n}="{_escapa(v)}"' for n, v in zip(nomes, valores, strict=True)]
    if extra:
        pares.append(extra)
    return "{" + ",".join(pares) + "}" if pares else ""


class _Serie:
    tipo = "untyped"

    def __init__(self, nome: str, ajuda: str, rotulos: Sequence[str]) -> None:
        self.nome, self.ajuda, self.rotulos = nome, ajuda, tuple(rotulos)
        self._trava = threading.Lock()

    def _chave(self, valores: dict[str, str]) -> tuple[str, ...]:
        if set(valores) != set(self.rotulos):
            raise ValueError(f"{self.nome} espera {self.rotulos}, veio {sorted(valores)}")
        return tuple(str(valores[r]) for r in self.rotulos)

    def _cabecalho(self) -> list[str]:
        return [f"# HELP {self.nome} {_escapa(self.ajuda)}", f"# TYPE {self.nome} {self.tipo}"]


class Contador(_Serie):
    tipo = "counter"

    def __init__(self, nome: str, ajuda: str, rotulos: Sequence[str] = ()) -> None:
        super().__init__(nome, ajuda, rotulos)
        self._valores: dict[tuple[str, ...], float] = {}

    def inc(self, valor: float = 1.0, **rotulos: str) -> None:
        chave = self._chave(rotulos)
        with self._trava:
            self._valores[chave] = self._valores.get(chave, 0.0) + valor

    def valor(self, **rotulos: str) -> float:
        return self._valores.get(self._chave(rotulos), 0.0)

    def render(self) -> str:
        with self._trava:
            itens = sorted(self._valores.items())
        return "\n".join(self._cabecalho() + [
            f"{self.nome}{_rotulos(self.rotulos, k)} {_numero(v)}" for k, v in itens])


class Histograma(_Serie):
    tipo = "histogram"

    def __init__(self, nome: str, ajuda: str, rotulos: Sequence[str],
                 limites: Sequence[float]) -> None:
        super().__init__(nome, ajuda, rotulos)
        self.limites = tuple(float(x) for x in limites)
        self._dados: dict[tuple[str, ...], list[float]] = {}

    def observe(self, valor: float, **rotulos: str) -> None:
        chave = self._chave(rotulos)
        with self._trava:
            d = self._dados.setdefault(chave, [0.0] * (len(self.limites) + 3))
            i = next((i for i, lim in enumerate(self.limites) if valor <= lim),
                     len(self.limites))
            d[i] += 1
            d[-2] += valor
            d[-1] += 1

    def contagem(self, **rotulos: str) -> int:
        d = self._dados.get(self._chave(rotulos))
        return int(d[-1]) if d else 0

    def render(self) -> str:
        with self._trava:
            itens = sorted((k, list(v)) for k, v in self._dados.items())
        linhas = self._cabecalho()
        for chave, d in itens:
            acumulado = 0.0
            for i, lim in enumerate((*self.limites, math.inf)):
                acumulado += d[i]
                le = "+Inf" if math.isinf(lim) else _numero(lim)
                rot = _rotulos(self.rotulos, chave, 'le="' + le + '"')
                linhas.append(f"{self.nome}_bucket{rot} {_numero(acumulado)}")
            linhas.append(f"{self.nome}_sum{_rotulos(self.rotulos, chave)} {_numero(d[-2])}")
            linhas.append(f"{self.nome}_count{_rotulos(self.rotulos, chave)} {_numero(d[-1])}")
        return "\n".join(linhas)


HTTP_REQUISICOES = Contador("valtexa_ia_http_requests_total", "Requisições HTTP da IA.",
                            ("method", "route", "status"))
HTTP_DURACAO = Histograma("valtexa_ia_http_request_duration_seconds",
                          "Duração das requisições HTTP da IA (segundos).", ("method", "route"),
                          LIMITES_HTTP_S)
INFERENCIAS = Contador("valtexa_ia_inferences_total", "Imagens classificadas, por modelo.",
                       ("model",))
CHAMADAS = Contador("valtexa_ia_inference_calls_total", "Chamadas de inferência (lotes).",
                    ("model",))
ERROS = Contador("valtexa_ia_inference_errors_total",
                 "Chamadas de inferência que falharam, por modelo e tipo de erro.",
                 ("model", "error"))
LATENCIA = Histograma("valtexa_ia_inference_duration_seconds",
                      "Tempo de inferência por imagem (segundos), por modelo.", ("model",),
                      LIMITES_INFERENCIA_S)
CLASSES = Contador("valtexa_ia_predictions_total",
                   "Previsões por classe (a distribuição que o drift compara).",
                   ("model", "label"))

SERIES: tuple[_Serie, ...] = (HTTP_REQUISICOES, HTTP_DURACAO, INFERENCIAS, CHAMADAS, ERROS,
                              LATENCIA, CLASSES)


def tipo_de_erro(exc: BaseException) -> str:
    """Nome curto e de cardinalidade fixa (não a mensagem)."""
    nome = type(exc).__name__
    return {"ModelNotLoadedError": "modelo_ausente", "InvalidImageError": "imagem_invalida",
            "UnknownClassError": "classe_desconhecida"}.get(nome, "falha")


def registrar_inferencia(modelo: str, classes: Iterable[str], segundos: float,
                         imagens: int) -> None:
    CHAMADAS.inc(model=modelo)
    if imagens <= 0:
        return
    INFERENCIAS.inc(imagens, model=modelo)
    LATENCIA.observe(segundos / imagens, model=modelo)
    for rotulo in classes:
        CLASSES.inc(model=modelo, label=rotulo or "?")


def registrar_erro(modelo: str, exc: BaseException) -> None:
    CHAMADAS.inc(model=modelo)
    ERROS.inc(model=modelo, error=tipo_de_erro(exc))


def render(extras: Iterable[str] = ()) -> str:
    return "\n".join([s.render() for s in SERIES] + list(extras)) + "\n"


def modelos_carregados(estado: dict[str, bool]) -> str:
    linhas = ["# HELP valtexa_ia_model_loaded 1 se os pesos do modelo estão carregados.",
              "# TYPE valtexa_ia_model_loaded gauge"]
    linhas += [f'valtexa_ia_model_loaded{{model="{_escapa(k)}"}} {1 if v else 0}'
               for k, v in sorted(estado.items())]
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Middleware HTTP
# ---------------------------------------------------------------------------

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]


def molde_da_rota(scope: Scope) -> str:
    """O molde completo (com o prefixo do roteador), como no SaaS."""
    rota = scope.get("route")
    molde = getattr(rota, "path_format", None) or getattr(rota, "path", None)
    if not molde:
        return SEM_ROTA
    molde, caminho = str(molde), str(scope.get("path", ""))
    try:
        concreto = molde.format(**dict(scope.get("path_params") or {}))
    except (KeyError, IndexError, ValueError):
        concreto = None
    if concreto and caminho.endswith(concreto):
        return (caminho[: len(caminho) - len(concreto)] + molde)[:200]
    return molde[:200]


class MetricasMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        status = {"code": 500}

        async def enviar(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = int(message["status"])
            await send(message)

        inicio = perf_counter()
        try:
            await self.app(scope, receive, enviar)
        except Exception:
            status["code"] = 500
            raise
        finally:
            metodo, rota = str(scope.get("method", "GET")), molde_da_rota(scope)
            HTTP_REQUISICOES.inc(method=metodo, route=rota, status=str(status["code"]))
            HTTP_DURACAO.observe(perf_counter() - inicio, method=metodo, route=rota)


__all__ = [
    "CONTENT_TYPE",
    "MetricasMiddleware",
    "modelos_carregados",
    "registrar_erro",
    "registrar_inferencia",
    "render",
]
