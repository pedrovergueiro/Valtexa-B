"""O gêmeo digital do talhão pela API: tempo, ajuste, previsão e prescrição por risco.

* ``POST /fields/{id}/clima`` — a série da estação (registros horários ou dias prontos) ou,
  com ``CLIMA_OPEN_METEO=true``, o histórico e a previsão do Open-Meteo pela posição das
  plantas. ``GET`` resume o que há, dia a dia por fonte. **Por padrão a API não chama nada de
  fora**: sem a flag, buscar o Open-Meteo responde 409.
* ``POST /fields/{id}/gemeo/ajuste?doenca=`` — ajusta o modelo epidemiológico às passadas
  fechadas (:mod:`app.gemeo.ajuste`) e treina a rede residual com o portão temporal
  (:mod:`app.gemeo.rede`). Demora (dezenas de segundos num talhão de milhares de pés): roda
  fora do lock da fusão — o Rover continua mandando quadros —, um ajuste por talhão de cada
  vez (o segundo recebe 409).
* ``GET /fields/{id}/gemeo`` — o grafo do talhão e os modelos ajustados, com o diagnóstico.
* ``GET /fields/{id}/gemeo/previsao?doenca=`` — risco por planta, janelas de infecção, focos de
  agora e do horizonte (:mod:`app.gemeo.previsao`). Sem ajuste, usa o prior e avisa.
* ``POST /fields/{id}/gemeo/prescricao`` — a prescrição sobre o risco previsto, com a data de
  aplicação e os cenários (:mod:`app.gemeo.prescricao_risco`).

Estado em ``FIELDS_DIR/gemeo/<talhão>/``: ``clima.json`` e ``modelo-<doença>.json``.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import HTTP_422, settings
from app.fields_api import EconomiaIn, ProdutoIn, _known, _require_fields, fields
from app.gemeo import ajuste, clima, grafo, prescricao_risco, previsao, rede
from app.gemeo.epidemia import PRESETS, ModeloAjustado
from app.plant_fusion import _safe_filename
from app.prescricao import Economia, PrescricaoImpossivelError, Produto, resumo_em_texto

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fields", tags=["gêmeo digital"])

MAX_REGISTROS = 24 * 400
"""Teto de registros horários por envio (um ano e pouco)."""
MAX_DIAS = 800


class _Entrada(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class RegistroHorarioIn(_Entrada):
    time: datetime
    temperatura_c: float = Field(ge=-30, le=60)
    ur: float = Field(ge=0, le=100)
    chuva_mm: float = Field(default=0.0, ge=0, le=300)
    molhamento: float | None = Field(
        default=None, ge=0, le=1, description="Fração da hora com folha molhada (sensor).")
    vento_ms: float | None = Field(default=None, ge=0, le=60, description="Vento médio (m/s).")
    vento_dir_graus: float | None = Field(
        default=None, ge=0, le=360, description="De onde o vento vem (graus, meteorológico).")


class DiaIn(_Entrada):
    data: date
    t_media: float = Field(ge=-30, le=60)
    t_min: float = Field(ge=-30, le=60)
    t_max: float = Field(ge=-30, le=60)
    ur_media: float = Field(ge=0, le=100)
    chuva_mm: float = Field(ge=0, le=1000)
    molhamento_h: float = Field(ge=0, le=24, description="Horas de folha molhada no dia.")
    molhamento_medido: bool = False
    vento_ms: float | None = Field(default=None, ge=0, le=60)
    vento_u_ms: float | None = Field(default=None, ge=-60, le=60,
                                     description="Componente para leste (para onde o ar vai).")
    vento_v_ms: float | None = Field(default=None, ge=-60, le=60,
                                     description="Componente para norte.")


class ClimaIn(_Entrada):
    registros_horarios: list[RegistroHorarioIn] = Field(default_factory=list,
                                                        max_length=MAX_REGISTROS)
    dias: list[DiaIn] = Field(default_factory=list, max_length=MAX_DIAS)
    buscar_open_meteo: bool = Field(
        default=False, description="Busca histórico e previsão (exige CLIMA_OPEN_METEO=true).")
    dias_previsao: int = Field(default=16, ge=1, le=16)


class PrescricaoRiscoIn(_Entrada):
    doenca: str = Field(min_length=1)
    produto: ProdutoIn
    economia: EconomiaIn
    residual_dias: int = Field(ge=1, le=120, description="Proteção depois de aplicado (bula).")
    curativo: float = Field(default=0.0, ge=0, le=1,
                            description="Fração do latente que o produto elimina (bula/ensaio).")
    dias_ate_colheita: int | None = Field(default=None, ge=0)
    horizonte_dias: int = Field(default=21, ge=1, le=90)
    membros: int = Field(default=8, ge=1, le=64)
    hoje: date | None = None


class ProdutoBulaIn(_Entrada):
    nome: str = Field(min_length=1, max_length=160)
    registro_mapa: str = Field(min_length=1, max_length=80)
    culturas: list[str] = Field(min_length=1, max_length=30)
    alvos: list[str] = Field(min_length=1, max_length=30)
    dose_min_l_ha: float = Field(gt=0, le=5000)
    dose_l_ha: float = Field(gt=0, le=5000)
    dose_max_l_ha: float = Field(gt=0, le=5000)
    carencia_dias: int = Field(ge=0, le=365)
    residual_dias: int = Field(default=14, ge=1, le=120)
    vento_max_ms: float | None = Field(default=None, gt=0, le=20)


class EquipamentoIn(_Entrada):
    nome: str = Field(min_length=1, max_length=120)
    vazao_nominal_l_min: float = Field(gt=0, le=50)
    faixa_por_bico_m: float = Field(gt=0, le=10)
    bicos_por_lado: int = Field(ge=1, le=60)
    velocidade_m_s: float = Field(gt=0, le=10)
    velocidade_min_m_s: float = Field(gt=0, le=10)
    velocidade_max_m_s: float = Field(gt=0, le=10)
    ciclo_min: float = Field(default=0.2, gt=0, lt=1)
    ciclo_max: float = Field(default=1.0, gt=0, le=1)
    frequencia_hz: float = Field(default=10.0, gt=0, le=100)


class RegrasIn(_Entrada):
    limiar_risco: float = Field(default=0.15, gt=0, lt=1)
    limiar_dose_cheia: float = Field(default=0.5, gt=0, le=1)
    incerteza_para_revisar: float = Field(default=0.4, gt=0, le=1)
    vento_max_ms: float = Field(default=2.8, gt=0, le=20)
    temperatura_max_c: float = Field(default=30.0, ge=0, le=60)
    umidade_min: float = Field(default=55.0, ge=0, le=100)


class ExclusaoIn(_Entrada):
    row_id: str = Field(min_length=1, max_length=80)
    x_inicio_m: float
    x_fim_m: float
    motivo: str = Field(min_length=1, max_length=200)


class AplicacaoAnteriorIn(_Entrada):
    data: date
    row_ids: list[str] = Field(default_factory=list, max_length=2000)
    residual_dias: int = Field(ge=1, le=120)


class AplicacaoLocalizadaIn(_Entrada):
    doenca: str = Field(min_length=1)
    produto: ProdutoBulaIn
    equipamento: EquipamentoIn
    regras: RegrasIn = Field(default_factory=RegrasIn)
    trecho_m: float = Field(default=5.0, ge=1, le=100)
    exclusoes: list[ExclusaoIn] = Field(default_factory=list, max_length=5000)
    aplicacoes_anteriores: list[AplicacaoAnteriorIn] = Field(default_factory=list,
                                                             max_length=200)
    data_aplicacao: date | None = None
    dias_ate_colheita: int | None = Field(default=None, ge=0)
    horizonte_dias: int = Field(default=14, ge=1, le=90)
    membros: int = Field(default=8, ge=1, le=64)
    hoje: date | None = None


# ---------------------------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------------------------


class GemeoStore:
    """Arquivos do gêmeo por talhão, e um ajuste por talhão de cada vez."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.escrita_clima = threading.Lock()
        """Serializa ler-juntar-gravar da série: dois envios ao mesmo tempo não se apagam."""
        self._ajustando: set[str] = set()

    @staticmethod
    def pasta(field_id: str) -> Path:
        if fields.directory is None:
            raise RuntimeError("FIELDS_DIR não configurado.")
        return fields.directory / "gemeo" / _safe_filename(field_id)[: -len(".json")]

    def serie(self, field_id: str) -> clima.SerieClimatica:
        caminho = self.pasta(field_id) / "clima.json"
        return clima.SerieClimatica.load(caminho) if caminho.exists() else (
            clima.SerieClimatica.vazia())

    def salvar_serie(self, field_id: str, serie: clima.SerieClimatica) -> None:
        serie.save(self.pasta(field_id) / "clima.json")

    def caminho_modelo(self, field_id: str, doenca: str) -> Path:
        return self.pasta(field_id) / f"modelo-{_safe_filename(doenca)}"

    def modelo(self, field_id: str, doenca: str) -> dict[str, Any] | None:
        caminho = self.caminho_modelo(field_id, doenca)
        if not caminho.exists():
            return None
        return json.loads(caminho.read_text(encoding="utf-8"))

    def salvar_modelo(self, field_id: str, doenca: str, conteudo: dict[str, Any]) -> None:
        caminho = self.caminho_modelo(field_id, doenca)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        temporario = caminho.with_suffix(".tmp")
        temporario.write_text(json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")
        temporario.replace(caminho)

    def externas(self, field_id: str) -> list[dict[str, Any]]:
        caminho = self.pasta(field_id) / "observacoes_externas.json"
        if not caminho.exists():
            return []
        return json.loads(caminho.read_text(encoding="utf-8"))

    def acrescentar_externa(self, field_id: str, passada: dict[str, Any]) -> bool:
        """Acrescenta; a mesma (fonte, pass_id) de novo substitui (reenvio idempotente)."""
        with self.escrita_clima:
            atuais = [e for e in self.externas(field_id)
                      if (e["fonte"], e["pass_id"]) != (passada["fonte"], passada["pass_id"])]
            substituiu = len(atuais) != len(self.externas(field_id))
            atuais.append(passada)
            caminho = self.pasta(field_id) / "observacoes_externas.json"
            caminho.parent.mkdir(parents=True, exist_ok=True)
            temporario = caminho.with_suffix(".tmp")
            temporario.write_text(json.dumps(atuais, ensure_ascii=False), encoding="utf-8")
            temporario.replace(caminho)
        return substituiu

    def comecar_ajuste(self, chave: str) -> bool:
        with self._lock:
            if chave in self._ajustando:
                return False
            self._ajustando.add(chave)
            return True

    def terminar_ajuste(self, chave: str) -> None:
        with self._lock:
            self._ajustando.discard(chave)


gemeo = GemeoStore()


def _grafo(field_id: str) -> grafo.GrafoLavoura:
    store = _require_fields()
    _known(store, field_id)
    with fields._lock:
        observacoes = store.pass_observations(field_id)
    externas = gemeo.externas(field_id)
    if externas:
        observacoes, _ = grafo.juntar_externas(observacoes, externas)
    try:
        return grafo.construir(observacoes)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def _doenca(g: grafo.GrafoLavoura, doenca: str) -> None:
    if doenca not in g.doencas:
        raise HTTPException(HTTP_422, f"{doenca!r} não é doença do talhão: {', '.join(g.doencas)}.")
    if doenca not in PRESETS:
        raise HTTPException(HTTP_422, f"Sem modelo epidemiológico para {doenca!r}: disponíveis "
                            f"{', '.join(sorted(PRESETS))}.")


def _serie_com_dados(field_id: str) -> clima.SerieClimatica:
    serie = gemeo.serie(field_id)
    if not serie.dias:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "Talhão sem série climática: mande a da estação em POST "
                            f"/fields/{field_id}/clima (ou ligue CLIMA_OPEN_METEO).")
    return serie


def _com_dispersao(g: grafo.GrafoLavoura, salvo: dict[str, Any] | None) -> None:
    """Liga no grafo a dispersão pelo vento escolhida no ajuste, se houver."""
    escolha = (salvo or {}).get("dispersao") or {}
    if escolha.get("fracao_aerea") and escolha.get("parametros"):
        from app.gemeo.dispersao import ParametrosDispersao

        g.dispersao = (ParametrosDispersao(**escolha["parametros"]),
                       float(escolha["azimute_rua_graus"]))


def _modelo(field_id: str, doenca: str) -> tuple[ModeloAjustado, rede.RedeResidual | None,
                                                 dict[str, Any] | None]:
    salvo = gemeo.modelo(field_id, doenca)
    if salvo is None:
        return ModeloAjustado(PRESETS[doenca], diagnostico={
            "identificado": False, "motivo": "sem ajuste: POST /gemeo/ajuste"}), None, None
    residual = None if salvo.get("rede") is None else rede.RedeResidual.from_dict(salvo["rede"])
    return ModeloAjustado.from_dict(salvo["modelo"]), residual, salvo


# ---------------------------------------------------------------------------------------------
# Clima
# ---------------------------------------------------------------------------------------------


def _resumo_serie(serie: clima.SerieClimatica) -> dict[str, Any]:
    datas = sorted(serie.dias)
    fontes: dict[str, int] = {}
    for d in serie.dias.values():
        fontes[d.fonte] = fontes.get(d.fonte, 0) + 1
    return {"lat": serie.lat, "lon": serie.lon, "dias": len(datas),
            "inicio": datas[0].isoformat() if datas else None,
            "fim": datas[-1].isoformat() if datas else None, "por_fonte": fontes}


@router.get("/{field_id}/clima", summary="A série climática do talhão, por fonte")
def get_clima(field_id: str, dias: bool = Query(default=False)) -> dict[str, Any]:
    _require_fields()
    serie = gemeo.serie(field_id)
    saida = _resumo_serie(serie)
    if dias:
        saida["serie"] = serie.to_dict()["dias"]
    return saida


@router.post("/{field_id}/clima", summary="Acrescenta tempo ao talhão (estação ou Open-Meteo)")
def post_clima(field_id: str, pedido: ClimaIn) -> dict[str, Any]:
    """A estação vence o Open-Meteo, e o observado vence a previsão, dia a dia.

    ``registros_horarios``: dia com menos de 18 horas fica de fora; molhamento sem sensor é
    estimado (UR ≥ 90% ou chuva). ``buscar_open_meteo``: só com ``CLIMA_OPEN_METEO=true``
    (a API não chama serviço de fora por padrão) e com plantas georreferenciadas.
    """
    store = _require_fields()
    novos: list[clima.DiaClimatico] = []
    if pedido.registros_horarios:
        novos += clima.dias_de_registros_horarios(
            [r.model_dump() | {"time": r.time.isoformat()} for r in pedido.registros_horarios],
            "estacao")
    for d in pedido.dias:
        if not d.t_min <= d.t_media <= d.t_max:
            raise HTTPException(HTTP_422, f"{d.data}: exige t_min <= t_media <= t_max.")
        novos.append(clima.DiaClimatico(**d.model_dump(), fonte="estacao"))
    lat = lon = None
    if pedido.buscar_open_meteo:
        if not settings.clima_open_meteo:
            raise HTTPException(status.HTTP_409_CONFLICT,
                                "Busca externa desligada: ligue CLIMA_OPEN_METEO=true para o "
                                "gêmeo buscar o tempo no Open-Meteo.")
        _known(store, field_id)
        with fields._lock:
            observacoes = store.pass_observations(field_id)
        lat, lon = clima.centro([p["lat"] for p in observacoes["plants"]],
                                [p["lon"] for p in observacoes["plants"]])
        if lat is None or lon is None:
            raise HTTPException(status.HTTP_409_CONFLICT,
                                "Plantas sem posição GPS: sem onde buscar o tempo.")
        hoje = datetime.now(UTC).date()
        tempos = [o["time"] for p in observacoes["plants"] for o in p["observations"].values()
                  if o.get("time")]
        inicio = min((datetime.fromisoformat(str(t).replace("Z", "+00:00")).date()
                      for t in tempos), default=hoje) - timedelta(days=7)
        try:
            novos += clima.open_meteo(lat, lon, inicio=inicio, hoje=hoje,
                                      dias_previsao=pedido.dias_previsao)
        except Exception as exc:  # rede, HTTP, corpo inesperado
            logger.warning("Open-Meteo falhou para %s: %s", field_id, exc)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                                f"Open-Meteo não respondeu: {exc}") from exc
    if not novos:
        raise HTTPException(HTTP_422, "Nada a acrescentar: mande registros_horarios, dias ou "
                            "buscar_open_meteo.")
    with gemeo.escrita_clima:
        serie = gemeo.serie(field_id)
        if lat is not None:
            serie.lat, serie.lon = lat, lon
        serie.junta(novos)
        gemeo.salvar_serie(field_id, serie)
    return {"acrescentados": len(novos), **_resumo_serie(serie)}


# ---------------------------------------------------------------------------------------------
# Gêmeo
# ---------------------------------------------------------------------------------------------


@router.get("/{field_id}/gemeo", summary="O grafo do talhão e os modelos ajustados")
def get_gemeo(field_id: str) -> dict[str, Any]:
    g = _grafo(field_id)
    modelos = {}
    for d in g.doencas:
        salvo = gemeo.modelo(field_id, d) if d in PRESETS else None
        modelos[d] = None if salvo is None else {
            "ajustado_em": salvo.get("ajustado_em"), "passadas": salvo.get("passadas"),
            "parametros": salvo["modelo"]["parametros"],
            "diagnostico": salvo["modelo"]["diagnostico"], "rede": salvo.get("rede_diagnostico")}
    return {**g.resumo(), "clima": _resumo_serie(gemeo.serie(field_id)), "modelos": modelos}


@router.post("/{field_id}/gemeo/ajuste", summary="Ajusta o modelo epidemiológico às passadas")
def post_ajuste(field_id: str, doenca: str = Query(min_length=1),
                max_iter: int = Query(default=250, ge=20, le=2000),
                dispersao: bool = Query(
                    default=False, description="Escolhe também a parte do contágio que vai "
                    "pelo vento (exige vento na série e o azimute das ruas)."),
                azimute_rua_graus: float | None = Query(default=None, ge=0, lt=360)
                ) -> dict[str, Any]:
    """Ajusta às passadas. Com ``dispersao=true``, depois do ajuste escolhe a fração aérea
    ``ω`` (:func:`app.gemeo.ajuste.escolher_dispersao`) — desligada por padrão: os parâmetros
    físicos da dispersão são hipótese, e ``ω = 0`` é o resultado quando o vento não ajuda."""
    if dispersao and azimute_rua_graus is None:
        raise HTTPException(HTTP_422, "dispersao=true exige azimute_rua_graus (o das ruas).")
    g = _grafo(field_id)
    _doenca(g, doenca)
    serie = _serie_com_dados(field_id)
    chave = f"{field_id}|{doenca}"
    if not gemeo.comecar_ajuste(chave):
        raise HTTPException(status.HTTP_409_CONFLICT, "Ajuste deste talhão já em andamento.")
    try:
        try:
            modelo = ajuste.ajustar(g, doenca, serie, max_iter=max_iter)
        except ValueError as exc:  # série sem dado nenhum para algum dia
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        residual, diag_rede = None, {"adotada": False, "motivo": "uma passada só"}
        if g.t >= 2:
            fim = g.inicio + timedelta(days=int(g.dia_obs.max()) + 1)
            hist = rede.historico(g, doenca, serie.periodo(g.inicio, fim), modelo.parametros)
            residual, diag_rede = rede.ajustar_rede(g, hist, modelo.parametros)
        escolha = None
        if dispersao:
            from app.gemeo.dispersao import ParametrosDispersao

            escolha = ajuste.escolher_dispersao(g, doenca, serie, modelo, ParametrosDispersao(),
                                                float(azimute_rua_graus))
        conteudo = {"modelo": modelo.to_dict(), "dispersao": escolha,
                    "rede": None if residual is None else residual.to_dict(),
                    "rede_diagnostico": diag_rede, "passadas": g.passadas,
                    "ajustado_em": datetime.now(UTC).isoformat()}
        gemeo.salvar_modelo(field_id, doenca, conteudo)
    finally:
        gemeo.terminar_ajuste(chave)
    return {"field_id": field_id, "doenca": doenca, "passadas": g.passadas,
            "parametros": modelo.parametros.to_dict(), "diagnostico": modelo.diagnostico,
            "rede": diag_rede, "dispersao": escolha}


def _previsao(field_id: str, doenca: str, hoje: date | None, horizonte_dias: int,
              membros: int) -> tuple[grafo.GrafoLavoura, previsao.Previsao, ModeloAjustado,
                                     clima.SerieClimatica]:
    g = _grafo(field_id)
    _doenca(g, doenca)
    serie = _serie_com_dados(field_id)
    modelo, residual, salvo = _modelo(field_id, doenca)
    _com_dispersao(g, salvo)
    try:
        prev = previsao.prever(g, doenca, modelo, serie, hoje=hoje, horizonte_dias=horizonte_dias,
                               membros=membros, rede=residual)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if salvo is not None and salvo.get("passadas") != g.passadas:
        prev.avisos.append(f"Modelo ajustado com {len(salvo.get('passadas') or [])} passada(s); "
                           f"o talhão tem {g.t}: refaça o ajuste para usar as novas.")
    return g, prev, modelo, serie


@router.get("/{field_id}/gemeo/previsao", summary="Onde a doença vai estar, com a incerteza")
def get_previsao(
    field_id: str,
    doenca: str = Query(min_length=1),
    horizonte_dias: int = Query(default=21, ge=1, le=90),
    membros: int = Query(default=16, ge=1, le=64),
    hoje: date | None = Query(default=None),
    plantas: str = Query(default="em_risco", pattern="^(todas|em_risco|nenhuma)$"),
    risco_minimo: float = Query(default=0.05, ge=0, le=1),
) -> dict[str, Any]:
    g, prev, _modelo_usado, _serie = _previsao(field_id, doenca, hoje, horizonte_dias, membros)
    return prev.to_dict(g, plantas=plantas, risco_minimo=risco_minimo)


@router.post("/{field_id}/gemeo/prescricao", summary="Prescrição sobre o risco previsto")
def post_prescricao(field_id: str, pedido: PrescricaoRiscoIn,
                    formato: str = Query(default="json", pattern="^(json|geojson)$")
                    ) -> JSONResponse:
    """Prescreve onde a doença **vai** estar, com a data de aplicação (antes da próxima janela
    de infecção) e os cenários rodados no gêmeo. É recomendação: nada é pulverizado sem o
    agrônomo aprovar. **422** quando a bula, a carência ou o registro proíbem."""
    g, prev, modelo, serie = _previsao(field_id, pedido.doenca, pedido.hoje,
                                       pedido.horizonte_dias, pedido.membros)
    try:
        resultado = prescricao_risco.prescrever_por_risco(
            g, prev, modelo, serie,
            produto=Produto(**pedido.produto.model_dump(exclude={"culturas", "doencas"}),
                            culturas=tuple(pedido.produto.culturas),
                            doencas=tuple(pedido.produto.doencas)),
            economia=Economia(**pedido.economia.model_dump()),
            residual_dias=pedido.residual_dias, curativo=pedido.curativo,
            dias_ate_colheita=pedido.dias_ate_colheita, membros=pedido.membros)
    except PrescricaoImpossivelError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc
    if formato == "geojson":
        return JSONResponse(resultado.prescricao.to_geojson(), media_type="application/geo+json")
    return JSONResponse({**resultado.to_dict(), "resumo": resumo_em_texto(resultado.prescricao),
                         "previsao": prev.resumo()})


@router.post("/{field_id}/gemeo/aplicacao-localizada",
             summary="Mapa de aplicação localizada: risco por trecho, taxa e PWM por bico")
def post_aplicacao_localizada(field_id: str, pedido: AplicacaoLocalizadaIn) -> dict[str, Any]:
    """A prescrição variável preditiva (:mod:`app.aplicacao_localizada`): o risco previsto
    vira taxa por trecho de rua (dentro da bula) e ciclo de PWM por bico (dentro da máquina),
    com as verificações determinísticas e o hash do mapa — que o SaaS prende à aprovação do
    agrônomo. **Proposta**: não libera nada."""
    from app import aplicacao_localizada as al

    g, prev, _modelo_usado, serie = _previsao(field_id, pedido.doenca, pedido.hoje,
                                              pedido.horizonte_dias, pedido.membros)
    quando = pedido.data_aplicacao or prev.hoje
    try:
        produto = al.ProdutoDaBula(**pedido.produto.model_dump(exclude={"culturas", "alvos"}),
                                   culturas=tuple(pedido.produto.culturas),
                                   alvos=tuple(pedido.produto.alvos))
        corpo = al.mapa_de_aplicacao(
            g, prev, produto, al.Equipamento(**pedido.equipamento.model_dump()),
            al.Regras(**pedido.regras.model_dump()), trecho_m=pedido.trecho_m,
            exclusoes=[al.Exclusao(**e.model_dump()) for e in pedido.exclusoes],
            anteriores=[al.AplicacaoAnterior(a.data, tuple(a.row_ids), a.residual_dias)
                        for a in pedido.aplicacoes_anteriores],
            data_aplicacao=quando, dia=serie.dias.get(quando),
            dias_ate_colheita=pedido.dias_ate_colheita)
    except al.AplicacaoImpossivelError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc
    return {"field_id": field_id, **corpo, "previsao": prev.resumo()}


# ---------------------------------------------------------------------------------------------
# Enxame: evidência de outras fontes e prioridades de inspeção
# ---------------------------------------------------------------------------------------------

MAX_PLANTAS_EXTERNAS = 200_000


class ObservacaoExternaIn(_Entrada):
    """O que outra fonte (drone, agrônomo) viu, planta a planta: a LLR por doença."""

    fonte: str = Field(pattern="^[a-z][a-z0-9_]{0,31}$")
    pass_id: str = Field(min_length=1, max_length=80)
    time: datetime
    evidencia: dict[str, dict[str, float]] = Field(
        description="{doença: {plant_id: LLR}} — log da razão de verossimilhança medida.")


@router.post("/{field_id}/gemeo/observacoes",
             summary="Evidência por planta de outra fonte (drone, agrônomo)")
def post_observacoes(field_id: str, pedido: ObservacaoExternaIn) -> dict[str, Any]:
    """A fonte precisa de canal conhecido (:data:`app.gemeo.epidemia.CANAIS`). **O Rover não
    entra por aqui**: a evidência dele já chega pela fusão por planta, e aceitá-la de novo
    seria contá-la duas vezes."""
    from app.gemeo.epidemia import CANAIS, FONTE_ROVER

    if pedido.fonte == FONTE_ROVER:
        raise HTTPException(HTTP_422, "A evidência do Rover entra pela fusão por planta "
                            "(/frames, /predict/rover-frame), não por aqui: contaria duas vezes.")
    if pedido.fonte not in CANAIS:
        raise HTTPException(HTTP_422, f"Fonte sem canal de observação: {pedido.fonte!r} "
                            f"(conhecidas: {', '.join(sorted(CANAIS))}).")
    total = sum(len(v) for v in pedido.evidencia.values())
    if total > MAX_PLANTAS_EXTERNAS:
        raise HTTPException(HTTP_422, f"Mais de {MAX_PLANTAS_EXTERNAS} medidas num envio.")
    if any(not -50 <= llr <= 50 for v in pedido.evidencia.values() for llr in v.values()):
        raise HTTPException(HTTP_422, "LLR fora de [-50, 50]: confira a escala (log natural).")
    g = _grafo(field_id)
    desconhecidas_doenca = [d for d in pedido.evidencia if d not in g.doencas]
    if desconhecidas_doenca:
        raise HTTPException(HTTP_422, f"Doenças fora do talhão: {', '.join(desconhecidas_doenca)}.")
    passada = {"fonte": pedido.fonte, "pass_id": pedido.pass_id,
               "time": pedido.time.isoformat(), "evidencia": pedido.evidencia}
    conhecidas = set(g.plant_ids)
    fora = sum(1 for v in pedido.evidencia.values() for pid in v if pid not in conhecidas)
    substituiu = gemeo.acrescentar_externa(field_id, passada)
    return {"field_id": field_id, "fonte": pedido.fonte, "pass_id": pedido.pass_id,
            "medidas": total, "plantas_desconhecidas": fora, "substituiu": substituiu}


class LevantamentoIn(_Entrada):
    """Um levantamento aéreo processado: a área coberta e as regiões suspeitas."""

    fonte: str = Field(default="drone", pattern="^[a-z][a-z0-9_]{0,31}$")
    pass_id: str = Field(min_length=1, max_length=80)
    time: datetime
    doenca: str = Field(min_length=1, max_length=64)
    cobertura: list[list[float]] = Field(min_length=3, max_length=10_000,
                                         description="Polígono [[lon, lat], ...] voado.")
    anomalias: list[list[list[float]]] = Field(default_factory=list, max_length=10_000,
                                               description="Regiões suspeitas, [[lon, lat], ...].")


@router.post("/{field_id}/gemeo/levantamento",
             summary="Levantamento aéreo importado: regiões suspeitas viram evidência por planta")
def post_levantamento(field_id: str, pedido: LevantamentoIn) -> dict[str, Any]:
    """A área coberta e as regiões suspeitas de um voo viram LLR por planta no canal da fonte
    (:mod:`app.gemeo.levantamento`) e entram como uma passada externa — o mesmo caminho de
    ``/gemeo/observacoes``. Reenviar o mesmo ``pass_id`` substitui."""
    from app.gemeo import levantamento
    from app.gemeo.epidemia import CANAIS, FONTE_ROVER

    if pedido.fonte == FONTE_ROVER or pedido.fonte not in CANAIS:
        raise HTTPException(HTTP_422, f"Fonte sem canal de levantamento: {pedido.fonte!r} "
                            f"(conhecidas: {', '.join(sorted(CANAIS))}).")
    for poligono in [pedido.cobertura, *pedido.anomalias]:
        if len(poligono) < 3 or any(len(p) != 2 or not (-180 <= p[0] <= 180 and
                                                         -90 <= p[1] <= 90) for p in poligono):
            raise HTTPException(HTTP_422, "Polígono inválido: ao menos 3 pontos [lon, lat].")
    g = _grafo(field_id)
    if pedido.doenca not in g.doencas:
        raise HTTPException(HTTP_422, f"{pedido.doenca!r} não é doença do talhão: "
                            f"{', '.join(g.doencas)}.")
    canal = CANAIS[pedido.fonte]
    evid, conta = levantamento.evidencia(g.plant_ids, g.lat, g.lon, pedido.cobertura,
                                         pedido.anomalias, canal)
    if not evid:
        raise HTTPException(HTTP_422, "Nenhuma planta com GPS dentro da área coberta: "
                            "confira a cobertura (lon, lat) do levantamento.")
    passada = {"fonte": pedido.fonte, "pass_id": pedido.pass_id,
               "time": pedido.time.isoformat(), "evidencia": {pedido.doenca: evid}}
    substituiu = gemeo.acrescentar_externa(field_id, passada)
    return {"field_id": field_id, "fonte": pedido.fonte, "pass_id": pedido.pass_id,
            "doenca": pedido.doenca, **conta, "substituiu": substituiu,
            "canal": {"sensibilidade": canal.sensibilidade, "alarme_falso": canal.alarme_falso,
                      "origem": canal.origem},
            "aviso": "Região suspeita do voo tratada como sinal da doença pedida pelo canal "
                     f"{pedido.fonte!r} ({canal.origem}): vigor não é específico de doença."}


@router.get("/{field_id}/gemeo/prioridades",
            summary="Onde vale mais olhar: tarefas por valor da informação")
def get_prioridades(
    field_id: str,
    doenca: str = Query(min_length=1),
    canal: str = Query(default="rover", pattern="^[a-z][a-z0-9_]{0,31}$"),
    dias_ate_visita: int = Query(default=2, ge=1, le=30),
    membros: int = Query(default=8, ge=1, le=64),
    hoje: date | None = Query(default=None),
    segmento_m: float = Query(default=20.0, gt=1, le=500),
    bloco_ruas: int = Query(default=6, ge=1, le=100),
    bloco_m: float = Query(default=40.0, gt=1, le=1000),
    custo_tratar: float | None = Query(default=None, gt=0),
    perda_planta: float | None = Query(default=None, gt=0),
    eficacia: float | None = Query(default=None, gt=0, le=1),
    max_tarefas: int = Query(default=100, ge=1, le=10_000),
    com_plantas: bool = Query(default=False),
) -> dict[str, Any]:
    """Com ``custo_tratar``, ``perda_planta`` e ``eficacia`` (os três), o valor é o da decisão
    tratar / não tratar, em dinheiro; sem eles, a informação em bits. Ver
    :mod:`app.gemeo.prioridade`."""
    from app.gemeo import prioridade
    from app.gemeo.epidemia import canal_de

    economia_dada = [custo_tratar, perda_planta, eficacia]
    if any(v is not None for v in economia_dada) and not all(v is not None for v in economia_dada):
        raise HTTPException(HTTP_422, "custo_tratar, perda_planta e eficacia vão juntos.")
    g, prev, modelo, _serie = _previsao(field_id, doenca, hoje, dias_ate_visita, membros)
    try:
        c = canal_de(modelo.parametros, canal)
    except KeyError as exc:
        raise HTTPException(HTTP_422, str(exc)) from exc
    economia = None if custo_tratar is None else prioridade.Economia(
        custo_tratar, perda_planta, eficacia)
    resultado = prioridade.priorizar(g, prev, modelo, c, nome_canal=canal, economia=economia,
                                     segmento_m=segmento_m, bloco_ruas=bloco_ruas,
                                     bloco_m=bloco_m)
    return resultado.to_dict(g, max_tarefas=max_tarefas, com_plantas=com_plantas)
