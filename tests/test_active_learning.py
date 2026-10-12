"""Testes do módulo de aprendizado ativo (fila de revisão do agrônomo)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pytest

from app.active_learning import (
    TRACK_OUT_OF_SCOPE,
    ReviewItem,
    badge_select,
    make_synthetic_crop_data,
    rarity_from_counts,
    score_items,
    select_for_review,
    select_out_of_scope,
    simulate_labeling,
)

CLASSES = ["cordana", "pestalotiopsis", "saudavel", "sigatoka"]


def _item(
    item_id: str,
    probs: dict[str, float],
    *,
    novelty: float | None = 0.3,
    pset: list[str] | None = None,
    in_scope: bool | None = True,
    embedding: np.ndarray | None = None,
    field_id: str | None = "T1",
    model_key: str = "banana_v3",
    label: str | None = None,
) -> dict[str, Any]:
    pred = max(probs, key=probs.__getitem__)
    return {
        "id": item_id,
        "model_key": model_key,
        "crop": "banana",
        "predicted_class": pred,
        "confidence": probs[pred],
        "probabilities": probs,
        "novelty_score": novelty,
        "prediction_set": pset,
        "needs_review": False,
        "in_scope": in_scope,
        "created_at": f"2026-09-27T10:{int(item_id.split('-')[-1]) % 60:02d}:00",
        "embedding": None if embedding is None else embedding.tolist(),
        "metadata": {"field_id": field_id, "plant_id": f"P{item_id}", "camera": "rover-1"},
        "label": label,
    }


def _random_pool(
    n: int = 120, dim: int = 32, seed: int = 0, n_fields: int = 3
) -> list[dict[str, Any]]:
    """Pool sintético com embeddings, probabilidades softmax variadas e vários talhões."""
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        logits = rng.normal(scale=rng.uniform(0.3, 4.0), size=len(CLASSES))
        p = np.exp(logits - logits.max())
        p /= p.sum()
        pool.append(
            _item(
                f"it-{i}",
                dict(zip(CLASSES, p.tolist(), strict=True)),
                novelty=float(rng.uniform(0, 2)),
                embedding=rng.normal(size=dim),
                field_id=f"T{i % n_fields}",
            )
        )
    return pool


class TestScores:
    """Pontuações individuais nos casos óbvios."""

    def test_confident_scores_below_ambiguous(self) -> None:
        """Uma folha com 0,97 numa classe deve pontuar menos que uma dividida 0,48/0,46."""
        confident = _item("it-1", {"sigatoka": 0.97, "cordana": 0.01, "saudavel": 0.02})
        ambiguous = _item("it-2", {"sigatoka": 0.48, "cordana": 0.46, "saudavel": 0.06})
        s = score_items([confident, ambiguous])
        assert s["it-1"].uncertainty < s["it-2"].uncertainty
        assert s["it-1"].total < s["it-2"].total
        assert any(r.startswith("incerteza alta: margem 0,02") for r in s["it-2"].reasons)

    def test_high_novelty_raises_score(self) -> None:
        """Mesmas probabilidades, novidade maior -> pontuação maior e motivo 'novidade 1,8'."""
        probs = {"sigatoka": 0.8, "cordana": 0.1, "saudavel": 0.1}
        s = score_items([_item("it-1", probs, novelty=0.2), _item("it-2", probs, novelty=1.8)])
        assert s["it-2"].novelty > s["it-1"].novelty
        assert s["it-2"].total > s["it-1"].total
        assert "novidade 1,8" in s["it-2"].reasons

    def test_conformal_set_empty_and_large_count_as_uncertain(self) -> None:
        """Conjunto conformal vazio ou com várias classes aumenta a incerteza."""
        probs = {"sigatoka": 0.7, "cordana": 0.2, "saudavel": 0.1}
        s = score_items(
            [
                _item("it-1", probs, pset=["sigatoka"]),
                _item("it-2", probs, pset=["sigatoka", "cordana", "saudavel"]),
                _item("it-3", probs, pset=[]),
            ]
        )
        assert s["it-1"].uncertainty < s["it-2"].uncertainty
        assert s["it-1"].uncertainty < s["it-3"].uncertainty
        assert "conjunto conformal ambíguo: 3 classes" in s["it-2"].reasons

    def test_rare_class_scores_higher(self) -> None:
        """Classe prevista rara entre os rotulados do mesmo modelo recebe raridade alta."""
        labeled = [_item(f"lab-{i}", {"sigatoka": 1.0}, label="sigatoka") for i in range(50)]
        labeled.append(_item("lab-99", {"cordana": 1.0}, label="cordana"))
        probs_s = {"sigatoka": 0.7, "cordana": 0.3}
        probs_c = {"cordana": 0.7, "sigatoka": 0.3}
        s = score_items([_item("it-1", probs_s), _item("it-2", probs_c)], labeled)
        assert s["it-2"].rarity > 0.9
        assert s["it-1"].rarity == pytest.approx(0.0)
        assert any(r.startswith("classe rara: cordana") for r in s["it-2"].reasons)

    def test_rarity_uniform_without_labels(self) -> None:
        """Sem rótulos a raridade é zero para todas as classes (nada a corrigir)."""
        assert set(rarity_from_counts({}, ["a", "b", "c"]).values()) == {0.0}


class TestSelection:
    """Montagem do lote principal."""

    @pytest.mark.parametrize("strategy", ["badge", "coreset", "uncertainty", "random"])
    def test_out_of_scope_never_in_main_batch(self, strategy: str) -> None:
        """Itens com in_scope False não competem com folhas, mesmo sendo muito incertos."""
        pool = _random_pool(40)
        for i in range(10):
            pool[i]["in_scope"] = False
            pool[i]["probabilities"] = {c: 0.25 for c in CLASSES}
        out_ids = {p["id"] for p in pool[:10]}
        sel = select_for_review(pool, budget=20, strategy=strategy, seed=1)
        assert not out_ids & {s.id for s in sel}
        oos = select_out_of_scope(pool, budget=5)
        assert len(oos) == 5
        assert all(s.track == TRACK_OUT_OF_SCOPE and s.id in out_ids for s in oos)

    @pytest.mark.parametrize("strategy", ["badge", "coreset", "uncertainty", "random"])
    def test_quotas_respected(self, strategy: str) -> None:
        """Cotas por classe prevista e por talhão nunca são ultrapassadas."""
        pool = _random_pool(150, n_fields=4)
        sel = select_for_review(
            pool, budget=16, strategy=strategy, seed=3, max_per_class=5, max_per_field=0.25
        )
        by_id = {p["id"]: p for p in pool}
        classes = Counter(by_id[s.id]["predicted_class"] for s in sel)
        fields = Counter(by_id[s.id]["metadata"]["field_id"] for s in sel)
        assert max(classes.values()) <= 5
        assert max(fields.values()) <= 4  # 0,25 x 16
        assert len(sel) == 16

    @pytest.mark.parametrize("strategy", ["badge", "coreset", "uncertainty", "random"])
    def test_near_duplicates_not_selected_together(self, strategy: str) -> None:
        """Quadros vizinhos da mesma planta (embedding quase igual) não entram juntos."""
        rng = np.random.default_rng(7)
        base = rng.normal(size=32)
        ambiguous = {"sigatoka": 0.34, "cordana": 0.33, "saudavel": 0.33}
        dups = [
            _item(f"dup-{i}", ambiguous, novelty=1.9, embedding=base + 0.01 * rng.normal(size=32))
            for i in range(6)
        ]
        pool = dups + _random_pool(40, seed=2)
        sel = select_for_review(pool, budget=15, strategy=strategy, seed=0)
        chosen_dups = [s.id for s in sel if s.id.startswith("dup-")]
        assert len(chosen_dups) <= 1
        if strategy == "uncertainty":
            assert len(chosen_dups) == 1  # o mais informativo do grupo entra; os outros não

    def test_near_duplicate_of_labeled_is_skipped(self) -> None:
        """Um pendente praticamente idêntico a um item já rotulado não é pedido de novo."""
        rng = np.random.default_rng(1)
        emb = rng.normal(size=32)
        labeled = [_item("lab-1", {"sigatoka": 1.0}, embedding=emb, label="sigatoka")]
        pending = [
            _item("it-1", {"sigatoka": 0.5, "cordana": 0.5}, embedding=emb * 1.001),
            _item("it-2", {"sigatoka": 0.9, "cordana": 0.1}, embedding=rng.normal(size=32)),
        ]
        sel = select_for_review(pending, labeled, budget=2, strategy="uncertainty")
        assert [s.id for s in sel] == ["it-2"]

    @pytest.mark.parametrize("strategy", ["badge", "random", "coreset", "uncertainty"])
    def test_deterministic_with_seed(self, strategy: str) -> None:
        """Mesma seed -> mesmo lote, na mesma ordem."""
        pool = _random_pool(80)
        a = select_for_review(pool, budget=12, strategy=strategy, seed=42)
        b = select_for_review(pool, budget=12, strategy=strategy, seed=42)
        assert [s.id for s in a] == [s.id for s in b]
        assert [s.position for s in a] == list(range(1, 13))

    @pytest.mark.parametrize("strategy", ["badge", "coreset"])
    def test_returns_exactly_budget_distinct(self, strategy: str) -> None:
        """BADGE e core-set devolvem exatamente ``budget`` itens distintos."""
        pool = _random_pool(100)
        labeled = [
            {**p, "id": f"lab-{i}", "label": p["predicted_class"]}
            for i, p in enumerate(_random_pool(30, seed=9))
        ]
        sel = select_for_review(pool, labeled, budget=25, strategy=strategy, seed=5)
        assert len(sel) == 25
        assert len({s.id for s in sel}) == 25
        assert sel[0].strategy == strategy

    def test_reasons_in_portuguese(self) -> None:
        """Os motivos explicam a escolha: diversidade no BADGE, incerteza no item ambíguo."""
        sel = select_for_review(_random_pool(60), budget=10, strategy="badge", seed=0)
        assert "maior gradiente esperado do lote" in sel[0].reasons
        assert all(
            any(r.startswith("diversidade: longe dos já escolhidos") for r in s.reasons)
            for s in sel[1:]
        )

    def test_fallback_without_embeddings(self) -> None:
        """Sem embeddings, BADGE cai para o ranking por pontuação e avisa em ``strategy``."""
        pool = _random_pool(30)
        for p in pool:
            p["embedding"] = None
        sel = select_for_review(pool, budget=5, strategy="badge")
        assert len(sel) == 5
        assert sel[0].strategy.startswith("uncertainty (fallback")
        scores = [s.score for s in sel]
        assert scores == sorted(scores, reverse=True)

    def test_accepts_dataclass_items_and_multiple_models(self) -> None:
        """Aceita ReviewItem e mistura de modelos com número de classes e dimensões diferentes."""
        rng = np.random.default_rng(0)
        pool = [ReviewItem.from_mapping(p) for p in _random_pool(20)]
        for i in range(20):
            p = rng.dirichlet(np.ones(3))
            pool.append(
                ReviewItem(
                    id=f"citros-{i}",
                    model_key="citros_v1",
                    crop="citros",
                    predicted_class=["cancro", "greening", "saudavel"][int(p.argmax())],
                    confidence=float(p.max()),
                    probabilities=dict(zip(["cancro", "greening", "saudavel"], p, strict=True)),
                    embedding=rng.normal(size=16),
                )
            )
        for strategy in ("badge", "coreset"):
            sel = select_for_review(pool, budget=12, strategy=strategy, seed=0)
            assert len({s.id for s in sel}) == 12
            assert {s.model_key for s in sel} == {"banana_v3", "citros_v1"}

    def test_invalid_strategy(self) -> None:
        """Estratégia desconhecida gera erro claro."""
        with pytest.raises(ValueError, match="estratégia desconhecida"):
            select_for_review(_random_pool(5), budget=2, strategy="nope")  # type: ignore[arg-type]


class TestBadgeKernel:
    """A identidade de Kronecker substitui o embedding de gradiente materializado."""

    def test_matches_explicit_gradient_embedding(self) -> None:
        """Com o mesmo sorteio, a semeadura implícita coincide com k-means++ em g explícito."""
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(4), size=30)
        a = p - np.eye(4)[p.argmax(1)]
        h = rng.normal(size=(30, 8))
        g = np.einsum("ic,id->icd", a, h).reshape(30, -1)
        implicit = [i for i, _ in badge_select(a, h, 10, np.random.default_rng(3))]
        explicit = [i for i, _ in badge_select(g, np.ones((30, 1)), 10, np.random.default_rng(3))]
        assert implicit == explicit
        assert implicit[0] == int(np.argmax(np.linalg.norm(g, axis=1)))


class TestSimulator:
    """Simulador de ganho em dados sintéticos com uma classe rara."""

    def test_active_beats_random_on_average(self) -> None:
        """BADGE ou incerteza superam o aleatório na acurácia média com os mesmos rótulos.

        Média sobre 3 conjuntos sintéticos x 4 seeds; exigimos só uma vantagem pequena (0,5 ponto)
        para não ser frágil — nos experimentos a diferença fica em torno de 2-3 pontos.
        """
        gains = []
        for ds in range(3):
            x, y = make_synthetic_crop_data(n=900, seed=ds)
            res = simulate_labeling(
                x, y, ("random", "uncertainty", "badge"), rounds=8, seeds=range(4), epochs=200
            )
            best = max(res.mean_accuracy("uncertainty"), res.mean_accuracy("badge"))
            gains.append(best - res.mean_accuracy("random"))
        assert np.mean(gains) > 0.005

    def test_curve_shape(self) -> None:
        """A curva tem um ponto por rodada, começa igual para todos e cresce com os rótulos."""
        x, y = make_synthetic_crop_data(n=400, seed=1)
        res = simulate_labeling(x, y, rounds=4, seeds=(0, 1), epochs=150)
        for pts in res.curves.values():
            assert [p.n_labels for p in pts] == [10, 20, 30, 40, 50]
            assert pts[-1].accuracy_mean > pts[0].accuracy_mean
        firsts = {pts[0].accuracy_mean for pts in res.curves.values()}
        assert len(firsts) == 1  # mesmo lote inicial para todas as estratégias
        assert "rótulos" in res.table()
