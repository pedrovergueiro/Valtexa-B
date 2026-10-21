"""PINN para a dispersão de esporos: recuperar D e λ de armadilhas de esporo (dado sintético).

    python -m training.pinn_dispersao --sementes 0 1 2 --saida pinn.json   (torch; CPU basta)

A arquitetura do roteiro (6.2), nos passos que dá para fazer sem dado de campo:

1. **o modelo físico** é o de :mod:`app.gemeo.dispersao` em regime permanente,
   ``u·∇C = D∇²C - λC + S``, com fontes ``S`` suavizadas (gaussianas de 0,5 m) nas plantas que
   esporulam e o vento ``u`` conhecido (anemômetro);
2. **a verdade** sai da solução numérica (convolução com a função de Green numa grade de
   0,25 m) com ``D`` e ``λ`` escolhidos;
3. **a medição**: 40 armadilhas de esporo em pontos sorteados, com 10% de ruído
   multiplicativo — o que um talhão instrumentado daria;
4. **a PINN**: uma rede ``C(x, y)`` e os parâmetros ``log D``, ``log λ`` aprendidos juntos,
   pela perda nos dados (em log) mais o resíduo da equação em pontos de colocação; e, para
   comparar, o **ajuste pelo solucionador**: a mesma física resolvida numericamente para cada
   ``(D, λ)`` candidato, com Nelder-Mead no erro das armadilhas (dois parâmetros);
5. **a validação**: o ``D`` e o ``λ`` recuperados contra os verdadeiros, e o erro do campo em
   pontos que nenhuma armadilha mediu, contra uma rede só de dados (sem física).

O que **não** está aqui, e o roteiro exige antes de qualquer uso: série real de armadilha e
anemômetro no talhão, os mecanismos do patógeno (produção, sobrevivência, deposição,
infecção), e a comparação das previsões com as passadas seguintes (erro e calibração). Este
script prova o mecanismo de estimação; não estima nada do café.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from app.gemeo.dispersao import green

LADO_X, LADO_Y = 60.0, 40.0
SIGMA_FONTE = 0.5
VERDADE = {"difusao_m2_s": 0.8, "deposicao_1_s": 0.04}
VENTO = (0.6, 0.15)


def campo_verdadeiro(rng, h: float = 0.25):
    """``(xs, ys, C, fontes, q)``: a solução numérica com fontes gaussianas."""
    fontes = np.column_stack([rng.uniform(15, 30, 4), rng.uniform(12, 28, 4)])
    q = rng.uniform(0.5, 1.5, len(fontes))
    xs, ys = np.arange(0, LADO_X, h), np.arange(0, LADO_Y, h)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    s = sum(qi * np.exp(-((gx - fx) ** 2 + (gy - fy) ** 2) / (2 * SIGMA_FONTE ** 2))
            / (2 * math.pi * SIGMA_FONTE ** 2) for (fx, fy), qi in zip(fontes, q, strict=True))
    ny, nx = s.shape
    oy, ox = np.mgrid[-(ny - 1):ny, -(nx - 1):nx]
    nucleo = green(ox * h, oy * h, *VENTO, VERDADE["difusao_m2_s"], VERDADE["deposicao_1_s"],
                   h / 2.0)
    forma = (3 * ny, 3 * nx)
    conv = np.fft.irfft2(np.fft.rfft2(s * h * h, forma) * np.fft.rfft2(nucleo, forma), forma)
    return xs, ys, conv[ny - 1:2 * ny - 1, nx - 1:2 * nx - 1], fontes, q


def _interpola(xs, ys, c, pts):
    from scipy.interpolate import RegularGridInterpolator

    return RegularGridInterpolator((ys, xs), c)(pts[:, ::-1])


def treina(pts, obs, fontes, q, com_fisica: bool, passos: int, semente: int):
    import torch

    torch.manual_seed(semente)
    escala = torch.tensor([LADO_X, LADO_Y])
    rede = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 1))
    log_d = torch.nn.Parameter(torch.tensor(math.log(0.3)))  # chute longe da verdade
    log_l = torch.nn.Parameter(torch.tensor(math.log(0.2)))
    params = list(rede.parameters()) + ([log_d, log_l] if com_fisica else [])
    otim = torch.optim.Adam(params, lr=2e-3)
    p_obs = torch.tensor(pts, dtype=torch.float32)
    y_obs = torch.log(torch.tensor(obs, dtype=torch.float32))
    f = torch.tensor(fontes, dtype=torch.float32)
    qq = torch.tensor(q, dtype=torch.float32)
    ux, uy = VENTO

    def log_c(p):
        return rede(p / escala * 2 - 1).squeeze(-1)

    def fonte(p):
        d2 = ((p[:, None, :] - f[None]) ** 2).sum(-1)
        return (qq * torch.exp(-d2 / (2 * SIGMA_FONTE ** 2))).sum(-1) / (
            2 * math.pi * SIGMA_FONTE ** 2)

    gerador = torch.Generator().manual_seed(semente)
    for _ in range(passos):
        otim.zero_grad()
        perda = ((log_c(p_obs) - y_obs) ** 2).mean()
        if com_fisica:
            # Colocação: metade espalhada, metade perto das fontes (onde o campo muda rápido).
            unif = torch.rand(512, 2, generator=gerador) * escala
            perto = f[torch.randint(len(f), (512,), generator=gerador)] + torch.randn(
                512, 2, generator=gerador) * 2.0
            p = torch.cat([unif, perto]).requires_grad_(True)
            # Em C linear. (A forma em log, dividida por C, foi tentada e piorou: medido em
            # 19/10/2026, D errado em 52-76% e o campo pior que a rede só de dados.)
            c = torch.exp(log_c(p))
            gc = torch.autograd.grad(c.sum(), p, create_graph=True)[0]
            cxx = torch.autograd.grad(gc[:, 0].sum(), p, create_graph=True)[0][:, 0]
            cyy = torch.autograd.grad(gc[:, 1].sum(), p, create_graph=True)[0][:, 1]
            d, lam = torch.exp(log_d), torch.exp(log_l)
            residuo = ux * gc[:, 0] + uy * gc[:, 1] - d * (cxx + cyy) + lam * c - fonte(p)
            perda = perda + 10.0 * (residuo ** 2).mean()
        perda.backward()
        otim.step()
    def prever(pontos):
        with torch.no_grad():
            return torch.exp(log_c(torch.tensor(pontos, dtype=torch.float32))).numpy()

    return prever, float(torch.exp(log_d).detach()), float(torch.exp(log_l).detach())


def campo_com(fontes, q, difusao: float, deposicao: float, h: float = 0.25):
    xs, ys = np.arange(0, LADO_X, h), np.arange(0, LADO_Y, h)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    s = sum(qi * np.exp(-((gx - fx) ** 2 + (gy - fy) ** 2) / (2 * SIGMA_FONTE ** 2))
            / (2 * math.pi * SIGMA_FONTE ** 2) for (fx, fy), qi in zip(fontes, q, strict=True))
    ny, nx = s.shape
    oy, ox = np.mgrid[-(ny - 1):ny, -(nx - 1):nx]
    nucleo = green(ox * h, oy * h, *VENTO, difusao, deposicao, h / 2.0)
    forma = (3 * ny, 3 * nx)
    conv = np.fft.irfft2(np.fft.rfft2(s * h * h, forma) * np.fft.rfft2(nucleo, forma), forma)
    return xs, ys, conv[ny - 1:2 * ny - 1, nx - 1:2 * nx - 1]


def ajusta_por_solver(pts, obs, fontes, q):
    """``(D, λ, prever)`` minimizando o erro em log nas armadilhas, com o solucionador."""
    from scipy.optimize import minimize

    alvo = np.log(obs)

    def erro(x):
        xs, ys, c = campo_com(fontes, q, math.exp(x[0]), math.exp(x[1]), h=0.5)
        return float(np.mean((np.log(np.maximum(_interpola(xs, ys, c, pts), 1e-300))
                              - alvo) ** 2))

    r = minimize(erro, np.log([0.3, 0.2]), method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 300})
    d, lam = math.exp(r.x[0]), math.exp(r.x[1])
    xs, ys, c = campo_com(fontes, q, d, lam)
    return d, lam, lambda pontos: _interpola(xs, ys, c, pontos)


def experimento(semente: int, passos: int, armadilhas: int) -> dict:
    rng = np.random.default_rng(semente)
    xs, ys, c, fontes, q = campo_verdadeiro(rng)
    pts = np.column_stack([rng.uniform(5, LADO_X - 5, armadilhas),
                           rng.uniform(5, LADO_Y - 5, armadilhas)])
    verdade = _interpola(xs, ys, c, pts)
    obs = verdade * np.exp(rng.normal(0, 0.1, len(pts)))
    teste = np.column_stack([rng.uniform(5, LADO_X - 5, 400), rng.uniform(5, LADO_Y - 5, 400)])
    c_teste = _interpola(xs, ys, c, teste)
    saida = {"semente": semente, "armadilhas": armadilhas}
    for nome, fisica in (("pinn", True), ("so_dados", False)):
        prever, d, lam = treina(pts, obs, fontes, q, fisica, passos, semente)
        erro = np.abs(np.log(np.maximum(prever(teste), 1e-12)) - np.log(c_teste))
        saida[nome] = {"erro_log_mediano_teste": round(float(np.median(erro)), 4),
                       "erro_log_p90_teste": round(float(np.percentile(erro, 90)), 4)}
        if fisica:
            saida[nome] |= {"D": round(d, 4), "lambda": round(lam, 5),
                            "erro_D": round(d / VERDADE["difusao_m2_s"] - 1, 4),
                            "erro_lambda": round(lam / VERDADE["deposicao_1_s"] - 1, 4)}
    d, lam, prever = ajusta_por_solver(pts, obs, fontes, q)
    erro = np.abs(np.log(np.maximum(prever(teste), 1e-12)) - np.log(c_teste))
    saida["solver"] = {"erro_log_mediano_teste": round(float(np.median(erro)), 4),
                       "erro_log_p90_teste": round(float(np.percentile(erro, 90)), 4),
                       "D": round(d, 4), "lambda": round(lam, 5),
                       "erro_D": round(d / VERDADE["difusao_m2_s"] - 1, 4),
                       "erro_lambda": round(lam / VERDADE["deposicao_1_s"] - 1, 4)}
    print(json.dumps(saida, ensure_ascii=False), flush=True)
    return saida


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--passos", type=int, default=4000)
    p.add_argument("--armadilhas", type=int, default=40)
    p.add_argument("--saida", type=Path, default=None)
    args = p.parse_args(argv)
    execucoes = [experimento(s, args.passos, args.armadilhas) for s in args.sementes]
    resumo = {"verdade": VERDADE, "vento": VENTO, "execucoes": execucoes}
    if args.saida:
        args.saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
