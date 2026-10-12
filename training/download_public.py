"""Baixa datasets públicos do Mendeley Data e do Harvard Dataverse.

::

    python -m training.download_public mendeley 8d9fv6kpt3 --out ~/valtexa-raw/lemon
    python -m training.download_public dataverse doi:10.7910/DVN/LQUWXW --out ~/valtexa-raw/tz

Retoma de onde parou: arquivo que já existe com o tamanho certo é pulado. Serve
aos datasets de citros e banana do rover (ver MODELS.md), mas não sabe nada sobre
eles — só sobre as duas APIs públicas.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MENDELEY = "https://data.mendeley.com/public-api/datasets"
DATAVERSE = "https://dataverse.harvard.edu/api"
WORKERS = 4
"""Downloads simultâneos. Mais que isso não acelera e incomoda o servidor."""

HEADERS = {"User-Agent": "valtexa-dataset-downloader/1.0 (pesquisa)", "Accept": "*/*"}
"""O Mendeley recusa (403) o User-Agent padrão do urllib."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa datasets públicos (Mendeley, Dataverse)")
    parser.add_argument("source", choices=("mendeley", "dataverse"))
    parser.add_argument("dataset", help="Id do Mendeley (ex.: 8d9fv6kpt3) ou DOI do Dataverse.")
    parser.add_argument("--version", type=int, default=1, help="Versão do dataset no Mendeley.")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.source == "mendeley":
        files = list(mendeley_files(args.dataset, args.version))
    else:
        files = list(dataverse_files(args.dataset))

    total = sum(size for _, size, _ in files)
    print(f"{len(files)} arquivos, {total / 1e9:.2f} GB -> {args.out}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, ok in enumerate(pool.map(lambda f: fetch(args.out, *f), files), start=1):
            if done % 200 == 0 or done == len(files):
                print(f"  {done}/{len(files)}", flush=True)
            if not ok:
                print("  falhou um arquivo (ver acima); rode de novo para retomar")


def mendeley_files(dataset: str, version: int):
    """O dataset inteiro num zip só.

    A listagem arquivo a arquivo da API pública para em 1.000 por pasta, sem
    paginação — o dataset grande de limão (17.609 imagens) chegaria com 12.955.
    O zip é o único caminho completo; a URL redireciona para o S3 com validade
    de 5 minutos, então é pedida na hora de baixar.
    """
    url = f"https://data.mendeley.com/public-api/zip/{dataset}/download/{version}"
    yield f"{dataset}-v{version}.zip", 0, url


def dataverse_files(doi: str):
    query = urllib.parse.urlencode({"persistentId": doi})
    data = _get_json(f"{DATAVERSE}/datasets/:persistentId/?{query}")["data"]["latestVersion"]
    for entry in data["files"]:
        info = entry["dataFile"]
        folder = entry.get("directoryLabel") or ""
        relative = "/".join(p for p in (folder, info["filename"]) if p)
        yield relative, int(info["filesize"]), f"{DATAVERSE}/access/datafile/{info['id']}"


def fetch(out: Path, relative: str, size: int, url: str, attempts: int = 4) -> bool:
    target = out / relative
    if target.exists() and (size == 0 or target.stat().st_size == size):
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                open(partial, "wb") as sink,
            ):
                while chunk := response.read(1 << 20):
                    sink.write(chunk)
            partial.replace(target)
            return True
        except OSError as exc:
            print(f"  {relative}: tentativa {attempt} falhou ({exc})")
            time.sleep(2 * attempt)
    return False


def _get_json(url: str):
    request = urllib.request.Request(url, headers={**HEADERS, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


if __name__ == "__main__":
    main()
