#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA NMF-KL / MA-XRF — software desktop para matrizes espectrais transpostas
Autor: gerado para fluxo MA-XRF + NMF-KL + PCA

Objetivo:
    Carregar matrizes CSV exportadas pelo pré-processador NMF-KL/MA-XRF,
    validar os dados, permitir seleção de famílias/componentes, controlar
    normalização/escala, rodar PCA com scikit-learn e exportar a rodada completa.

Dependências:
    pip install numpy pandas scikit-learn matplotlib scipy

    Observação: scipy é usado apenas para o dendrograma/cluster hierárquico.
    Se não estiver instalado, o PCA e as demais métricas continuam funcionando.

Entrada esperada principal:
    obra,Pb_prep_E1.800,Pb_prep_E1.812,Zn_prep_E1.800,...
    MAN,0.001,0.002,0.004,...
    FIG,0.002,0.003,0.005,...

Também aceita matriz por componente:
    obra,H00_E1.800,H00_E1.812,H01_E1.800,...

Regras implementadas:
    - não altera o arquivo original;
    - não normaliza sem escolha explícita;
    - permite normalizar individualmente por área cada perfil H/família antes da PCA;
    - não consolida famílias repetidas automaticamente;
    - não executa PCA com apenas uma obra;
    - remove variância zero por padrão, mas permite desligar;
    - permite limitar a matriz efetivamente usada no PCA por dígitos significativos;
    - exporta a matriz exata usada no PCA;
    - separa visualmente contribuições de canais com loadings positivos e negativos;
    - exporta scores, loadings, contribuição dos loadings, variância, distâncias, configuração e gráficos.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Erro ao importar scikit-learn. Instale com: pip install scikit-learn\n"
        f"Detalhe: {exc}"
    )

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
except Exception:  # pragma: no cover
    dendrogram = None
    linkage = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
# Em execução desktop normal, TkAgg é o backend esperado.
# Em ambientes sem display (testes, importação estática, servidores), forçar TkAgg
# pode impedir até a importação do módulo. Por isso a escolha é protegida.
try:  # pragma: no cover - depende do ambiente gráfico disponível
    matplotlib.use("TkAgg")
except Exception:
    pass
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


APP_TITLE = "PCA NMF-KL / MA-XRF"
APP_VERSION = "2026-09-16_espaco_referencia_fases_dendrograma_refs"
CORE_ELEMENTS = {"cr", "pb", "co", "fe", "zn"}
EXCLUDE_KEYWORDS = (
    "restauro",
    "artefato",
    "artifact",
    "subfracao",
    "subfração",
)
MAX_TABLE_ROWS = 1000
ROLE_REFERENCE = "Referência"
ROLE_SUPPLEMENTARY = "Suplementar"
ROLE_EXCLUDED = "Excluir"
ROLE_OPTIONS = (ROLE_REFERENCE, ROLE_SUPPLEMENTARY, ROLE_EXCLUDED)
PCA_ROLE_COLUMN = "papel_pca"
SYNTHESIS_SCALE_LINEAR = "Linear"
SYNTHESIS_SCALE_ASINH = "Comprimida simétrica (asinh)"
SYNTHESIS_SCALE_OPTIONS = (SYNTHESIS_SCALE_LINEAR, SYNTHESIS_SCALE_ASINH)
LOADING_POSITIVE_COLOR = "#2C7BB6"
LOADING_NEGATIVE_COLOR = "#D7191C"
CHI2_95_2D = 5.991464547107979
CHI2_95_3D = 7.814727903251179


def matrix_feature_columns(matrix_df: pd.DataFrame) -> List[str]:
    """Colunas numéricas efetivamente usadas para PCA/projeção, ignorando metadados internos."""
    return [c for c in matrix_df.columns if c not in {"obra", PCA_ROLE_COLUMN}]


def _first_problem_cells(mask: pd.DataFrame, max_items: int = 8) -> List[str]:
    """Retorna exemplos compactos de células problemáticas no formato coluna@linha_csv."""
    examples: List[str] = []
    if mask is None or mask.empty:
        return examples
    for col in mask.columns:
        if not bool(mask[col].any()):
            continue
        rows = mask.index[mask[col].to_numpy()].tolist()
        for row_idx in rows:
            # +2 porque o índice 0 corresponde à primeira linha de dados; a linha 1 é o cabeçalho.
            examples.append(f"{col}@linha_{int(row_idx) + 2}")
            if len(examples) >= max_items:
                return examples
    return examples


def has_duplicate_labels(labels: Sequence[str]) -> bool:
    """True quando há rótulos repetidos após conversão para string e remoção de espaços."""
    cleaned = [str(x).strip() for x in labels]
    return len(cleaned) != len(set(cleaned))


# -----------------------------------------------------------------------------
# Utilidades gerais
# -----------------------------------------------------------------------------


def safe_slug(text: str, max_len: int = 80) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "sem_nome"


def fmt_pct(value: float) -> str:
    """Percentual em pt-BR: 32,8%."""
    try:
        return f"{float(value) * 100:.1f}%".replace(".", ",")
    except Exception:
        return "n/d"


def try_read_text_sample(path: Path) -> Tuple[str, str]:
    """Retorna (encoding, sample)."""
    encodings = ("utf-8-sig", "utf-8", "latin1", "cp1252")
    last_exc = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                sample = f.read(8192)
            return enc, sample
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise RuntimeError(f"Não foi possível ler o arquivo como texto. Último erro: {last_exc}")


def detect_delimiter(sample: str) -> str:
    """Detecta separador CSV de modo conservador."""
    if not sample:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {sep: first_line.count(sep) for sep in [",", ";", "\t", "|"]}
        return max(counts, key=counts.get) if any(counts.values()) else ","


def read_header_columns(path: Path, encoding: str, delimiter: str) -> List[str]:
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        return [str(c).strip() for c in next(reader)]


def find_duplicates(seq: Sequence[str]) -> List[str]:
    seen = set()
    duplicates = []
    for item in seq:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return duplicates


def extract_family_from_variable(col_name: str) -> str:
    """
    Extrai família/componente a partir do nome da variável.

    Exemplos:
        Pb_prep_E1.800       -> Pb_prep
        MAN_Pb_prep_E1.800   -> MAN_Pb_prep  (mantém prefixo, se existir)
        H00_E1.800           -> H00
        Zn_base_rep01_E8.640 -> Zn_base_rep01
    """
    name = str(col_name).strip()
    # Padrão principal: algo_E1.800 ou algo_E-0.100
    m = re.match(r"^(.+?)_E[-+]?\d+(?:[\.,]\d+)?$", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: remove a última parte se ela parece energia sem E.
    m2 = re.match(r"^(.+?)_[-+]?\d+(?:[\.,]\d+)?$", name)
    if m2:
        return m2.group(1)
    return name


def extract_energy_from_variable(col_name: str) -> Optional[float]:
    """Extrai a energia numérica do sufixo de uma variável espectral.

    Formatos aceitos incluem ``Pb_misto_E1.800`` e ``H00_E-0,100``.
    O retorno é ``None`` quando o nome não contém um sufixo de energia válido.
    """
    name = str(col_name).strip()
    match = re.search(r"_E([-+]?\d+(?:[\.,]\d+)?)$", name, flags=re.IGNORECASE)
    if match is None:
        return None
    try:
        value = float(match.group(1).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def normalize_spectral_groups_by_area(
    X: pd.DataFrame,
    col_to_family: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normaliza por área cada perfil espectral (linha × família/componente).

    Cada bloco de colunas pertencente a uma família é ordenado por energia e
    integrado pelo método trapezoidal. Blocos inteiramente nulos representam
    ausência da família e permanecem nulos. Qualquer bloco presente deve ter ao
    menos dois canais, energias distintas, valores não negativos e área positiva.

    A função retorna a matriz normalizada, mantendo a ordem original das colunas,
    e uma tabela de auditoria com a área anterior à transformação.
    """
    if X is None or X.empty:
        raise ValueError("A normalização por área requer uma matriz não vazia.")

    normalized = X.copy().astype(float)
    if not np.isfinite(normalized.to_numpy(dtype=float)).all():
        raise ValueError("A normalização por área recebeu valores não finitos.")

    negative_mask = normalized < 0
    if bool(negative_mask.any().any()):
        examples = _first_problem_cells(negative_mask, max_items=8)
        suffix = f" Exemplos: {', '.join(examples)}." if examples else ""
        raise ValueError(
            "A normalização individual por área dos perfis H exige valores não negativos."
            + suffix
        )

    grouped_columns: Dict[str, List[str]] = {}
    for col in normalized.columns:
        family = str(col_to_family.get(col, extract_family_from_variable(col))).strip() or "sem_grupo"
        grouped_columns.setdefault(family, []).append(col)

    audit_rows: List[Dict[str, object]] = []
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # compatibilidade com NumPy anterior a 2.0
        trapezoid = np.trapz

    for family, family_columns in grouped_columns.items():
        parsed = [(col, extract_energy_from_variable(col)) for col in family_columns]
        invalid_columns = [col for col, energy in parsed if energy is None]
        if invalid_columns:
            examples = ", ".join(invalid_columns[:6])
            more = f" (+{len(invalid_columns) - 6})" if len(invalid_columns) > 6 else ""
            raise ValueError(
                f"Não foi possível normalizar a família '{family}' por área porque há colunas "
                f"sem sufixo de energia reconhecível: {examples}{more}. "
                "Use nomes como Familia_E1.800 ou desative essa opção."
            )

        ordered = sorted(((col, float(energy)) for col, energy in parsed), key=lambda item: item[1])
        ordered_columns = [col for col, _energy in ordered]
        energies = np.asarray([energy for _col, energy in ordered], dtype=float)
        if len(energies) < 2:
            raise ValueError(
                f"A família '{family}' possui somente um canal. São necessários pelo menos dois "
                "canais para calcular área espectral pelo método trapezoidal."
            )
        if bool(np.any(np.diff(energies) <= 0)):
            raise ValueError(
                f"A família '{family}' possui energias repetidas ou não estritamente crescentes. "
                "Revise os nomes das colunas antes da normalização por área."
            )

        block = normalized.loc[:, ordered_columns].to_numpy(dtype=float)
        for row_position, row_index in enumerate(normalized.index):
            spectrum = block[row_position]
            if bool(np.all(spectrum == 0.0)):
                area = 0.0
                status = "ausente_bloco_zero"
            else:
                area = float(trapezoid(spectrum, x=energies))
                if not np.isfinite(area) or area <= 0.0:
                    raise ValueError(
                        f"A família '{family}' na linha de índice {row_index} apresenta dados não nulos, "
                        f"mas área trapezoidal inválida ({area!r}). Revise o espectro."
                    )
                block[row_position] = spectrum / area
                status = "normalizado_area_unitaria"

            audit_rows.append({
                "indice_linha_original": row_index,
                "familia_componente": family,
                "n_canais": int(len(ordered_columns)),
                "energia_min": float(energies[0]),
                "energia_max": float(energies[-1]),
                "area_antes_normalizacao": area,
                "status": status,
            })

        normalized.loc[:, ordered_columns] = block

    return normalized, pd.DataFrame(audit_rows)


def base_family_name(family: str) -> str:
    """Remove sufixos de repetição para alerta, sem modificar os dados."""
    base = str(family)
    base = re.sub(r"_rep\d+$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_repeti[cç][aã]o\d+$", "", base, flags=re.IGNORECASE)
    return base


def family_has_excluded_keyword(family: str) -> bool:
    f = str(family).lower()
    f_norm = f.replace("ç", "c").replace("ã", "a")
    return any(k.replace("ç", "c").replace("ã", "a") in f_norm for k in EXCLUDE_KEYWORDS)


def family_is_core(family: str) -> bool:
    """Identifica núcleo comum: Cr, Pb, Co, Fe, Zn e variações."""
    f = base_family_name(str(family)).lower()
    tokens = re.split(r"[_\-\s]+", f)
    if tokens and tokens[0] in CORE_ELEMENTS:
        return True
    # Fallback para nomes com prefixo de obra, ex.: MAN_Pb_prep.
    return any(tok in CORE_ELEMENTS for tok in tokens)


def convert_numeric_dataframe(raw_feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[int]], Dict[str, int]]:
    """
    Converte colunas para numérico aceitando vírgula decimal.

    A versão inicial convertia e inseria coluna por coluna, o que fragmentava
    o DataFrame quando a matriz tinha milhares de variáveis espectrais.
    Aqui a limpeza é feita em bloco e o DataFrame final é criado de uma vez,
    evitando o PerformanceWarning e reduzindo bastante o tempo de carregamento.

    Retorna:
        numeric_df
        non_numeric_positions: coluna -> índices de linha problemáticos
        empty_counts: coluna -> quantidade de células vazias
    """
    if raw_feature_df.empty:
        return pd.DataFrame(index=raw_feature_df.index), {}, {}

    # Preserva nomes/ordem das colunas e trabalha em bloco.
    as_text = raw_feature_df.astype(str).apply(lambda s: s.str.strip())
    empty_mask = raw_feature_df.isna() | as_text.eq("")
    empty_counts = {str(col): int(empty_mask[col].sum()) for col in raw_feature_df.columns}

    # Remove espaços internos e aceita vírgula decimal.
    cleaned = as_text.replace(r"\s+", "", regex=True).replace(",", ".", regex=True)
    cleaned = cleaned.mask(empty_mask, np.nan)

    # pd.to_numeric ainda trabalha por coluna internamente, mas sem inserir
    # repetidamente em um DataFrame existente.
    numeric = cleaned.apply(pd.to_numeric, errors="coerce").astype(float).copy()

    bad_mask = numeric.isna() & (~empty_mask)
    non_numeric_positions: Dict[str, List[int]] = {}
    bad_cols = [col for col in raw_feature_df.columns if bool(bad_mask[col].any())]
    for col in bad_cols:
        # Guardar índices 1-based considerando linha de dados, não cabeçalho.
        rows = raw_feature_df.index[bad_mask[col].to_numpy()].tolist()[:20]
        non_numeric_positions[str(col)] = [int(i) + 2 for i in rows]

    return numeric, non_numeric_positions, empty_counts


def round_array_to_significant_digits(values: np.ndarray, digits: int) -> np.ndarray:
    """
    Arredonda valores numéricos para N dígitos significativos.

    Diferente de arredondar por casas decimais fixas, este procedimento preserva
    a escala relativa de valores muito pequenos e muito grandes. Exemplos:
        0.0003847291846 com 6 dígitos -> 0.000384729
        12345.6789 com 6 dígitos     -> 12345.7

    Zeros permanecem zeros. Valores não finitos são preservados para que as
    validações existentes continuem acusando qualquer problema antes do PCA.
    """
    try:
        digits = int(digits)
    except Exception as exc:
        raise ValueError("O número de dígitos significativos deve ser inteiro.") from exc

    if digits < 3 or digits > 15:
        raise ValueError("Use entre 3 e 15 dígitos significativos. Valores maiores não são fisicamente úteis em float64.")

    arr = np.asarray(values, dtype=float)
    out = arr.copy()
    mask = np.isfinite(out) & (out != 0.0)
    if not np.any(mask):
        return out

    vals = out[mask]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        exponents = np.floor(np.log10(np.abs(vals)))
        factors = np.power(10.0, digits - 1 - exponents)
        valid = np.isfinite(factors) & (factors > 0)
        rounded_vals = vals.copy()
        rounded_vals[valid] = np.round(vals[valid] * factors[valid]) / factors[valid]

    out[mask] = rounded_vals
    return out


def round_dataframe_to_significant_digits(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    """Aplica arredondamento por dígitos significativos preservando índice e colunas."""
    rounded = round_array_to_significant_digits(df.to_numpy(dtype=float), digits)
    return pd.DataFrame(rounded, columns=df.columns, index=df.index)


def _distance_long_template(
    labels: Sequence[str],
    values: np.ndarray,
    metric_label: str,
    distance_col: str,
    distances: np.ndarray,
) -> pd.DataFrame:
    """Transforma uma matriz de distâncias par-a-par em formato longo."""
    rows = []
    labels = [str(x) for x in labels]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rows.append({
                "metrica": metric_label,
                "obra_1": labels[i],
                "obra_2": labels[j],
                distance_col: float(distances[i, j]) if np.isfinite(distances[i, j]) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(distance_col, ascending=False).reset_index(drop=True)


def distance_long_to_square(df: pd.DataFrame, labels: Sequence[str], distance_col: str) -> pd.DataFrame:
    """Converte tabela longa de distâncias para matriz quadrada obra × obra."""
    labels = [str(x) for x in labels]
    square = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels, dtype=float)
    if df is None or df.empty:
        square.insert(0, "obra", labels)
        return square.reset_index(drop=True)
    for _, row in df.iterrows():
        a = str(row["obra_1"])
        b = str(row["obra_2"])
        val = row[distance_col]
        if a in square.index and b in square.columns:
            square.loc[a, b] = val
            square.loc[b, a] = val
    square.insert(0, "obra", labels)
    return square.reset_index(drop=True)


def pairwise_euclidean_long(labels: Sequence[str], values: np.ndarray, metric_label: str) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    diff = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return _distance_long_template(labels, values, metric_label, "distancia", distances)


def pairwise_cosine_long(labels: Sequence[str], values: np.ndarray, metric_label: str) -> pd.DataFrame:
    """Distância do cosseno = 1 - similaridade do cosseno."""
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    denom = np.outer(norms, norms)
    sim = np.full((values.shape[0], values.shape[0]), np.nan, dtype=float)
    valid = denom > 0
    sim[valid] = (values @ values.T)[valid] / denom[valid]
    # Pequenos erros numéricos podem gerar 1.00000000002 ou -1.00000000002.
    sim = np.clip(sim, -1.0, 1.0)
    distances = 1.0 - sim
    np.fill_diagonal(distances, 0.0)
    return _distance_long_template(labels, values, metric_label, "distancia", distances)


def pairwise_mahalanobis_long(
    labels: Sequence[str],
    values: np.ndarray,
    metric_label: str,
    covariance_values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Mahalanobis no espaço dos scores, com covariância opcional ajustada só nas referências."""
    values = np.asarray(values, dtype=float)
    cov_source = np.asarray(covariance_values, dtype=float) if covariance_values is not None else values
    if values.ndim != 2 or values.shape[0] < 2 or cov_source.ndim != 2 or cov_source.shape[0] < 2:
        return pd.DataFrame(columns=["metrica", "obra_1", "obra_2", "distancia"])
    cov = np.cov(cov_source, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    inv_cov = np.linalg.pinv(cov)
    distances = np.zeros((values.shape[0], values.shape[0]), dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[0]):
            diff = values[i] - values[j]
            val = float(diff @ inv_cov @ diff.T)
            distances[i, j] = math.sqrt(max(val, 0.0))
    return _distance_long_template(labels, values, metric_label, "distancia", distances)


def make_pairs_rank_tables(distance_tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gera tabelas de pares mais próximos e mais distantes para cada métrica."""
    closest_parts = []
    farthest_parts = []
    for metric_name, df in distance_tables.items():
        if df is None or df.empty or "distancia" not in df.columns:
            continue
        work = df.dropna(subset=["distancia"]).copy()
        if work.empty:
            continue
        asc = work.sort_values("distancia", ascending=True).reset_index(drop=True)
        asc.insert(0, "rank_mais_proximo", np.arange(1, len(asc) + 1))
        desc = work.sort_values("distancia", ascending=False).reset_index(drop=True)
        desc.insert(0, "rank_mais_distante", np.arange(1, len(desc) + 1))
        closest_parts.append(asc)
        farthest_parts.append(desc)
    closest = pd.concat(closest_parts, ignore_index=True) if closest_parts else pd.DataFrame()
    farthest = pd.concat(farthest_parts, ignore_index=True) if farthest_parts else pd.DataFrame()
    return closest, farthest



def make_loading_contribution_tables(loadings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula contribuição quadrática dos loadings por família/componente/grupo.

    A contribuição de um grupo g em uma componente principal PCk é:
        soma(loadings_g^2) / soma(loadings_todos^2) * 100

    O quadrado é usado porque loadings positivos e negativos não devem se cancelar.
    Essa tabela serve para avaliar se uma PCA está dominada por poucas variáveis
    ou se os pesos estão distribuídos de forma mais equilibrada.
    """
    if loadings_df is None or loadings_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pc_cols = [c for c in loadings_df.columns if re.match(r"^PC\d+$", str(c))]
    if not pc_cols:
        return pd.DataFrame(), pd.DataFrame()

    if "familia_componente" in loadings_df.columns:
        group_col = "familia_componente"
    elif "variavel" in loadings_df.columns:
        group_col = "familia_componente"
        loadings_df = loadings_df.copy()
        loadings_df[group_col] = loadings_df["variavel"].map(extract_family_from_variable)
    else:
        group_col = "familia_componente"
        loadings_df = loadings_df.copy()
        loadings_df[group_col] = "sem_grupo"

    if "variavel" not in loadings_df.columns:
        loadings_df = loadings_df.copy()
        loadings_df["variavel"] = [f"variavel_{i + 1}" for i in range(len(loadings_df))]

    work_base = loadings_df.copy()
    work_base[group_col] = work_base[group_col].astype(str).str.strip().replace("", "sem_grupo")

    detailed_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for pc in pc_cols:
        values = pd.to_numeric(work_base[pc], errors="coerce").astype(float)
        sq_values = values.pow(2)
        total_sq = float(sq_values.sum(skipna=True))
        if not np.isfinite(total_sq) or total_sq <= 0:
            continue

        work = work_base[["variavel", group_col]].copy()
        work["_loading"] = values
        work["_loading_quadrado"] = sq_values
        work["_abs_loading"] = values.abs()

        pc_rows: List[Dict[str, object]] = []
        for group_name, sub in work.groupby(group_col, sort=False, dropna=False):
            if sub.empty:
                continue
            sum_sq = float(sub["_loading_quadrado"].sum(skipna=True))
            contribution_pct = (sum_sq / total_sq) * 100.0 if total_sq > 0 else np.nan

            idx_max = sub["_abs_loading"].idxmax()
            max_abs_loading = float(sub.loc[idx_max, "_abs_loading"])
            signed_loading_at_max = float(sub.loc[idx_max, "_loading"])
            variable_at_max = str(sub.loc[idx_max, "variavel"])

            pos_sq = float(sub.loc[sub["_loading"] >= 0, "_loading_quadrado"].sum(skipna=True))
            neg_sq = float(sub.loc[sub["_loading"] < 0, "_loading_quadrado"].sum(skipna=True))
            positive_contribution_pct = (pos_sq / total_sq) * 100.0
            negative_contribution_pct = (neg_sq / total_sq) * 100.0
            signed_balance_pct = positive_contribution_pct - negative_contribution_pct
            positive_fraction_in_group = (pos_sq / sum_sq) * 100.0 if sum_sq > 0 else np.nan
            negative_fraction_in_group = (neg_sq / sum_sq) * 100.0 if sum_sq > 0 else np.nan
            if np.isclose(pos_sq, neg_sq):
                predominant_sign = "misto"
            elif pos_sq > neg_sq:
                predominant_sign = "positivo"
            else:
                predominant_sign = "negativo"

            pc_rows.append({
                "PC": pc,
                "tipo_agregacao": group_col,
                "grupo_variavel": str(group_name) if str(group_name).strip() else "sem_grupo",
                "n_variaveis": int(len(sub)),
                "soma_loading_quadrado": sum_sq,
                "contribuicao_percentual": contribution_pct,
                "contribuicao_positiva_percentual": positive_contribution_pct,
                "contribuicao_negativa_percentual": negative_contribution_pct,
                "balanco_assinado_percentual": signed_balance_pct,
                "fracao_positiva_no_grupo_percentual": positive_fraction_in_group,
                "fracao_negativa_no_grupo_percentual": negative_fraction_in_group,
                "loading_abs_max": max_abs_loading,
                "variavel_maior_abs": variable_at_max,
                "loading_assinado_variavel_maior_abs": signed_loading_at_max,
                "sinal_predominante": predominant_sign,
            })

        pc_rows = sorted(pc_rows, key=lambda r: float(r["contribuicao_percentual"]), reverse=True)
        for rank, row in enumerate(pc_rows, start=1):
            row["rank_no_PC"] = rank
            detailed_rows.append(row)

        if pc_rows:
            top1 = pc_rows[0]
            top2_pct = float(sum(float(r["contribuicao_percentual"]) for r in pc_rows[:2]))
            top3_pct = float(sum(float(r["contribuicao_percentual"]) for r in pc_rows[:3]))
            top1_pct = float(top1["contribuicao_percentual"])
            second = pc_rows[1] if len(pc_rows) > 1 else None
            third = pc_rows[2] if len(pc_rows) > 2 else None

            if top1_pct >= 75.0 or top2_pct >= 85.0:
                dominance = "alta"
                interpretation = "PC concentrado em poucos grupos; usar como alerta de dominância."
            elif top1_pct >= 60.0 or top2_pct >= 75.0:
                dominance = "moderada"
                interpretation = "PC com dominância moderada; exige justificativa físico-química."
            else:
                dominance = "distribuida"
                interpretation = "PC com contribuição mais distribuída entre grupos."

            summary_rows.append({
                "PC": pc,
                "tipo_agregacao": group_col,
                "n_grupos": int(len(pc_rows)),
                "soma_quadrados_total_PC": total_sq,
                "maior_grupo": top1["grupo_variavel"],
                "maior_contribuicao_percentual": top1_pct,
                "segundo_grupo": second["grupo_variavel"] if second else "",
                "segunda_contribuicao_percentual": float(second["contribuicao_percentual"]) if second else np.nan,
                "terceiro_grupo": third["grupo_variavel"] if third else "",
                "terceira_contribuicao_percentual": float(third["contribuicao_percentual"]) if third else np.nan,
                "top2_percentual": top2_pct,
                "top3_percentual": top3_pct,
                "indice_dominancia": dominance,
                "interpretacao": interpretation,
            })

    detailed = pd.DataFrame(detailed_rows)
    summary = pd.DataFrame(summary_rows)

    detailed_cols = [
        "PC", "rank_no_PC", "tipo_agregacao", "grupo_variavel", "n_variaveis",
        "contribuicao_percentual", "contribuicao_positiva_percentual",
        "contribuicao_negativa_percentual", "balanco_assinado_percentual",
        "fracao_positiva_no_grupo_percentual", "fracao_negativa_no_grupo_percentual",
        "soma_loading_quadrado", "loading_abs_max",
        "variavel_maior_abs", "loading_assinado_variavel_maior_abs", "sinal_predominante",
    ]
    summary_cols = [
        "PC", "tipo_agregacao", "n_grupos", "soma_quadrados_total_PC",
        "maior_grupo", "maior_contribuicao_percentual", "segundo_grupo",
        "segunda_contribuicao_percentual", "terceiro_grupo", "terceira_contribuicao_percentual",
        "top2_percentual", "top3_percentual", "indice_dominancia", "interpretacao",
    ]
    if not detailed.empty:
        detailed = detailed[detailed_cols]
    if not summary.empty:
        summary = summary[summary_cols]
    return detailed, summary


# -----------------------------------------------------------------------------
# Estruturas de estado
# -----------------------------------------------------------------------------


@dataclass
class LoadState:
    path: Optional[Path] = None
    encoding: str = ""
    delimiter: str = ""
    original_df: Optional[pd.DataFrame] = None
    numeric_df: Optional[pd.DataFrame] = None
    obra_col: str = "obra"
    feature_cols: List[str] = field(default_factory=list)
    families: List[str] = field(default_factory=list)
    col_to_family: Dict[str, str] = field(default_factory=dict)
    header_duplicates: List[str] = field(default_factory=list)
    stripped_duplicates: List[str] = field(default_factory=list)
    non_numeric_positions: Dict[str, List[int]] = field(default_factory=dict)
    empty_counts: Dict[str, int] = field(default_factory=dict)
    zero_variance_cols: List[str] = field(default_factory=list)
    repeated_family_variants: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class PCAState:
    selected_families: List[str] = field(default_factory=list)
    selected_columns: List[str] = field(default_factory=list)
    removed_zero_variance: List[str] = field(default_factory=list)
    matrix_used: Optional[pd.DataFrame] = None
    scores: Optional[pd.DataFrame] = None
    loadings: Optional[pd.DataFrame] = None
    loading_contributions: Optional[pd.DataFrame] = None
    loading_dominance: Optional[pd.DataFrame] = None
    synthesis_scores: Optional[pd.DataFrame] = None
    synthesis_phase_centroids: Optional[pd.DataFrame] = None
    synthesis_phase_distances: Optional[pd.DataFrame] = None
    pca_mean: Optional[pd.DataFrame] = None
    preprocess_parameters: Optional[pd.DataFrame] = None
    area_normalization_audit: Optional[pd.DataFrame] = None
    variance: Optional[pd.DataFrame] = None
    distances: Optional[pd.DataFrame] = None
    distance_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    distance_square_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    closest_pairs: Optional[pd.DataFrame] = None
    farthest_pairs: Optional[pd.DataFrame] = None
    config: Optional[pd.DataFrame] = None
    figures: Dict[str, Figure] = field(default_factory=dict)
    explained_ratio: Optional[np.ndarray] = None
    score_values: Optional[np.ndarray] = None
    obra_labels: List[str] = field(default_factory=list)
    reference_obras: List[str] = field(default_factory=list)
    supplementary_obras: List[str] = field(default_factory=list)
    excluded_obras: List[str] = field(default_factory=list)


@dataclass
class MetadataState:
    path: Optional[Path] = None
    encoding: str = ""
    delimiter: str = ""
    df: Optional[pd.DataFrame] = None
    obra_col: str = "obra"
    fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Widgets auxiliares
# -----------------------------------------------------------------------------


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)
        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        # Windows/macOS/Linux fallback simples.
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass


class ReadOnlyText(ttk.Frame):
    def __init__(self, parent, height: int = 12):
        super().__init__(parent)
        self.text = tk.Text(self, height=height, wrap="word", undo=False)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")
        self.text.configure(state="disabled")

    def set_text(self, value: str):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", value)
        self.text.configure(state="disabled")


# -----------------------------------------------------------------------------
# Aplicação principal
# -----------------------------------------------------------------------------


class PCANMFXRFApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x860")
        self.minsize(1120, 720)

        self.load_state = LoadState()
        self.pca_state = PCAState()
        self.metadata_state = MetadataState()
        self.family_vars: Dict[str, tk.BooleanVar] = {}
        self.obra_role_vars: Dict[int, tk.StringVar] = {}
        self._suspend_pca_invalidation = False
        self._pca_input_trace_ids: List[str] = []

        self.var_normalize_family_area = tk.BooleanVar(value=False)
        self.var_normalize_rows = tk.BooleanVar(value=False)
        self.var_zscore = tk.BooleanVar(value=False)
        self.var_apply_sig_digits = tk.BooleanVar(value=False)
        self.var_sig_digits = tk.IntVar(value=10)
        self.var_remove_zero_var = tk.BooleanVar(value=True)
        self.var_n_components = tk.IntVar(value=2)
        self.var_metric_euclidean_scores = tk.BooleanVar(value=True)
        self.var_metric_cosine_matrix = tk.BooleanVar(value=True)
        self.var_metric_cosine_scores = tk.BooleanVar(value=False)
        self.var_metric_mahalanobis_scores = tk.BooleanVar(value=False)
        self.var_metric_dendrogram = tk.BooleanVar(value=False)
        self.var_color_by = tk.StringVar(value="Nenhum")
        self.var_marker_by = tk.StringVar(value=PCA_ROLE_COLUMN)
        self.var_ellipse_by = tk.StringVar(value="Nenhum")
        self.var_show_legend = tk.BooleanVar(value=True)
        self.var_show_labels = tk.BooleanVar(value=True)
        self.var_synthesis_phase_colors = tk.BooleanVar(value=True)
        self.var_synthesis_scale = tk.StringVar(value=SYNTHESIS_SCALE_ASINH)
        self.var_reference_space_label = tk.StringVar(value="Espaço de referência")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._configure_style()
        self._build_ui()
        self._register_static_pca_input_traces()
        self._set_status("Carregue uma matriz CSV para iniciar.")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", padding=4)
        style.configure("TLabelframe", padding=8)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Warn.TLabel", foreground="#8a5a00")
        style.configure("Error.TLabel", foreground="#9b1c1c")

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side="top", fill="x")
        ttk.Label(top, text="PCA NMF-KL / MA-XRF", style="Title.TLabel").pack(side="left", padx=8)
        ttk.Label(
            top,
            text="Matrizes espectrais transpostas por família ou componente",
            style="Subtitle.TLabel",
        ).pack(side="left", padx=12)

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(side="top", fill="both", expand=True, padx=8, pady=4)

        self.left_scroll = ScrollableFrame(main)
        self.right = ttk.Frame(main)
        main.add(self.left_scroll, weight=0)
        main.add(self.right, weight=1)

        self._build_left_panel(self.left_scroll.content)
        self._build_right_panel(self.right)

        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True, padx=8)
        self.progress_bar = ttk.Progressbar(
            bottom,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            length=280,
        )
        self.progress_bar.pack(side="right", padx=8, pady=4)

    def _build_left_panel(self, parent):
        # 1. Carregar dados
        lf_load = ttk.LabelFrame(parent, text="1. Carregar dados")
        lf_load.pack(fill="x", padx=4, pady=4)
        ttk.Button(lf_load, text="Carregar matriz CSV", command=self.load_csv, style="Accent.TButton").pack(fill="x")
        self.lbl_file = ttk.Label(lf_load, text="Nenhum arquivo carregado", wraplength=330)
        self.lbl_file.pack(fill="x", pady=(6, 0))

        # 2. Conferir matriz
        lf_summary = ttk.LabelFrame(parent, text="2. Conferir matriz")
        lf_summary.pack(fill="x", padx=4, pady=4)
        self.lbl_summary = ttk.Label(lf_summary, text="Aguardando arquivo.", justify="left", wraplength=330)
        self.lbl_summary.pack(fill="x")
        ttk.Button(lf_summary, text="Atualizar validação", command=self.refresh_validation_report).pack(fill="x", pady=(6, 0))

        # 3. Seleção de famílias/componentes
        lf_fam = ttk.LabelFrame(parent, text="3. Selecionar famílias/componentes")
        lf_fam.pack(fill="both", expand=False, padx=4, pady=4)
        btnrow1 = ttk.Frame(lf_fam)
        btnrow1.pack(fill="x")
        ttk.Button(btnrow1, text="Núcleo comum", command=self.select_core_families).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(btnrow1, text="Todas", command=self.select_all_families).pack(side="left", fill="x", expand=True, padx=(2, 0))
        btnrow2 = ttk.Frame(lf_fam)
        btnrow2.pack(fill="x", pady=(4, 4))
        ttk.Button(btnrow2, text="Remover restauro/artefato/subfração", command=self.remove_excluded_families).pack(fill="x")
        ttk.Button(lf_fam, text="Limpar seleção", command=self.clear_family_selection).pack(fill="x", pady=(0, 6))

        self.family_container = ttk.Frame(lf_fam)
        self.family_container.pack(fill="both", expand=True)
        ttk.Label(self.family_container, text="Carregue um CSV para listar as famílias.").pack(anchor="w")

        # 4. Papel das obras no PCA
        lf_roles = ttk.LabelFrame(parent, text="4. Papel das obras no PCA")
        lf_roles.pack(fill="x", padx=4, pady=4)
        ttk.Label(
            lf_roles,
            text=(
                "Referência define os eixos do PCA. Suplementar é apenas projetada. "
                "Excluir não entra na rodada."
            ),
            wraplength=330,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))
        role_btns = ttk.Frame(lf_roles)
        role_btns.pack(fill="x", pady=(0, 4))
        ttk.Button(role_btns, text="Todas referência", command=lambda: self.set_all_obra_roles(ROLE_REFERENCE)).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(role_btns, text="Todas suplementar", command=lambda: self.set_all_obra_roles(ROLE_SUPPLEMENTARY)).pack(side="left", fill="x", expand=True, padx=(2, 0))
        ttk.Button(lf_roles, text="Atualizar limite de PCs", command=self._update_summary_labels).pack(fill="x", pady=(0, 4))
        self.lbl_role_summary = ttk.Label(lf_roles, text="Carregue um CSV para definir o papel das obras.", wraplength=330, justify="left")
        self.lbl_role_summary.pack(fill="x", pady=(0, 4))
        self.obra_role_container = ttk.Frame(lf_roles)
        self.obra_role_container.pack(fill="x")
        ttk.Label(self.obra_role_container, text="Nenhuma obra carregada.").pack(anchor="w")

        # 5. Configurar PCA
        lf_cfg = ttk.LabelFrame(parent, text="5. Configurar PCA")
        lf_cfg.pack(fill="x", padx=4, pady=4)
        ttk.Checkbutton(
            lf_cfg,
            text="Normalizar cada perfil H/família por área",
            variable=self.var_normalize_family_area,
        ).pack(anchor="w")
        ttk.Label(
            lf_cfg,
            text=(
                "Integra cada bloco família×energia separadamente; famílias ausentes (zeros) permanecem nulas. "
                "Use para comparar formas de H entre NMFs independentes."
            ),
            wraplength=330,
            justify="left",
        ).pack(anchor="w", padx=(20, 0), pady=(0, 2))
        ttk.Checkbutton(
            lf_cfg,
            text="Normalizar também a linha/obra inteira por soma",
            variable=self.var_normalize_rows,
        ).pack(anchor="w")
        ttk.Checkbutton(lf_cfg, text="Aplicar z-score nas colunas", variable=self.var_zscore).pack(anchor="w")
        ttk.Checkbutton(
            lf_cfg,
            text="Limitar matriz final por dígitos significativos",
            variable=self.var_apply_sig_digits,
        ).pack(anchor="w")
        sigrow = ttk.Frame(lf_cfg)
        sigrow.pack(fill="x", pady=(2, 0))
        ttk.Label(sigrow, text="Dígitos significativos:").pack(side="left")
        self.spin_sig_digits = ttk.Spinbox(sigrow, from_=3, to=15, textvariable=self.var_sig_digits, width=6)
        self.spin_sig_digits.pack(side="left", padx=6)
        ttk.Label(sigrow, text="recomendado: 8–12").pack(side="left")
        ttk.Checkbutton(lf_cfg, text="Remover variáveis de variância zero", variable=self.var_remove_zero_var).pack(anchor="w", pady=(4, 0))

        pcrow = ttk.Frame(lf_cfg)
        pcrow.pack(fill="x", pady=(6, 0))
        ttk.Label(pcrow, text="Nº de PCs:").pack(side="left")
        self.spin_components = ttk.Spinbox(pcrow, from_=1, to=2, textvariable=self.var_n_components, width=6)
        self.spin_components.pack(side="left", padx=6)
        self.lbl_pc_limit = ttk.Label(lf_cfg, text="Limite: n/d")
        self.lbl_pc_limit.pack(anchor="w", pady=(4, 0))

        lf_meta = ttk.LabelFrame(parent, text="6. Metadados e agrupamentos visuais")
        lf_meta.pack(fill="x", padx=4, pady=4)
        ttk.Button(lf_meta, text="Carregar metadados CSV", command=self.load_metadata_csv).pack(fill="x")
        self.lbl_metadata = ttk.Label(lf_meta, text="Nenhum arquivo de metadados carregado", wraplength=330, justify="left")
        self.lbl_metadata.pack(fill="x", pady=(6, 4))

        ttk.Label(lf_meta, text="Colorir pontos por:").pack(anchor="w")
        self.cbo_color_by = ttk.Combobox(lf_meta, state="readonly", values=["Nenhum", PCA_ROLE_COLUMN], textvariable=self.var_color_by)
        self.cbo_color_by.pack(fill="x", pady=(0, 4))

        ttk.Label(lf_meta, text="Marcador por:").pack(anchor="w")
        self.cbo_marker_by = ttk.Combobox(lf_meta, state="readonly", values=["Nenhum", PCA_ROLE_COLUMN], textvariable=self.var_marker_by)
        self.cbo_marker_by.pack(fill="x", pady=(0, 4))

        ttk.Label(lf_meta, text="Desenhar regiões de dispersão 95% por:").pack(anchor="w")
        self.cbo_ellipse_by = ttk.Combobox(lf_meta, state="readonly", values=["Nenhum", PCA_ROLE_COLUMN], textvariable=self.var_ellipse_by)
        self.cbo_ellipse_by.pack(fill="x", pady=(0, 4))
        ttk.Label(
            lf_meta,
            text=(
                "Regra fixa: 3 ou mais obras = elipse/elipsoide de 95% por qui-quadrado; "
                "2 obras = somente o eixo principal maior de 95%, pontilhado; 1 obra = sem região."
            ),
            wraplength=330,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        ttk.Checkbutton(lf_meta, text="Mostrar legenda (arrastável)", variable=self.var_show_legend).pack(anchor="w")
        ttk.Checkbutton(lf_meta, text="Mostrar rótulos das obras (arrastáveis)", variable=self.var_show_labels).pack(anchor="w")
        ttk.Label(
            lf_meta,
            text="Depois de gerar o gráfico, clique e arraste os rótulos e as caixas de legenda para reposicioná-los.",
            wraplength=330,
            justify="left",
        ).pack(anchor="w", pady=(2, 4))
        ttk.Checkbutton(
            lf_meta,
            text="Mostrar cores e regiões de fase na Síntese PCA",
            variable=self.var_synthesis_phase_colors,
        ).pack(anchor="w")

        ttk.Label(lf_meta, text="Nome do espaço PCA:").pack(anchor="w", pady=(6, 0))
        self.entry_reference_space_label = ttk.Entry(
            lf_meta,
            textvariable=self.var_reference_space_label,
        )
        self.entry_reference_space_label.pack(fill="x", pady=(0, 4))
        self.entry_reference_space_label.bind(
            "<Return>",
            lambda _event: self.refresh_visualization(silent=True),
        )
        ttk.Label(
            lf_meta,
            text=(
                "Rótulo livre usado nas figuras da Síntese PCA. "
                "Ex.: Espaço Tarsila, Espaço de referência ou o nome de outro artista."
            ),
            wraplength=330,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(lf_meta, text="Escala visual dos gráficos PCA:").pack(anchor="w", pady=(6, 0))
        self.cbo_synthesis_scale = ttk.Combobox(
            lf_meta,
            state="readonly",
            values=SYNTHESIS_SCALE_OPTIONS,
            textvariable=self.var_synthesis_scale,
        )
        self.cbo_synthesis_scale.pack(fill="x", pady=(0, 4))
        self.cbo_synthesis_scale.bind("<<ComboboxSelected>>", lambda _event: self.refresh_visualization(silent=True))
        ttk.Label(
            lf_meta,
            text="A escala asinh comprime apenas a visualização nos gráficos PC1×PC2, PC1×PC3, PC1×PC2×PC3 e Síntese. Scores, distâncias e centroides usam os valores originais.",
            wraplength=330,
            justify="left",
        ).pack(anchor="w")
        ttk.Button(lf_meta, text="Atualizar visualização", command=self.refresh_visualization).pack(fill="x", pady=(6, 0))

        lf_metrics = ttk.LabelFrame(parent, text="7. Métricas de distância")
        lf_metrics.pack(fill="x", padx=4, pady=4)
        ttk.Label(
            lf_metrics,
            text="As métricas nos scores usam todos os PCs gerados. Para PC1–PC3, rode com Nº de PCs = 3.",
            wraplength=330,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Checkbutton(lf_metrics, text="Euclidiana nos scores selecionados", variable=self.var_metric_euclidean_scores).pack(anchor="w")
        ttk.Checkbutton(lf_metrics, text="Cosseno na matriz usada no PCA", variable=self.var_metric_cosine_matrix).pack(anchor="w")
        ttk.Checkbutton(lf_metrics, text="Cosseno nos scores selecionados", variable=self.var_metric_cosine_scores).pack(anchor="w")
        ttk.Checkbutton(lf_metrics, text="Mahalanobis nos scores selecionados", variable=self.var_metric_mahalanobis_scores).pack(anchor="w")
        ttk.Checkbutton(lf_metrics, text="Cluster hierárquico / dendrograma", variable=self.var_metric_dendrogram).pack(anchor="w")

        # 8. Rodar e exportar
        lf_run = ttk.LabelFrame(parent, text="8. Rodar e exportar")
        lf_run.pack(fill="x", padx=4, pady=4)
        ttk.Button(lf_run, text="Rodar PCA", command=self.run_pca, style="Accent.TButton").pack(fill="x")
        ttk.Button(lf_run, text="Exportar pacote tese (essencial)", command=self.export_essential_run).pack(fill="x", pady=(6, 0))
        ttk.Button(lf_run, text="Exportar rodada detalhada", command=self.export_full_run).pack(fill="x", pady=(3, 0))
        ttk.Separator(lf_run).pack(fill="x", pady=6)
        ttk.Button(lf_run, text="Exportar matriz usada", command=lambda: self.export_single("matrix")).pack(fill="x")
        ttk.Button(lf_run, text="Exportar scores", command=lambda: self.export_single("scores")).pack(fill="x", pady=(3, 0))
        ttk.Button(lf_run, text="Exportar loadings", command=lambda: self.export_single("loadings")).pack(fill="x", pady=(3, 0))
        ttk.Button(lf_run, text="Exportar contribuição dos loadings", command=lambda: self.export_single("loading_contributions")).pack(fill="x", pady=(3, 0))
        ttk.Button(lf_run, text="Exportar síntese PCA", command=lambda: self.export_single("synthesis")).pack(fill="x", pady=(3, 0))
        ttk.Button(lf_run, text="Exportar distâncias", command=lambda: self.export_single("distances")).pack(fill="x", pady=(3, 0))
        ttk.Button(lf_run, text="Exportar gráficos", command=lambda: self.export_single("figures")).pack(fill="x", pady=(3, 0))

    def _build_right_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        self.tab_validation = ttk.Frame(self.notebook)
        self.tab_pc12 = ttk.Frame(self.notebook)
        self.tab_pc13 = ttk.Frame(self.notebook)
        self.tab_pc123 = ttk.Frame(self.notebook)
        self.tab_synthesis = ttk.Frame(self.notebook)
        self.tab_loading_contrib = ttk.Frame(self.notebook)
        self.tab_variance = ttk.Frame(self.notebook)
        self.tab_scores = ttk.Frame(self.notebook)
        self.tab_loadings = ttk.Frame(self.notebook)
        self.tab_distances = ttk.Frame(self.notebook)
        self.tab_dendrogram = ttk.Frame(self.notebook)
        self.tab_config = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_validation, text="Validação")
        self.notebook.add(self.tab_pc12, text="PC1 × PC2")
        self.notebook.add(self.tab_pc13, text="PC1 × PC3")
        self.notebook.add(self.tab_pc123, text="PC1 × PC2 × PC3")
        self.notebook.add(self.tab_synthesis, text="Síntese PCA")
        self.notebook.add(self.tab_loading_contrib, text="Contrib. Loadings")
        self.notebook.add(self.tab_variance, text="Variância")
        self.notebook.add(self.tab_scores, text="Scores")
        self.notebook.add(self.tab_loadings, text="Loadings")
        self.notebook.add(self.tab_distances, text="Distâncias")
        self.notebook.add(self.tab_dendrogram, text="Dendrograma")
        self.notebook.add(self.tab_config, text="Configuração")

        # A Síntese PCA possui visualizações próprias, sem substituir os
        # gráficos gerais das abas PC1×PC2, PC1×PC3 e PC1×PC2×PC3.
        self.synthesis_notebook = ttk.Notebook(self.tab_synthesis)
        self.synthesis_notebook.pack(fill="both", expand=True)
        self.tab_synthesis_pc12 = ttk.Frame(self.synthesis_notebook)
        self.tab_synthesis_pc13 = ttk.Frame(self.synthesis_notebook)
        self.tab_reference_space_3d = ttk.Frame(self.synthesis_notebook)
        self.synthesis_notebook.add(self.tab_synthesis_pc12, text="PC1 × PC2")
        self.synthesis_notebook.add(self.tab_synthesis_pc13, text="PC1 × PC3")
        self.synthesis_notebook.add(self.tab_reference_space_3d, text="Espaço PCA 3D")

        self.validation_text = ReadOnlyText(self.tab_validation, height=30)
        self.validation_text.pack(fill="both", expand=True)
        self.validation_text.set_text("Carregue uma matriz CSV para iniciar a validação.")

        self.config_text = ReadOnlyText(self.tab_config, height=30)
        self.config_text.pack(fill="both", expand=True)
        self.config_text.set_text("A configuração da rodada aparecerá após executar o PCA.")

        self._init_tree_tabs()
        self._init_plot_tabs()

    def _init_tree_tabs(self):
        self.tree_scores = self._make_tree(self.tab_scores)
        self.tree_loadings = self._make_tree(self.tab_loadings)
        self.tree_distances = self._make_tree(self.tab_distances)

    def _init_plot_tabs(self):
        for tab, msg in [
            (self.tab_pc12, "O gráfico PC1 × PC2 aparecerá após executar o PCA."),
            (self.tab_pc13, "O gráfico PC1 × PC3 aparecerá quando houver pelo menos 3 PCs."),
            (self.tab_pc123, "O gráfico tridimensional PC1 × PC2 × PC3 aparecerá quando houver pelo menos 3 PCs."),
            (self.tab_synthesis_pc12, "A figura-síntese PC1 × PC2 aparecerá após executar o PCA."),
            (self.tab_synthesis_pc13, "A figura-síntese PC1 × PC3 aparecerá quando houver pelo menos 3 PCs."),
            (self.tab_reference_space_3d, "O espaço PCA tridimensional aparecerá quando houver pelo menos 3 PCs."),
            (self.tab_loading_contrib, "O gráfico de contribuição dos loadings aparecerá após executar o PCA."),
            (self.tab_variance, "O gráfico de variância explicada aparecerá após executar o PCA."),
            (self.tab_dendrogram, "O dendrograma aparecerá se a opção de cluster hierárquico estiver marcada."),
        ]:
            ttk.Label(tab, text=msg).pack(anchor="center", pady=30)

    def _make_tree(self, parent) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, show="headings")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    def _set_progress(self, value: Optional[float] = None, text: Optional[str] = None):
        """Atualiza a única barra de progresso usada por carregamento, PCA e exportações."""
        if value is not None:
            self.progress_var.set(max(0.0, min(100.0, float(value))))
        if text is not None:
            self.status_var.set(text)
        self.update_idletasks()

    def _reset_progress_later(self, delay_ms: int = 700):
        self.after(delay_ms, lambda: self.progress_var.set(0.0))

    def _has_pca_results(self) -> bool:
        """Retorna True quando há resultados calculados que ficariam obsoletos ao mudar entradas do PCA."""
        p = getattr(self, "pca_state", None)
        return bool(
            p is not None
            and (
                getattr(p, "matrix_used", None) is not None
                or getattr(p, "scores", None) is not None
                or getattr(p, "score_values", None) is not None
            )
        )

    def _invalidate_pca_results(self, reason: str = "Parâmetro do PCA alterado."):
        """Limpa resultados quando o usuário altera uma entrada que exige nova rodada.

        Controles puramente visuais, como cor, rótulos e escala asinh, não chamam
        este método. Assim, evita-se o erro silencioso de exportar resultados antigos
        após mudar família, normalização, z-score, número de PCs ou métricas.
        """
        if bool(getattr(self, "_suspend_pca_invalidation", False)):
            return
        if not self._has_pca_results():
            return
        self.pca_state = PCAState()
        if hasattr(self, "_clear_results_tabs"):
            self._clear_results_tabs()
        if hasattr(self, "_set_status"):
            self._set_status(f"{reason} Rode o PCA novamente para atualizar os resultados.")

    def _bind_pca_input_trace(self, var, reason: str):
        """Vincula uma variável Tk a invalidação conservadora dos resultados do PCA."""
        try:
            trace_id = var.trace_add("write", lambda *_args, r=reason: self._invalidate_pca_results(r))
            self._pca_input_trace_ids.append(trace_id)
        except Exception:
            # Variáveis Tk de versões antigas podem não suportar trace_add; a GUI continua funcional.
            pass

    def _register_static_pca_input_traces(self):
        """Registra mudanças em opções que alteram cálculo, não apenas visualização."""
        tracked = [
            (self.var_normalize_family_area, "Normalização individual por área dos perfis H alterada."),
            (self.var_normalize_rows, "Normalização por soma alterada."),
            (self.var_zscore, "Opção de z-score alterada."),
            (self.var_apply_sig_digits, "Controle de dígitos significativos alterado."),
            (self.var_sig_digits, "Número de dígitos significativos alterado."),
            (self.var_remove_zero_var, "Remoção de variância zero alterada."),
            (self.var_n_components, "Número de PCs alterado."),
            (self.var_metric_euclidean_scores, "Métricas de distância alteradas."),
            (self.var_metric_cosine_matrix, "Métricas de distância alteradas."),
            (self.var_metric_cosine_scores, "Métricas de distância alteradas."),
            (self.var_metric_mahalanobis_scores, "Métricas de distância alteradas."),
            (self.var_metric_dendrogram, "Opção de dendrograma alterada."),
        ]
        for var, reason in tracked:
            self._bind_pca_input_trace(var, reason)

    # ------------------------------------------------------------------
    # Carregamento e validação
    # ------------------------------------------------------------------

    def load_csv(self):
        path_str = filedialog.askopenfilename(
            title="Selecionar matriz CSV",
            filetypes=[("CSV", "*.csv"), ("Texto", "*.txt"), ("Todos os arquivos", "*.*")],
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            self._set_progress(2, "Iniciando carregamento do CSV...")
            load_state = self._read_and_validate_csv(path)
            self._set_progress(88, "Atualizando interface com famílias/componentes...")
            self.load_state = load_state
            self.pca_state = PCAState()
            self._populate_family_checkboxes()
            self._populate_obra_role_controls()
            self._set_progress(94, "Gerando relatório de validação...")
            self._update_summary_labels()
            self.refresh_validation_report()
            self._clear_results_tabs()
            self._set_progress(100, "CSV carregado e validado.")
            self._reset_progress_later()
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Erro ao carregar CSV", f"Não foi possível carregar o arquivo.\n\n{exc}")
            self._set_progress(0, "Erro ao carregar CSV.")

    def _read_and_validate_csv(self, path: Path) -> LoadState:
        self._set_progress(5, "Lendo amostra do arquivo e detectando encoding...")
        encoding, sample = try_read_text_sample(path)
        self._set_progress(10, "Detectando separador CSV...")
        delimiter = detect_delimiter(sample)
        self._set_progress(15, "Conferindo cabeçalho...")
        header_cols = read_header_columns(path, encoding, delimiter)
        header_duplicates = find_duplicates(header_cols)

        # dtype=str preserva conteúdo para validação; keep_default_na=False preserva vazios como "".
        self._set_progress(25, "Carregando tabela CSV na memória...")
        df = pd.read_csv(path, sep=delimiter, encoding=encoding, dtype=str, keep_default_na=False, engine="python")
        df.columns = [str(c).strip() for c in df.columns]
        stripped_duplicates = find_duplicates(list(df.columns))

        self._set_progress(38, "Validando estrutura básica da matriz...")
        state = LoadState(path=path, encoding=encoding, delimiter=delimiter)
        state.header_duplicates = header_duplicates
        state.stripped_duplicates = stripped_duplicates
        state.original_df = df.copy()

        # Coluna obra/Obra.
        obra_candidates = [c for c in df.columns if str(c).strip().lower() == "obra"]
        if not obra_candidates:
            state.errors.append('Não existe coluna "obra" ou "Obra".')
            # Ainda retorna estado para mostrar relatório.
            return state
        if len(obra_candidates) > 1:
            state.errors.append(f"Há mais de uma coluna identificada como obra: {obra_candidates}")
            return state

        obra_col = obra_candidates[0]
        state.obra_col = "obra"
        if obra_col != "obra":
            df = df.rename(columns={obra_col: "obra"})

        df["obra"] = df["obra"].astype(str).str.strip()
        blank_obra_rows = df.index[df["obra"].eq("")].tolist()
        if blank_obra_rows:
            state.errors.append(
                "Há nome(s) de obra vazio(s) na coluna 'obra'. Linhas CSV: "
                + ", ".join(str(int(i) + 2) for i in blank_obra_rows[:20])
            )

        feature_cols = [c for c in df.columns if c != "obra"]
        state.feature_cols = feature_cols

        if len(df) < 2:
            state.errors.append("PCA não pode ser executado com apenas uma obra. É necessário carregar pelo menos duas obras.")
        if df["obra"].nunique(dropna=True) < 2:
            state.errors.append("A matriz possui menos de duas obras distintas.")
        if not feature_cols:
            state.errors.append("A matriz não possui colunas numéricas de variáveis após a coluna obra.")

        duplicate_obras = df.loc[df["obra"].duplicated(keep=False), "obra"].tolist()
        if duplicate_obras:
            state.errors.append(
                "Há nomes de obras repetidos nas linhas: "
                + ", ".join(map(str, sorted(set(duplicate_obras))))
                + ". Use rótulos únicos para evitar ambiguidade em scores, distâncias e metadados."
            )

        # Conversão numérica.
        self._set_progress(55, "Convertendo variáveis espectrais para números...")
        numeric_df, non_numeric, empty_counts = convert_numeric_dataframe(df[feature_cols])
        self._set_progress(68, "Conferindo valores vazios e não numéricos...")
        state.numeric_df = numeric_df
        state.non_numeric_positions = non_numeric
        state.empty_counts = empty_counts
        state.original_df = df.copy()

        if non_numeric:
            qtd_cols = len(non_numeric)
            examples = []
            for col, rows in list(non_numeric.items())[:5]:
                examples.append(f"{col} (linhas {rows[:5]})")
            state.errors.append(
                f"Foram encontrados valores não numéricos em {qtd_cols} coluna(s). Exemplos: " + "; ".join(examples)
            )

        empty_total = sum(empty_counts.values())
        if empty_total > 0:
            cols_with_empty = [f"{c}: {n}" for c, n in empty_counts.items() if n > 0]
            state.errors.append(
                f"Foram encontradas {empty_total} célula(s) vazia(s). Exemplos: " + "; ".join(cols_with_empty[:8])
            )

        if numeric_df is not None and not numeric_df.empty:
            inf_mask = pd.DataFrame(
                np.isinf(numeric_df.to_numpy(dtype=float)),
                index=numeric_df.index,
                columns=numeric_df.columns,
            )
            if bool(inf_mask.any().any()):
                examples = _first_problem_cells(inf_mask)
                state.errors.append(
                    "Foram encontrados valores infinitos (+inf/-inf) na matriz. "
                    "Isso impede PCA confiável. Exemplos: " + "; ".join(examples)
                )

            negative_mask = numeric_df < 0
            if bool(negative_mask.any().any()):
                examples = _first_problem_cells(negative_mask)
                state.warnings.append(
                    "Há valores negativos na matriz. Para H/W/famílias de NMF-KL isso geralmente deve ser revisado "
                    "ou justificado metodologicamente. Exemplos: " + "; ".join(examples)
                )

        if header_duplicates:
            state.errors.append("Há colunas duplicadas no cabeçalho original: " + ", ".join(header_duplicates[:20]))
        if stripped_duplicates:
            state.errors.append("Há colunas duplicadas após remover espaços: " + ", ".join(stripped_duplicates[:20]))

        # Famílias.
        self._set_progress(76, "Detectando famílias/componentes a partir dos nomes das colunas...")
        col_to_family = {col: extract_family_from_variable(col) for col in feature_cols}
        families = sorted(set(col_to_family.values()), key=lambda x: str(x).lower())
        state.col_to_family = col_to_family
        state.families = families

        # Alerta de famílias repetidas por variantes _repXX.
        base_to_variants: Dict[str, List[str]] = {}
        for fam in families:
            base = base_family_name(fam)
            base_to_variants.setdefault(base, []).append(fam)
        repeated = {base: sorted(vars_) for base, vars_ in base_to_variants.items() if len(vars_) > 1}
        state.repeated_family_variants = repeated
        if repeated:
            examples = [f"{base}: {', '.join(vars_)}" for base, vars_ in list(repeated.items())[:6]]
            state.warnings.append(
                "Possíveis famílias repetidas detectadas. Nada foi consolidado automaticamente. "
                + " | ".join(examples)
            )

        # Matriz com mais variáveis que obras.
        n_rows = len(df)
        n_vars = len(feature_cols)
        if n_vars > n_rows:
            state.warnings.append(
                f"A matriz tem mais colunas de variáveis ({n_vars}) do que obras/linhas ({n_rows}). "
                "Isso é esperado em dados espectrais, mas limita o número útil de PCs a n_obras - 1."
            )

        # Variância zero.
        self._set_progress(82, "Verificando variáveis com variância zero...")
        if numeric_df is not None and not numeric_df.empty:
            if np.isfinite(numeric_df.to_numpy(dtype=float)).all():
                variances = numeric_df.var(axis=0, ddof=0, skipna=True)
                zero_cols = variances.index[(variances.fillna(0) == 0)].tolist()
                state.zero_variance_cols = zero_cols
                if zero_cols:
                    state.warnings.append(
                        f"Foram detectadas {len(zero_cols)} variável(is) com variância zero. "
                        "Por padrão, elas serão removidas antes do PCA."
                    )
            else:
                state.warnings.append(
                    "A checagem de variância zero foi pulada porque a matriz contém valor não finito. "
                    "Corrija esse erro e recarregue o CSV."
                )

        # Limite matemático.
        if n_rows >= 2 and n_vars >= 1:
            max_pc = max(1, min(n_rows - 1, n_vars))
            self.var_n_components.set(min(2, max_pc))
            self.spin_components.configure(from_=1, to=max_pc)

        self._set_progress(86, "Finalizando validação da matriz...")
        return state

    def refresh_validation_report(self):
        state = self.load_state
        if state.original_df is None:
            self.validation_text.set_text("Carregue uma matriz CSV para iniciar a validação.")
            return

        df = state.original_df
        n_rows = len(df)
        n_unique = df["obra"].nunique(dropna=True) if "obra" in df.columns else 0
        n_vars = len(state.feature_cols)
        role_counts = self._role_counts() if hasattr(self, "obra_role_vars") else {}
        n_ref = role_counts.get(ROLE_REFERENCE, n_rows)
        n_sup = role_counts.get(ROLE_SUPPLEMENTARY, 0)
        n_excl = role_counts.get(ROLE_EXCLUDED, 0)
        max_pc = self._max_pc_from_current_roles(n_vars) if hasattr(self, "obra_role_vars") else (max(0, min(n_rows - 1, n_vars)) if n_rows >= 2 and n_vars >= 1 else 0)

        lines = []
        lines.append("RELATÓRIO DE VALIDAÇÃO")
        lines.append("=" * 80)
        lines.append(f"Arquivo: {state.path}")
        lines.append(f"Encoding detectado: {state.encoding}")
        lines.append(f"Separador detectado: {repr(state.delimiter)}")
        lines.append("")
        lines.append("Estrutura:")
        lines.append(f"  - linhas/obras: {n_rows}")
        lines.append(f"  - obras distintas: {n_unique}")
        lines.append(f"  - variáveis espectrais: {n_vars}")
        lines.append(f"  - famílias/componentes detectados: {len(state.families)}")
        lines.append(f"  - obras Referência: {n_ref}")
        lines.append(f"  - obras Suplementares: {n_sup}")
        lines.append(f"  - obras Excluídas: {n_excl}")
        lines.append(f"  - máximo útil de PCs pelas Referências: {max_pc}")
        lines.append("")
        lines.append("Papel atual das obras no PCA:")
        roles = self.get_obra_roles() if hasattr(self, "get_obra_roles") else [ROLE_REFERENCE] * n_rows
        if "obra" in df.columns:
            for obra, role in zip(df["obra"].astype(str).tolist(), roles):
                lines.append(f"  - {obra}: {role}")
        else:
            lines.append("  - não disponível")
        lines.append("")

        lines.append("Famílias/componentes detectados:")
        if state.families:
            for fam in state.families:
                n_cols = sum(1 for c in state.feature_cols if state.col_to_family.get(c) == fam)
                lines.append(f"  - {fam} ({n_cols} variável/is)")
        else:
            lines.append("  - nenhuma família detectada")
        lines.append("")

        lines.append("Checagens obrigatórias:")
        lines.append(f"  1. coluna obra/Obra: {'OK' if 'obra' in df.columns else 'ERRO'}")
        lines.append(f"  2. pelo menos 2 obras: {'OK' if n_rows >= 2 and n_unique >= 2 else 'ERRO'}")
        lines.append(f"  3. valores não numéricos: {'ERRO' if state.non_numeric_positions else 'OK'}")
        lines.append(f"  4. células vazias: {'ERRO' if sum(state.empty_counts.values()) > 0 else 'OK'}")
        duplicate_obra_status = 'ERRO' if ('obra' in df.columns and df['obra'].duplicated(keep=False).any()) else 'OK'
        blank_obra_status = 'ERRO' if ('obra' in df.columns and df['obra'].astype(str).str.strip().eq('').any()) else 'OK'
        lines.append(f"  5. nomes de obras vazios: {blank_obra_status}")
        lines.append(f"  6. nomes de obras repetidos: {duplicate_obra_status}")
        lines.append(f"  7. colunas duplicadas: {'ERRO' if state.header_duplicates or state.stripped_duplicates else 'OK'}")
        lines.append(f"  8. possíveis famílias repetidas: {'ALERTA' if state.repeated_family_variants else 'OK'}")
        lines.append(f"  9. número suficiente de Referências para PCA: {'OK' if max_pc >= 1 else 'ERRO'}")
        lines.append(f"  10. mais variáveis que obras: {'ALERTA' if n_vars > n_rows else 'OK'}")
        lines.append(f"  11. variância zero: {'ALERTA' if state.zero_variance_cols else 'OK'}")
        lines.append("")

        if state.errors:
            lines.append("ERROS QUE IMPEDEM PCA:")
            for err in state.errors:
                lines.append(f"  - {err}")
            lines.append("")
        else:
            lines.append("ERROS QUE IMPEDEM PCA: nenhum")
            lines.append("")

        if state.warnings:
            lines.append("ALERTAS:")
            for warn in state.warnings:
                lines.append(f"  - {warn}")
            lines.append("")
        else:
            lines.append("ALERTAS: nenhum")
            lines.append("")

        if state.repeated_family_variants:
            lines.append("TRATAMENTO DE FAMÍLIAS REPETIDAS:")
            lines.append("  O padrão desta versão é usar separadamente. Nenhuma média, soma ou exclusão é feita automaticamente.")
            for base, variants in state.repeated_family_variants.items():
                lines.append(f"  - {base}: {', '.join(variants)}")
            lines.append("")

        if state.zero_variance_cols:
            lines.append("VARIÁVEIS COM VARIÂNCIA ZERO, primeiros 40 nomes:")
            for col in state.zero_variance_cols[:40]:
                lines.append(f"  - {col}")
            if len(state.zero_variance_cols) > 40:
                lines.append(f"  ... +{len(state.zero_variance_cols) - 40} variável/is")

        self.validation_text.set_text("\n".join(lines))
        self._update_summary_labels()

    def _update_summary_labels(self):
        state = self.load_state
        if state.original_df is None:
            self.lbl_file.configure(text="Nenhum arquivo carregado")
            self.lbl_summary.configure(text="Aguardando arquivo.")
            self.lbl_pc_limit.configure(text="Limite: n/d")
            if hasattr(self, "lbl_role_summary"):
                self.lbl_role_summary.configure(text="Carregue um CSV para definir o papel das obras.")
            return
        df = state.original_df
        n_rows = len(df)
        n_unique = df["obra"].nunique(dropna=True) if "obra" in df.columns else 0
        n_vars = len(state.feature_cols)
        role_counts = self._role_counts() if hasattr(self, "obra_role_vars") else {}
        n_ref = role_counts.get(ROLE_REFERENCE, n_rows)
        n_sup = role_counts.get(ROLE_SUPPLEMENTARY, 0)
        n_excl = role_counts.get(ROLE_EXCLUDED, 0)
        max_pc = self._max_pc_from_current_roles(n_vars) if hasattr(self, "obra_role_vars") else (max(0, min(n_rows - 1, n_vars)) if n_rows >= 2 and n_vars >= 1 else 0)
        self.lbl_file.configure(text=str(state.path.name if state.path else "Arquivo carregado"))
        self.lbl_summary.configure(
            text=(
                f"Obras/linhas: {n_rows}\n"
                f"Obras distintas: {n_unique}\n"
                f"Referência: {n_ref} | Suplementar: {n_sup} | Excluída: {n_excl}\n"
                f"Variáveis: {n_vars}\n"
                f"Famílias/componentes: {len(state.families)}\n"
                f"Erros: {len(state.errors)} | Alertas: {len(state.warnings)}"
            )
        )
        self.lbl_pc_limit.configure(text=f"Limite: n_components ≤ min(n_referências - 1, n_variáveis) = {max_pc}")
        if hasattr(self, "_update_role_summary_label"):
            self._update_role_summary_label()
        if max_pc >= 1:
            self.spin_components.configure(from_=1, to=max_pc)
            current = int(self.var_n_components.get() or 1)
            self.var_n_components.set(min(max(current, 1), max_pc))
        else:
            self.spin_components.configure(from_=1, to=1)


    # ------------------------------------------------------------------
    # Seleção de famílias
    # ------------------------------------------------------------------

    def _populate_family_checkboxes(self):
        for child in self.family_container.winfo_children():
            child.destroy()
        self.family_vars.clear()

        families = self.load_state.families
        if not families:
            ttk.Label(self.family_container, text="Nenhuma família/componente detectado.").pack(anchor="w")
            return

        # Seleção inicial: núcleo comum sem restauro/artefato/subfração.
        default_selected = [f for f in families if family_is_core(f) and not family_has_excluded_keyword(f)]
        # Para matriz por componente (H00, H01...), não há núcleo comum: selecionar tudo que não é excluído.
        if not default_selected:
            default_selected = [f for f in families if not family_has_excluded_keyword(f)]

        for fam in families:
            var = tk.BooleanVar(value=fam in default_selected)
            self.family_vars[fam] = var
            self._bind_pca_input_trace(var, "Seleção de famílias/componentes alterada.")
            cb = ttk.Checkbutton(self.family_container, text=fam, variable=var)
            cb.pack(anchor="w", fill="x")

    def selected_families(self) -> List[str]:
        return [fam for fam, var in self.family_vars.items() if bool(var.get())]

    def select_all_families(self):
        for var in self.family_vars.values():
            var.set(True)

    def clear_family_selection(self):
        for var in self.family_vars.values():
            var.set(False)

    def select_core_families(self):
        for fam, var in self.family_vars.items():
            var.set(family_is_core(fam) and not family_has_excluded_keyword(fam))

    def remove_excluded_families(self):
        for fam, var in self.family_vars.items():
            if family_has_excluded_keyword(fam):
                var.set(False)

    # ------------------------------------------------------------------
    # Papel das obras no PCA: referência, suplementar ou excluir
    # ------------------------------------------------------------------

    def _populate_obra_role_controls(self):
        """Cria uma linha por obra para definir se entra no PCA ou se será apenas projetada."""
        if not hasattr(self, "obra_role_container"):
            return
        for child in self.obra_role_container.winfo_children():
            child.destroy()
        self.obra_role_vars.clear()

        state = self.load_state
        if state.original_df is None or "obra" not in state.original_df.columns:
            ttk.Label(self.obra_role_container, text="Nenhuma obra carregada.").pack(anchor="w")
            self._update_role_summary_label()
            return

        header = ttk.Frame(self.obra_role_container)
        header.pack(fill="x")
        ttk.Label(header, text="Obra", width=18).pack(side="left", padx=(0, 4))
        ttk.Label(header, text="Papel", width=18).pack(side="left")

        for row_idx, obra in enumerate(state.original_df["obra"].astype(str).tolist()):
            row = ttk.Frame(self.obra_role_container)
            row.pack(fill="x", pady=1)
            label = obra if obra.strip() else f"linha_{row_idx + 2}"
            ttk.Label(row, text=label, width=18).pack(side="left", padx=(0, 4))
            var = tk.StringVar(value=ROLE_REFERENCE)
            self.obra_role_vars[row_idx] = var
            cbo = ttk.Combobox(row, state="readonly", values=ROLE_OPTIONS, textvariable=var, width=18)
            cbo.pack(side="left", fill="x", expand=True)
            cbo.bind("<<ComboboxSelected>>", lambda _event: self._on_obra_roles_changed())

        self._update_role_summary_label()

    def _on_obra_roles_changed(self):
        """Atualiza a GUI quando o papel de uma obra muda.

        A escolha Referência/Suplementar/Excluir só pode ser aplicada de forma
        metodologicamente correta durante uma nova rodada de PCA. Se o usuário
        alterar a lista suspensa depois de já ter rodado o PCA, os resultados
        antigos ficam obsoletos. Para evitar interpretação errada, eles são
        limpos e a interface pede uma nova execução.
        """
        pca_state = getattr(self, "pca_state", None)
        had_results = bool(
            pca_state is not None
            and (
                getattr(pca_state, "matrix_used", None) is not None
                or getattr(pca_state, "scores", None) is not None
                or getattr(pca_state, "score_values", None) is not None
            )
        )

        if had_results:
            self.pca_state = PCAState()
            if hasattr(self, "_clear_results_tabs"):
                self._clear_results_tabs()
            if hasattr(self, "_set_status"):
                self._set_status(
                    "Papel das obras alterado. Rode o PCA novamente para aplicar Referência/Suplementar/Excluir."
                )

        self._update_role_summary_label()
        self._update_summary_labels()

    def set_all_obra_roles(self, role: str):
        if role not in ROLE_OPTIONS:
            return
        for var in self.obra_role_vars.values():
            var.set(role)
        self._on_obra_roles_changed()

    def get_obra_roles(self) -> List[str]:
        """Retorna o papel de cada linha na ordem da matriz original."""
        state = self.load_state
        if state.original_df is None:
            return []
        roles = []
        for row_idx in range(len(state.original_df)):
            var = self.obra_role_vars.get(row_idx)
            role = var.get() if var is not None else ROLE_REFERENCE
            if role not in ROLE_OPTIONS:
                role = ROLE_REFERENCE
            roles.append(role)
        return roles

    def _role_counts(self) -> Dict[str, int]:
        roles = self.get_obra_roles()
        return {role: int(sum(1 for r in roles if r == role)) for role in ROLE_OPTIONS}

    def _reference_row_indices(self) -> List[int]:
        return [i for i, role in enumerate(self.get_obra_roles()) if role == ROLE_REFERENCE]

    def _included_row_indices(self) -> List[int]:
        return [i for i, role in enumerate(self.get_obra_roles()) if role != ROLE_EXCLUDED]

    def _max_pc_from_current_roles(self, n_vars: Optional[int] = None) -> int:
        state = self.load_state
        if state.original_df is None:
            return 0
        ref_indices = self._reference_row_indices()
        if n_vars is None:
            n_vars = len(state.feature_cols)
        return max(0, min(len(ref_indices) - 1, int(n_vars))) if len(ref_indices) >= 2 and n_vars >= 1 else 0

    def _update_role_summary_label(self):
        if not hasattr(self, "lbl_role_summary"):
            return
        state = self.load_state
        if state.original_df is None:
            self.lbl_role_summary.configure(text="Carregue um CSV para definir o papel das obras.")
            return
        counts = self._role_counts()
        max_pc = self._max_pc_from_current_roles()
        self.lbl_role_summary.configure(
            text=(
                f"Referência: {counts.get(ROLE_REFERENCE, 0)} | "
                f"Suplementar: {counts.get(ROLE_SUPPLEMENTARY, 0)} | "
                f"Excluída: {counts.get(ROLE_EXCLUDED, 0)}\n"
                f"Limite atual: n_components ≤ {max_pc} usando apenas as referências."
            )
        )

    # ------------------------------------------------------------------
    # Metadados e agrupamentos visuais
    # ------------------------------------------------------------------

    def load_metadata_csv(self):
        path_str = filedialog.askopenfilename(
            title="Selecionar metadados CSV",
            filetypes=[("CSV", "*.csv"), ("Texto", "*.txt"), ("Todos os arquivos", "*.*")],
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            self._set_progress(5, "Carregando metadados...")
            encoding, sample = try_read_text_sample(path)
            delimiter = detect_delimiter(sample)
            df = pd.read_csv(path, sep=delimiter, encoding=encoding, dtype=str, keep_default_na=False, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            obra_candidates = [c for c in df.columns if str(c).strip().lower() == "obra"]
            if not obra_candidates:
                raise ValueError('O arquivo de metadados precisa ter coluna "obra" ou "Obra".')
            if len(obra_candidates) > 1:
                raise ValueError(f"Há mais de uma coluna identificada como obra nos metadados: {obra_candidates}")
            obra_col = obra_candidates[0]
            if obra_col != "obra":
                df = df.rename(columns={obra_col: "obra"})
            df["obra"] = df["obra"].astype(str).str.strip()
            fields = [c for c in df.columns if c != "obra"]
            warnings = []
            if not fields:
                warnings.append("O arquivo foi carregado, mas não possui campos adicionais além de 'obra'.")
            if df["obra"].duplicated().any():
                dups = sorted(set(df.loc[df["obra"].duplicated(keep=False), "obra"].tolist()))
                warnings.append("Há obras repetidas nos metadados; a primeira ocorrência será usada no merge: " + ", ".join(map(str, dups[:10])))
                df = df.drop_duplicates(subset=["obra"], keep="first").copy()
            self.metadata_state = MetadataState(
                path=path,
                encoding=encoding,
                delimiter=delimiter,
                df=df,
                fields=fields,
                warnings=warnings,
            )
            self._update_metadata_controls(reset_defaults=True)
            self._set_progress(100, "Metadados carregados.")
            self._reset_progress_later()
            if self.pca_state.scores is not None:
                self.refresh_visualization(silent=True)
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Erro ao carregar metadados", str(exc))
            self._set_progress(0, "Erro ao carregar metadados.")

    def _find_metadata_field(self, *keywords: str) -> Optional[str]:
        state = self.metadata_state
        for field in state.fields:
            norm = str(field).strip().lower()
            if any(k.lower() == norm or k.lower() in norm for k in keywords):
                return field
        return None

    def _update_metadata_controls(self, reset_defaults: bool = False):
        state = self.metadata_state
        options = ["Nenhum", PCA_ROLE_COLUMN] + [f for f in state.fields if f != PCA_ROLE_COLUMN]
        if hasattr(self, "cbo_color_by"):
            self.cbo_color_by.configure(values=options)
            self.cbo_marker_by.configure(values=options)
            self.cbo_ellipse_by.configure(values=options)

        defaults = {
            "color": self._find_metadata_field("fase"),
            "marker": self._find_metadata_field("autenticidade", "status") or PCA_ROLE_COLUMN,
            "ellipse": self._find_metadata_field("fase"),
        }
        if reset_defaults or self.var_color_by.get() not in options:
            self.var_color_by.set(defaults["color"] or "Nenhum")
        if reset_defaults or self.var_marker_by.get() not in options:
            self.var_marker_by.set(defaults["marker"] or PCA_ROLE_COLUMN)
        if reset_defaults or self.var_ellipse_by.get() not in options:
            self.var_ellipse_by.set(defaults["ellipse"] or "Nenhum")

        if hasattr(self, "lbl_metadata"):
            if state.df is None:
                self.lbl_metadata.configure(text="Nenhum arquivo de metadados carregado")
            else:
                txt = (
                    f"Arquivo: {state.path.name if state.path else 'metadados'}\n"
                    f"Obras nos metadados: {len(state.df)}\n"
                    f"Campos disponíveis: {', '.join(state.fields) if state.fields else 'nenhum'}"
                )
                if state.warnings:
                    txt += "\nAlertas: " + " | ".join(state.warnings)
                self.lbl_metadata.configure(text=txt)

    def refresh_visualization(self, silent: bool = False):
        if self.pca_state.scores is None:
            if not silent:
                messagebox.showinfo("Sem PCA", "Rode o PCA antes de atualizar a visualização.")
            return
        try:
            self._set_progress(50, "Atualizando gráficos com metadados e elipses...")
            self._render_results()
            self._set_progress(100, "Visualização atualizada.")
            self._reset_progress_later()
        except Exception as exc:
            traceback.print_exc()
            if not silent:
                messagebox.showerror("Erro ao atualizar visualização", str(exc))
            self._set_progress(0, "Erro ao atualizar visualização.")

    def _scores_with_metadata(self) -> pd.DataFrame:
        if self.pca_state.scores is None:
            return pd.DataFrame()
        base = self.pca_state.scores.copy()
        meta = self.metadata_state.df
        if meta is None or meta.empty:
            return base
        meta_use = meta.copy()
        # Evita duplicar a coluna interna papel_pca se o CSV de metadados tiver o mesmo nome.
        if PCA_ROLE_COLUMN in meta_use.columns:
            meta_use = meta_use.drop(columns=[PCA_ROLE_COLUMN])
        merged = base.merge(meta_use, on="obra", how="left")
        return merged

    def _build_group_summary(self) -> pd.DataFrame:
        merged = self._scores_with_metadata()
        if merged.empty or self.metadata_state.df is None:
            return pd.DataFrame()
        rows = []
        has_pc3 = bool(
            self.pca_state.score_values is not None
            and self.pca_state.score_values.ndim == 2
            and self.pca_state.score_values.shape[1] >= 3
        )
        ellipse_field = self.var_ellipse_by.get()
        fields_to_describe = []
        for value in [self.var_color_by.get(), self.var_marker_by.get(), ellipse_field]:
            if value and value != "Nenhum" and value in merged.columns and value not in fields_to_describe:
                fields_to_describe.append(value)
        for field in fields_to_describe:
            counts = merged[field].fillna("Sem grupo").astype(str).value_counts(dropna=False)
            for group, count in counts.items():
                ellipse_drawn = "não"
                note = ""
                ellipsoid_drawn = "não"
                ellipsoid_note = ""
                if field == ellipse_field:
                    if count >= 3:
                        ellipse_drawn = "sim"
                        note = "elipse 95% por χ²(2)"
                    elif count == 2:
                        ellipse_drawn = "eixo principal"
                        note = "2 obras; somente eixo maior 95% por χ²(2), pontilhado"
                    elif count == 1:
                        note = "1 obra; sem elipse/eixo"
                    if not has_pc3:
                        ellipsoid_note = "rodada sem PC3; região 3D não disponível"
                    elif count >= 3:
                        ellipsoid_drawn = "sim"
                        ellipsoid_note = "elipsoide 95% por χ²(3)"
                        if count == 3:
                            ellipsoid_note += "; covariância pode ser degenerada/planar"
                    elif count == 2:
                        ellipsoid_drawn = "eixo principal"
                        ellipsoid_note = "2 obras; somente eixo maior 95% por χ²(3), pontilhado"
                    elif count == 1:
                        ellipsoid_note = "1 obra; sem elipsoide/eixo"
                rows.append({
                    "criterio": field,
                    "grupo": group,
                    "n_obras": int(count),
                    "elipse_desenhada": ellipse_drawn,
                    "observacao": note,
                    "elipsoide_3D_desenhado": ellipsoid_drawn,
                    "observacao_elipsoide_3D": ellipsoid_note,
                })
        return pd.DataFrame(rows)

    def _draw_group_ellipse(
        self,
        ax,
        points: np.ndarray,
        color: str,
        overall_scale: float,
        scale_mode: str = SYNTHESIS_SCALE_LINEAR,
        factor: float = 1.0,
    ):
        """Desenha região 2D de 95% por qui-quadrado no espaço original dos scores.

        - n >= 3: contorno elíptico com semi-eixos sqrt(χ²_0.95,2 * λ_i).
        - n == 2: desenha somente o eixo principal maior, com o mesmo semi-comprimento
          sqrt(χ²_0.95,2 * λ_1), em linha pontilhada.

        A escala asinh, quando ativa, é aplicada apenas depois da construção da
        geometria no espaço original do PCA.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        n_points = int(len(pts))
        if n_points < 2:
            return None

        center = pts.mean(axis=0)
        cov = np.asarray(np.cov(pts, rowvar=False, ddof=1), dtype=float)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        if float(eigvals[0]) <= 1e-15:
            return None

        if n_points == 2:
            semilength = math.sqrt(CHI2_95_2D * float(eigvals[0]))
            t = np.linspace(-semilength, semilength, 240)
            axis_raw = center[None, :] + t[:, None] * eigvecs[:, 0][None, :]
            axis_plot = self._synthesis_visual_transform(axis_raw, scale_mode, factor)
            line, = ax.plot(
                axis_plot[:, 0],
                axis_plot[:, 1],
                color=color,
                linewidth=1.8,
                alpha=0.92,
                linestyle="--",
            )
            return {
                "artist": line,
                "tipo": "eixo_principal_95",
                "n_obras": n_points,
                "threshold": CHI2_95_2D,
                "autovalor_principal": float(eigvals[0]),
                "semicomprimento": float(semilength),
            }

        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        unit_circle = np.vstack([np.cos(theta), np.sin(theta)])
        radii = np.sqrt(CHI2_95_2D * eigvals)
        ellipse_raw = center[:, None] + eigvecs @ (radii[:, None] * unit_circle)
        ellipse_plot = self._synthesis_visual_transform(ellipse_raw.T, scale_mode, factor)
        line, = ax.plot(
            ellipse_plot[:, 0],
            ellipse_plot[:, 1],
            color=color,
            linewidth=1.8,
            alpha=0.92,
            linestyle="-",
        )
        return {
            "artist": line,
            "tipo": "elipse_95",
            "n_obras": n_points,
            "threshold": CHI2_95_2D,
            "autovalores": eigvals,
            "semieixos": radii,
        }

    def _draw_group_ellipsoid_3d(
        self,
        ax,
        points: np.ndarray,
        color: str,
        overall_scale: float,
        scale_mode: str = SYNTHESIS_SCALE_LINEAR,
        factor: float = 1.0,
    ):
        """Desenha região 3D de 95% por qui-quadrado no espaço original dos scores.

        - n >= 3: elipsoide com semi-eixos sqrt(χ²_0.95,3 * λ_i).
        - n == 2: somente o eixo principal maior, com o mesmo critério χ²(3),
          desenhado como linha pontilhada.

        Com exatamente três obras a covariância 3D tem posto no máximo 2; portanto,
        o elipsoide pode ser matematicamente degenerado/planar. Nenhuma espessura
        artificial é adicionada. A escala asinh é aplicada somente após a construção
        da geometria no espaço original do PCA.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return None
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        n_points = int(len(pts))
        if n_points < 2:
            return None

        center = pts.mean(axis=0)
        cov = np.asarray(np.cov(pts, rowvar=False, ddof=1), dtype=float)
        if cov.shape != (3, 3) or not np.isfinite(cov).all():
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        basis = eigvecs[:, order]
        if float(eigvals[0]) <= 1e-15:
            return None

        if n_points == 2:
            semilength = math.sqrt(CHI2_95_3D * float(eigvals[0]))
            t = np.linspace(-semilength, semilength, 240)
            axis_raw = center[None, :] + t[:, None] * basis[:, 0][None, :]
            axis_plot = self._synthesis_visual_transform(axis_raw, scale_mode, factor)
            line, = ax.plot(
                axis_plot[:, 0],
                axis_plot[:, 1],
                axis_plot[:, 2],
                color=color,
                linewidth=1.9,
                alpha=0.95,
                linestyle="--",
            )
            return {
                "line": line,
                "n_obras": n_points,
                "tipo": "eixo_principal_95",
                "aproximado": False,
                "degenerado": True,
                "threshold": CHI2_95_3D,
                "autovalor_principal": float(eigvals[0]),
                "semicomprimento": float(semilength),
            }

        radii = np.sqrt(CHI2_95_3D * eigvals)
        positive_rank = int(np.sum(eigvals > max(float(eigvals[0]) * 1e-12, 1e-15)))
        degenerate = positive_rank < 3

        u = np.linspace(0.0, 2.0 * np.pi, 49)
        v = np.linspace(0.0, np.pi, 25)
        sphere = np.stack(
            [
                np.outer(np.cos(u), np.sin(v)),
                np.outer(np.sin(u), np.sin(v)),
                np.outer(np.ones_like(u), np.cos(v)),
            ],
            axis=0,
        )
        sphere_flat = sphere.reshape(3, -1)
        transformed_raw = center[:, None] + basis @ (radii[:, None] * sphere_flat)
        transformed_plot = self._synthesis_visual_transform(transformed_raw.T, scale_mode, factor).T
        x_mesh = transformed_plot[0].reshape(sphere.shape[1:])
        y_mesh = transformed_plot[1].reshape(sphere.shape[1:])
        z_mesh = transformed_plot[2].reshape(sphere.shape[1:])

        surface = ax.plot_surface(
            x_mesh,
            y_mesh,
            z_mesh,
            color=color,
            alpha=0.09 if degenerate else 0.13,
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )
        wireframe = ax.plot_wireframe(
            x_mesh,
            y_mesh,
            z_mesh,
            rstride=4,
            cstride=4,
            color=color,
            alpha=0.34 if degenerate else 0.24,
            linewidth=0.42,
            linestyle="--" if degenerate else "-",
        )
        return {
            "surface": surface,
            "wireframe": wireframe,
            "n_obras": n_points,
            "tipo": "elipsoide_95",
            "aproximado": False,
            "degenerado": degenerate,
            "rank_covariancia": positive_rank,
            "threshold": CHI2_95_3D,
            "autovalores": eigvals,
            "semieixos": radii,
        }

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def _ensure_loaded_and_valid(self) -> bool:
        state = self.load_state
        if state.original_df is None or state.numeric_df is None:
            messagebox.showwarning("Sem matriz", "Carregue uma matriz CSV antes de rodar o PCA.")
            return False
        if state.errors:
            messagebox.showerror(
                "Validação impediu o PCA",
                "Corrija os erros da matriz antes de rodar o PCA. Veja a aba Validação.",
            )
            self.notebook.select(self.tab_validation)
            return False
        return True

    def _get_significant_digits_setting(self) -> int:
        """Lê e valida a quantidade de dígitos significativos solicitada na GUI."""
        try:
            digits = int(self.var_sig_digits.get())
        except Exception as exc:
            raise ValueError("Informe um número inteiro de dígitos significativos.") from exc
        if digits < 3 or digits > 15:
            raise ValueError("A quantidade de dígitos significativos deve ficar entre 3 e 15.")
        return digits

    def _prepare_matrix_for_pca(
        self,
    ) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, pd.DataFrame]:
        """
        Prepara a matriz comum para PCA/projeção.

        Diferença metodológica importante desta versão:
        - linhas Referência definem variância zero, z-score e PCA;
        - linhas Suplementar recebem exatamente as mesmas transformações e são projetadas depois;
        - linhas Excluir não entram na rodada.
        """
        state = self.load_state
        assert state.original_df is not None
        assert state.numeric_df is not None

        selected_fams = self.selected_families()
        if not selected_fams:
            raise ValueError("Nenhuma família/componente foi selecionada para o PCA.")

        selected_columns = [
            col for col in state.feature_cols if state.col_to_family.get(col) in selected_fams
        ]
        if not selected_columns:
            raise ValueError("A seleção de famílias não contém nenhuma variável espectral.")

        removal_reasons: Dict[str, str] = {col: "" for col in selected_columns}
        zscore_mean_map: Dict[str, float] = {}
        zscore_std_map: Dict[str, float] = {}
        zscore_scale_map: Dict[str, float] = {}
        area_normalization_audit = pd.DataFrame()

        roles_all = self.get_obra_roles()
        included_indices = [i for i, role in enumerate(roles_all) if role != ROLE_EXCLUDED]
        reference_indices = [i for i, role in enumerate(roles_all) if role == ROLE_REFERENCE]
        supplementary_indices = [i for i, role in enumerate(roles_all) if role == ROLE_SUPPLEMENTARY]

        if len(reference_indices) < 2:
            raise ValueError(
                "PCA com amostras suplementares exige pelo menos duas obras marcadas como Referência. "
                "Marque o conjunto que definirá o modelo como Referência e as demais observações como Suplementar."
            )
        if not included_indices:
            raise ValueError("Nenhuma obra foi incluída na rodada. Marque pelo menos duas obras como Referência.")

        X_all_original = state.numeric_df[selected_columns].copy()
        if bool(X_all_original.isna().any().any()):
            raise ValueError("A matriz preparada ainda contém NaN. Verifique células vazias ou não numéricas.")
        if not np.isfinite(X_all_original.to_numpy(dtype=float)).all():
            raise ValueError("A matriz preparada contém valores infinitos ou não finitos. Revise o CSV antes do PCA.")

        X = X_all_original.iloc[included_indices].copy()
        roles_included = [roles_all[i] for i in included_indices]
        obras_included = state.original_df.iloc[included_indices]["obra"].astype(str).tolist()
        ref_pos = [pos for pos, role in enumerate(roles_included) if role == ROLE_REFERENCE]

        if bool(self.var_normalize_family_area.get()):
            selected_col_to_family = {
                col: state.col_to_family.get(col, extract_family_from_variable(col))
                for col in selected_columns
            }
            X, area_normalization_audit = normalize_spectral_groups_by_area(
                X,
                selected_col_to_family,
            )
            if not area_normalization_audit.empty:
                obra_by_index = dict(zip(X.index.tolist(), obras_included))
                role_by_index = dict(zip(X.index.tolist(), roles_included))
                area_normalization_audit.insert(
                    1,
                    "obra",
                    area_normalization_audit["indice_linha_original"].map(obra_by_index),
                )
                area_normalization_audit.insert(
                    2,
                    PCA_ROLE_COLUMN,
                    area_normalization_audit["indice_linha_original"].map(role_by_index),
                )

        if bool(self.var_normalize_rows.get()):
            row_sums = X.sum(axis=1)
            bad_mask = (~np.isfinite(row_sums.values)) | np.isclose(row_sums.values, 0) | (row_sums.values <= 0)
            bad_rows = row_sums.index[bad_mask].tolist()
            if bad_rows:
                obras_bad = state.original_df.loc[bad_rows, "obra"].tolist()
                raise ValueError(
                    "Não é possível normalizar por soma porque há linha(s) com soma menor ou igual a zero, "
                    "zero numérico ou valor não finito: " + ", ".join(map(str, obras_bad))
                )
            if bool((X < 0).any().any()):
                obras_neg = state.original_df.loc[X.index[(X < 0).any(axis=1)].tolist(), "obra"].astype(str).tolist()
                raise ValueError(
                    "A normalização por soma foi bloqueada porque há valores negativos na(s) obra(s): "
                    + ", ".join(obras_neg)
                    + ". Revise a matriz ou desative a normalização por soma."
                )
            X = X.div(row_sums, axis=0)

        removed_zero: List[str] = []

        # Variância zero é avaliada apenas nas Referências, pois só elas definem o espaço PCA.
        if bool(self.var_remove_zero_var.get()):
            X_ref_for_var = X.iloc[ref_pos]
            variances = X_ref_for_var.var(axis=0, ddof=0, skipna=True)
            removed_zero = variances.index[(variances.fillna(0) == 0)].tolist()
            if removed_zero:
                for col in removed_zero:
                    removal_reasons[col] = "variancia_zero_referencias_pre_zscore"
                X = X.drop(columns=removed_zero)

        if X.shape[1] < 1:
            raise ValueError("Após remover variância zero nas Referências, nenhuma variável restou para o PCA.")

        if bool(self.var_zscore.get()):
            scaler = StandardScaler(with_mean=True, with_std=True)
            zscore_columns = list(X.columns)
            X_ref_df = X.iloc[ref_pos].copy()
            zscore_means = X_ref_df.mean(axis=0)
            zscore_stds = X_ref_df.std(axis=0, ddof=0)
            X_ref = X_ref_df.values.astype(float)
            scaler.fit(X_ref)
            zscore_mean_map = {col: float(zscore_means[col]) for col in zscore_columns}
            zscore_std_map = {col: float(zscore_stds[col]) for col in zscore_columns}
            zscore_scale_map = {col: float(scale) for col, scale in zip(zscore_columns, scaler.scale_)}
            X_values = scaler.transform(X.values.astype(float))
            X = pd.DataFrame(X_values, columns=X.columns, index=X.index)

        # Controle explícito de precisão numérica para testes de robustez.
        # Aplicado à matriz final usada para PCA/projeção. Se uma variável virar constante
        # nas Referências após o arredondamento, ela é removida quando a opção estiver ativa.
        if bool(self.var_apply_sig_digits.get()):
            digits = self._get_significant_digits_setting()
            X = round_dataframe_to_significant_digits(X, digits)

            if bool(self.var_remove_zero_var.get()):
                X_ref_after_digits = X.iloc[ref_pos]
                variances_after_digits = X_ref_after_digits.var(axis=0, ddof=0, skipna=True)
                removed_after_digits = variances_after_digits.index[(variances_after_digits.fillna(0) == 0)].tolist()
                if removed_after_digits:
                    for col in removed_after_digits:
                        removal_reasons[col] = "variancia_zero_referencias_apos_digitos"
                    X = X.drop(columns=removed_after_digits)
                    for col in removed_after_digits:
                        if col not in removed_zero:
                            removed_zero.append(col)

            if X.shape[1] < 1:
                raise ValueError(
                    "Após aplicar dígitos significativos e remover variância zero nas Referências, "
                    "nenhuma variável restou para o PCA. Aumente os dígitos ou revise a seleção."
                )

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("A matriz final usada no PCA contém valores não finitos após as transformações.")

        matrix_used = X.copy()
        matrix_used.insert(0, PCA_ROLE_COLUMN, roles_included)
        matrix_used.insert(0, "obra", obras_included)

        final_feature_cols = list(X.columns)
        X_ref_final = X.iloc[ref_pos].copy()
        final_mean = X_ref_final.mean(axis=0)
        final_std = X_ref_final.std(axis=0, ddof=0)
        final_min = X_ref_final.min(axis=0)
        final_max = X_ref_final.max(axis=0)
        preprocess_rows: List[Dict[str, object]] = []
        for col in selected_columns:
            used = col in final_feature_cols
            preprocess_rows.append({
                "variavel": col,
                "familia_componente": state.col_to_family.get(col, ""),
                "usada_no_PCA": bool(used),
                "motivo_remocao": removal_reasons.get(col, "") if not used else "",
                "normalizacao_area_individual_H_aplicada": bool(self.var_normalize_family_area.get()),
                "metodo_normalizacao_area_H": (
                    "integração trapezoidal por obra e família/componente; blocos ausentes permanecem zero"
                    if bool(self.var_normalize_family_area.get()) else "não aplicado"
                ),
                "normalizacao_por_soma_linha_aplicada": bool(self.var_normalize_rows.get()),
                "zscore_aplicado": bool(self.var_zscore.get()),
                "zscore_media_referencias_antes_transformacao": zscore_mean_map.get(col, np.nan),
                "zscore_desvio_referencias_antes_transformacao": zscore_std_map.get(col, np.nan),
                "zscore_escala_usada_sklearn": zscore_scale_map.get(col, np.nan),
                "digitos_significativos_aplicados": bool(self.var_apply_sig_digits.get()),
                "digitos_significativos": self._get_significant_digits_setting() if bool(self.var_apply_sig_digits.get()) else "não aplicado",
                "media_final_referencias_apos_preprocessamento": float(final_mean[col]) if used else np.nan,
                "desvio_final_referencias_apos_preprocessamento": float(final_std[col]) if used else np.nan,
                "min_final_referencias_apos_preprocessamento": float(final_min[col]) if used else np.nan,
                "max_final_referencias_apos_preprocessamento": float(final_max[col]) if used else np.nan,
            })
        preprocess_parameters = pd.DataFrame(preprocess_rows)
        return (
            matrix_used,
            selected_columns,
            removed_zero,
            preprocess_parameters,
            area_normalization_audit,
        )


    def run_pca(self):
        if not self._ensure_loaded_and_valid():
            return
        try:
            self._set_progress(5, "Preparando matriz para PCA/projeção...")
            state = self.load_state
            assert state.original_df is not None

            (
                matrix_used,
                selected_columns,
                removed_zero,
                preprocess_parameters,
                area_normalization_audit,
            ) = self._prepare_matrix_for_pca()
            self._set_progress(25, "Matriz preparada; separando Referências e Suplementares...")
            obras = matrix_used["obra"].astype(str).tolist()
            roles = matrix_used[PCA_ROLE_COLUMN].astype(str).tolist()
            feature_cols = matrix_feature_columns(matrix_used)
            X_all = matrix_used[feature_cols].values.astype(float)
            if has_duplicate_labels(obras):
                raise ValueError(
                    "Há obras incluídas com rótulos repetidos. Use nomes únicos antes de calcular scores e distâncias."
                )
            if not np.isfinite(X_all).all():
                raise ValueError("A matriz usada no PCA contém valores não finitos.")

            ref_mask = np.array([role == ROLE_REFERENCE for role in roles], dtype=bool)
            sup_mask = np.array([role == ROLE_SUPPLEMENTARY for role in roles], dtype=bool)
            obras_ref = [obra for obra, keep in zip(obras, ref_mask) if keep]
            obras_sup = [obra for obra, keep in zip(obras, sup_mask) if keep]

            n_ref = int(ref_mask.sum())
            n_vars = len(feature_cols)
            if n_ref < 2 or len(set(obras_ref)) < 2:
                raise ValueError(
                    "PCA não pode ser ajustado com menos de duas obras de Referência distintas. "
                    "Defina ao menos duas Referências e marque como Suplementar o que será apenas projetado."
                )
            max_components = min(n_ref - 1, n_vars)
            if max_components < 1:
                raise ValueError("Número insuficiente de obras de Referência ou variáveis para PCA.")
            requested = int(self.var_n_components.get() or max_components)
            n_components = min(max(requested, 1), max_components)
            self.var_n_components.set(n_components)
            if bool(self.var_metric_mahalanobis_scores.get()) and n_ref <= n_components + 1:
                warn = (
                    "Mahalanobis foi solicitada com poucas obras de Referência em relação ao número de PCs. "
                    "A métrica será exportada, mas deve ser interpretada como exploratória."
                )
                if warn not in self.load_state.warnings:
                    self.load_state.warnings.append(warn)

            self._set_progress(45, "Ajustando PCA somente nas obras de Referência...")
            pca = PCA(n_components=n_components, svd_solver="full")
            X_ref = X_all[ref_mask]
            pca.fit(X_ref)
            scores_values = pca.transform(X_all)
            explained = pca.explained_variance_ratio_

            self._set_progress(62, "Montando scores, loadings e variância...")
            pc_cols = [f"PC{i + 1}" for i in range(n_components)]
            scores_df = pd.DataFrame(scores_values, columns=pc_cols)
            scores_df.insert(0, PCA_ROLE_COLUMN, roles)
            scores_df.insert(0, "obra", obras)

            loadings_df = pd.DataFrame(pca.components_.T, columns=pc_cols)
            loadings_df.insert(0, "familia_componente", [state.col_to_family.get(col, "") for col in feature_cols])
            loadings_df.insert(0, "variavel", feature_cols)
            loadings_df.insert(0, "tipo_valor", "peso/eigenvector sklearn PCA")

            pca_mean_df = pd.DataFrame({
                "variavel": feature_cols,
                "familia_componente": [state.col_to_family.get(col, "") for col in feature_cols],
                "media_usada_pelo_PCA_referencias": pca.mean_,
            })

            self._set_progress(66, "Calculando contribuição quadrática dos loadings...")
            loading_contrib_df, loading_dominance_df = make_loading_contribution_tables(loadings_df)

            variance_df = pd.DataFrame({
                "PC": pc_cols,
                "variancia_explicada": explained,
                "variancia_explicada_%": explained * 100,
                "variancia_acumulada": np.cumsum(explained),
                "variancia_acumulada_%": np.cumsum(explained) * 100,
                "autovalor_variancia": pca.explained_variance_,
                "singular_value": pca.singular_values_,
                "base_calculo": "Somente obras de Referência",
            })

            self._set_progress(74, "Calculando métricas de distância entre obras...")
            distance_tables, distance_square_tables, combined_distances, closest_pairs, farthest_pairs = self._compute_distance_metrics(
                obras=obras,
                score_values=scores_values,
                matrix_used=matrix_used,
                n_components=n_components,
                roles=roles,
                reference_mask=ref_mask,
            )
            self._set_progress(80, "Registrando configuração da rodada...")
            config_df = self._make_config_dataframe(
                n_components=n_components,
                selected_columns=selected_columns,
                removed_zero=removed_zero,
                matrix_used=matrix_used,
                distance_tables=distance_tables,
            )

            self.pca_state = PCAState(
                selected_families=self.selected_families(),
                selected_columns=selected_columns,
                removed_zero_variance=removed_zero,
                matrix_used=matrix_used,
                scores=scores_df,
                loadings=loadings_df,
                loading_contributions=loading_contrib_df,
                loading_dominance=loading_dominance_df,
                pca_mean=pca_mean_df,
                preprocess_parameters=preprocess_parameters,
                area_normalization_audit=area_normalization_audit,
                variance=variance_df,
                distances=combined_distances,
                distance_tables=distance_tables,
                distance_square_tables=distance_square_tables,
                closest_pairs=closest_pairs,
                farthest_pairs=farthest_pairs,
                config=config_df,
                explained_ratio=explained,
                score_values=scores_values,
                obra_labels=obras,
                reference_obras=obras_ref,
                supplementary_obras=obras_sup,
                excluded_obras=[state.original_df.iloc[i]["obra"] for i, role in enumerate(self.get_obra_roles()) if role == ROLE_EXCLUDED],
            )

            self._prepare_loading_contribution_figures(loading_contrib_df)

            if bool(self.var_metric_dendrogram.get()):
                self._set_progress(84, "Preparando dendrograma das obras de Referência...")
                scores_dendro = scores_values[ref_mask]
                labels_dendro = [obra for obra, keep in zip(obras, ref_mask) if keep]
                self._prepare_dendrogram_figure(scores_values=scores_dendro, obras=labels_dendro)

            self._set_progress(88, "Renderizando gráficos e tabelas na interface...")
            self._render_results()
            self._set_progress(100, "PCA concluído.")
            self._reset_progress_later()
            messagebox.showinfo(
                "PCA concluído",
                (
                    f"PCA ajustado com {n_ref} obra(s) de Referência, "
                    f"{len(obras_sup)} Suplementar(es), {n_components} PC(s) e {n_vars} variável(is)."
                ),
            )
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Erro ao rodar PCA", str(exc))
            self._set_progress(0, "Erro ao rodar PCA.")


    def _compute_distance_metrics(
        self,
        obras: List[str],
        score_values: np.ndarray,
        matrix_used: pd.DataFrame,
        n_components: int,
        roles: Optional[List[str]] = None,
        reference_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Calcula métricas numéricas de diferença entre obras.

        Para Mahalanobis, a covariância é estimada somente nas Referências quando há
        amostras suplementares, evitando que a obra questionada altere a métrica.
        """
        distance_tables: Dict[str, pd.DataFrame] = {}
        distance_square_tables: Dict[str, pd.DataFrame] = {}
        pc_label = f"PC1-PC{n_components}"
        roles = roles or [ROLE_REFERENCE] * len(obras)
        reference_mask = np.asarray(reference_mask, dtype=bool) if reference_mask is not None else np.array([r == ROLE_REFERENCE for r in roles], dtype=bool)

        def _add_roles(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            role_map = {str(obra): role for obra, role in zip(obras, roles)}
            out = df.copy()
            out.insert(3, "papel_obra_1", out["obra_1"].astype(str).map(role_map))
            out.insert(4, "papel_obra_2", out["obra_2"].astype(str).map(role_map))
            return out

        if bool(self.var_metric_euclidean_scores.get()):
            key = f"euclidiana_scores_{pc_label}"
            df = pairwise_euclidean_long(obras, score_values, f"Euclidiana nos scores ({pc_label})")
            df = _add_roles(df)
            distance_tables[key] = df
            distance_square_tables[f"matriz_{key}"] = distance_long_to_square(df, obras, "distancia")

        if bool(self.var_metric_cosine_matrix.get()):
            X_matrix = matrix_used[matrix_feature_columns(matrix_used)].values.astype(float)
            key = "cosseno_matriz_usada"
            df = pairwise_cosine_long(obras, X_matrix, "Cosseno na matriz usada para PCA/projeção")
            df = _add_roles(df)
            distance_tables[key] = df
            distance_square_tables[f"matriz_{key}"] = distance_long_to_square(df, obras, "distancia")

        if bool(self.var_metric_cosine_scores.get()):
            key = f"cosseno_scores_{pc_label}"
            df = pairwise_cosine_long(obras, score_values, f"Cosseno nos scores ({pc_label})")
            df = _add_roles(df)
            distance_tables[key] = df
            distance_square_tables[f"matriz_{key}"] = distance_long_to_square(df, obras, "distancia")

        if bool(self.var_metric_mahalanobis_scores.get()):
            key = f"mahalanobis_scores_{pc_label}_cov_referencias"
            cov_values = score_values[reference_mask] if int(reference_mask.sum()) >= 2 else score_values
            df = pairwise_mahalanobis_long(
                obras,
                score_values,
                f"Mahalanobis nos scores ({pc_label}; covariância das Referências)",
                covariance_values=cov_values,
            )
            df = _add_roles(df)
            distance_tables[key] = df
            distance_square_tables[f"matriz_{key}"] = distance_long_to_square(df, obras, "distancia")

        if not distance_tables:
            # Evita uma aba vazia caso o usuário desmarque tudo.
            key = f"euclidiana_scores_{pc_label}"
            df = pairwise_euclidean_long(obras, score_values, f"Euclidiana nos scores ({pc_label})")
            df = _add_roles(df)
            distance_tables[key] = df
            distance_square_tables[f"matriz_{key}"] = distance_long_to_square(df, obras, "distancia")

        combined = pd.concat(distance_tables.values(), ignore_index=True) if distance_tables else pd.DataFrame()
        closest, farthest = make_pairs_rank_tables(distance_tables)
        return distance_tables, distance_square_tables, combined, closest, farthest



    @staticmethod
    def _draw_signed_loading_contribution_axis(
        ax,
        sub: pd.DataFrame,
        title: str,
        show_legend: bool = True,
    ):
        """Desenha contribuição quadrática separando canais por sinal do loading.

        As parcelas à esquerda e à direita são magnitudes não negativas calculadas
        com loading². O lado do gráfico conserva apenas o sinal original do loading;
        o rótulo entre colchetes conserva a contribuição total do grupo.
        """
        ordered = sub.iloc[::-1].copy()
        labels = ordered["grupo_variavel"].astype(str).tolist()
        totals = pd.to_numeric(
            ordered["contribuicao_percentual"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        positive = pd.to_numeric(
            ordered["contribuicao_positiva_percentual"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        negative = pd.to_numeric(
            ordered["contribuicao_negativa_percentual"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)

        y = np.arange(len(labels))
        ax.barh(
            y,
            positive,
            color=LOADING_POSITIVE_COLOR,
            alpha=0.92,
            label="Canais com loading positivo (+)",
        )
        ax.barh(
            y,
            -negative,
            color=LOADING_NEGATIVE_COLOR,
            alpha=0.88,
            label="Canais com loading negativo (−)",
        )
        display_labels = [
            f"{label}  [{total:.1f}%]".replace(".", ",")
            for label, total in zip(labels, totals)
        ]
        ax.set_yticks(y)
        ax.set_yticklabels(display_labels, fontsize=8)
        ax.axvline(0.0, color="#444444", linewidth=0.9, zorder=3)
        ax.set_xlabel("Contribuição quadrática por sinal (%)   ← negativo | positivo →")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_axisbelow(True)

        for yi, pos_value, neg_value in zip(y, positive, negative):
            if np.isfinite(pos_value) and pos_value >= 0.05:
                ax.annotate(
                    (f"{pos_value:.1f}% (+)").replace(".", ","),
                    xy=(pos_value, yi),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    color=LOADING_POSITIVE_COLOR,
                )
            if np.isfinite(neg_value) and neg_value >= 0.05:
                ax.annotate(
                    (f"{neg_value:.1f}% (−)").replace(".", ","),
                    xy=(-neg_value, yi),
                    xytext=(-4, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=7.5,
                    color=LOADING_NEGATIVE_COLOR,
                )

        max_side = max(
            float(np.nanmax(positive)) if positive.size else 0.0,
            float(np.nanmax(negative)) if negative.size else 0.0,
            1.0,
        )
        axis_limit = max(5.0, max_side * 1.32)
        ax.set_xlim(-axis_limit, axis_limit)
        # Os dois lados mostram magnitudes de contribuição (sempre não negativas);
        # esquerda/direita codificam somente o sinal original dos loadings.
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{abs(value):g}"))
        if show_legend:
            ax.legend(loc="lower right", fontsize=8, frameon=True)


    def _prepare_loading_contribution_figures(self, loading_contrib_df: pd.DataFrame, top_n: int = 12, max_pcs: int = 3):
        """Cria gráficos por PC com contribuição total decomposta por sinal."""
        if loading_contrib_df is None or loading_contrib_df.empty:
            return
        required = {
            "PC", "grupo_variavel", "contribuicao_percentual", "rank_no_PC",
            "contribuicao_positiva_percentual", "contribuicao_negativa_percentual",
        }
        if not required.issubset(set(loading_contrib_df.columns)):
            return

        pc_order = [str(pc) for pc in loading_contrib_df["PC"].drop_duplicates().tolist()]
        for pc in pc_order[:max(1, int(max_pcs))]:
            sub = loading_contrib_df[loading_contrib_df["PC"].astype(str) == str(pc)]
            sub = sub.sort_values("rank_no_PC", ascending=True).head(int(top_n)).copy()
            if sub.empty:
                continue

            fig = Figure(figsize=(10.4, 5.6), dpi=100)
            ax = fig.add_subplot(111)
            self._draw_signed_loading_contribution_axis(
                ax,
                sub,
                title=f"Contribuição quadrática dos loadings — {pc}",
                show_legend=True,
            )
            fig.text(
                0.5,
                0.01,
                "Entre colchetes: contribuição total da família. O sinal global do eixo PCA é arbitrário.",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            self.pca_state.figures[f"grafico_contribuicao_loadings_{safe_slug(pc)}"] = fig


    def _render_loading_contribution_view(self, top_n: int = 10, max_pcs: int = 3):
        """Mostra, em aba própria, os gráficos de contribuição quadrática dos loadings.

        Esta visualização foi criada para comparar rapidamente se PC1/PC2/PC3 estão
        dominados por uma família/componente/grupo ou se a contribuição está mais
        distribuída. A tabela completa continua disponível em CSV.
        """
        tab = self.tab_loading_contrib
        for child in tab.winfo_children():
            child.destroy()

        contrib = self.pca_state.loading_contributions
        if contrib is None or contrib.empty:
            ttk.Label(
                tab,
                text="A contribuição dos loadings ainda não está disponível. Rode o PCA novamente.",
            ).pack(anchor="center", pady=30)
            return

        required = {
            "PC", "grupo_variavel", "contribuicao_percentual", "rank_no_PC",
            "contribuicao_positiva_percentual", "contribuicao_negativa_percentual",
        }
        if not required.issubset(set(contrib.columns)):
            ttk.Label(tab, text="Tabela de contribuição dos loadings incompleta.").pack(anchor="center", pady=30)
            return

        pc_order = [str(pc) for pc in contrib["PC"].drop_duplicates().tolist()]
        pc_order = pc_order[:max(1, int(max_pcs))]
        n_pcs = len(pc_order)
        if n_pcs == 0:
            ttk.Label(tab, text="Nenhum PC disponível para visualização dos loadings.").pack(anchor="center", pady=30)
            return

        fig = Figure(figsize=(10.8, max(3.6, 3.35 * n_pcs)), dpi=100)
        axes = fig.subplots(n_pcs, 1, squeeze=False).ravel()
        dominance = self.pca_state.loading_dominance

        for ax, pc in zip(axes, pc_order):
            sub = contrib[contrib["PC"].astype(str) == pc].sort_values("rank_no_PC").head(int(top_n)).copy()
            title = f"{pc} — contribuição quadrática dos loadings"
            if dominance is not None and not dominance.empty and "PC" in dominance.columns:
                row = dominance[dominance["PC"].astype(str) == pc]
                if not row.empty:
                    idx = str(row.iloc[0].get("indice_dominancia", ""))
                    top1 = row.iloc[0].get("maior_contribuicao_percentual", np.nan)
                    try:
                        title += f" | dominância: {idx} | maior: {float(top1):.1f}%".replace(".", ",")
                    except Exception:
                        title += f" | dominância: {idx}"
            self._draw_signed_loading_contribution_axis(
                ax,
                sub,
                title=title,
                show_legend=(ax is axes[0]),
            )

        fig.suptitle(
            "Contribuição dos loadings por PC — direção dos canais (+/−)",
            fontsize=12,
        )
        fig.text(
            0.5,
            0.006,
            "Entre colchetes: contribuição total da família. O sinal global de cada PC pode inverter sem alterar a PCA.",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.025, 1, 0.965))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.pca_state.figures["figura_contribuicao_loadings_resumo"] = fig

    def _prepare_dendrogram_figure(self, scores_values: np.ndarray, obras: List[str]):
        """Cria dendrograma hierárquico a partir dos scores gerados."""
        if linkage is None or dendrogram is None:
            self.load_state.warnings.append(
                "Dendrograma solicitado, mas scipy não está instalado. Instale com: pip install scipy"
            )
            return
        if len(obras) < 3:
            self.load_state.warnings.append("Dendrograma não gerado: são necessárias pelo menos 3 obras.")
            return
        Z = linkage(scores_values, method="ward", metric="euclidean")
        fig = Figure(figsize=(8.0, 5.2), dpi=100)
        ax = fig.add_subplot(111)
        dendrogram(Z, labels=[str(o) for o in obras], ax=ax)
        ax.set_title("Cluster hierárquico das obras de Referência — scores PCA")
        ax.set_ylabel("Distância Ward")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        self.pca_state.figures["dendrograma_obras"] = fig

    def _make_config_dataframe(
        self,
        n_components: int,
        selected_columns: List[str],
        removed_zero: List[str],
        matrix_used: pd.DataFrame,
        distance_tables: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        state = self.load_state
        selected_fams = self.selected_families()
        role_counts = matrix_used[PCA_ROLE_COLUMN].value_counts().to_dict() if PCA_ROLE_COLUMN in matrix_used.columns else {}
        obras_ref = matrix_used.loc[matrix_used[PCA_ROLE_COLUMN] == ROLE_REFERENCE, "obra"].astype(str).tolist() if PCA_ROLE_COLUMN in matrix_used.columns else []
        obras_sup = matrix_used.loc[matrix_used[PCA_ROLE_COLUMN] == ROLE_SUPPLEMENTARY, "obra"].astype(str).tolist() if PCA_ROLE_COLUMN in matrix_used.columns else []
        obras_excl = [str(state.original_df.iloc[i]["obra"]) for i, role in enumerate(self.get_obra_roles()) if role == ROLE_EXCLUDED] if state.original_df is not None else []
        entries = {
            "data_hora": datetime.now().isoformat(timespec="seconds"),
            "software": APP_TITLE,
            "versao_correcao": APP_VERSION,
            "arquivo_origem": str(state.path),
            "encoding_detectado": state.encoding,
            "separador_detectado": repr(state.delimiter),
            "n_linhas_obras": len(state.original_df) if state.original_df is not None else "",
            "n_obras_distintas": state.original_df["obra"].nunique() if state.original_df is not None and "obra" in state.original_df.columns else "",
            "n_variaveis_origem": len(state.feature_cols),
            "n_variaveis_selecionadas_antes_preprocessamento": len(selected_columns),
            "n_variaveis_usadas_no_pca": len(matrix_feature_columns(matrix_used)),
            "parametros_preprocessamento_exportados": "parametros_preprocessamento_PCA.csv",
            "n_obras_referencia_fit_pca": role_counts.get(ROLE_REFERENCE, 0),
            "n_obras_suplementares_projetadas": role_counts.get(ROLE_SUPPLEMENTARY, 0),
            "n_obras_excluidas": len(obras_excl),
            "obras_referencia_fit_pca": " | ".join(obras_ref),
            "obras_suplementares_projetadas": " | ".join(obras_sup),
            "obras_excluidas": " | ".join(obras_excl),
            "n_componentes_principais": n_components,
            "rotulo_editavel_espaco_pca": self._reference_space_label(),
            "regioes_sintese_definidas_por": "Somente obras de Referência; suplementares apenas projetadas",
            "natureza_regioes_sintese": "Descritiva; não constitui fronteira de autoria ou classificação",
            "familias_selecionadas": " | ".join(selected_fams),
            "normalizacao_area_individual_perfis_H": bool(self.var_normalize_family_area.get()),
            "metodo_normalizacao_area_H": (
                "trapezoidal por obra e família/componente; blocos ausentes permanecem zero"
                if bool(self.var_normalize_family_area.get()) else "não aplicado"
            ),
            "normalizacao_por_soma_linha": bool(self.var_normalize_rows.get()),
            "normalizacao_area_H_e_soma_obra_combinadas": bool(
                self.var_normalize_family_area.get() and self.var_normalize_rows.get()
            ),
            "zscore_colunas": bool(self.var_zscore.get()),
            "zscore_ajustado_em": "Somente obras de Referência" if bool(self.var_zscore.get()) else "não aplicado",
            "limitar_digitos_significativos": bool(self.var_apply_sig_digits.get()),
            "digitos_significativos": self._get_significant_digits_setting() if bool(self.var_apply_sig_digits.get()) else "não aplicado",
            "etapa_digitos_significativos": (
                "após normalização por área dos H, normalização por soma e/ou z-score; imediatamente antes do PCA/projeção"
                if bool(self.var_apply_sig_digits.get()) else "não aplicado"
            ),
            "remover_variancia_zero": bool(self.var_remove_zero_var.get()),
            "metricas_distancia_calculadas": " | ".join(distance_tables.keys()) if distance_tables else "",
            "distancias_scores_calculadas_com": f"PC1-PC{n_components}",
            "euclidiana_scores": bool(self.var_metric_euclidean_scores.get()),
            "cosseno_matriz_usada": bool(self.var_metric_cosine_matrix.get()),
            "cosseno_scores": bool(self.var_metric_cosine_scores.get()),
            "mahalanobis_scores": bool(self.var_metric_mahalanobis_scores.get()),
            "dendrograma_cluster_hierarquico": bool(self.var_metric_dendrogram.get()),
            "dendrograma_base_calculo": "Somente obras de Referência; método Ward nos scores PCA",
            "arquivo_metadados": str(self.metadata_state.path) if self.metadata_state.path else "",
            "metadados_campos": " | ".join(self.metadata_state.fields),
            "colorir_pontos_por": self.var_color_by.get(),
            "marcador_por": self.var_marker_by.get(),
            "elipses_por": self.var_ellipse_by.get(),
            "elipsoides_3D_por": self.var_ellipse_by.get(),
            "mostrar_legenda": bool(self.var_show_legend.get()),
            "mostrar_rotulos": bool(self.var_show_labels.get()),
            "colorir_fases_sintese_pca": bool(self.var_synthesis_phase_colors.get()),
            "regioes_fase_sintese_2D": "Somente fases com pelo menos três obras de Referência e covariância 2D de posto completo",
            "regra_grupos_2_obras": "somente eixo principal maior 95% por qui-quadrado, pontilhado",
            "qui_quadrado_95_2D_df2": CHI2_95_2D,
            "qui_quadrado_95_3D_df3": CHI2_95_3D,
            "escala_visual_graficos_pca": self.var_synthesis_scale.get(),
            "observacao_escala_visual_graficos_pca": (
                "Quando a escala visual comprimida simétrica (asinh) está ativada, a transformação é aplicada somente ao desenho "
                "dos gráficos PC1×PC2, PC1×PC3, PC1×PC2×PC3 e da figura-síntese. Scores, centroides, elipses de compatibilidade "
                "e distâncias continuam sendo calculados nos valores originais do PCA."
            ),
            "observacao_nuvens_elipsoidais_3D": (
                "As regiões 3D são construídas nos scores originais pela regra de 95% com χ²(3): "
                "semieixo_i = sqrt(χ²_0.95,3 * λ_i). Para 2 obras, somente o eixo principal maior é mostrado, pontilhado. "
                "Para 3 ou mais obras, o elipsoide é desenhado sem espessura artificial; com exatamente 3 obras ele pode "
                "ser degenerado/planar. A escala asinh, quando usada, altera somente a representação visual posterior."
            ),
            "observacao_metricas": (
                "Distâncias nos scores são calculadas com todos os PCs gerados na rodada. "
                "A Mahalanobis usa a covariância das Referências quando há Suplementares. "
                "Para obter PC1-PC3, definir Nº de PCs = 3 antes de rodar o PCA."
            ),
            "observacao_loadings": (
                "O arquivo loadings_PCA.csv registra pca.components_.T, isto é, pesos/eigenvectors das variáveis "
                "em cada componente principal no padrão sklearn. Quando necessário, interpretar como pesos PCA, "
                "não como prova direta de correlação variável-PC."
            ),
            "observacao_contribuicao_loadings": (
                "Os arquivos contribuicao_loadings_por_PC.csv e resumo_dominancia_loadings.csv calculam, para cada PC, "
                "a soma dos loadings ao quadrado por família/componente/grupo dividida pela soma total dos loadings ao quadrado. "
                "A contribuição total é decomposta em parcelas de canais com loading positivo e negativo. "
                "Essa porcentagem mede concentração/dominância dos pesos dentro de cada PC; não é a variância explicada do PCA. "
                "O sinal global de cada eixo PCA é arbitrário e pode inverter entre rodadas sem mudar as relações geométricas."
            ),
            "observacao_figura_sintese_pca": (
                "As figuras da Síntese PCA são visualizações adicionais: as regiões descritivas são calculadas somente "
                "com as obras de Referência, enquanto as observações Suplementares são apenas projetadas. O rótulo do "
                "espaço é editável e não altera os cálculos. As regiões não constituem fronteiras de autoria ou classificação."
            ),
            "observacao_normalizacao_soma": (
                "Quando ativada, a normalização por soma é bloqueada se houver soma menor/igual a zero, "
                "valor não finito ou qualquer valor negativo nas linhas incluídas."
            ),
            "observacao_normalizacao_area_H": (
                "Quando ativada, cada bloco família/componente × energia de cada obra é integrado pelo método trapezoidal "
                "e dividido por sua própria área. Blocos inteiramente nulos são preservados como ausência. A transformação "
                "remove a escala arbitrária de H para priorizar forma espectral; não representa abundância da componente, "
                "que exigiria informação de W. Se combinada com a normalização global por soma da obra, a ponderação relativa "
                "também passa a depender do número de famílias presentes; essa combinação deve ser usada somente com justificativa."
            ),
            "variaveis_variancia_zero_removidas_qtd": len(removed_zero),
            "variaveis_variancia_zero_removidas": " | ".join(removed_zero),
            "alertas_validacao": " | ".join(state.warnings),
            "erros_validacao": " | ".join(state.errors),
            "familias_repetidas_detectadas": json.dumps(state.repeated_family_variants, ensure_ascii=False),
            "observacao_metodologica": (
                "PCA executado com sklearn.decomposition.PCA. O ajuste dos eixos principais é feito somente "
                "com as obras marcadas como Referência; obras Suplementares recebem as mesmas transformações "
                "e são projetadas por pca.transform, sem participar do fit. A matriz registrada em "
                "matriz_usada_no_PCA.csv contém a matriz final usada para PCA/projeção, incluindo a coluna papel_pca."
            ),
        }
        return pd.DataFrame([{"chave": k, "valor": v} for k, v in entries.items()])

    # ------------------------------------------------------------------
    # Renderização de resultados
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Figura-síntese PC1 × PC2 recentrada no conjunto de Referência
    # ------------------------------------------------------------------

    def _reference_space_label(self) -> str:
        """Retorna o nome editável do espaço PCA com fallback neutro."""
        value = ""
        if hasattr(self, "var_reference_space_label"):
            value = str(self.var_reference_space_label.get()).strip()
        return value or "Espaço de referência"

    def _synthesis_phase_field(self, merged: pd.DataFrame) -> Optional[str]:
        """Escolhe o campo de fase a partir dos metadados, sem exigir um nome rígido."""
        candidates = [self.var_ellipse_by.get(), self.var_color_by.get()]
        found = self._find_metadata_field("fase", "phase")
        if found:
            candidates.append(found)
        for field in candidates:
            if field and field != "Nenhum" and field in merged.columns:
                return field
        return None

    def _canonical_phase_label(self, value: str) -> str:
        """Normaliza rótulos frequentes de fase apenas para cor/legenda visual."""
        txt = str(value).strip()
        low = txt.lower()
        low_ascii = (
            low.replace("á", "a").replace("à", "a").replace("â", "a").replace("ã", "a")
               .replace("é", "e").replace("ê", "e")
               .replace("í", "i")
               .replace("ó", "o").replace("ô", "o").replace("õ", "o")
               .replace("ú", "u").replace("ç", "c")
        )
        if "antrop" in low_ascii and "social" in low_ascii:
            return "Antropofágica/Social"
        if "pau" in low_ascii and "brasil" in low_ascii and "neo" in low_ascii:
            return "(Neo)-Pau-Brasil"
        if "pau" in low_ascii and "brasil" in low_ascii:
            return "Pau-Brasil"
        if "antrop" in low_ascii:
            return "Antropofágica"
        if "social" in low_ascii:
            return "Social"
        if "paris" in low_ascii:
            return "Parisiense"
        if "primeir" in low_ascii and "ano" in low_ascii:
            return "Primeiros anos"
        return txt or "Sem fase"

    def _phase_rgb(self, phase: str) -> Tuple[float, float, float]:
        """Paleta baseada em cores primárias e derivações intuitivas.

        A ideia é favorecer leitura humana de sobreposições nos halos da
        figura-síntese. Até 3 grupos, as cores-base são vermelho, azul e
        amarelo. Para mais grupos, entram cores derivadas perceptualmente
        compatíveis (verde, laranja, violeta).
        """
        canonical = self._canonical_phase_label(phase)
        preferred = {
            "Antropofágica": "#d9534f",   # vermelho
            "Antropofágica/Social": "#8b6ad9",  # violeta
            "Primeiros anos": "#4f81d9",  # azul
            "(Neo)-Pau-Brasil": "#e3bf3a", # amarelo
            "Pau-Brasil": "#e3bf3a",      # amarelo
            "Parisiense": "#6fbf73",       # verde
            "Social": "#6fbf73",          # verde derivado
        }
        if canonical in preferred:
            return matplotlib.colors.to_rgb(preferred[canonical])

        fallback_palette = [
            "#d9534f",  # vermelho
            "#4f81d9",  # azul
            "#e3bf3a",  # amarelo
            "#6fbf73",  # verde
            "#e68a3f",  # laranja
            "#8b6ad9",  # violeta
            "#d96aa7",  # magenta suave
            "#6fc2b0",  # ciano/verde-água suave
        ]
        idx = sum(ord(ch) for ch in canonical) % len(fallback_palette)
        return matplotlib.colors.to_rgb(fallback_palette[idx])

    def _phase_halo_alpha(self, rank: int = 0) -> float:
        """Transparência base dos halos por fase.

        Um pequeno decréscimo com o rank evita que gráficos com muitas fases
        fiquem pesados demais visualmente.
        """
        return max(0.18, 0.30 - 0.018 * int(rank))

    def _phase_is_informative_for_halo(self, phase: str) -> bool:
        """Decide se uma fase deve gerar halo visual.

        Rótulos vazios ou genéricos como "Não identificada" não representam
        um domínio técnico-material interpretável e, por isso, não devem
        colorir o fundo da figura-síntese.
        """
        txt = str(phase).strip()
        if not txt:
            return False
        low = txt.lower()
        low_ascii = (
            low.replace("á", "a").replace("à", "a").replace("â", "a").replace("ã", "a")
               .replace("é", "e").replace("ê", "e")
               .replace("í", "i")
               .replace("ó", "o").replace("ô", "o").replace("õ", "o")
               .replace("ú", "u").replace("ç", "c")
        )
        generic_terms = (
            "sem fase",
            "nao identificada",
            "não identificada",
            "nao identificado",
            "não identificado",
            "indefinida",
            "indefinido",
            "nan",
            "none",
        )
        return low_ascii not in generic_terms

    def _enable_interactive_drag(
        self,
        canvas,
        annotations: Optional[Sequence] = None,
        legends: Optional[Sequence] = None,
    ) -> None:
        """Habilita arraste de rótulos e legendas no canvas interativo.

        Os controladores são guardados no próprio canvas para permanecerem
        ativos enquanto a aba existir. As novas posições passam a fazer parte
        da Figure e, portanto, são preservadas na exportação do PNG enquanto
        a visualização não for regenerada.
        """
        controllers = []
        for annotation in annotations or []:
            if annotation is None:
                continue
            try:
                controller = annotation.draggable(use_blit=False)
                if controller is not None:
                    controllers.append(controller)
            except Exception:
                continue

        for legend in legends or []:
            if legend is None:
                continue
            try:
                controller = legend.set_draggable(True, use_blit=False, update="loc")
                if controller is not None:
                    controllers.append(controller)
            except Exception:
                continue

        # Evita que os controladores sejam descartados pelo coletor de lixo.
        canvas._pca_drag_controllers = controllers

    def _repel_texts_data(
        self,
        fig: Figure,
        ax,
        texts: List,
        anchors: List[Tuple[float, float]],
        max_iter: int = 120,
        expand_px: float = 2.0,
    ):
        """Afasta rótulos entre si em coordenadas de dados, sem depender de libs externas.

        Correção de robustez:
        algumas versões/backends do Matplotlib criam a Figure inicialmente com
        FigureCanvasBase, que não possui get_renderer(). Para medir colisões de
        texto antes de inserir a figura no Tkinter, usamos um canvas Agg
        temporário quando necessário.
        """
        if not texts:
            return

        def _get_renderer_safe():
            try:
                fig.canvas.draw()
            except Exception:
                pass

            get_renderer = getattr(fig.canvas, "get_renderer", None)
            if callable(get_renderer):
                return get_renderer()

            renderer = getattr(fig.canvas, "renderer", None)
            if renderer is not None:
                return renderer

            agg_canvas = FigureCanvasAgg(fig)
            agg_canvas.draw()
            return agg_canvas.get_renderer()

        renderer = _get_renderer_safe()
        ax_bbox = ax.get_window_extent(renderer=renderer)
        margin = 6.0

        def _move_text_px(txt, dx_px: float, dy_px: float):
            x, y = txt.get_position()
            xd, yd = ax.transData.transform((x, y))
            xd += dx_px
            yd += dy_px
            xd = min(max(xd, ax_bbox.x0 + margin), ax_bbox.x1 - margin)
            yd = min(max(yd, ax_bbox.y0 + margin), ax_bbox.y1 - margin)
            xn, yn = ax.transData.inverted().transform((xd, yd))
            txt.set_position((float(xn), float(yn)))

        for _ in range(max_iter):
            renderer = _get_renderer_safe()
            bboxes = [t.get_window_extent(renderer=renderer).expanded(1.03, 1.10) for t in texts]
            moved = False

            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    bi = bboxes[i]
                    bj = bboxes[j]
                    if not bi.overlaps(bj):
                        continue

                    overlap_x = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                    overlap_y = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                    if overlap_x <= 0 or overlap_y <= 0:
                        continue

                    cix = 0.5 * (bi.x0 + bi.x1)
                    ciy = 0.5 * (bi.y0 + bi.y1)
                    cjx = 0.5 * (bj.x0 + bj.x1)
                    cjy = 0.5 * (bj.y0 + bj.y1)

                    sx = 1.0 if cix <= cjx else -1.0
                    sy = 1.0 if ciy <= cjy else -1.0

                    if abs(overlap_x) >= abs(overlap_y):
                        shift_x = 0.5 * overlap_x + expand_px
                        _move_text_px(texts[i], -sx * shift_x, 0.25 * -sy * expand_px)
                        _move_text_px(texts[j], sx * shift_x, 0.25 * sy * expand_px)
                    else:
                        shift_y = 0.5 * overlap_y + expand_px
                        _move_text_px(texts[i], 0.25 * -sx * expand_px, -sy * shift_y)
                        _move_text_px(texts[j], 0.25 * sx * expand_px, sy * shift_y)

                    moved = True

            # Pequena força radial para afastar textos do próprio ponto quando necessário.
            if anchors:
                for txt, (ax0, ay0) in zip(texts, anchors):
                    bbox = txt.get_window_extent(renderer=renderer)
                    anchor_px = ax.transData.transform((ax0, ay0))
                    if (bbox.x0 - 2) <= anchor_px[0] <= (bbox.x1 + 2) and (bbox.y0 - 2) <= anchor_px[1] <= (bbox.y1 + 2):
                        x, y = txt.get_position()
                        x_px, y_px = ax.transData.transform((x, y))
                        vx = x_px - anchor_px[0]
                        vy = y_px - anchor_px[1]
                        norm = max((vx ** 2 + vy ** 2) ** 0.5, 1.0)
                        _move_text_px(txt, 8.0 * vx / norm, 8.0 * vy / norm)
                        moved = True

            if not moved:
                break

    def _add_repelled_labels(
        self,
        fig: Figure,
        ax,
        anchors: List[Tuple[float, float]],
        labels: List[str],
        fontsize: float = 8.5,
        text_color: str = "#111111",
        zorder: int = 8,
    ) -> List:
        """Cria rótulos afastados, arrastáveis e ligados aos respectivos pontos."""
        if not anchors or not labels:
            return []
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = max((xmax - xmin) * 0.010, 0.03)
        dy = max((ymax - ymin) * 0.010, 0.03)
        temporary_texts = []
        for (x, y), label in zip(anchors, labels):
            txt = ax.text(
                x + dx,
                y + dy,
                str(label),
                fontsize=fontsize,
                color=text_color,
                zorder=zorder,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.6},
            )
            temporary_texts.append(txt)

        self._repel_texts_data(fig, ax, temporary_texts, anchors)
        final_positions = [txt.get_position() for txt in temporary_texts]
        for txt in temporary_texts:
            txt.remove()

        annotations = []
        for (x0, y0), (xt, yt), label in zip(anchors, final_positions, labels):
            annotation = ax.annotate(
                str(label),
                xy=(float(x0), float(y0)),
                xytext=(float(xt), float(yt)),
                textcoords="data",
                fontsize=fontsize,
                color=text_color,
                zorder=zorder,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.6},
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#666666",
                    "linewidth": 0.45,
                    "alpha": 0.55,
                    "shrinkA": 1.5,
                    "shrinkB": 2.5,
                },
            )
            annotations.append(annotation)
        return annotations

    def _ellipse_parameters_from_cov(self, cov: np.ndarray, threshold: float = 5.991) -> Tuple[float, float, float]:
        """Retorna largura, altura e ângulo da elipse de Mahalanobis no plano 2D."""
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            cov = np.eye(2, dtype=float)
        cov = cov + np.eye(2) * 1e-12
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = np.maximum(eigvals[order], 1e-12)
        eigvecs = eigvecs[:, order]
        angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
        # largura/altura completas, não semi-eixos.
        width, height = 2.0 * np.sqrt(float(threshold) * eigvals)
        return float(width), float(height), angle

    def _build_synthesis_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
        """Monta coordenadas recentradas, centroides de fase e distâncias para a figura-síntese.

        A figura é deliberadamente adicional: não substitui os gráficos PCA originais.
        Referências definem o centro, a elipse e os centroides de fase. Suplementares são apenas avaliadas.
        """
        p = self.pca_state
        if p.scores is None or p.explained_ratio is None:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
        scores = self._scores_with_metadata()
        if scores.empty or "PC1" not in scores.columns or "PC2" not in scores.columns:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        work = scores.copy()
        work["PC1"] = pd.to_numeric(work["PC1"], errors="coerce")
        work["PC2"] = pd.to_numeric(work["PC2"], errors="coerce")
        work = work.dropna(subset=["PC1", "PC2"]).copy()
        if work.empty or PCA_ROLE_COLUMN not in work.columns:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        ref_mask = work[PCA_ROLE_COLUMN].astype(str).eq(ROLE_REFERENCE)
        if int(ref_mask.sum()) < 2:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        ref_xy_raw = work.loc[ref_mask, ["PC1", "PC2"]].to_numpy(dtype=float)
        center_raw = ref_xy_raw.mean(axis=0)
        work["PC1_centrado"] = work["PC1"].astype(float) - float(center_raw[0])
        work["PC2_centrado"] = work["PC2"].astype(float) - float(center_raw[1])
        xy = work[["PC1_centrado", "PC2_centrado"]].to_numpy(dtype=float)
        ref_xy = work.loc[ref_mask, ["PC1_centrado", "PC2_centrado"]].to_numpy(dtype=float)

        # Elipse 95% para 2 dimensões: χ²(2, 0.95) ≈ 5.991. Com n pequeno, tratar como domínio visual.
        ellipse_threshold = 5.991
        cov = np.cov(ref_xy, rowvar=False, ddof=1) if ref_xy.shape[0] >= 2 else np.eye(2)
        cov = np.atleast_2d(cov).astype(float)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            cov = np.eye(2, dtype=float)
        inv_cov = np.linalg.pinv(cov + np.eye(2) * 1e-12)
        mahal2 = np.einsum("ij,jk,ik->i", xy, inv_cov, xy)
        mahal2 = np.maximum(mahal2, 0.0)
        work["distancia_euclidiana_centro_referencias"] = np.linalg.norm(xy, axis=1)
        work["mahalanobis2_centro_referencias_PC1_PC2"] = mahal2
        work["mahalanobis_centro_referencias_PC1_PC2"] = np.sqrt(mahal2)
        work["dentro_regiao_descritiva_referencias_PC1_PC2"] = mahal2 <= ellipse_threshold
        work["fora_regiao_descritiva_referencias_PC1_PC2"] = ~work[
            "dentro_regiao_descritiva_referencias_PC1_PC2"
        ]
        work["criterio_regiao_descritiva"] = (
            "elipse χ²(2,0.95) no plano PC1×PC2; uso exclusivamente descritivo"
        )

        phase_field = self._find_metadata_field("fase", "phase")
        if not phase_field:
            color_candidate = self.var_color_by.get()
            if color_candidate not in {"", "Nenhum", PCA_ROLE_COLUMN} and color_candidate in work.columns:
                phase_field = color_candidate
        phase_centroids_rows = []
        distances_rows = []
        if phase_field:
            work["fase_sintese"] = work[phase_field].fillna("").astype(str).map(self._canonical_phase_label)
            ref_with_phase = work.loc[ref_mask & work["fase_sintese"].ne("") & work["fase_sintese"].ne("Sem fase")].copy()
            for phase, sub in ref_with_phase.groupby("fase_sintese", sort=False):
                pts = sub[["PC1_centrado", "PC2_centrado"]].to_numpy(dtype=float)
                if len(pts) < 1:
                    continue
                cx, cy = pts.mean(axis=0)
                dists = np.linalg.norm(pts - np.array([[cx, cy]], dtype=float), axis=1)
                spread_mean = float(np.mean(dists)) if len(dists) else 0.0
                spread_max = float(np.max(dists)) if len(dists) else 0.0
                spread_std = float(np.std(dists, ddof=0)) if len(dists) else 0.0
                phase_centroids_rows.append({
                    "fase": phase,
                    "campo_metadado_fase": phase_field,
                    "n_obras_referencia": int(len(sub)),
                    "PC1_centroide_centrado": float(cx),
                    "PC2_centroide_centrado": float(cy),
                    "distancia_euclidiana_origem": float(np.hypot(cx, cy)),
                    "raio_medio_fase": spread_mean,
                    "raio_maximo_fase": spread_max,
                    "desvio_radial_fase": spread_std,
                })

            centroid_lookup = {row["fase"]: np.array([row["PC1_centroide_centrado"], row["PC2_centroide_centrado"]], dtype=float) for row in phase_centroids_rows}
            closest_phase = []
            closest_dist = []
            for idx, row in work.iterrows():
                point = np.array([float(row["PC1_centrado"]), float(row["PC2_centrado"])], dtype=float)
                per_point = []
                for phase, centroid in centroid_lookup.items():
                    metodo = "centroide completo da fase"
                    centroid_used = centroid.copy()
                    # Leave-one-out apenas para obra de Referência quando a distância é até a própria fase.
                    if str(row.get(PCA_ROLE_COLUMN, "")) == ROLE_REFERENCE and str(row.get("fase_sintese", "")) == phase:
                        same_phase = ref_with_phase[ref_with_phase["fase_sintese"].astype(str).eq(phase)]
                        same_phase = same_phase[same_phase["obra"].astype(str).ne(str(row["obra"]))]
                        if len(same_phase) >= 1:
                            centroid_used = same_phase[["PC1_centrado", "PC2_centrado"]].to_numpy(dtype=float).mean(axis=0)
                            metodo = "leave-one-out na própria fase"
                    dist = float(np.linalg.norm(point - centroid_used))
                    distances_rows.append({
                        "obra": row["obra"],
                        PCA_ROLE_COLUMN: row.get(PCA_ROLE_COLUMN, ""),
                        "fase_metadado_obra": row.get("fase_sintese", ""),
                        "fase_centroide": phase,
                        "metodo_centroide": metodo,
                        "PC1_centrado_obra": float(point[0]),
                        "PC2_centrado_obra": float(point[1]),
                        "PC1_centroide_fase": float(centroid_used[0]),
                        "PC2_centroide_fase": float(centroid_used[1]),
                        "distancia_euclidiana_ao_centroide_fase": dist,
                    })
                    per_point.append((phase, dist))
                if per_point:
                    phase_min, dist_min = min(per_point, key=lambda x: x[1])
                    closest_phase.append(phase_min)
                    closest_dist.append(dist_min)
                else:
                    closest_phase.append("")
                    closest_dist.append(np.nan)
            work["fase_mais_proxima_PC1_PC2"] = closest_phase
            work["distancia_fase_mais_proxima_PC1_PC2"] = closest_dist
        else:
            work["fase_sintese"] = ""
            work["fase_mais_proxima_PC1_PC2"] = ""
            work["distancia_fase_mais_proxima_PC1_PC2"] = np.nan

        phase_centroids = pd.DataFrame(phase_centroids_rows)
        phase_distances = pd.DataFrame(distances_rows)
        metadata = {
            "center_raw": center_raw,
            "cov": cov,
            "inv_cov": inv_cov,
            "ellipse_threshold": ellipse_threshold,
            "phase_field": phase_field or "",
        }
        return work, phase_centroids, phase_distances, metadata

    def _auto_asinh_factor_for_synthesis(self, scores_centered: pd.DataFrame) -> float:
        """Fator automático para a escala visual asinh da figura-síntese.

        O fator é calculado a partir da dispersão das obras de Referência.
        Assim, a região central das referências fica legível e os pontos
        suplementares muito distantes são comprimidos apenas visualmente.
        """
        if scores_centered is None or scores_centered.empty:
            return 1.0
        work = scores_centered.copy()
        ref = work[work.get(PCA_ROLE_COLUMN, "").astype(str).eq(ROLE_REFERENCE)] if PCA_ROLE_COLUMN in work.columns else work
        xy_ref = ref[["PC1_centrado", "PC2_centrado"]].to_numpy(dtype=float) if not ref.empty else work[["PC1_centrado", "PC2_centrado"]].to_numpy(dtype=float)
        xy_all = work[["PC1_centrado", "PC2_centrado"]].to_numpy(dtype=float)
        xy_ref = xy_ref[np.all(np.isfinite(xy_ref), axis=1)]
        xy_all = xy_all[np.all(np.isfinite(xy_all), axis=1)]
        if xy_all.size == 0:
            return 1.0
        ref_radius = np.linalg.norm(xy_ref, axis=1) if xy_ref.size else np.array([], dtype=float)
        all_radius = np.linalg.norm(xy_all, axis=1)
        candidates = []
        if ref_radius.size:
            nonzero_ref = ref_radius[ref_radius > 1e-12]
            if nonzero_ref.size:
                candidates.append(float(np.percentile(nonzero_ref, 50)))
                candidates.append(float(np.percentile(nonzero_ref, 75)) * 0.65)
        nonzero_all = all_radius[all_radius > 1e-12]
        if nonzero_all.size:
            candidates.append(float(np.percentile(nonzero_all, 75)) * 0.25)
            candidates.append(float(np.percentile(nonzero_all, 95)) * 0.08)
        candidates = [c for c in candidates if np.isfinite(c) and c > 1e-12]
        return max(min(candidates) if candidates else 1.0, 1e-9)

    def _synthesis_visual_transform(self, xy: np.ndarray, scale_mode: str, factor: float) -> np.ndarray:
        """Transforma coordenadas apenas para desenho da figura-síntese."""
        arr = np.asarray(xy, dtype=float)
        if str(scale_mode) == SYNTHESIS_SCALE_ASINH:
            return np.arcsinh(arr / max(float(factor), 1e-12))
        return arr.copy()

    def _synthesis_visual_inverse(self, xy_visual: np.ndarray, scale_mode: str, factor: float) -> np.ndarray:
        """Transformação inversa da escala visual, usada para o fundo/gradiente."""
        arr = np.asarray(xy_visual, dtype=float)
        if str(scale_mode) == SYNTHESIS_SCALE_ASINH:
            return np.sinh(arr) * max(float(factor), 1e-12)
        return arr.copy()

    def _ellipse_points_from_cov(self, cov: np.ndarray, threshold: float, n_points: int = 360) -> np.ndarray:
        """Retorna pontos da elipse de Mahalanobis/dispersão no espaço original PC1×PC2."""
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (2, 2):
            cov = np.eye(2, dtype=float)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov + np.eye(2) * 1e-12)
            order = eigvals.argsort()[::-1]
            eigvals = np.maximum(eigvals[order], 1e-12)
            eigvecs = eigvecs[:, order]
            theta = np.linspace(0, 2 * np.pi, int(n_points))
            circle = np.vstack([np.cos(theta), np.sin(theta)])
            radii = np.sqrt(eigvals * max(float(threshold), 1e-12))
            pts = (eigvecs @ (radii[:, None] * circle)).T
            return pts
        except Exception:
            theta = np.linspace(0, 2 * np.pi, int(n_points))
            r = math.sqrt(max(float(threshold), 1.0))
            return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    def _make_synthesis_background(
        self,
        ax,
        extent: Tuple[float, float, float, float],
        centroids: pd.DataFrame,
        cov: np.ndarray,
        threshold: float,
        scale_mode: str = SYNTHESIS_SCALE_LINEAR,
        factor: float = 1.0,
    ):
        """Desenha halos locais por fase e atenuação externa ao domínio das referências.

        Nesta versão, os halos são calculados no espaço visual do gráfico.
        Isso evita que a escala asinh transforme um raio pequeno nos scores
        originais em uma coloração quase global. Cada halo nasce no centróide
        da fase, é mais intenso no núcleo e desaparece radialmente com corte
        explícito a cerca de 3 sigma.
        """
        xmin, xmax, ymin, ymax = extent
        nx, ny = 340, 280
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        Xg, Yg = np.meshgrid(xs, ys)

        # Fundo branco: sem mistura global de cores.
        bg = np.ones((ny, nx, 3), dtype=float)

        visual_span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
        min_sigma_visual = max(visual_span * 0.035, 1e-4)
        max_sigma_visual = max(visual_span * 0.130, min_sigma_visual)

        use_phase_colors = bool(getattr(self, "var_synthesis_phase_colors", None).get()) if hasattr(self, "var_synthesis_phase_colors") else True

        if use_phase_colors and centroids is not None and not centroids.empty:
            cent = centroids.copy()
            informative_rows = []
            for _, row in cent.iterrows():
                phase = str(row.get("fase", ""))
                if self._phase_is_informative_for_halo(phase):
                    informative_rows.append(row)

            for rank, row in enumerate(informative_rows):
                phase = str(row["fase"])
                cx_raw = float(row["PC1_centroide_centrado"])
                cy_raw = float(row["PC2_centroide_centrado"])

                centroid_visual = self._synthesis_visual_transform(
                    np.array([[cx_raw, cy_raw]], dtype=float),
                    scale_mode,
                    factor,
                )[0]
                cx, cy = float(centroid_visual[0]), float(centroid_visual[1])

                # Estima o raio visual a partir da dispersão real da fase.
                radius_mean = float(row.get("raio_medio_fase", 0.0) or 0.0)
                radius_max = float(row.get("raio_maximo_fase", 0.0) or 0.0)
                radius_std = float(row.get("desvio_radial_fase", 0.0) or 0.0)
                radius_raw = max(radius_mean * 1.55, radius_max * 0.95, radius_std * 2.1, 0.0)

                sigma_candidates = []
                if radius_raw > 0:
                    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
                    samples_raw = np.column_stack([
                        cx_raw + radius_raw * np.cos(theta),
                        cy_raw + radius_raw * np.sin(theta),
                    ])
                    samples_visual = self._synthesis_visual_transform(samples_raw, scale_mode, factor)
                    sample_dists = np.linalg.norm(samples_visual - centroid_visual.reshape(1, 2), axis=1)
                    sample_dists = sample_dists[np.isfinite(sample_dists)]
                    if len(sample_dists):
                        sigma_candidates.append(float(np.nanmedian(sample_dists)))

                n_ref = int(row.get("n_obras_referencia", 1) or 1)
                if n_ref <= 1:
                    sigma_candidates.append(visual_span * 0.045)
                elif n_ref == 2:
                    sigma_candidates.append(visual_span * 0.055)
                else:
                    sigma_candidates.append(visual_span * 0.040)

                sigma = max([min_sigma_visual] + sigma_candidates)
                sigma = min(sigma, max_sigma_visual)

                dist2_visual = (Xg - cx) ** 2 + (Yg - cy) ** 2
                dist_visual = np.sqrt(dist2_visual)
                intensity = np.exp(-dist2_visual / (2.0 * sigma ** 2))

                # Corte radial: fora disso não há cor da fase.
                cutoff = 3.15 * sigma
                intensity = np.where(dist_visual <= cutoff, intensity, 0.0)

                color = np.array(self._phase_rgb(phase), dtype=float)

                # Alpha suficientemente forte no centro, mas local.
                alpha = np.clip(self._phase_halo_alpha(rank) * intensity, 0.0, 0.34)
                bg = bg * (1.0 - alpha[..., None]) + color * alpha[..., None]

                # Núcleo mais escuro exatamente sobre o centróide.
                core_sigma = max(sigma * 0.36, min_sigma_visual * 0.55)
                core_intensity = np.exp(-dist2_visual / (2.0 * core_sigma ** 2))
                core_intensity = np.where(dist_visual <= 2.25 * core_sigma, core_intensity, 0.0)
                core_alpha = np.clip(0.13 * core_intensity, 0.0, 0.14)
                bg = bg * (1.0 - core_alpha[..., None]) + color * core_alpha[..., None]

        # Região externa ao domínio de referência continua calculada nos scores originais.
        try:
            grid_visual = np.stack([Xg, Yg], axis=-1)
            grid_raw = self._synthesis_visual_inverse(grid_visual, scale_mode, factor)
            cov = np.asarray(cov, dtype=float)
            inv_cov = np.linalg.pinv(cov + np.eye(2) * 1e-12)
            m2 = np.einsum("...i,ij,...j->...", grid_raw, inv_cov, grid_raw)
            outside = np.clip((m2 - threshold) / max(threshold, 1e-9), 0.0, 1.0)
            outside_color = np.array(matplotlib.colors.to_rgb("#d8a2a2"), dtype=float)
            alpha = 0.070 * outside[..., None]
            bg = bg * (1 - alpha) + outside_color * alpha
        except Exception:
            pass

        ax.imshow(bg, extent=extent, origin="lower", aspect="auto", zorder=0)

    def _render_synthesis_plot(self):
        """Renderiza a figura científica recentrada PC1×PC2 sem alterar os gráficos originais."""
        p = self.pca_state
        tab = self.tab_synthesis_pc12
        for child in tab.winfo_children():
            child.destroy()

        if p.scores is None or p.explained_ratio is None or len(p.explained_ratio) < 2:
            ttk.Label(tab, text="A figura-síntese requer pelo menos PC1 e PC2.").pack(anchor="center", pady=30)
            return

        scores_centered, phase_centroids, phase_distances, meta = self._build_synthesis_tables()
        p.synthesis_scores = scores_centered
        p.synthesis_phase_centroids = phase_centroids
        p.synthesis_phase_distances = phase_distances
        if scores_centered.empty:
            ttk.Label(tab, text="Não há dados suficientes para gerar a figura-síntese.").pack(anchor="center", pady=30)
            return

        scale_mode = self.var_synthesis_scale.get() if hasattr(self, "var_synthesis_scale") else SYNTHESIS_SCALE_LINEAR
        if scale_mode not in SYNTHESIS_SCALE_OPTIONS:
            scale_mode = SYNTHESIS_SCALE_LINEAR
        factor = self._auto_asinh_factor_for_synthesis(scores_centered) if scale_mode == SYNTHESIS_SCALE_ASINH else 1.0

        fig = Figure(figsize=(8.4, 6.4), dpi=100)
        ax = fig.add_subplot(111)

        raw_xy = scores_centered[["PC1_centrado", "PC2_centrado"]].to_numpy(dtype=float)
        plot_xy = self._synthesis_visual_transform(raw_xy, scale_mode, factor)
        scores_plot = scores_centered.copy()
        scores_plot["PC1_plot"] = plot_xy[:, 0]
        scores_plot["PC2_plot"] = plot_xy[:, 1]

        cov = np.asarray(meta.get("cov", np.eye(2)), dtype=float)
        threshold = float(meta.get("ellipse_threshold", 5.991))
        ellipse_raw = self._ellipse_points_from_cov(cov, threshold)
        ellipse_plot = self._synthesis_visual_transform(ellipse_raw, scale_mode, factor)

        centroid_plot_points = np.empty((0, 2), dtype=float)
        if phase_centroids is not None and not phase_centroids.empty:
            centroid_raw = phase_centroids[["PC1_centroide_centrado", "PC2_centroide_centrado"]].to_numpy(dtype=float)
            centroid_plot_points = self._synthesis_visual_transform(centroid_raw, scale_mode, factor)

        combined = [plot_xy, ellipse_plot]
        if centroid_plot_points.size:
            combined.append(centroid_plot_points)
        combined_xy = np.vstack([a for a in combined if a.size])
        span_x = max(float(np.ptp(combined_xy[:, 0])) if len(combined_xy) > 1 else 1.0, 1.0)
        span_y = max(float(np.ptp(combined_xy[:, 1])) if len(combined_xy) > 1 else 1.0, 1.0)
        pad_x = span_x * 0.18
        pad_y = span_y * 0.18
        xmin = float(np.min(combined_xy[:, 0])) - pad_x
        xmax = float(np.max(combined_xy[:, 0])) + pad_x
        ymin = float(np.min(combined_xy[:, 1])) - pad_y
        ymax = float(np.max(combined_xy[:, 1])) + pad_y
        extent = (xmin, xmax, ymin, ymax)

        self._make_synthesis_background(ax, extent, phase_centroids, cov, threshold, scale_mode=scale_mode, factor=factor)

        # Domínio técnico-material das Referências. Em escala asinh, a elipse é desenhada
        # como curva transformada, pois uma elipse no dado original não permanece elipse no desenho.
        ax.plot(
            ellipse_plot[:, 0],
            ellipse_plot[:, 1],
            color="#333333",
            linewidth=1.35,
            linestyle="--",
            alpha=0.88,
            zorder=2,
            label="Região descritiva das referências",
        )

        # Centroides de fase: marcadores sutis, sem vetores. O halo já comunica a região.
        if phase_centroids is not None and not phase_centroids.empty:
            for _, row in phase_centroids.iterrows():
                phase = str(row["fase"])
                raw_centroid = np.array([[float(row["PC1_centroide_centrado"]), float(row["PC2_centroide_centrado"])]], dtype=float)
                cx, cy = self._synthesis_visual_transform(raw_centroid, scale_mode, factor)[0]
                color = self._phase_rgb(phase)
                ax.scatter([cx], [cy], s=34, marker="o", facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.95, zorder=4)

        # Pontos das obras.
        label_anchors: List[Tuple[float, float]] = []
        label_texts: List[str] = []
        use_phase_colors = bool(self.var_synthesis_phase_colors.get()) if hasattr(self, "var_synthesis_phase_colors") else True
        for _, row in scores_plot.iterrows():
            role = str(row.get(PCA_ROLE_COLUMN, ""))
            x = float(row["PC1_plot"])
            y = float(row["PC2_plot"])
            low_compat = bool(row.get("fora_regiao_descritiva_referencias_PC1_PC2", False))
            phase = str(row.get("fase_sintese", ""))
            edge = "#b22222" if low_compat else "black"
            lw = 1.9 if low_compat else 0.9
            face = self._phase_rgb(phase) if (use_phase_colors and role == ROLE_REFERENCE and self._phase_is_informative_for_halo(phase)) else "#bdbdbd"
            if role == ROLE_REFERENCE:
                ax.scatter([x], [y], s=76, marker="o", facecolor=face, edgecolor=edge, linewidth=lw, alpha=0.97, zorder=6)
            else:
                ax.scatter([x], [y], s=96, marker="o", facecolor="none", edgecolor=edge, linewidth=max(lw, 1.5), alpha=0.99, zorder=7)
            if bool(self.var_show_labels.get()):
                label_anchors.append((x, y))
                label_texts.append(str(row["obra"]))

        # Origem interpretativa.
        ax.scatter([0], [0], s=26, marker="+", color="black", linewidths=1.2, zorder=8)
        ax.annotate("centro das referências", (0, 0), xytext=(6, -14), textcoords="offset points", fontsize=8, color="black")
        ax.axhline(0, linewidth=0.8, color="black", alpha=0.55)
        ax.axvline(0, linewidth=0.8, color="black", alpha=0.55)
        ax.grid(True, alpha=0.18, linewidth=0.6)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        draggable_labels: List = []
        if label_anchors and label_texts:
            draggable_labels = self._add_repelled_labels(
                fig,
                ax,
                label_anchors,
                label_texts,
                fontsize=8.5,
                text_color="#111111",
                zorder=8,
            )
        if scale_mode == SYNTHESIS_SCALE_ASINH:
            ax.set_xlabel(f"PC1 recentrado — escala visual asinh ({fmt_pct(float(p.explained_ratio[0]))})")
            ax.set_ylabel(f"PC2 recentrado — escala visual asinh ({fmt_pct(float(p.explained_ratio[1]))})")
            ax.text(
                0.01,
                0.01,
                f"Escala visual: asinh(score/{factor:.3g}). Halos locais são recurso visual; distâncias/centroides usam scores originais.",
                transform=ax.transAxes,
                fontsize=7.5,
                ha="left",
                va="bottom",
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
                zorder=10,
            )
        else:
            ax.set_xlabel(f"PC1 recentrado ({fmt_pct(float(p.explained_ratio[0]))})")
            ax.set_ylabel(f"PC2 recentrado ({fmt_pct(float(p.explained_ratio[1]))})")
        ax.set_title(f"{self._reference_space_label()} — projeção PC1 × PC2")

        # Legendas da figura-síntese: separar categorias e fases para evitar sobreposição no meio.
        use_phase_colors = bool(self.var_synthesis_phase_colors.get()) if hasattr(self, "var_synthesis_phase_colors") else True
        draggable_legends: List = []
        if bool(self.var_show_legend.get()):
            category_handles = [
                Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                       markerfacecolor="#bdbdbd", markeredgecolor="black", color="black",
                       label="Obra de referência"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                       markerfacecolor="none", markeredgecolor="black", color="black",
                       label="Observação suplementar"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                       markerfacecolor="none", markeredgecolor="#b22222", color="#b22222",
                       markeredgewidth=1.9, label="Fora da região descritiva"),
                Line2D([0], [0], linestyle="--", linewidth=1.35, color="#333333",
                       label="Região descritiva das referências"),
            ]
            legend_classes = ax.legend(
                handles=category_handles,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.985),
                borderaxespad=0.0,
                fontsize=8,
                frameon=True,
                title=None,
            )
            ax.add_artist(legend_classes)
            draggable_legends.append(legend_classes)

            phase_handles: List[Patch] = []
            if use_phase_colors and phase_centroids is not None and not phase_centroids.empty:
                phases_order = []
                for phase in phase_centroids["fase"].astype(str).tolist():
                    phase = str(phase)
                    if self._phase_is_informative_for_halo(phase) and phase not in phases_order:
                        phases_order.append(phase)
                for rank, phase in enumerate(phases_order):
                    phase_handles.append(
                        Patch(
                            facecolor=self._phase_rgb(phase),
                            edgecolor="none",
                            alpha=self._phase_halo_alpha(rank),
                            label=phase,
                        )
                    )
            if phase_handles:
                legend_phases = ax.legend(
                    handles=phase_handles,
                    loc="upper left",
                    bbox_to_anchor=(0.015, 0.985),
                    borderaxespad=0.0,
                    fontsize=8,
                    frameon=True,
                    title="Fases",
                )
                ax.add_artist(legend_phases)
                draggable_legends.append(legend_phases)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        self._enable_interactive_drag(canvas, draggable_labels, draggable_legends)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        p.figures["figura_sintese_PC1_PC2_recentrada"] = fig
    def _clear_results_tabs(self):
        self._clear_plot_tab(self.tab_pc12, "O gráfico PC1 × PC2 aparecerá após executar o PCA.")
        self._clear_plot_tab(self.tab_pc13, "O gráfico PC1 × PC3 aparecerá quando houver pelo menos 3 PCs.")
        self._clear_plot_tab(self.tab_pc123, "O gráfico tridimensional PC1 × PC2 × PC3 aparecerá quando houver pelo menos 3 PCs.")
        self._clear_plot_tab(self.tab_synthesis_pc12, "A figura-síntese PC1 × PC2 aparecerá após executar o PCA.")
        self._clear_plot_tab(self.tab_synthesis_pc13, "A figura-síntese PC1 × PC3 aparecerá quando houver pelo menos 3 PCs.")
        self._clear_plot_tab(self.tab_reference_space_3d, "O espaço PCA tridimensional aparecerá quando houver pelo menos 3 PCs.")
        self._clear_plot_tab(self.tab_loading_contrib, "O gráfico de contribuição dos loadings aparecerá após executar o PCA.")
        self._clear_plot_tab(self.tab_variance, "O gráfico de variância explicada aparecerá após executar o PCA.")
        self._clear_plot_tab(self.tab_dendrogram, "O dendrograma aparecerá se a opção de cluster hierárquico estiver marcada.")
        self._load_dataframe_into_tree(self.tree_scores, pd.DataFrame())
        self._load_dataframe_into_tree(self.tree_loadings, pd.DataFrame())
        self._load_dataframe_into_tree(self.tree_distances, pd.DataFrame())
        self.config_text.set_text("A configuração da rodada aparecerá após executar o PCA.")

    def _render_results(self):
        pca_state = self.pca_state
        if pca_state.scores is None or pca_state.loadings is None:
            return

        self._render_score_plot(pc_x=0, pc_y=1, tab=self.tab_pc12, key="grafico_PC1_PC2")
        self._render_score_plot(pc_x=0, pc_y=2, tab=self.tab_pc13, key="grafico_PC1_PC3")
        self._render_score_plot_3d(tab=self.tab_pc123, key="grafico_PC1_PC2_PC3")
        self._render_reference_space_2d(
            pc_x=0,
            pc_y=1,
            tab=self.tab_synthesis_pc12,
            key="espaco_referencia_PC1_PC2",
        )
        self._render_reference_space_2d(
            pc_x=0,
            pc_y=2,
            tab=self.tab_synthesis_pc13,
            key="espaco_referencia_PC1_PC3",
        )
        self._render_score_plot_3d(
            tab=self.tab_reference_space_3d,
            key="espaco_referencia_PC1_PC2_PC3",
            reference_space=True,
        )
        self._render_loading_contribution_view()
        self._render_variance_plot()
        self._render_dendrogram_plot()

        self._load_dataframe_into_tree(self.tree_scores, pca_state.scores)

        # A tabela de loadings pode ser muito grande; mostrar os principais, exportar tudo.
        loadings_display = pca_state.loadings.copy()
        pc_cols = [c for c in loadings_display.columns if c.startswith("PC")]
        if pc_cols:
            loadings_display["max_abs_loading"] = loadings_display[pc_cols].abs().max(axis=1)
            loadings_display = loadings_display.sort_values("max_abs_loading", ascending=False)
        self._load_dataframe_into_tree(self.tree_loadings, loadings_display.head(MAX_TABLE_ROWS))

        self._load_dataframe_into_tree(self.tree_distances, pca_state.distances if pca_state.distances is not None else pd.DataFrame())
        if pca_state.config is not None:
            text = "CONFIGURAÇÃO DA RODADA\n" + "=" * 80 + "\n"
            for _, row in pca_state.config.iterrows():
                text += f"{row['chave']}: {row['valor']}\n"
            if pca_state.loadings is not None and len(pca_state.loadings) > MAX_TABLE_ROWS:
                text += (
                    f"\nObservação: a aba Loadings mostra até {MAX_TABLE_ROWS} linhas ordenadas por contribuição absoluta. "
                    "A exportação contém a tabela completa.\n"
                )
            self.config_text.set_text(text)

    def _clear_plot_tab(self, tab, message: str):
        for child in tab.winfo_children():
            child.destroy()
        ttk.Label(tab, text=message).pack(anchor="center", pady=30)

    def _current_visual_scale_mode(self) -> str:
        """Modo de escala visual escolhido na interface para os gráficos PCA."""
        scale_mode = self.var_synthesis_scale.get() if hasattr(self, "var_synthesis_scale") else SYNTHESIS_SCALE_LINEAR
        if scale_mode not in SYNTHESIS_SCALE_OPTIONS:
            scale_mode = SYNTHESIS_SCALE_LINEAR
        return scale_mode

    def _auto_asinh_factor_for_scores_xy(self, xy: np.ndarray, roles: Optional[Sequence[str]] = None) -> float:
        """Fator automático para escala asinh nos gráficos de scores.

        O fator é estimado principalmente pela dispersão das obras de Referência,
        para abrir visualmente a região central sem alterar os scores originais.
        """
        arr = np.asarray(xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            return 1.0
        arr = arr[np.all(np.isfinite(arr), axis=1)]
        if arr.size == 0:
            return 1.0

        ref_arr = arr
        if roles is not None:
            role_values = np.asarray([str(r) for r in roles])
            if len(role_values) == len(np.asarray(xy)):
                finite_mask = np.all(np.isfinite(np.asarray(xy, dtype=float)), axis=1)
                role_values = role_values[finite_mask]
                candidate_ref = arr[role_values == ROLE_REFERENCE]
                if candidate_ref.size:
                    ref_arr = candidate_ref

        # Recentra apenas para estimar escala visual; não muda os scores desenhados.
        ref_center = np.nanmedian(ref_arr, axis=0) if ref_arr.size else np.nanmedian(arr, axis=0)
        ref_radius = np.linalg.norm(ref_arr - ref_center, axis=1) if ref_arr.size else np.array([], dtype=float)
        all_radius = np.linalg.norm(arr - ref_center, axis=1)

        candidates = []
        nonzero_ref = ref_radius[ref_radius > 1e-12]
        if nonzero_ref.size:
            candidates.append(float(np.percentile(nonzero_ref, 50)))
            candidates.append(float(np.percentile(nonzero_ref, 75)) * 0.65)
        nonzero_all = all_radius[all_radius > 1e-12]
        if nonzero_all.size:
            candidates.append(float(np.percentile(nonzero_all, 75)) * 0.25)
            candidates.append(float(np.percentile(nonzero_all, 95)) * 0.08)
        candidates = [c for c in candidates if np.isfinite(c) and c > 1e-12]
        return max(min(candidates) if candidates else 1.0, 1e-9)

    def _render_score_plot(self, pc_x: int, pc_y: int, tab, key: str):
        pca_state = self.pca_state
        for child in tab.winfo_children():
            child.destroy()

        if pca_state.score_values is None or pca_state.explained_ratio is None:
            ttk.Label(tab, text="Execute o PCA para gerar o gráfico.").pack(anchor="center", pady=30)
            return
        if pca_state.score_values.shape[1] <= max(pc_x, pc_y):
            ttk.Label(tab, text=f"Este gráfico requer pelo menos {max(pc_x, pc_y) + 1} PCs.").pack(anchor="center", pady=30)
            return

        explained = pca_state.explained_ratio
        merged = self._scores_with_metadata()
        if merged.empty:
            ttk.Label(tab, text="Não há dados suficientes para gerar o gráfico.").pack(anchor="center", pady=30)
            return

        x_col = f"PC{pc_x + 1}"
        y_col = f"PC{pc_y + 1}"
        if x_col not in merged.columns or y_col not in merged.columns:
            ttk.Label(tab, text="Scores insuficientes para este gráfico.").pack(anchor="center", pady=30)
            return

        fig = Figure(figsize=(7.2, 5.2), dpi=100)
        ax = fig.add_subplot(111)

        color_field = self.var_color_by.get()
        marker_field = self.var_marker_by.get()
        ellipse_field = self.var_ellipse_by.get()
        use_color = color_field != "Nenhum" and color_field in merged.columns
        use_marker = marker_field != "Nenhum" and marker_field in merged.columns
        use_ellipse = ellipse_field != "Nenhum" and ellipse_field in merged.columns

        default_colors = matplotlib.rcParams.get("axes.prop_cycle", None)
        color_list = default_colors.by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"]) if default_colors else ["C0", "C1", "C2", "C3"]
        marker_list = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "8", "*"]

        color_values = merged[color_field].fillna("Sem grupo").astype(str) if use_color else pd.Series(["Todos"] * len(merged))
        marker_values = merged[marker_field].fillna("Sem grupo").astype(str) if use_marker else pd.Series(["Todos"] * len(merged))
        unique_colors = list(dict.fromkeys(color_values.tolist()))
        unique_markers = list(dict.fromkeys(marker_values.tolist()))
        color_map = {grp: color_list[i % len(color_list)] for i, grp in enumerate(unique_colors)}
        marker_map = {grp: marker_list[i % len(marker_list)] for i, grp in enumerate(unique_markers)}
        if use_marker and marker_field == PCA_ROLE_COLUMN:
            marker_map[ROLE_REFERENCE] = "o"
            marker_map[ROLE_SUPPLEMENTARY] = "*"
            marker_map[ROLE_EXCLUDED] = "x"

        plot_df = merged.copy()
        plot_df["__color_group__"] = color_values.values
        plot_df["__marker_group__"] = marker_values.values

        # Escala visual escolhida na GUI. A transformação altera apenas o desenho.
        # Scores, distâncias, centroides e tabelas permanecem nos valores originais.
        scale_mode = self._current_visual_scale_mode()
        raw_xy = plot_df[[x_col, y_col]].to_numpy(dtype=float)
        roles = plot_df[PCA_ROLE_COLUMN].astype(str).tolist() if PCA_ROLE_COLUMN in plot_df.columns else None
        factor = self._auto_asinh_factor_for_scores_xy(raw_xy, roles) if scale_mode == SYNTHESIS_SCALE_ASINH else 1.0
        plot_xy = self._synthesis_visual_transform(raw_xy, scale_mode, factor)
        x_plot_col = f"__{x_col}_plot__"
        y_plot_col = f"__{y_col}_plot__"
        plot_df[x_plot_col] = plot_xy[:, 0]
        plot_df[y_plot_col] = plot_xy[:, 1]

        legend_used = set()
        for (cg, mg), sub in plot_df.groupby(["__color_group__", "__marker_group__"], sort=False):
            label_parts = []
            if use_color:
                label_parts.append(f"{color_field}={cg}")
            if use_marker:
                label_parts.append(f"{marker_field}={mg}")
            label = " | ".join(label_parts) if label_parts else None
            if label in legend_used:
                label = None
            else:
                legend_used.add(label)
            ax.scatter(
                sub[x_plot_col].values,
                sub[y_plot_col].values,
                s=130 if (use_marker and marker_field == PCA_ROLE_COLUMN and mg == ROLE_SUPPLEMENTARY) else 55,
                color=color_map.get(cg, "C0"),
                marker=marker_map.get(mg, "o"),
                label=label,
                alpha=0.95,
            )

        draggable_labels: List = []
        if bool(self.var_show_labels.get()):
            for _, row in plot_df.iterrows():
                annotation = ax.annotate(
                    str(row["obra"]),
                    (row[x_plot_col], row[y_plot_col]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.6},
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#666666",
                        "linewidth": 0.45,
                        "alpha": 0.55,
                        "shrinkA": 1.5,
                        "shrinkB": 2.5,
                    },
                )
                draggable_labels.append(annotation)

        overall_scale = max(
            float(np.ptp(plot_df[x_plot_col].values)) if len(plot_df) > 1 else 0.0,
            float(np.ptp(plot_df[y_plot_col].values)) if len(plot_df) > 1 else 0.0,
            1.0,
        )
        if use_ellipse:
            ellipse_series = plot_df[ellipse_field].fillna("Sem grupo").astype(str)
            plot_df["__ellipse_group__"] = ellipse_series.values
            for group, sub in plot_df.groupby("__ellipse_group__", sort=False):
                pts_raw = sub[[x_col, y_col]].to_numpy(dtype=float)
                if len(pts_raw) < 2:
                    continue
                if use_color and ellipse_field == color_field:
                    ecolor = color_map.get(group, "C0")
                else:
                    idx = list(dict.fromkeys(plot_df["__ellipse_group__"].tolist())).index(group)
                    ecolor = color_list[idx % len(color_list)]
                self._draw_group_ellipse(
                    ax,
                    pts_raw,
                    ecolor,
                    overall_scale,
                    scale_mode=scale_mode,
                    factor=factor,
                )

        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.grid(True, alpha=0.25)
        if scale_mode == SYNTHESIS_SCALE_ASINH:
            ax.set_xlabel(f"PC{pc_x + 1} — escala visual asinh ({fmt_pct(explained[pc_x])})")
            ax.set_ylabel(f"PC{pc_y + 1} — escala visual asinh ({fmt_pct(explained[pc_y])})")
            ax.text(
                0.01,
                0.01,
                f"Escala visual: asinh(score/{factor:.3g}). Cálculos/tabelas usam scores originais.",
                transform=ax.transAxes,
                fontsize=7.5,
                ha="left",
                va="bottom",
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
            )
        else:
            ax.set_xlabel(f"PC{pc_x + 1} ({fmt_pct(explained[pc_x])})")
            ax.set_ylabel(f"PC{pc_y + 1} ({fmt_pct(explained[pc_y])})")
        title = f"Scores PCA — PC{pc_x + 1} × PC{pc_y + 1}"
        if scale_mode == SYNTHESIS_SCALE_ASINH:
            title += " | escala visual asinh"
        if use_ellipse:
            title += f" | regiões 95% por {ellipse_field}"
        ax.set_title(title)
        draggable_legends: List = []
        if bool(self.var_show_legend.get()) and (use_color or use_marker):
            legend = ax.legend(loc="best", fontsize=8)
            draggable_legends.append(legend)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        self._enable_interactive_drag(canvas, draggable_labels, draggable_legends)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        pca_state.figures[key] = fig

    def _render_reference_space_2d(self, pc_x: int, pc_y: int, tab, key: str):
        """Renderiza uma projeção 2D do espaço PCA definido pelas Referências.

        A média, a covariância e a região descritiva são calculadas somente com
        as obras marcadas como Referência. Observações suplementares recebem as
        mesmas coordenadas do PCA e são apenas projetadas na figura.
        """
        p = self.pca_state
        for child in tab.winfo_children():
            child.destroy()

        required_pc_count = max(pc_x, pc_y) + 1
        if p.scores is None or p.explained_ratio is None:
            ttk.Label(tab, text="Execute o PCA para gerar esta projeção.").pack(anchor="center", pady=30)
            return
        if len(p.explained_ratio) < required_pc_count:
            ttk.Label(
                tab,
                text=f"Esta projeção requer pelo menos {required_pc_count} componentes principais.",
            ).pack(anchor="center", pady=30)
            return

        x_col = f"PC{pc_x + 1}"
        y_col = f"PC{pc_y + 1}"
        merged = self._scores_with_metadata()
        if merged.empty or any(col not in merged.columns for col in [x_col, y_col, PCA_ROLE_COLUMN]):
            ttk.Label(tab, text="Scores ou papéis no PCA insuficientes para gerar a síntese.").pack(anchor="center", pady=30)
            return

        work = merged.copy()
        work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[x_col, y_col]).copy()
        ref_mask = work[PCA_ROLE_COLUMN].astype(str).eq(ROLE_REFERENCE)
        reference_raw = work.loc[ref_mask, [x_col, y_col]].to_numpy(dtype=float)
        if len(reference_raw) < 3:
            ttk.Label(
                tab,
                text="A região 2D requer pelo menos três obras de Referência.",
            ).pack(anchor="center", pady=30)
            return

        center = reference_raw.mean(axis=0)
        all_centered = work[[x_col, y_col]].to_numpy(dtype=float) - center.reshape(1, 2)
        work["__x_centered_space__"] = all_centered[:, 0]
        work["__y_centered_space__"] = all_centered[:, 1]
        reference_centered = reference_raw - center.reshape(1, 2)
        covariance = np.asarray(np.cov(reference_centered, rowvar=False, ddof=1), dtype=float)
        covariance_rank = int(np.linalg.matrix_rank(covariance, tol=1e-12))
        region_available = bool(
            covariance.shape == (2, 2)
            and np.isfinite(covariance).all()
            and covariance_rank == 2
        )

        scale_mode = self._current_visual_scale_mode()
        factor = (
            self._auto_asinh_factor_for_scores_xy(all_centered, work[PCA_ROLE_COLUMN].astype(str).tolist())
            if scale_mode == SYNTHESIS_SCALE_ASINH
            else 1.0
        )
        plot_xy = self._synthesis_visual_transform(all_centered, scale_mode, factor)
        work["__x_plot_space__"] = plot_xy[:, 0]
        work["__y_plot_space__"] = plot_xy[:, 1]

        ellipse_plot = np.empty((0, 2), dtype=float)
        mahal2 = np.full(len(work), np.nan, dtype=float)
        if region_available:
            ellipse_raw = self._ellipse_points_from_cov(covariance, CHI2_95_2D)
            ellipse_plot = self._synthesis_visual_transform(ellipse_raw, scale_mode, factor)
            inverse_covariance = np.linalg.pinv(covariance + np.eye(2) * 1e-12)
            mahal2 = np.einsum("ij,jk,ik->i", all_centered, inverse_covariance, all_centered)
            mahal2 = np.maximum(mahal2, 0.0)
        work["__outside_reference_region__"] = mahal2 > CHI2_95_2D

        fig = Figure(figsize=(8.4, 6.4), dpi=100)
        ax = fig.add_subplot(111)
        if ellipse_plot.size:
            ax.plot(
                ellipse_plot[:, 0],
                ellipse_plot[:, 1],
                color="#444444",
                linewidth=1.55,
                linestyle="--",
                alpha=0.92,
                label="Região descritiva das referências (95%)",
                zorder=2,
            )

        phase_field = self._find_metadata_field("fase", "phase")
        if not phase_field:
            color_candidate = self.var_color_by.get()
            if color_candidate not in {"", "Nenhum", PCA_ROLE_COLUMN} and color_candidate in work.columns:
                phase_field = color_candidate
        if phase_field and phase_field in work.columns:
            work["__phase_space__"] = work[phase_field].fillna("Sem fase").astype(str)
        else:
            work["__phase_space__"] = "Referência"

        use_phase_colors = bool(self.var_synthesis_phase_colors.get())
        phase_regions_drawn: List[str] = []
        if use_phase_colors and phase_field:
            reference_phase_df = work.loc[ref_mask].copy()
            for phase, phase_df in reference_phase_df.groupby("__phase_space__", sort=False):
                if len(phase_df) < 3:
                    continue
                phase_points = phase_df[
                    ["__x_centered_space__", "__y_centered_space__"]
                ].to_numpy(dtype=float)
                phase_center = phase_points.mean(axis=0)
                phase_covariance = np.asarray(
                    np.cov(phase_points, rowvar=False, ddof=1),
                    dtype=float,
                )
                if (
                    phase_covariance.shape != (2, 2)
                    or not np.isfinite(phase_covariance).all()
                    or int(np.linalg.matrix_rank(phase_covariance, tol=1e-12)) < 2
                ):
                    continue
                phase_ellipse_raw = (
                    self._ellipse_points_from_cov(phase_covariance, CHI2_95_2D)
                    + phase_center.reshape(1, 2)
                )
                phase_ellipse_plot = self._synthesis_visual_transform(
                    phase_ellipse_raw,
                    scale_mode,
                    factor,
                )
                phase_color = matplotlib.colors.to_hex(self._phase_rgb(str(phase)))
                ax.fill(
                    phase_ellipse_plot[:, 0],
                    phase_ellipse_plot[:, 1],
                    facecolor=phase_color,
                    edgecolor=phase_color,
                    linewidth=1.2,
                    alpha=0.14,
                    zorder=1,
                )
                ax.plot(
                    phase_ellipse_plot[:, 0],
                    phase_ellipse_plot[:, 1],
                    color=phase_color,
                    linewidth=1.25,
                    alpha=0.82,
                    zorder=2,
                )
                phase_regions_drawn.append(str(phase))

        phase_order: List[str] = []
        label_anchors: List[Tuple[float, float]] = []
        label_texts: List[str] = []
        for _, row in work.iterrows():
            role = str(row.get(PCA_ROLE_COLUMN, ""))
            phase = str(row.get("__phase_space__", "Sem fase"))
            if use_phase_colors and role == ROLE_REFERENCE and phase not in phase_order:
                phase_order.append(phase)
            x = float(row["__x_plot_space__"])
            y = float(row["__y_plot_space__"])
            outside = bool(row.get("__outside_reference_region__", False))
            edge_color = "#b22222" if outside else "#202020"
            line_width = 1.9 if outside else 0.9
            if role == ROLE_REFERENCE:
                face_color = (
                    matplotlib.colors.to_hex(self._phase_rgb(phase))
                    if use_phase_colors
                    else "#bdbdbd"
                )
                marker = "o"
                size = 78
            elif role == ROLE_SUPPLEMENTARY:
                face_color = "none"
                marker = "*"
                size = 150
            else:
                face_color = "none"
                marker = "x"
                size = 82
            ax.scatter(
                [x], [y],
                s=size,
                marker=marker,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
                alpha=0.98,
                zorder=6 if role == ROLE_REFERENCE else 7,
            )
            if bool(self.var_show_labels.get()):
                label_anchors.append((x, y))
                label_texts.append(str(row["obra"]))

        ax.scatter([0.0], [0.0], s=30, marker="+", color="black", linewidths=1.25, zorder=8)
        ax.annotate(
            "centro das referências",
            (0.0, 0.0),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )
        ax.axhline(0.0, linewidth=0.8, color="black", alpha=0.55)
        ax.axvline(0.0, linewidth=0.8, color="black", alpha=0.55)
        ax.grid(True, alpha=0.20, linewidth=0.6)

        draggable_labels: List = []
        if label_anchors:
            draggable_labels = self._add_repelled_labels(
                fig,
                ax,
                label_anchors,
                label_texts,
                fontsize=8.5,
                text_color="#111111",
                zorder=8,
            )

        explained = p.explained_ratio
        scale_suffix = " — escala visual asinh" if scale_mode == SYNTHESIS_SCALE_ASINH else ""
        ax.set_xlabel(f"{x_col}{scale_suffix} ({fmt_pct(float(explained[pc_x]))})")
        ax.set_ylabel(f"{y_col}{scale_suffix} ({fmt_pct(float(explained[pc_y]))})")
        ax.set_title(f"{self._reference_space_label()} — projeção {x_col} × {y_col}")

        note = "Região calculada somente com as obras de Referência; não constitui fronteira de autoria."
        if not region_available:
            note = "Região não desenhada: a covariância das Referências não possui posto completo."
        if phase_regions_drawn:
            note += " Regiões coloridas de fase: somente grupos com pelo menos três Referências."
        if scale_mode == SYNTHESIS_SCALE_ASINH:
            note += f" Escala visual: asinh(score/{factor:.3g})."
        ax.text(
            0.01,
            0.01,
            note,
            transform=ax.transAxes,
            fontsize=7.4,
            ha="left",
            va="bottom",
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
            zorder=10,
        )

        draggable_legends: List = []
        if bool(self.var_show_legend.get()):
            category_handles = [
                Line2D([0], [0], marker="o", linestyle="None", markersize=7.5,
                       markerfacecolor="#bdbdbd", markeredgecolor="#202020",
                       label="Obra de referência"),
                Line2D([0], [0], marker="*", linestyle="None", markersize=10,
                       markerfacecolor="none", markeredgecolor="#202020",
                       label="Observação suplementar"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=7.5,
                       markerfacecolor="none", markeredgecolor="#b22222", markeredgewidth=1.9,
                       label="Fora da região descritiva"),
            ]
            if ellipse_plot.size:
                category_handles.append(
                    Line2D([0], [0], linestyle="--", linewidth=1.55, color="#444444",
                           label="Região descritiva das referências (95%)")
                )
            legend_classes = ax.legend(
                handles=category_handles,
                loc="upper right",
                fontsize=8,
                frameon=True,
            )
            ax.add_artist(legend_classes)
            draggable_legends.append(legend_classes)

            phase_handles = [
                Patch(
                    facecolor=matplotlib.colors.to_hex(self._phase_rgb(phase)),
                    edgecolor="#202020",
                    linewidth=0.5,
                    label=phase,
                )
                for phase in phase_order
            ]
            if phase_handles:
                legend_phases = ax.legend(
                    handles=phase_handles,
                    loc="upper left",
                    fontsize=8,
                    frameon=True,
                    title="Fases",
                )
                ax.add_artist(legend_phases)
                draggable_legends.append(legend_phases)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        self._enable_interactive_drag(canvas, draggable_labels, draggable_legends)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        p.figures[key] = fig

    def _auto_asinh_factor_for_scores_xyz(self, xyz: np.ndarray, roles: Optional[Sequence[str]] = None) -> float:
        """Fator automático da escala asinh para PC1 × PC2 × PC3.

        O critério é o mesmo dos gráficos 2D, agora usando a distância radial
        tridimensional. A transformação permanece estritamente visual.
        """
        arr_full = np.asarray(xyz, dtype=float)
        if arr_full.ndim != 2 or arr_full.shape[1] != 3 or arr_full.size == 0:
            return 1.0

        finite_mask = np.all(np.isfinite(arr_full), axis=1)
        arr = arr_full[finite_mask]
        if arr.size == 0:
            return 1.0

        ref_arr = arr
        if roles is not None:
            role_values = np.asarray([str(r) for r in roles])
            if len(role_values) == len(arr_full):
                candidate_ref = arr[role_values[finite_mask] == ROLE_REFERENCE]
                if candidate_ref.size:
                    ref_arr = candidate_ref

        ref_center = np.nanmedian(ref_arr, axis=0) if ref_arr.size else np.nanmedian(arr, axis=0)
        ref_radius = np.linalg.norm(ref_arr - ref_center, axis=1) if ref_arr.size else np.array([], dtype=float)
        all_radius = np.linalg.norm(arr - ref_center, axis=1)

        candidates = []
        nonzero_ref = ref_radius[ref_radius > 1e-12]
        if nonzero_ref.size:
            candidates.append(float(np.percentile(nonzero_ref, 50)))
            candidates.append(float(np.percentile(nonzero_ref, 75)) * 0.65)
        nonzero_all = all_radius[all_radius > 1e-12]
        if nonzero_all.size:
            candidates.append(float(np.percentile(nonzero_all, 75)) * 0.25)
            candidates.append(float(np.percentile(nonzero_all, 95)) * 0.08)
        candidates = [c for c in candidates if np.isfinite(c) and c > 1e-12]
        return max(min(candidates) if candidates else 1.0, 1e-9)

    def _render_score_plot_3d(self, tab, key: str, reference_space: bool = False):
        """Renderiza PC1 × PC2 × PC3 com câmera controlável.

        No modo ``reference_space``, desenha uma única região tridimensional
        descritiva calculada exclusivamente com as obras de Referência. Esse
        modo é usado na subaba da Síntese PCA e não depende do agrupamento
        selecionado no controle de elipses por metadados.
        """
        pca_state = self.pca_state
        for child in tab.winfo_children():
            child.destroy()

        if pca_state.score_values is None or pca_state.explained_ratio is None:
            ttk.Label(tab, text="Execute o PCA para gerar o gráfico tridimensional.").pack(anchor="center", pady=30)
            return
        if pca_state.score_values.shape[1] < 3:
            ttk.Label(tab, text="Este gráfico requer pelo menos 3 PCs.").pack(anchor="center", pady=30)
            return

        merged = self._scores_with_metadata()
        required_cols = ["PC1", "PC2", "PC3"]
        if merged.empty or any(col not in merged.columns for col in required_cols):
            ttk.Label(tab, text="Scores insuficientes para gerar PC1 × PC2 × PC3.").pack(anchor="center", pady=30)
            return

        fig = Figure(figsize=(7.8, 6.1), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        color_field = self.var_color_by.get()
        marker_field = self.var_marker_by.get()
        ellipse_field = self.var_ellipse_by.get()
        use_color = color_field != "Nenhum" and color_field in merged.columns
        use_marker = marker_field != "Nenhum" and marker_field in merged.columns
        use_ellipsoid = reference_space or (ellipse_field != "Nenhum" and ellipse_field in merged.columns)

        default_colors = matplotlib.rcParams.get("axes.prop_cycle", None)
        color_list = default_colors.by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"]) if default_colors else ["C0", "C1", "C2", "C3"]
        marker_list = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "8", "*"]

        color_values = merged[color_field].fillna("Sem grupo").astype(str) if use_color else pd.Series(["Todos"] * len(merged))
        marker_values = merged[marker_field].fillna("Sem grupo").astype(str) if use_marker else pd.Series(["Todos"] * len(merged))
        unique_colors = list(dict.fromkeys(color_values.tolist()))
        unique_markers = list(dict.fromkeys(marker_values.tolist()))
        color_map = {grp: color_list[i % len(color_list)] for i, grp in enumerate(unique_colors)}
        marker_map = {grp: marker_list[i % len(marker_list)] for i, grp in enumerate(unique_markers)}
        if use_marker and marker_field == PCA_ROLE_COLUMN:
            marker_map[ROLE_REFERENCE] = "o"
            marker_map[ROLE_SUPPLEMENTARY] = "*"
            marker_map[ROLE_EXCLUDED] = "x"

        plot_df = merged.copy()
        plot_df["__color_group__"] = color_values.values
        plot_df["__marker_group__"] = marker_values.values

        scale_mode = self._current_visual_scale_mode()
        raw_xyz = plot_df[required_cols].to_numpy(dtype=float)
        roles = plot_df[PCA_ROLE_COLUMN].astype(str).tolist() if PCA_ROLE_COLUMN in plot_df.columns else None
        factor = self._auto_asinh_factor_for_scores_xyz(raw_xyz, roles) if scale_mode == SYNTHESIS_SCALE_ASINH else 1.0
        plot_xyz = self._synthesis_visual_transform(raw_xyz, scale_mode, factor)
        plot_cols = ["__PC1_plot_3d__", "__PC2_plot_3d__", "__PC3_plot_3d__"]
        for idx, col in enumerate(plot_cols):
            plot_df[col] = plot_xyz[:, idx]

        overall_scale = max(
            float(np.ptp(plot_xyz[:, 0])) if len(plot_xyz) > 1 else 0.0,
            float(np.ptp(plot_xyz[:, 1])) if len(plot_xyz) > 1 else 0.0,
            float(np.ptp(plot_xyz[:, 2])) if len(plot_xyz) > 1 else 0.0,
            1e-9,
        )

        cloud_handles: List = []
        degenerate_regions = 0
        axes_2points = 0
        clouds_drawn = 0
        reference_region_note = ""
        if reference_space:
            if PCA_ROLE_COLUMN not in plot_df.columns:
                reference_region_note = "Região 3D indisponível: a coluna de papel no PCA não foi encontrada."
            else:
                reference_df = plot_df.loc[
                    plot_df[PCA_ROLE_COLUMN].astype(str).eq(ROLE_REFERENCE)
                ]
                reference_points = reference_df[required_cols].to_numpy(dtype=float)
                reference_points = reference_points[np.all(np.isfinite(reference_points), axis=1)]
                covariance_rank = 0
                if len(reference_points) >= 4:
                    covariance_rank = int(
                        np.linalg.matrix_rank(
                            np.cov(reference_points, rowvar=False, ddof=1),
                            tol=1e-12,
                        )
                    )
                if len(reference_points) < 4 or covariance_rank < 3:
                    reference_region_note = (
                        "Região 3D não desenhada: são necessárias ao menos quatro "
                        "referências e covariância de posto completo."
                    )
                else:
                    cloud_color = "#555555"
                    cloud_info = self._draw_group_ellipsoid_3d(
                        ax,
                        reference_points,
                        cloud_color,
                        overall_scale,
                        scale_mode=scale_mode,
                        factor=factor,
                    )
                    if cloud_info is not None and not bool(cloud_info.get("degenerado", False)):
                        clouds_drawn = 1
                        cloud_handles.append(
                            Patch(
                                facecolor=cloud_color,
                                edgecolor=cloud_color,
                                alpha=0.16,
                                label="Região descritiva das referências (95%)",
                            )
                        )
                    else:
                        reference_region_note = (
                            "Região 3D não desenhada: a covariância das referências "
                            "não definiu um elipsoide tridimensional completo."
                        )
        elif use_ellipsoid:
            ellipse_series = plot_df[ellipse_field].fillna("Sem grupo").astype(str)
            plot_df["__ellipse_group__"] = ellipse_series.values
            ellipse_groups = list(dict.fromkeys(plot_df["__ellipse_group__"].tolist()))
            for group, sub in plot_df.groupby("__ellipse_group__", sort=False):
                pts_raw = sub[required_cols].to_numpy(dtype=float)
                if len(pts_raw) < 2:
                    continue
                if use_color and ellipse_field == color_field:
                    cloud_color = color_map.get(group, "C0")
                else:
                    cloud_color = color_list[ellipse_groups.index(group) % len(color_list)]
                cloud_info = self._draw_group_ellipsoid_3d(
                    ax,
                    pts_raw,
                    cloud_color,
                    overall_scale,
                    scale_mode=scale_mode,
                    factor=factor,
                )
                if cloud_info is None:
                    continue
                clouds_drawn += 1
                if cloud_info.get("tipo") == "eixo_principal_95":
                    axes_2points += 1
                if bool(cloud_info.get("degenerado", False)) and cloud_info.get("tipo") == "elipsoide_95":
                    degenerate_regions += 1
                if not (use_color and ellipse_field == color_field):
                    if cloud_info.get("tipo") == "eixo_principal_95":
                        cloud_handles.append(
                            Line2D(
                                [0], [0],
                                color=cloud_color,
                                linewidth=1.9,
                                linestyle="--",
                                label=f"{ellipse_field}={group} (eixo maior 95%)",
                            )
                        )
                    else:
                        cloud_handles.append(
                            Patch(
                                facecolor=cloud_color,
                                edgecolor=cloud_color,
                                alpha=0.20,
                                label=f"{ellipse_field}={group} (elipsoide 95%)",
                            )
                        )

        # Os pontos são desenhados depois das superfícies para permanecerem legíveis.
        legend_used = set()
        for (cg, mg), sub in plot_df.groupby(["__color_group__", "__marker_group__"], sort=False):
            label_parts = []
            if use_color:
                label_parts.append(f"{color_field}={cg}")
            if use_marker:
                label_parts.append(f"{marker_field}={mg}")
            label = " | ".join(label_parts) if label_parts else None
            if label in legend_used:
                label = None
            else:
                legend_used.add(label)
            ax.scatter(
                sub[plot_cols[0]].values,
                sub[plot_cols[1]].values,
                sub[plot_cols[2]].values,
                s=145 if (use_marker and marker_field == PCA_ROLE_COLUMN and mg == ROLE_SUPPLEMENTARY) else 62,
                color=color_map.get(cg, "C0"),
                marker=marker_map.get(mg, "o"),
                label=label,
                alpha=0.98,
                depthshade=True,
            )

        if bool(self.var_show_labels.get()):
            ranges = np.ptp(plot_xyz, axis=0) if len(plot_xyz) > 1 else np.zeros(3, dtype=float)
            offsets = np.maximum(ranges * 0.012, overall_scale * 0.006)
            for _, row in plot_df.iterrows():
                ax.text(
                    float(row[plot_cols[0]]) + float(offsets[0]),
                    float(row[plot_cols[1]]) + float(offsets[1]),
                    float(row[plot_cols[2]]) + float(offsets[2]),
                    str(row["obra"]),
                    fontsize=8.5,
                    color="#111111",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.6},
                )

        # Linhas que passam pela origem auxiliam a comparação precisa dos sinais.
        x_min, x_max = float(np.nanmin(plot_xyz[:, 0])), float(np.nanmax(plot_xyz[:, 0]))
        y_min, y_max = float(np.nanmin(plot_xyz[:, 1])), float(np.nanmax(plot_xyz[:, 1]))
        z_min, z_max = float(np.nanmin(plot_xyz[:, 2])), float(np.nanmax(plot_xyz[:, 2]))
        ax.plot([x_min, x_max], [0.0, 0.0], [0.0, 0.0], color="#666666", linewidth=0.7, alpha=0.75)
        ax.plot([0.0, 0.0], [y_min, y_max], [0.0, 0.0], color="#666666", linewidth=0.7, alpha=0.75)
        ax.plot([0.0, 0.0], [0.0, 0.0], [z_min, z_max], color="#666666", linewidth=0.7, alpha=0.75)

        explained = pca_state.explained_ratio
        scale_suffix = " [asinh]" if scale_mode == SYNTHESIS_SCALE_ASINH else ""
        ax.set_xlabel(f"PC1 ({fmt_pct(explained[0])}){scale_suffix}", labelpad=9)
        ax.set_ylabel(f"PC2 ({fmt_pct(explained[1])}){scale_suffix}", labelpad=9)
        ax.set_zlabel(f"PC3 ({fmt_pct(explained[2])}){scale_suffix}", labelpad=9)
        title = (
            f"{self._reference_space_label()} — PC1 × PC2 × PC3"
            if reference_space
            else "Scores PCA — PC1 × PC2 × PC3"
        )
        if scale_mode == SYNTHESIS_SCALE_ASINH:
            title += " | escala visual asinh"
        if clouds_drawn:
            if reference_space:
                title += " | região descritiva das referências"
            else:
                title += f" | regiões 95% por {ellipse_field}"
        ax.set_title(title, pad=16)
        ax.grid(True, alpha=0.25)

        initial_elev = 24.0
        initial_azim = -58.0
        ax.view_init(elev=initial_elev, azim=initial_azim)
        data_ranges = np.ptp(plot_xyz, axis=0)
        if np.all(np.isfinite(data_ranges)) and np.any(data_ranges > 1e-12):
            ax.set_box_aspect(np.maximum(data_ranges, 1e-12))

        draggable_legends: List = []
        if bool(self.var_show_legend.get()) and (use_color or use_marker or cloud_handles):
            handles, labels = ax.get_legend_handles_labels()
            handles.extend(cloud_handles)
            labels.extend([h.get_label() for h in cloud_handles])
            unique_entries = []
            seen_labels = set()
            for handle, label in zip(handles, labels):
                if not label or label in seen_labels:
                    continue
                seen_labels.add(label)
                unique_entries.append((handle, label))
            if unique_entries:
                legend = ax.legend(
                    handles=[item[0] for item in unique_entries],
                    labels=[item[1] for item in unique_entries],
                    loc="upper left",
                    fontsize=8,
                )
                draggable_legends.append(legend)

        notes = ["Arraste o gráfico para girar; o PNG preserva a orientação exibida."]
        if scale_mode == SYNTHESIS_SCALE_ASINH:
            notes.append(f"Escala visual: asinh(score/{factor:.3g}); cálculos e tabelas usam scores originais.")
        if reference_space and clouds_drawn:
            notes.append(
                f"Região global descritiva das Referências: χ²(3)={CHI2_95_3D:.3f}; "
                "não constitui fronteira de autoria ou classificação."
            )
        elif reference_space and reference_region_note:
            notes.append(reference_region_note)
        elif clouds_drawn:
            cloud_note = (
                f"Regiões 3D de 95%: χ²(3)={CHI2_95_3D:.3f}; "
                "2 obras mostram somente o eixo principal maior pontilhado."
            )
            if axes_2points:
                cloud_note += f" Grupos com 2 obras: {axes_2points}."
            if degenerate_regions:
                cloud_note += f" Elipsoides degenerados/planares: {degenerate_regions}."
            notes.append(cloud_note)
        fig.text(
            0.01,
            0.012,
            "\n".join(notes),
            fontsize=7.4,
            ha="left",
            va="bottom",
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
        )
        bottom_margin = 0.105 if len(notes) >= 3 else 0.075
        fig.tight_layout(rect=(0.0, bottom_margin, 1.0, 1.0))

        canvas = FigureCanvasTkAgg(fig, master=tab)
        controls = ttk.Frame(tab)
        controls.pack(side="top", fill="x", padx=4, pady=(2, 0))

        preset_row = ttk.Frame(controls)
        preset_row.pack(fill="x")
        ttk.Label(preset_row, text="Vistas:").pack(side="left", padx=(0, 4))

        elev_var = tk.DoubleVar(master=tab, value=initial_elev)
        azim_var = tk.DoubleVar(master=tab, value=initial_azim)

        def redraw_canvas():
            draw_idle = getattr(canvas, "draw_idle", None)
            if callable(draw_idle):
                draw_idle()
            else:
                canvas.draw()

        def apply_view(elev: float, azim: float):
            elev_value = max(-90.0, min(90.0, float(elev)))
            azim_value = ((float(azim) + 180.0) % 360.0) - 180.0
            ax.view_init(elev=elev_value, azim=azim_value)
            elev_var.set(round(elev_value, 1))
            azim_var.set(round(azim_value, 1))
            redraw_canvas()

        view_presets = [
            ("Inicial", initial_elev, initial_azim),
            ("PC1 × PC2", 90.0, -90.0),
            ("PC1 × PC3", 0.0, -90.0),
            ("PC2 × PC3", 0.0, 0.0),
            ("Vista oposta", initial_elev, 122.0),
        ]
        for label, elev, azim in view_presets:
            ttk.Button(
                preset_row,
                text=label,
                command=lambda e=elev, a=azim: apply_view(e, a),
            ).pack(side="left", padx=2)

        angle_row = ttk.Frame(controls)
        angle_row.pack(fill="x", pady=(2, 0))
        ttk.Label(angle_row, text="Elevação (°):").pack(side="left")
        ttk.Spinbox(angle_row, from_=-90, to=90, increment=1, textvariable=elev_var, width=7).pack(side="left", padx=(3, 10))
        ttk.Label(angle_row, text="Azimute (°):").pack(side="left")
        ttk.Spinbox(angle_row, from_=-180, to=180, increment=1, textvariable=azim_var, width=7).pack(side="left", padx=(3, 6))

        def apply_numeric_view():
            try:
                apply_view(float(elev_var.get()), float(azim_var.get()))
            except Exception:
                messagebox.showwarning(
                    "Ângulos inválidos",
                    "Informe elevação entre −90° e 90° e azimute entre −180° e 180°.",
                )

        ttk.Button(angle_row, text="Aplicar ângulos", command=apply_numeric_view).pack(side="left", padx=2)

        def export_current_3d_png():
            initial_name = (
                f"{safe_slug(self._reference_space_label())}_PC1_PC2_PC3.png"
                if reference_space
                else "grafico_PC1_PC2_PC3.png"
            )
            file_str = filedialog.asksaveasfilename(
                title="Exportar a vista tridimensional atual",
                defaultextension=".png",
                initialfile=initial_name,
                filetypes=[("Imagem PNG", "*.png")],
            )
            if not file_str:
                return
            try:
                fig.savefig(file_str, dpi=400, bbox_inches="tight")
                messagebox.showinfo(
                    "Exportação concluída",
                    f"Vista tridimensional salva em:\n{file_str}",
                )
            except Exception as exc:
                messagebox.showerror("Erro ao exportar PNG 3D", str(exc))

        ttk.Button(angle_row, text="Exportar PNG 3D", command=export_current_3d_png).pack(side="left", padx=(6, 2))
        ttk.Label(
            angle_row,
            text="Os ângulos acompanham o giro com o mouse.",
        ).pack(side="left", padx=(10, 0))

        def sync_angles_after_mouse(_event=None):
            elev_var.set(round(float(ax.elev), 1))
            azim_var.set(round((((float(ax.azim) + 180.0) % 360.0) - 180.0), 1))

        connection_id = canvas.mpl_connect("button_release_event", sync_angles_after_mouse)
        canvas.draw()
        self._enable_interactive_drag(canvas, legends=draggable_legends)
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas._pca_3d_view_controls = {
            "elevation_var": elev_var,
            "azimuth_var": azim_var,
            "connection_id": connection_id,
            "apply_view": apply_view,
        }
        pca_state.figures[key] = fig


    def _render_dendrogram_plot(self):
        tab = self.tab_dendrogram
        for child in tab.winfo_children():
            child.destroy()
        fig = self.pca_state.figures.get("dendrograma_obras")
        if fig is None:
            ttk.Label(tab, text="Dendrograma não gerado nesta rodada.").pack(anchor="center", pady=30)
            return
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _render_variance_plot(self):
        pca_state = self.pca_state
        tab = self.tab_variance
        for child in tab.winfo_children():
            child.destroy()

        if pca_state.variance is None or pca_state.explained_ratio is None:
            ttk.Label(tab, text="Execute o PCA para gerar o gráfico.").pack(anchor="center", pady=30)
            return

        explained = np.asarray(pca_state.explained_ratio, dtype=float)
        labels = [f"PC{i + 1}" for i in range(len(explained))]
        x = np.arange(len(explained), dtype=float)

        explained_pct = explained * 100.0
        cumulative_pct = np.cumsum(explained_pct)

        fig = Figure(figsize=(7.2, 5.2), dpi=100)
        ax = fig.add_subplot(111)
        bars = ax.bar(x, explained_pct, label="Variância individual")
        ax.plot(x, cumulative_pct, marker="o", label="Variância acumulada")

        # Percentual individual é escrito dentro do topo de cada barra para não
        # colidir com a curva acumulada (em PC1, ambos possuem exatamente o mesmo valor).
        for bar, individual_pct in zip(bars, explained_pct):
            if not np.isfinite(individual_pct):
                continue
            bar_height = float(bar.get_height())
            if bar_height >= 3.0:
                y_text = bar_height
                offset = (0, -4)
                va = "top"
                color = "white"
            else:
                y_text = bar_height
                offset = (0, 3)
                va = "bottom"
                color = "#1f1f1f"
            ax.annotate(
                f"{individual_pct:.1f}%".replace(".", ","),
                xy=(bar.get_x() + bar.get_width() / 2.0, y_text),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
                color=color,
            )

        # Mostra sobre cada ponto da curva o percentual acumulado até aquele PC.
        # Ex.: no ponto PC3, o rótulo corresponde a PC1 + PC2 + PC3.
        for xi, yi in zip(x, cumulative_pct):
            ax.annotate(
                f"Σ={yi:.1f}%".replace(".", ","),
                xy=(xi, yi),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Variância explicada (%)")
        ax.set_title("Variância explicada e acumulada")
        ax.grid(True, axis="y", alpha=0.25)
        legend = ax.legend(loc="best")

        max_y = float(np.nanmax(cumulative_pct)) if cumulative_pct.size else 100.0
        ax.set_ylim(0, max(100.0, max_y) + 8.0)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        self._enable_interactive_drag(canvas, legends=[legend])
        canvas.get_tk_widget().pack(fill="both", expand=True)
        pca_state.figures["grafico_variancia_explicada"] = fig

    def _load_dataframe_into_tree(self, tree: ttk.Treeview, df: pd.DataFrame):
        tree.delete(*tree.get_children())
        tree["columns"] = []
        if df is None or df.empty:
            return

        columns = [str(c) for c in df.columns]
        tree["columns"] = columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=max(90, min(240, len(col) * 9)), anchor="center")

        for _, row in df.iterrows():
            values = []
            for val in row.tolist():
                if isinstance(val, float):
                    if math.isfinite(val):
                        values.append(f"{val:.8g}")
                    else:
                        values.append(str(val))
                else:
                    values.append(str(val))
            tree.insert("", "end", values=values)

    # ------------------------------------------------------------------
    # Exportações
    # ------------------------------------------------------------------

    def _ensure_pca_done(self) -> bool:
        if self.pca_state.matrix_used is None:
            messagebox.showwarning("Sem PCA", "Rode o PCA antes de exportar resultados.")
            return False
        return True

    def _make_export_dir(self, base_folder: Path) -> Path:
        state = self.load_state
        stem = safe_slug(state.path.stem if state.path else "matriz")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"PCA_{stem}_{timestamp}"
        out_dir = base_folder / base_name
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        for idx in range(2, 100):
            candidate = base_folder / f"{base_name}_{idx:02d}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
        raise FileExistsError("Não foi possível criar uma pasta única de exportação para esta rodada.")


    def export_essential_run(self):
        """Exporta um pacote reduzido para tese/artigo, sem dezenas de tabelas auxiliares.

        A exportação detalhada continua disponível no botão específico. Este pacote
        prioriza os arquivos usados para justificar a metodologia e interpretar a PCA.
        """
        if not self._ensure_pca_done():
            return
        folder_str = filedialog.askdirectory(title="Selecionar pasta para exportar o pacote essencial")
        if not folder_str:
            return
        try:
            self._set_progress(5, "Criando pacote essencial...")
            state = self.load_state
            stem = safe_slug(state.path.stem if state.path else "matriz")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(folder_str) / f"PCA_{stem}_{timestamp}_ESSENCIAL"
            out_dir.mkdir(parents=True, exist_ok=False)
            self._write_essential_outputs(out_dir)
            self._set_progress(100, f"Pacote essencial exportado: {out_dir}")
            self._reset_progress_later()
            messagebox.showinfo("Exportação concluída", f"Pacote essencial exportado em:\n{out_dir}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Erro ao exportar", str(exc))
            self._set_progress(0, "Erro ao exportar pacote essencial.")

    def _write_essential_outputs(self, out_dir: Path):
        """Escreve apenas os arquivos centrais para seleção metodológica e tese."""
        p = self.pca_state
        assert p.scores is not None
        assert p.loadings is not None
        assert p.variance is not None

        dados_dir = out_dir / "dados"
        figs_dir = out_dir / "figuras"
        dados_dir.mkdir(parents=True, exist_ok=True)
        figs_dir.mkdir(parents=True, exist_ok=True)

        self._set_progress(15, "Exportando CSVs essenciais...")
        p.scores.to_csv(dados_dir / "01_scores_PCA.csv", index=False, encoding="utf-8-sig")
        p.loadings.to_csv(dados_dir / "02_loadings_PCA.csv", index=False, encoding="utf-8-sig")
        p.variance.to_csv(dados_dir / "03_variancia_explicada.csv", index=False, encoding="utf-8-sig")
        if p.preprocess_parameters is not None and not p.preprocess_parameters.empty:
            p.preprocess_parameters.to_csv(dados_dir / "00_parametros_preprocessamento_PCA.csv", index=False, encoding="utf-8-sig")
        if p.area_normalization_audit is not None and not p.area_normalization_audit.empty:
            p.area_normalization_audit.to_csv(
                dados_dir / "00b_auditoria_normalizacao_area_H.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if p.loading_contributions is not None and not p.loading_contributions.empty:
            p.loading_contributions.to_csv(dados_dir / "04_contribuicao_loadings_por_PC.csv", index=False, encoding="utf-8-sig")
        if p.loading_dominance is not None and not p.loading_dominance.empty:
            p.loading_dominance.to_csv(dados_dir / "05_resumo_dominancia_loadings.csv", index=False, encoding="utf-8-sig")

        self._set_progress(40, "Exportando síntese PCA essencial...")
        scores_centered, phase_centroids, phase_distances, _meta = self._build_synthesis_tables()
        if scores_centered is not None and not scores_centered.empty:
            scores_centered.to_csv(dados_dir / "06_scores_PCA_recentrados_sintese.csv", index=False, encoding="utf-8-sig")
        if phase_centroids is not None and not phase_centroids.empty:
            phase_centroids.to_csv(dados_dir / "07_centroides_fases_PCA_recentrado.csv", index=False, encoding="utf-8-sig")
        if phase_distances is not None and not phase_distances.empty:
            phase_distances.to_csv(dados_dir / "08_distancias_aos_centroides_fases.csv", index=False, encoding="utf-8-sig")
        if p.config is not None:
            p.config.to_csv(dados_dir / "09_configuracao_rodada_PCA.csv", index=False, encoding="utf-8-sig")

        self._set_progress(65, "Atualizando figuras essenciais...")
        # Gera apenas figuras ainda ausentes. Se o usuário já arrastou rótulos
        # ou legendas, reutilizar a Figure existente preserva essas posições.
        if "espaco_referencia_PC1_PC2" not in p.figures:
            self._render_reference_space_2d(0, 1, self.tab_synthesis_pc12, "espaco_referencia_PC1_PC2")
        if len(p.explained_ratio) >= 3 and "espaco_referencia_PC1_PC3" not in p.figures:
            self._render_reference_space_2d(0, 2, self.tab_synthesis_pc13, "espaco_referencia_PC1_PC3")
        if len(p.explained_ratio) >= 3 and "espaco_referencia_PC1_PC2_PC3" not in p.figures:
            self._render_score_plot_3d(
                self.tab_reference_space_3d,
                "espaco_referencia_PC1_PC2_PC3",
                reference_space=True,
            )
        if "figura_contribuicao_loadings_resumo" not in p.figures:
            self._render_loading_contribution_view()

        essential_keys = [
            "grafico_PC1_PC2",
            "grafico_PC1_PC3",
            "grafico_PC1_PC2_PC3",
            "espaco_referencia_PC1_PC2",
            "espaco_referencia_PC1_PC3",
            "espaco_referencia_PC1_PC2_PC3",
            "figura_contribuicao_loadings_resumo",
            "grafico_variancia_explicada",
        ]
        # Inclui também os gráficos individuais de contribuição dos primeiros PCs, se existirem.
        essential_keys.extend([k for k in p.figures.keys() if str(k).startswith("grafico_contribuicao_loadings_PC")])

        saved = set()
        for key in essential_keys:
            fig = p.figures.get(key)
            if fig is None or key in saved:
                continue
            if str(key).startswith("espaco_referencia_"):
                suffix = str(key).replace("espaco_referencia_", "")
                filename = f"{safe_slug(self._reference_space_label())}_{safe_slug(suffix)}.png"
            else:
                filename = f"{safe_slug(key)}.png"
            fig.savefig(figs_dir / filename, dpi=400, bbox_inches="tight")
            saved.add(key)

    def export_full_run(self):
        if not self._ensure_pca_done():
            return
        folder_str = filedialog.askdirectory(title="Selecionar pasta para exportar a rodada PCA")
        if not folder_str:
            return
        try:
            self._set_progress(5, "Criando pasta de exportação...")
            out_dir = self._make_export_dir(Path(folder_str))
            self._write_all_outputs(out_dir)
            self._set_progress(100, f"Rodada exportada: {out_dir}")
            self._reset_progress_later()
            messagebox.showinfo("Exportação concluída", f"Rodada PCA exportada em:\n{out_dir}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Erro ao exportar", str(exc))
            self._set_progress(0, "Erro ao exportar rodada.")

    def export_single(self, kind: str):
        if not self._ensure_pca_done():
            return
        pca_state = self.pca_state
        try:
            if kind == "figures":
                folder_str = filedialog.askdirectory(title="Selecionar pasta para exportar os gráficos")
                if not folder_str:
                    return
                out = Path(folder_str)
                self._set_progress(20, "Exportando gráficos...")
                self._save_figures(out)
                self._set_progress(100, f"Gráficos exportados: {out}")
                self._reset_progress_later()
                messagebox.showinfo("Exportação concluída", f"Gráficos exportados em:\n{out}")
                return

            if kind == "loading_contributions":
                folder_str = filedialog.askdirectory(title="Selecionar pasta para exportar contribuição dos loadings")
                if not folder_str:
                    return
                out = Path(folder_str)
                out.mkdir(parents=True, exist_ok=True)
                if pca_state.loading_contributions is None or pca_state.loading_contributions.empty:
                    raise ValueError("A tabela de contribuição dos loadings não está disponível. Rode o PCA novamente.")
                self._set_progress(25, "Exportando contribuição dos loadings...")
                pca_state.loading_contributions.to_csv(out / "contribuicao_loadings_por_PC.csv", index=False, encoding="utf-8-sig")
                if pca_state.loading_dominance is not None and not pca_state.loading_dominance.empty:
                    pca_state.loading_dominance.to_csv(out / "resumo_dominancia_loadings.csv", index=False, encoding="utf-8-sig")
                self._set_progress(100, f"Contribuição dos loadings exportada: {out}")
                self._reset_progress_later()
                messagebox.showinfo("Exportação concluída", f"Contribuição dos loadings exportada em:\n{out}")
                return

            if kind == "synthesis":
                folder_str = filedialog.askdirectory(title="Selecionar pasta para exportar a síntese PCA")
                if not folder_str:
                    return
                out = Path(folder_str)
                out.mkdir(parents=True, exist_ok=True)
                self._set_progress(20, "Preparando/exportando síntese PCA...")
                scores_centered, phase_centroids, phase_distances, _meta = self._build_synthesis_tables()
                if scores_centered.empty:
                    raise ValueError("A síntese PCA não pôde ser gerada. Rode PCA com pelo menos PC1 e PC2.")
                scores_centered.to_csv(out / "scores_PCA_recentrados_sintese.csv", index=False, encoding="utf-8-sig")
                if phase_centroids is not None and not phase_centroids.empty:
                    phase_centroids.to_csv(out / "centroides_fases_PCA_recentrado.csv", index=False, encoding="utf-8-sig")
                if phase_distances is not None and not phase_distances.empty:
                    phase_distances.to_csv(out / "distancias_aos_centroides_fases.csv", index=False, encoding="utf-8-sig")
                # Não regenera figuras já abertas: isso preserva rótulos,
                # legendas e a orientação tridimensional escolhida pelo usuário.
                if "espaco_referencia_PC1_PC2" not in self.pca_state.figures:
                    self._render_reference_space_2d(0, 1, self.tab_synthesis_pc12, "espaco_referencia_PC1_PC2")
                if len(self.pca_state.explained_ratio) >= 3:
                    if "espaco_referencia_PC1_PC3" not in self.pca_state.figures:
                        self._render_reference_space_2d(0, 2, self.tab_synthesis_pc13, "espaco_referencia_PC1_PC3")
                    if "espaco_referencia_PC1_PC2_PC3" not in self.pca_state.figures:
                        self._render_score_plot_3d(
                            self.tab_reference_space_3d,
                            "espaco_referencia_PC1_PC2_PC3",
                            reference_space=True,
                        )
                label_slug = safe_slug(self._reference_space_label())
                for figure_key, suffix in [
                    ("espaco_referencia_PC1_PC2", "PC1_PC2"),
                    ("espaco_referencia_PC1_PC3", "PC1_PC3"),
                    ("espaco_referencia_PC1_PC2_PC3", "PC1_PC2_PC3"),
                ]:
                    fig = self.pca_state.figures.get(figure_key)
                    if fig is not None:
                        fig.savefig(out / f"{label_slug}_{suffix}.png", dpi=400, bbox_inches="tight")
                self._set_progress(100, f"Síntese PCA exportada: {out}")
                self._reset_progress_later()
                messagebox.showinfo("Exportação concluída", f"Síntese PCA exportada em:\n{out}")
                return

            filename_map = {
                "matrix": "matriz_usada_no_PCA.csv",
                "scores": "scores_PCA.csv",
                "loadings": "loadings_PCA.csv",
                "distances": "distancias_obras_todas_metricas.csv",
            }
            data_map = {
                "matrix": pca_state.matrix_used,
                "scores": pca_state.scores,
                "loadings": pca_state.loadings,
                "distances": pca_state.distances,
            }
            if kind not in filename_map:
                raise ValueError(f"Tipo de exportação desconhecido: {kind}")
            default_name = filename_map[kind]
            file_str = filedialog.asksaveasfilename(
                title="Salvar arquivo CSV",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV", "*.csv")],
            )
            if not file_str:
                return
            df = data_map[kind]
            assert df is not None
            self._set_progress(35, "Salvando CSV...")
            df.to_csv(file_str, index=False, encoding="utf-8-sig")
            self._set_progress(100, f"Arquivo salvo: {file_str}")
            self._reset_progress_later()
            messagebox.showinfo("Exportação concluída", f"Arquivo salvo:\n{file_str}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Erro ao exportar", str(exc))
            self._set_progress(0, "Erro ao exportar arquivo.")

    def _write_all_outputs(self, out_dir: Path):
        p = self.pca_state
        assert p.matrix_used is not None
        assert p.scores is not None
        assert p.loadings is not None
        assert p.variance is not None
        assert p.distances is not None
        assert p.config is not None

        self._set_progress(12, "Exportando matriz usada no PCA/projeção...")
        p.matrix_used.to_csv(out_dir / "matriz_usada_no_PCA.csv", index=False, encoding="utf-8-sig")
        if PCA_ROLE_COLUMN in p.matrix_used.columns:
            p.matrix_used[p.matrix_used[PCA_ROLE_COLUMN] == ROLE_REFERENCE].to_csv(
                out_dir / "matriz_referencias_fit_PCA.csv", index=False, encoding="utf-8-sig"
            )
            p.matrix_used[p.matrix_used[PCA_ROLE_COLUMN] == ROLE_SUPPLEMENTARY].to_csv(
                out_dir / "matriz_suplementares_projetadas.csv", index=False, encoding="utf-8-sig"
            )
        self._set_progress(24, "Exportando scores...")
        p.scores.to_csv(out_dir / "scores_PCA.csv", index=False, encoding="utf-8-sig")
        self._set_progress(36, "Exportando loadings...")
        p.loadings.to_csv(out_dir / "loadings_PCA.csv", index=False, encoding="utf-8-sig")
        p.loadings.to_csv(out_dir / "pesos_componentes_PCA.csv", index=False, encoding="utf-8-sig")
        if p.loading_contributions is not None and not p.loading_contributions.empty:
            p.loading_contributions.to_csv(out_dir / "contribuicao_loadings_por_PC.csv", index=False, encoding="utf-8-sig")
        if p.loading_dominance is not None and not p.loading_dominance.empty:
            p.loading_dominance.to_csv(out_dir / "resumo_dominancia_loadings.csv", index=False, encoding="utf-8-sig")
        if p.pca_mean is not None:
            p.pca_mean.to_csv(out_dir / "media_variaveis_referencias_PCA.csv", index=False, encoding="utf-8-sig")
        if p.preprocess_parameters is not None and not p.preprocess_parameters.empty:
            p.preprocess_parameters.to_csv(out_dir / "parametros_preprocessamento_PCA.csv", index=False, encoding="utf-8-sig")
        if p.area_normalization_audit is not None and not p.area_normalization_audit.empty:
            p.area_normalization_audit.to_csv(
                out_dir / "auditoria_normalizacao_area_H.csv",
                index=False,
                encoding="utf-8-sig",
            )
        self._set_progress(48, "Exportando variância explicada...")
        p.variance.to_csv(out_dir / "variancia_explicada.csv", index=False, encoding="utf-8-sig")
        self._set_progress(58, "Exportando métricas de distância entre obras...")
        p.distances.to_csv(out_dir / "distancias_obras_todas_metricas.csv", index=False, encoding="utf-8-sig")
        # Mantém um arquivo legado com o nome anterior para compatibilidade, usando a primeira métrica disponível.
        if p.distance_tables:
            first_df = next(iter(p.distance_tables.values()))
            first_df.to_csv(out_dir / "distancias_obras.csv", index=False, encoding="utf-8-sig")
        for key, df in p.distance_tables.items():
            df.to_csv(out_dir / f"distancias_{safe_slug(key)}.csv", index=False, encoding="utf-8-sig")
        for key, df in p.distance_square_tables.items():
            df.to_csv(out_dir / f"{safe_slug(key)}.csv", index=False, encoding="utf-8-sig")
        if p.closest_pairs is not None and not p.closest_pairs.empty:
            p.closest_pairs.to_csv(out_dir / "pares_mais_proximos.csv", index=False, encoding="utf-8-sig")
        if p.farthest_pairs is not None and not p.farthest_pairs.empty:
            p.farthest_pairs.to_csv(out_dir / "pares_mais_distantes.csv", index=False, encoding="utf-8-sig")
        if self.metadata_state.df is not None:
            self._set_progress(70, "Exportando scores com metadados...")
            self._scores_with_metadata().to_csv(out_dir / "scores_PCA_com_metadados.csv", index=False, encoding="utf-8-sig")
            summary = self._build_group_summary()
            if not summary.empty:
                summary.to_csv(out_dir / "resumo_grupos_metadados.csv", index=False, encoding="utf-8-sig")
        self._set_progress(74, "Exportando síntese PCA recentrada...")
        scores_centered, phase_centroids, phase_distances, _meta = self._build_synthesis_tables()
        if scores_centered is not None and not scores_centered.empty:
            scores_centered.to_csv(out_dir / "scores_PCA_recentrados_sintese.csv", index=False, encoding="utf-8-sig")
        if phase_centroids is not None and not phase_centroids.empty:
            phase_centroids.to_csv(out_dir / "centroides_fases_PCA_recentrado.csv", index=False, encoding="utf-8-sig")
        if phase_distances is not None and not phase_distances.empty:
            phase_distances.to_csv(out_dir / "distancias_aos_centroides_fases.csv", index=False, encoding="utf-8-sig")
        self._set_progress(78, "Exportando configuração da rodada...")
        p.config.to_csv(out_dir / "configuracao_rodada_PCA.csv", index=False, encoding="utf-8-sig")
        self._set_progress(88, "Exportando gráficos...")
        self._save_figures(out_dir)

    def _save_figures(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        figures = self.pca_state.figures
        if not figures:
            raise ValueError("Nenhum gráfico disponível para exportação.")
        for key, fig in figures.items():
            if str(key).startswith("espaco_referencia_"):
                suffix = str(key).replace("espaco_referencia_", "")
                filename = f"{safe_slug(self._reference_space_label())}_{safe_slug(suffix)}.png"
            else:
                filename = f"{key}.png"
            # Ajustar nome para o padrão desejado.
            filename = filename.replace("grafico_PC1_PC2", "grafico_PC1_PC2")
            filename = filename.replace("grafico_PC1_PC3", "grafico_PC1_PC3")
            filename = filename.replace("grafico_variancia_explicada", "grafico_variancia_explicada")
            fig.savefig(out_dir / filename, dpi=400, bbox_inches="tight")


# -----------------------------------------------------------------------------
# Execução
# -----------------------------------------------------------------------------


def main():
    app = PCANMFXRFApp()
    app.mainloop()


if __name__ == "__main__":
    main()
