from __future__ import annotations

"""
Aplicação desktop para análise de hiper cubos de dados (.h5) com NMF/SVD.

Principais características desta versão:
- Interface em CustomTkinter com painel lateral compacto e rolável.
- Seleção de um ou vários arquivos HDF5.
- ROI por canal ou energia (via offset/gain, com calibração explicitamente ativável).
- Troca Canal/Energia na ROI por conversão real: E = offset + gain * canal.
- Binning espacial por soma.
- Fatoração global com subamostragem uniforme por obra.
- Exportação de CSVs restrita à ROI espectral efetivamente processada:
    1) espectros H individuais, um CSV por componente: SON-H00.csv, SON-H01.csv, etc.;
    2) mapa W vetorizado apenas quando há uma única obra/mapa em memória: W-SON.csv.
- Não exporta automaticamente planilha PCA nem mapas W por obra em lote múltiplo.
- Relatório automático de erro em caso de exceção.
- Tentativa de instalação automática de dependências ausentes.
- Não aplica subtração automática de background.
- v28: numeração visual das componentes inicia em 00, alinhada a H00/H01 na exportação.
- v46: orientação espectral controlável, semente/reinicialização reprodutível, botão de limpeza e manifesto metodológico.
- v46.1: valores iniciais: ROI por Energia de 1.8 a 16 keV, calibração ativa e binning 2.
- rawW-fastKL: W é exportado bruto; cálculo de KL completa é opcional e desativado por padrão.
- rawW-fastKL-Hconsolidado: exporta também a matriz H consolidada energia,H00,H01,...
- Harea1opt: opção de fixar a escala das componentes NMF com área(H)=1 e
  compensação exata W*=W*área(H), preservando o produto WH.
"""

import gc
import importlib
import os
import platform
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

APP_NAME = "Júlia 1"
APP_VERSION = "4.8.0"
DEFAULT_DATASET_PATH = "cube"
PREFERRED_DATASET_PATHS = ["cube", "/cube", "/entry/data/data"]
DEFAULT_ENERGIA_OFFSET = "-1.164291988537346"
DEFAULT_ENERGIA_GAIN = "0.012003290552661448"
PISO_NMF_KL = 1e-9


# ============================================================
# Utilidades de erro e relatório
# ============================================================

def pasta_base_relatorios() -> Path:
    """Define uma pasta robusta para salvar relatórios de erro."""
    try:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()



def gerar_relatorio_erro(exc: BaseException, contexto: str = "") -> Path:
    """Gera um relatório textual de erro para facilitar a depuração."""
    pasta = pasta_base_relatorios()
    pasta.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = pasta / f"relatorio_erro_ma_xrf_{timestamp}.txt"

    linhas = [
        f"Aplicação: {APP_NAME}",
        f"Versão: {APP_VERSION}",
        f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Sistema operacional: {platform.platform()}",
        f"Python: {sys.version}",
        "",
        "Contexto:",
        contexto or "(não informado)",
        "",
        f"Tipo da exceção: {type(exc).__name__}",
        f"Mensagem: {exc}",
        "",
        "Traceback completo:",
        traceback.format_exc(),
    ]

    try:
        caminho.write_text("\n".join(linhas), encoding="utf-8")
    except Exception:
        caminho = Path.cwd() / f"relatorio_erro_ma_xrf_{timestamp}.txt"
        caminho.write_text("\n".join(linhas), encoding="utf-8")

    return caminho


# ============================================================
# Bootstrap de dependências (instala apenas o que faltar)
# ============================================================

def instalar_pacote_se_necessario(nome_modulo: str, nome_pacote: str | None = None) -> bool:
    """Verifica se um módulo existe. Se não existir, tenta instalar via pip."""
    nome_pacote = nome_pacote or nome_modulo

    try:
        importlib.import_module(nome_modulo)
        return True
    except ModuleNotFoundError:
        pass

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                nome_pacote,
                "--disable-pip-version-check",
                "--no-input",
            ]
        )
        importlib.invalidate_caches()
        importlib.import_module(nome_modulo)
        return True
    except Exception:
        return False



def garantir_dependencias() -> None:
    """Instala automaticamente apenas as dependências ausentes."""
    dependencias = [
        ("numpy", "numpy"),
        ("h5py", "h5py"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "scikit-learn"),
        ("customtkinter", "customtkinter"),
    ]

    faltantes: list[str] = []
    for modulo, pacote in dependencias:
        if not instalar_pacote_se_necessario(modulo, pacote):
            faltantes.append(pacote)

    if faltantes:
        raise ModuleNotFoundError(
            "Não foi possível disponibilizar as seguintes dependências: " + ", ".join(faltantes)
        )


# ============================================================
# Imports com blindagem para relatório de erro
# ============================================================

try:
    garantir_dependencias()

    import customtkinter as ctk
    import h5py
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.decomposition import (
        MiniBatchNMF,
        NMF,
        TruncatedSVD,
        non_negative_factorization,
    )
    from tkinter import filedialog, messagebox
except Exception as exc:  # pragma: no cover
    caminho = gerar_relatorio_erro(exc, "Falha ao preparar/importar dependências da aplicação.")
    try:
        import tkinter as tk
        from tkinter import messagebox as _msg

        root = tk.Tk()
        root.withdraw()
        _msg.showerror(
            "Erro ao iniciar",
            "A aplicação não conseguiu iniciar.\n\n"
            f"Relatório salvo em:\n{caminho}\n\n"
            "O aplicativo tentou instalar automaticamente as dependências ausentes, mas não conseguiu concluir.\n"
            "Verifique internet, permissões do Python/pip e antivírus.",
        )
        root.destroy()
    except Exception:
        print("A aplicação não conseguiu iniciar.")
        print(f"Relatório salvo em: {caminho}")
    raise SystemExit(1)


# ============================================================
# Funções de processamento
# ============================================================

def preprocessar_e_normalizar(X_roi, metodo, canal_ar_index=None):
    """
    Pré-processa e normaliza a matriz 2D já recortada na ROI espectral
    (Pixels x Canais Úteis).

    Observações metodológicas:
    - Não há subtração automática de background nesta função.
    - O método por Argônio foi removido da interface e da lógica analítica.
    - Z-score não é aplicado aqui porque NMF/NMF-KL exige dados não negativos.
    """
    del canal_ar_index  # mantido apenas para compatibilidade interna de assinatura

    X_roi = np.asarray(X_roi, dtype=np.float64)

    # Blindagem física/matemática antes da normalização.
    # Isto apenas impede valores negativos na entrada do NMF; não é subtração de background.
    X_roi = np.clip(X_roi, 0, None)

    metodo_norm = (metodo or "Nenhum").strip().lower()

    if metodo_norm == "soma":
        somas = np.sum(X_roi, axis=1, keepdims=True)
        somas = np.where(somas == 0, 1e-9, somas)
        X_norm = X_roi / somas

    elif metodo_norm in ("área", "area"):
        if X_roi.shape[1] > 1:
            areas = np.sum(X_roi, axis=1, keepdims=True) - 0.5 * (
                X_roi[:, [0]] + X_roi[:, [-1]]
            )
        else:
            areas = np.sum(X_roi, axis=1, keepdims=True)
        areas = np.where(areas <= 0, 1e-9, areas)
        X_norm = X_roi / areas

    elif metodo_norm == "nenhum":
        X_norm = X_roi.copy()

    else:
        raise ValueError(f"Método de normalização inválido: {metodo}. Use Nenhum, Soma ou Área.")

    X_final = np.clip(X_norm, 1e-9, None)
    return X_final


def reescalar_componentes_nmf_area_unitaria(
    H: np.ndarray,
    W: np.ndarray | None = None,
    eixo_espectral: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Fixa a convenção de escala da NMF impondo área discreta unitária em cada H_k.

    A área usa integração trapezoidal no eixo espectral informado (energia ou
    canal). Sem eixo explícito, usa espaçamento unitário. Para cada componente k:

        a_k = área(H_k)
        H*_k = H_k / a_k
        W*_k = W_k * a_k

    Assim, W* @ H* é numericamente igual a W @ H (salvo arredondamento). Se W
    não for informado, a função retorna apenas H* e os fatores a_k necessários
    para compensá-lo posteriormente.
    """
    H_original = np.asarray(H, dtype=np.float64)
    if H_original.ndim != 2:
        raise ValueError(f"H deve ser uma matriz 2D. Shape encontrado: {H_original.shape}")
    if not np.all(np.isfinite(H_original)):
        raise ValueError("H contém valores não finitos; não é possível fixar sua escala.")
    if np.any(H_original < 0):
        raise ValueError("A convenção área(H)=1 é exclusiva para componentes NMF não negativas.")

    if eixo_espectral is not None:
        eixo = np.asarray(eixo_espectral, dtype=np.float64).ravel()
        if eixo.shape[0] != H_original.shape[1]:
            raise ValueError(
                "O eixo espectral e H têm números de canais incompatíveis: "
                f"eixo={eixo.shape[0]}, H={H_original.shape}."
            )
        if not np.all(np.isfinite(eixo)):
            raise ValueError("O eixo espectral contém valores não finitos.")
        if eixo.shape[0] > 1 and np.any(np.diff(eixo) <= 0):
            raise ValueError("O eixo espectral deve ser estritamente crescente.")
    else:
        eixo = None

    if H_original.shape[1] > 1 and eixo is not None:
        larguras = np.diff(eixo)
        fatores = np.sum(
            0.5 * (H_original[:, :-1] + H_original[:, 1:]) * larguras[np.newaxis, :],
            axis=1,
            dtype=np.float64,
        )
    elif H_original.shape[1] > 1:
        fatores = np.sum(H_original, axis=1, dtype=np.float64) - 0.5 * (
            H_original[:, 0] + H_original[:, -1]
        )
    else:
        fatores = np.sum(H_original, axis=1, dtype=np.float64)

    invalidas = np.flatnonzero((~np.isfinite(fatores)) | (fatores <= 0))
    if invalidas.size:
        nomes = ", ".join(f"H{i:02d}" for i in invalidas)
        raise ValueError(
            "Não foi possível impor área unitária porque há componente(s) "
            f"com área não positiva: {nomes}."
        )

    H_reescalado = H_original / fatores[:, np.newaxis]
    W_reescalado: np.ndarray | None = None
    if W is not None:
        W_original = np.asarray(W, dtype=np.float64)
        if W_original.ndim != 2:
            raise ValueError(f"W deve ser uma matriz 2D. Shape encontrado: {W_original.shape}")
        if W_original.shape[1] != H_original.shape[0]:
            raise ValueError(
                "W e H têm números de componentes incompatíveis para a compensação: "
                f"W={W_original.shape}, H={H_original.shape}."
            )
        W_reescalado = W_original * fatores[np.newaxis, :]

    return H_reescalado, W_reescalado, fatores


def calcular_kl_reconstrucao_em_blocos(X: np.ndarray, W: np.ndarray, H: np.ndarray, chunk_pixels: int = 5000) -> dict[str, float]:
    """Calcula D_KL(X || W H) em blocos, sem montar WH inteiro na memória."""
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or W.ndim != 2 or H.ndim != 2:
        raise ValueError(f"X, W e H devem ser 2D. Shapes: X={X.shape}, W={W.shape}, H={H.shape}")
    if X.shape[0] != W.shape[0]:
        raise ValueError(f"X e W têm números de pixels diferentes: X={X.shape}, W={W.shape}")
    if X.shape[1] != H.shape[1]:
        raise ValueError(f"X e H têm números de canais diferentes: X={X.shape}, H={H.shape}")
    if W.shape[1] != H.shape[0]:
        raise ValueError(f"W e H têm números de componentes diferentes: W={W.shape}, H={H.shape}")
    chunk_pixels = max(1, int(chunk_pixels))
    total_kl = 0.0
    total_l1 = 0.0
    total_x = 0.0
    eps = 1e-12
    for inicio in range(0, X.shape[0], chunk_pixels):
        fim = min(inicio + chunk_pixels, X.shape[0])
        Xb = np.clip(X[inicio:fim, :], eps, None)
        WH = np.clip(W[inicio:fim, :] @ H, eps, None)
        total_kl += float(np.sum(Xb * np.log(Xb / WH) - Xb + WH))
        total_l1 += float(np.sum(np.abs(Xb - WH)))
        total_x += float(np.sum(np.abs(Xb)))
    n_pixels, n_canais = X.shape
    return {
        "kl_total": float(total_kl),
        "kl_medio_por_pixel": float(total_kl / max(n_pixels, 1)),
        "kl_medio_por_entrada": float(total_kl / max(n_pixels * n_canais, 1)),
        "erro_relativo_L1": float(total_l1 / max(total_x, eps)),
    }


def aplicar_binning_espectral(cubo: np.ndarray, fator: int) -> np.ndarray:
    """Soma canais consecutivos após a ROI; descarta o bloco final incompleto."""
    if fator not in (1, 2, 4, 8):
        raise ValueError("O binning espectral deve ser 1, 2, 4 ou 8.")
    if fator == 1:
        return cubo
    n = cubo.shape[-1] // fator
    if n < 1:
        raise ValueError("Binning espectral maior que o número de canais da ROI.")
    return cubo[..., :n * fator].reshape(*cubo.shape[:-1], n, fator).sum(axis=-1)


def aplicar_binning_eixo(eixo: np.ndarray, fator: int) -> np.ndarray:
    """Representa cada grupo pelo centro médio em canal ou energia."""
    if fator not in (1, 2, 4, 8):
        raise ValueError("O binning espectral deve ser 1, 2, 4 ou 8.")
    if fator == 1:
        return eixo
    n = len(eixo) // fator
    if n < 1:
        raise ValueError("Binning espectral maior que o número de canais da ROI.")
    return np.asarray(eixo[:n * fator]).reshape(n, fator).mean(axis=1)


def aplicar_binning_espacial(cubo: np.ndarray, fator: int) -> np.ndarray:
    """
    Aplica binning espacial por soma em blocos fator x fator.
    Formato esperado do cubo: (Y, X, Canais).
    """
    if fator == 1:
        return cubo

    if fator not in (1, 2, 4, 8):
        raise ValueError("O binning espacial deve ser 1, 2, 4 ou 8.")

    y, x, c = cubo.shape
    y_novo = y // fator
    x_novo = x // fator

    if y_novo <= 0 or x_novo <= 0:
        raise ValueError("O fator de binning é incompatível com as dimensões do cubo.")

    # Trunca as bordas para caber no bloco exato.
    cubo_rec = cubo[: y_novo * fator, : x_novo * fator, :]

    # Executa a soma vetorizada em blocos tridimensionais.
    cubo_bin = cubo_rec.reshape(y_novo, fator, x_novo, fator, c)
    cubo_bin = cubo_bin.sum(axis=(1, 3))
    return cubo_bin



ORIENTACAO_AUTO_SEGURA = "Auto robusto (detectar canais)"
ORIENTACAO_YXC = "Y, X, Canais"
ORIENTACAO_CYX = "Canais, Y, X"
ORIENTACAO_AUTO_SHAPE = "Auto antigo por maior eixo"
ORIENTACOES_CUBO_GUI = [ORIENTACAO_AUTO_SEGURA, ORIENTACAO_YXC, ORIENTACAO_CYX, ORIENTACAO_AUTO_SHAPE]


def normalizar_orientacao_cubo(valor: str | None) -> str:
    """Normaliza a opção de orientação espectral escolhida na interface."""
    texto = str(valor or "").strip().lower()
    if texto in {"y, x, canais", "yxc", "y,x,c", "(y,x,c)", "yxchannels"}:
        return ORIENTACAO_YXC
    if texto in {"canais, y, x", "cyx", "c,y,x", "(c,y,x)", "channelsyx"}:
        return ORIENTACAO_CYX
    if "antigo" in texto or "maior" in texto:
        return ORIENTACAO_AUTO_SHAPE
    if "robust" in texto or "detectar" in texto:
        return ORIENTACAO_AUTO_SEGURA
    return ORIENTACAO_AUTO_SEGURA


def inferir_eixo_espectral_por_shape(
    shape: tuple[int, ...],
    orientacao_cubo: str | None = ORIENTACAO_AUTO_SEGURA,
) -> int | None:
    """
    Infere o eixo espectral a partir do shape do dataset 3D.

    Retorna:
        2 para cubos no formato (Y, X, Canais)
        0 para cubos no formato (Canais, Y, X)
        None quando a inferência não é segura

    Regra de segurança desta versão:
    - respeita sempre a escolha manual do usuário;
    - no modo automático, detecta primeiro assinaturas espectrais típicas
      (4096, 2048, 1024, 8192 etc.);
    - só prefere o último eixo quando não há forte evidência de que o primeiro
      eixo seja espectral.

    Esta correção evita o travamento observado quando um arquivo no formato
    (Canais, Y, X), por exemplo (4096, 600, 700), era interpretado como
    (Y, X, Canais) e a ROI era aplicada no eixo X, carregando um bloco enorme
    e incorreto na memória.
    """
    if len(shape) != 3:
        return None

    s0, s1, s2 = [int(v) for v in shape]
    modo = normalizar_orientacao_cubo(orientacao_cubo)

    if modo == ORIENTACAO_YXC:
        return 2
    if modo == ORIENTACAO_CYX:
        return 0

    # Modo legado: escolhe o maior eixo como espectral. Mantido para reproduzir
    # o comportamento anterior quando o usuário selecionar explicitamente essa opção.
    if modo == ORIENTACAO_AUTO_SHAPE:
        if s2 >= 16 and s2 >= s0 and s2 >= s1:
            return 2
        if s0 >= 16 and s0 >= s1 and s0 >= s2:
            return 0
        return None

    # Auto robusto: cobre os dois formatos mais comuns sem fazer leitura grande.
    # O eixo espectral costuma ter centenas/milhares de canais. Em dados CRONO,
    # 4096 é muito comum; após ROI/corte, o número pode cair para ~1000.
    canais_tipicos = {512, 1024, 1152, 1161, 1200, 1430, 2048, 4096, 8192}

    # Prioridade 1: dimensões exatamente compatíveis com números de canais usuais.
    s0_tipico = s0 in canais_tipicos
    s2_tipico = s2 in canais_tipicos
    if s0_tipico and not s2_tipico:
        return 0
    if s2_tipico and not s0_tipico:
        return 2

    # Prioridade 2: formato (Canais, Y, X) bruto, típico quando o primeiro eixo
    # é muito maior que os eixos espaciais. Ex.: (4096, 600, 700).
    if s0 >= 1500 and s0 > s2 * 1.5 and s0 >= s1:
        return 0

    # Prioridade 3: formato (Y, X, Canais), inclusive cubos costurados/ROI em que
    # Y pode ser ligeiramente maior que Canais. Ex.: (1200, 700, 1161).
    if s2 >= 16:
        return 2

    # Último recurso: comportamento antigo, mas só quando for claro.
    if s0 >= 16 and s0 >= s1 and s0 >= s2:
        return 0

    return None


def garantir_cubo_yxc(dataset_array: np.ndarray, orientacao_cubo: str | None = ORIENTACAO_AUTO_SEGURA) -> np.ndarray:
    """
    Garante que o dataset esteja no formato (Y, X, Canais).
    """
    if dataset_array.ndim != 3:
        raise ValueError(f"O dataset deve ter 3 dimensões. Shape encontrado: {dataset_array.shape}")

    eixo_espectral = inferir_eixo_espectral_por_shape(tuple(dataset_array.shape), orientacao_cubo)

    if eixo_espectral == 2:
        return dataset_array

    if eixo_espectral == 0:
        return np.transpose(dataset_array, (1, 2, 0))

    raise ValueError(
        "Não foi possível inferir automaticamente o eixo espectral do cubo. "
        f"Shape encontrado: {dataset_array.shape}."
    )


def numero_canais_dataset(dataset: h5py.Dataset, orientacao_cubo: str | None = ORIENTACAO_AUTO_SEGURA) -> int:
    """Retorna o número de canais espectrais do dataset 3D sem carregar o cubo inteiro."""
    if dataset.ndim != 3:
        raise ValueError(f"O dataset deve ter 3 dimensões. Shape encontrado: {dataset.shape}")

    eixo_espectral = inferir_eixo_espectral_por_shape(tuple(dataset.shape), orientacao_cubo)
    if eixo_espectral is None:
        raise ValueError(
            "Não foi possível inferir automaticamente o eixo espectral do cubo. "
            f"Shape encontrado: {dataset.shape}."
        )
    return int(dataset.shape[eixo_espectral])


def resolver_dataset_para_arquivo(caminho_h5: str, dataset_preferencial: str) -> str:
    """
    Resolve o caminho do dataset para um arquivo específico.

    Primeiro tenta usar o dataset escolhido no primeiro arquivo. Se esse caminho
    não existir no arquivo atual, procura automaticamente um dataset 3D numérico
    preferencial. Isso evita falha desnecessária quando arquivos do mesmo lote
    usam caminhos HDF5 equivalentes, mas não idênticos, como /cube e /entry/data/data.
    """
    with h5py.File(caminho_h5, "r") as h5f:
        if dataset_preferencial in h5f:
            return dataset_preferencial

    candidatos = listar_datasets_hdf5_3d(caminho_h5)
    caminho_resolvido = escolher_dataset_preferencial(candidatos)
    if caminho_resolvido:
        return caminho_resolvido

    raise KeyError(
        f"Dataset '{dataset_preferencial}' não localizado e nenhum dataset 3D numérico alternativo foi encontrado "
        f"no arquivo {Path(caminho_h5).name}."
    )


def carregar_roi_yxc(caminho_h5: str, dataset_path: str, roi_inicio: int, roi_fim: int, orientacao_cubo: str | None = ORIENTACAO_AUTO_SEGURA) -> np.ndarray:
    """
    Carrega apenas a ROI espectral do cubo e retorna no formato (Y, X, Canais_ROI).

    Para os formatos reconhecidos como (Y, X, C) ou (C, Y, X), a leitura é
    feita diretamente por slicing do HDF5, reduzindo o pico de RAM. Caso a
    orientação não seja reconhecida, a função cai para a leitura completa e usa
    garantir_cubo_yxc(), preservando compatibilidade com arquivos incomuns.
    """
    with h5py.File(caminho_h5, "r") as h5f:
        if dataset_path not in h5f:
            raise KeyError(f"Dataset '{dataset_path}' não localizado no arquivo {Path(caminho_h5).name}")

        dataset = h5f[dataset_path]
        if dataset.ndim != 3:
            raise ValueError(f"O dataset deve ter 3 dimensões. Shape encontrado: {dataset.shape}")

        eixo_espectral = inferir_eixo_espectral_por_shape(tuple(dataset.shape), orientacao_cubo)

        if eixo_espectral is not None:
            n_canais = int(dataset.shape[eixo_espectral])
            if roi_inicio < 0 or roi_fim <= roi_inicio or roi_fim > n_canais:
                raise ValueError(
                    f"ROI espectral inválida para {Path(caminho_h5).name}: "
                    f"início={roi_inicio}, fim exclusivo={roi_fim}, canais disponíveis={n_canais}."
                )

        if eixo_espectral == 2:
            return np.asarray(dataset[:, :, roi_inicio:roi_fim])

        if eixo_espectral == 0:
            bloco = np.asarray(dataset[roi_inicio:roi_fim, :, :])
            return np.transpose(bloco, (1, 2, 0))

        cubo = garantir_cubo_yxc(np.asarray(dataset), orientacao_cubo)
        n_canais = cubo.shape[-1]
        if roi_inicio < 0 or roi_fim <= roi_inicio or roi_fim > n_canais:
            raise ValueError(
                f"ROI espectral inválida para {Path(caminho_h5).name}: "
                f"início={roi_inicio}, fim exclusivo={roi_fim}, canais disponíveis={n_canais}."
            )
        return cubo[:, :, roi_inicio:roi_fim]


def listar_datasets_hdf5_3d(caminho_h5: str) -> list[tuple[str, tuple[int, ...], str]]:
    """Lista datasets numéricos tridimensionais candidatos a cubo MA-XRF."""
    candidatos: list[tuple[str, tuple[int, ...], str]] = []

    with h5py.File(caminho_h5, "r") as h5f:
        def _visitor(nome, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                try:
                    if np.issubdtype(obj.dtype, np.number):
                        candidatos.append(("/" + nome.lstrip("/"), tuple(obj.shape), str(obj.dtype)))
                except TypeError:
                    pass

        h5f.visititems(_visitor)

    candidatos.sort(key=lambda item: (item[0].count("/"), item[0]))
    return candidatos


def escolher_dataset_preferencial(candidatos: list[tuple[str, tuple[int, ...], str]]) -> str | None:
    """Retorna automaticamente o dataset preferencial quando possível."""
    if not candidatos:
        return None

    caminhos_normalizados = {caminho.lower().lstrip("/"): caminho for caminho, _, _ in candidatos}

    for preferido in PREFERRED_DATASET_PATHS:
        chave = preferido.lower().lstrip("/")
        if chave in caminhos_normalizados:
            return caminhos_normalizados[chave]

    if len(candidatos) == 1:
        return candidatos[0][0]

    return None


class DialogoEscolhaDataset(ctk.CTkToplevel):
    def __init__(self, master, candidatos: list[tuple[str, tuple[int, ...], str]]):
        super().__init__(master)
        self.title("Escolher dataset do cubo")
        self.geometry("760x420")
        self.minsize(680, 360)
        self.transient(master)
        self.grab_set()
        self.resultado: str | None = None
        self.candidatos = candidatos

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        lbl = ctk.CTkLabel(
            self,
            text=(
                "Escolha abaixo o dataset 3D que corresponde ao cubo MA-XRF.\n"
                "Na maioria dos casos, será o dataset com shape (Y, X, Canais) ou equivalente."
            ),
            justify="left",
            wraplength=700,
        )
        lbl.grid(row=0, column=0, padx=14, pady=(12, 8), sticky="ew")

        opcoes = [item[0] for item in candidatos]
        self.combo = ctk.CTkOptionMenu(self, values=opcoes, height=30)
        self.combo.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="ew")
        if opcoes:
            self.combo.set(opcoes[0])

        self.txt = ctk.CTkTextbox(self, height=220)
        self.txt.grid(row=2, column=0, padx=14, pady=(0, 8), sticky="nsew")
        self._preencher_texto()
        self.txt.configure(state="disabled")

        frame_btn = ctk.CTkFrame(self, fg_color="transparent")
        frame_btn.grid(row=3, column=0, padx=14, pady=(0, 12), sticky="e")

        btn_ok = ctk.CTkButton(frame_btn, text="Usar este dataset", command=self._confirmar, height=30)
        btn_ok.pack(side="left", padx=(0, 8))
        btn_cancel = ctk.CTkButton(frame_btn, text="Cancelar", command=self._cancelar, height=30, fg_color="#666666", hover_color="#555555")
        btn_cancel.pack(side="left")

        self.protocol("WM_DELETE_WINDOW", self._cancelar)
        self.wait_visibility()
        self.focus()

    def _preencher_texto(self):
        linhas = []
        for idx, (path, shape, dtype) in enumerate(self.candidatos, start=1):
            linhas.append(f"{idx}. {path}\n   shape={shape} | dtype={dtype}\n")
        self.txt.insert("1.0", "\n".join(linhas) if linhas else "Nenhum dataset 3D numérico encontrado.")

    def _confirmar(self):
        self.resultado = self.combo.get().strip() or None
        self.destroy()

    def _cancelar(self):
        self.resultado = None
        self.destroy()


def construir_eixo_energia_por_calibracao(n_canais: int, offset_txt: str, gain_txt: str) -> np.ndarray:
    """Constrói o eixo de energia a partir de calibração linear: E = offset + gain * canal."""
    if str(offset_txt).strip() == "" or str(gain_txt).strip() == "":
        raise ValueError("Para ROI em energia, informe offset e gain.")

    offset = float(offset_txt)
    gain = float(gain_txt)

    if gain == 0:
        raise ValueError("O gain não pode ser zero.")

    canais = np.arange(n_canais, dtype=np.float64)
    eixo = offset + gain * canais

    if not np.all(np.isfinite(eixo)):
        raise ValueError("A calibração gerou valores de energia inválidos.")

    return eixo



def interpretar_roi(
    modo_roi: str,
    valor_inicial: str,
    valor_final: str,
    n_canais: int,
    eixo_energia: np.ndarray | None,
) -> tuple[int, int, np.ndarray, str]:
    """
    Interpreta a ROI em modo Canal ou Energia.

    Retorna:
        idx_inicial, idx_final (inclusive), eixo_roi_plot, rotulo_eixo
    """
    if modo_roi == "Canal":
        i0 = int(float(valor_inicial))
        i1 = int(float(valor_final))

        i0 = max(0, min(i0, n_canais - 1))
        i1 = max(0, min(i1, n_canais - 1))

        if i1 < i0:
            raise ValueError("O canal final da ROI deve ser maior ou igual ao canal inicial.")

        eixo_roi = np.arange(i0, i1 + 1, dtype=np.float64)
        return i0, i1, eixo_roi, "Canal"

    if modo_roi == "Energia":
        if eixo_energia is None:
            raise ValueError("ROI em energia foi selecionada, mas offset/gain não foram informados.")

        e0 = float(valor_inicial)
        e1 = float(valor_final)

        if e1 < e0:
            raise ValueError("A energia final da ROI deve ser maior ou igual à energia inicial.")

        if np.any(np.diff(eixo_energia) < 0):
            raise ValueError("O eixo de energia deve estar em ordem crescente.")

        i0 = int(np.searchsorted(eixo_energia, e0, side="left"))
        i1 = int(np.searchsorted(eixo_energia, e1, side="right")) - 1

        i0 = max(0, min(i0, n_canais - 1))
        i1 = max(0, min(i1, n_canais - 1))

        if i1 < i0:
            raise ValueError("A ROI em energia não contém canais válidos.")

        eixo_roi = eixo_energia[i0 : i1 + 1].astype(np.float64)
        return i0, i1, eixo_roi, "Energia (keV)"

    raise ValueError(f"Modo de ROI inválido: {modo_roi}")


def tentar_construir_eixo_calibrado(n_canais: int, offset_txt: str, gain_txt: str) -> np.ndarray | None:
    """Tenta construir um eixo calibrado sem obrigar o modo ROI por energia."""
    if str(offset_txt).strip() == "" or str(gain_txt).strip() == "":
        return None
    return construir_eixo_energia_por_calibracao(n_canais, offset_txt, gain_txt)


def criar_colormap_temperature() -> LinearSegmentedColormap:
    """Cria uma colormap térmica customizada: azul -> ciano -> verde -> amarelo -> vermelho."""
    return LinearSegmentedColormap.from_list(
        "temperature_custom",
        [
            (0.00, "#0000FF"),  # azul
            (0.25, "#00FFFF"),  # ciano
            (0.50, "#00FF00"),  # verde
            (0.75, "#FFFF00"),  # amarelo
            (1.00, "#FF0000"),  # vermelho
        ],
        N=256,
    )


def resolver_colormap(nome: str):
    """Mapeia nomes amigáveis da GUI para colormaps válidos do Matplotlib."""
    nome_limpo = (nome or "viridis").strip().lower()
    if nome_limpo == "temperature":
        return criar_colormap_temperature()
    return nome_limpo


def abundancias_devem_ser_percentuais(metodo_norm: str) -> bool:
    """Define se as abundâncias W devem ser reescaladas internamente para visualização/exportação."""
    metodo = (metodo_norm or "").strip().lower()
    return metodo in ("soma", "área", "area")


def metodo_eh_svd(metodo_fatoracao: str) -> bool:
    """Identifica métodos baseados em SVD, que não devem ser descritos como abundância NMF."""
    return (metodo_fatoracao or "").strip().lower() == "truncatedsvd"


def sanitizar_nome_arquivo(texto: str, limite: int = 120) -> str:
    """Converte um nome de obra/caminho em um nome seguro para arquivo."""
    permitido = []
    for ch in str(texto):
        if ch.isalnum() or ch in ("-", "_", "."):
            permitido.append(ch)
        elif ch in (" ", "\t"):
            permitido.append("_")
    nome = "".join(permitido).strip("._-")
    while "__" in nome:
        nome = nome.replace("__", "_")
    if not nome:
        nome = "obra"
    return nome[:limite]




def prefixo_tres_letras_obra(nome_obra: str) -> str:
    """Gera o prefixo curto da obra para os CSVs: Sono -> SON."""
    nome_seguro = sanitizar_nome_arquivo(nome_obra)
    letras = [ch for ch in nome_seguro if ch.isalnum()]
    prefixo = "".join(letras[:3]).upper()
    return prefixo or "OBR"


def rotulo_normalizacao_pasta(metodo_norm: str) -> str:
    """Retorna o sufixo de normalização usado no nome da pasta de exportação."""
    metodo = (metodo_norm or "").strip().lower()
    if metodo in ("soma",):
        return "Nsoma"
    if metodo in ("área", "area"):
        return "Narea"
    if metodo in ("nenhum",):
        return "Nnenhum"
    return "N" + sanitizar_nome_arquivo(metodo_norm).lower()


def nome_pasta_exportacao_obra(
    nome_obra: str,
    n_componentes: int,
    fator_binning: int,
    metodo_norm: str,
    padronizar_escala_componentes: bool = False,
    fator_binning_espectral: int = 1,
) -> str:
    """Nomeia a pasta de exportação; acrescenta Harea1 quando a escala é fixada."""
    obra = sanitizar_nome_arquivo(nome_obra)
    sufixo_escala = "-Harea1" if padronizar_escala_componentes else ""
    return (
        f"{obra}-k{int(n_componentes)}-B{int(fator_binning)}-E{int(fator_binning_espectral)}-"
        f"{rotulo_normalizacao_pasta(metodo_norm)}{sufixo_escala}"
    )

def salvar_mapa_vetorizado_por_obra_csv(
    caminho_csv: str | Path,
    W_obra: np.ndarray,
    y_b: int,
    x_b: int,
    n_componentes: int,
    chunk_pixels: int = 250_000,
) -> Path:
    """
    Salva os mapas/score maps de uma obra em CSV vetorizado, sem criar um DataFrame gigante.

    Formato do arquivo:
        row,column,NNMA_Image_00,NNMA_Image_01,...

    Cada linha é um pixel pós-binning e cada coluna NNMA_Image_* é uma componente.
    """
    caminho = Path(caminho_csv)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    W = np.asarray(W_obra)
    n_pixels = int(y_b) * int(x_b)
    if W.ndim != 2 or W.shape[0] != n_pixels or W.shape[1] != n_componentes:
        raise ValueError(
            "W_obra não possui shape compatível com o mapa vetorizado: "
            f"W={W.shape}, esperado=({n_pixels}, {n_componentes})."
        )

    chunk_pixels = max(1, int(chunk_pixels))
    colunas_componentes = [f"NNMA_Image_{i:02d}" for i in range(n_componentes)]

    primeira = True
    for inicio in range(0, n_pixels, chunk_pixels):
        fim = min(inicio + chunk_pixels, n_pixels)
        indices = np.arange(inicio, fim, dtype=np.int64)
        dados: dict[str, np.ndarray] = {
            "row": (indices // int(x_b)).astype(np.int64, copy=False),
            "column": (indices % int(x_b)).astype(np.int64, copy=False),
        }
        bloco = W[inicio:fim, :]
        for i, nome_coluna in enumerate(colunas_componentes):
            dados[nome_coluna] = bloco[:, i]

        df = pd.DataFrame(dados)
        df.to_csv(caminho, index=False, sep=",", decimal=".", mode="w" if primeira else "a", header=primeira)
        primeira = False

    return caminho



@dataclass
class MetadadosObra:
    nome_arquivo: str
    dataset_path: str
    n_pixels_pos_binning: int


@dataclass
class ResultadoLote:
    H_global: np.ndarray
    dict_medias: dict[str, np.ndarray]
    eixo_roi: np.ndarray
    rotulo_eixo: str
    metodo_fatoracao: str
    info_textual: str
    nomes_obras: list[str]
    obra_visualizacao_nome: str | None = None
    mapas_abundancia: np.ndarray | None = None  # shape (Y, X, k) para visualização local
    abundancias_em_percentual: bool = False
    rotulo_abundancia: str = "Abundância relativa"
    contribuicao_relativa_componentes: np.ndarray | None = None
    fator_binning: int = 1
    fator_binning_espectral: int = 1
    metodo_norm: str = ""
    n_componentes: int = 0
    orientacao_cubo: str = ORIENTACAO_AUTO_SEGURA
    modo_semente: str = "Reprodutível (semente fixa)"
    random_state_usado: int | None = 42
    init_nmf_modo: str = "Auto estável (NNDSVDa quando possível)"
    init_nmf_efetivo: str = ""
    alerta_convergencia: str = ""
    pasta_mapas_por_obra: str | None = None
    arquivos_mapas_por_obra: list[str] | None = None
    manifesto_mapas_por_obra: str | None = None
    reconstruction_err_treino: float | None = None
    n_iter_treino: int | None = None
    metricas_reconstrucao_obras: list[dict[str, object]] | None = None
    padronizar_escala_componentes: bool = False
    fatores_escala_h: np.ndarray | None = None


def executar_fatoracao_global_lote(
    lista_caminhos_h5: list[str],
    dataset_path: str,
    fator_binning: int,
    roi_inicio: int,
    roi_fim: int,
    metodo_norm: str,
    canal_ar_index: int | None,
    metodologia: str,
    n_componentes: int,
    max_iter: int,
    eps: float,
    pixels_amostra_por_obra: int = 2000,
    progress_callback=None,
    armazenar_mapa_quando_unico: bool = True,
    salvar_mapas_por_obra: bool = False,
    pasta_mapas_por_obra: str | Path | None = None,
    prefixo_mapas_por_obra: str = "mapas_por_obra",
    rotulo_eixo_exportacao: str = "Canal",
    orientacao_cubo: str | None = ORIENTACAO_AUTO_SEGURA,
    random_state: int | None = 42,
    init_nmf_modo: str = "Auto estável (NNDSVDa quando possível)",
    calcular_kl_reconstrucao_completa: bool = False,
    padronizar_escala_componentes: bool = False,
    eixo_espectral_escala: np.ndarray | None = None,
    fator_binning_espectral: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray], str | None, np.ndarray | None, list[str], str | None, np.ndarray | None, str, str, float | None, int | None, list[dict[str, object]], np.ndarray | None]:
    """
    Executa a fatoração global combinada para múltiplas obras com resoluções espaciais diferentes.

    Etapa 1: subamostragem aleatória uniforme leve para extração da paleta global (H).
    Etapa 2: projeção sequencial por obra para gerar médias prontas para PCA.
    Opcionalmente, salva os mapas vetorizados de cada obra em CSV antes de apagar W da RAM.

    Observação:
    - roi_fim é tratado como EXCLUSIVO nesta função.
    """
    if not lista_caminhos_h5:
        raise ValueError("Nenhum arquivo HDF5 foi informado para o processamento em lote.")

    if pixels_amostra_por_obra <= 0:
        raise ValueError("O número de pixels amostrados por obra deve ser maior que zero.")

    if padronizar_escala_componentes and metodo_eh_svd(metodologia):
        raise ValueError(
            "A opção área(H)=1 com compensação de W é exclusiva para NMF. "
            "Desmarque-a para usar TruncatedSVD."
        )

    # Nenhuma multiplicação por 100 é aplicada ao array W_obra. Quando a
    # convenção área(H)=1 é ativada, W recebe somente a compensação exata a_k.
    reescalar_w_para_visualizacao = False
    orientacao_cubo = normalizar_orientacao_cubo(orientacao_cubo)
    init_nmf_modo_normalizado = str(init_nmf_modo or "").strip().lower()

    def report(frac: float, msg: str):
        if progress_callback is not None:
            progress_callback(float(max(0.0, min(1.0, frac))), msg)

    rng = np.random.default_rng(random_state)
    lista_matrizes_amostradas: list[np.ndarray] = []
    registro_obras: list[MetadadosObra] = []
    modelo_svd: TruncatedSVD | None = None
    nome_obra_visualizacao: str | None = None
    mapa_abundancia_visualizacao: np.ndarray | None = None
    arquivos_mapas_salvos: list[str] = []
    linhas_manifesto_mapas: list[dict[str, object]] = []
    caminho_manifesto_mapas: str | None = None
    alerta_convergencia = ""
    reconstruction_err_treino: float | None = None
    n_iter_treino: int | None = None
    metricas_reconstrucao_obras: list[dict[str, object]] = []
    soma_w_componentes_global: np.ndarray | None = None
    soma_w_total_global: float = 0.0
    fatores_escala_h: np.ndarray | None = None

    if salvar_mapas_por_obra and pasta_mapas_por_obra is None:
        raise ValueError("Para salvar mapas por obra, informe a pasta de exportação dos mapas.")
    pasta_mapas = Path(pasta_mapas_por_obra) if pasta_mapas_por_obra is not None else None
    if salvar_mapas_por_obra and pasta_mapas is not None:
        pasta_mapas.mkdir(parents=True, exist_ok=True)

    total_obras = len(lista_caminhos_h5)
    report(0.02, "Preparando a etapa 1: subamostragem estatística das obras...")

    # ------------------------------------------------------------
    # ETAPA 1: subamostragem uniforme para obter H_global
    # ------------------------------------------------------------
    for idx_obra, caminho in enumerate(lista_caminhos_h5, start=1):
        nome_breve = Path(caminho).name
        frac_base = 0.05 + 0.45 * ((idx_obra - 1) / max(total_obras, 1))
        report(frac_base, f"Etapa 1/2: lendo e amostrando {nome_breve} ({idx_obra}/{total_obras})...")

        dataset_obra = resolver_dataset_para_arquivo(caminho, dataset_path)
        cubo_roi = carregar_roi_yxc(caminho, dataset_obra, roi_inicio, roi_fim, orientacao_cubo)

        cubo_roi = aplicar_binning_espectral(cubo_roi, fator_binning_espectral)
        cubo_binnado = aplicar_binning_espacial(cubo_roi, fator_binning)
        y_b, x_b, c_b = cubo_binnado.shape

        n_pixels_total = y_b * x_b
        X_roi = cubo_binnado.reshape(n_pixels_total, c_b)
        X_tratado = preprocessar_e_normalizar(X_roi, metodo_norm, canal_ar_index)

        if metodologia in ("NMF-KL", "NMF-Euclidiana", "MiniBatchNMF-Fast"):
            X_tratado = np.clip(X_tratado, 0, None)
            X_tratado[X_tratado == 0] = PISO_NMF_KL

        n_amostras = min(pixels_amostra_por_obra, n_pixels_total)
        indices_aleatorios = rng.choice(n_pixels_total, size=n_amostras, replace=False)
        X_amostrado = X_tratado[indices_aleatorios, :]

        lista_matrizes_amostradas.append(X_amostrado)
        registro_obras.append(MetadadosObra(nome_arquivo=caminho, dataset_path=dataset_obra, n_pixels_pos_binning=n_pixels_total))

        del cubo_roi, cubo_binnado, X_roi, X_tratado, X_amostrado
        gc.collect()
        frac_fim = 0.05 + 0.45 * (idx_obra / max(total_obras, 1))
        report(frac_fim, f"Etapa 1/2: obra {idx_obra}/{total_obras} amostrada com sucesso.")

    X_global_amostrado = np.vstack(lista_matrizes_amostradas)
    n_amostras_global, n_features_global = X_global_amostrado.shape
    limite_nnsvd = min(n_amostras_global, n_features_global)
    if "aleat" in init_nmf_modo_normalizado or "random" in init_nmf_modo_normalizado:
        init_nmf = "random"
    else:
        init_nmf = "nndsvda" if n_componentes <= limite_nnsvd else "random"
    init_nmf_efetivo = init_nmf

    report(0.56, f"Treinando a paleta global com {metodologia}...")
    if metodologia == "NMF-KL":
        modelo = NMF(
            n_components=n_componentes,
            init=init_nmf,
            beta_loss="kullback-leibler",
            solver="mu",
            max_iter=max_iter,
            tol=eps,
            random_state=random_state,
        )
        modelo.fit(X_global_amostrado)
        reconstruction_err_treino = float(getattr(modelo, "reconstruction_err_", np.nan))
        n_iter_treino = int(getattr(modelo, "n_iter_", 0) or 0)
        if int(getattr(modelo, "n_iter_", 0) or 0) >= int(max_iter):
            alerta_convergencia = f"tolerancia não atingida em {max_iter} iterações!"

        H_global = modelo.components_

    elif metodologia == "MiniBatchNMF-Fast":
        # Nesta implementação, MiniBatchNMF usa inicialização aleatória para evitar
        # incompatibilidades e manter comportamento claro com beta_loss=KL.
        init_nmf_efetivo = "random"
        modelo = MiniBatchNMF(
            n_components=n_componentes,
            init="random",
            beta_loss="kullback-leibler",
            batch_size=1024,
            max_iter=max_iter,
            tol=eps,
            random_state=random_state,
        )
        modelo.fit(X_global_amostrado)
        reconstruction_err_treino = float(getattr(modelo, "reconstruction_err_", np.nan))
        n_iter_treino = int(getattr(modelo, "n_iter_", 0) or 0)
        if int(getattr(modelo, "n_iter_", 0) or 0) >= int(max_iter):
            alerta_convergencia = f"tolerancia não atingida em {max_iter} iterações!"
        H_global = modelo.components_

    elif metodologia == "NMF-Euclidiana":
        modelo = NMF(
            n_components=n_componentes,
            init=init_nmf,
            beta_loss="frobenius",
            solver="cd",
            max_iter=max_iter,
            tol=eps,
            random_state=random_state,
        )
        modelo.fit(X_global_amostrado)
        reconstruction_err_treino = float(getattr(modelo, "reconstruction_err_", np.nan))
        n_iter_treino = int(getattr(modelo, "n_iter_", 0) or 0)
        if int(getattr(modelo, "n_iter_", 0) or 0) >= int(max_iter):
            alerta_convergencia = f"tolerancia não atingida em {max_iter} iterações!"
        H_global = modelo.components_

    elif metodologia == "TruncatedSVD":
        limite_svd = min(n_amostras_global, n_features_global)
        if n_componentes >= limite_svd:
            raise ValueError(
                f"Para TruncatedSVD, k deve ser menor que min(n_amostras, n_features) = {limite_svd}."
            )

        modelo_svd = TruncatedSVD(
            n_components=n_componentes,
            algorithm="randomized",
            n_iter=max_iter,
            random_state=random_state,
        )
        modelo_svd.fit(X_global_amostrado)
        H_global = modelo_svd.components_
    else:
        raise ValueError(f"Metodologia inválida: {metodologia}")

    # Guarda a escala usada efetivamente na projeção. Quando a opção está
    # ativada, H_global passa a ser a versão canônica exportada/visualizada,
    # enquanto cada W projetado com H_para_projecao é compensado em seguida.
    H_para_projecao = np.asarray(H_global, dtype=np.float64)
    if padronizar_escala_componentes:
        H_global, _w_ausente, fatores_escala_h = reescalar_componentes_nmf_area_unitaria(
            H_para_projecao,
            eixo_espectral=eixo_espectral_escala,
        )
    else:
        H_global = H_para_projecao

    del X_global_amostrado, lista_matrizes_amostradas
    gc.collect()
    report(0.64, "Paleta global concluída. Iniciando a projeção por obra...")

    # ------------------------------------------------------------
    # ETAPA 2: projeção por obra para gerar uma linha por obra
    # ------------------------------------------------------------
    dict_medias_por_obra: dict[str, np.ndarray] = {}

    for idx_obra, registro in enumerate(registro_obras, start=1):
        nome_breve = Path(registro.nome_arquivo).name
        frac_base = 0.65 + 0.30 * ((idx_obra - 1) / max(total_obras, 1))
        report(frac_base, f"Etapa 2/2: projetando componentes em {nome_breve} ({idx_obra}/{total_obras})...")

        cubo_roi = carregar_roi_yxc(registro.nome_arquivo, registro.dataset_path, roi_inicio, roi_fim, orientacao_cubo)

        cubo_roi = aplicar_binning_espectral(cubo_roi, fator_binning_espectral)
        cubo_binnado = aplicar_binning_espacial(cubo_roi, fator_binning)
        y_b, x_b, c_b = cubo_binnado.shape

        X_roi = cubo_binnado.reshape(y_b * x_b, c_b)
        X_tratado = preprocessar_e_normalizar(X_roi, metodo_norm, canal_ar_index)

        if metodologia in ("NMF-KL", "NMF-Euclidiana", "MiniBatchNMF-Fast"):
            X_tratado = np.clip(X_tratado, 0, None)
            X_tratado[X_tratado == 0] = PISO_NMF_KL

            beta = "frobenius" if metodologia == "NMF-Euclidiana" else "kullback-leibler"
            solver = "cd" if metodologia == "NMF-Euclidiana" else "mu"

            W_obra, _, _ = non_negative_factorization(
                X_tratado,
                H=H_para_projecao,
                n_components=n_componentes,
                update_H=False,
                solver=solver,
                beta_loss=beta,
                tol=eps,
                max_iter=min(max_iter, 200),
                random_state=random_state,
            )
        else:
            if modelo_svd is None:
                raise RuntimeError("O modelo TruncatedSVD não foi treinado corretamente.")
            W_obra = modelo_svd.transform(X_tratado)

        if padronizar_escala_componentes:
            if fatores_escala_h is None:
                raise RuntimeError("Os fatores de escala de H não foram calculados.")
            W_obra = W_obra * fatores_escala_h[np.newaxis, :]

        if reescalar_w_para_visualizacao:
            # Compatibilidade defensiva: nesta versão fica sempre False.
            W_obra = W_obra * 100.0

        if metodologia in ("NMF-KL", "NMF-Euclidiana", "MiniBatchNMF-Fast"):
            # Cálculo da KL completa é útil para auditoria, mas é pesado:
            # faz um produto WH sobre todos os pixels e canais da ROI.
            # Por isso fica opcional e desativado por padrão.
            if calcular_kl_reconstrucao_completa:
                report(
                    min(0.98, frac_base + 0.30 / max(total_obras, 1) * 0.45),
                    f"Calculando KL completa da reconstrução para {nome_breve} (opcional, etapa lenta)...",
                )
                metricas_obra = calcular_kl_reconstrucao_em_blocos(
                    X=X_tratado, W=W_obra, H=H_global, chunk_pixels=5000
                )
                metricas_obra.update({
                    "arquivo": registro.nome_arquivo,
                    "nome_arquivo": Path(registro.nome_arquivo).name,
                    "dataset_hdf5": registro.dataset_path,
                    "n_pixels": int(y_b * x_b),
                    "n_canais": int(c_b),
                    "n_componentes": int(n_componentes),
                    "metodo_fatoracao": metodologia,
                    "normalizacao": metodo_norm,
                    "binning": int(fator_binning),
                    "w_exportado_bruto_sem_x100": not padronizar_escala_componentes,
                    "escala_componentes_area_h_unitaria": bool(padronizar_escala_componentes),
                })
                metricas_reconstrucao_obras.append(metricas_obra)

            soma_comp = np.sum(W_obra, axis=0, dtype=np.float64)
            if soma_w_componentes_global is None:
                soma_w_componentes_global = np.zeros(n_componentes, dtype=np.float64)
            soma_w_componentes_global += soma_comp
            soma_w_total_global += float(np.sum(soma_comp))

        vetor_medio_obra = np.mean(W_obra, axis=0)
        dict_medias_por_obra[registro.nome_arquivo] = vetor_medio_obra

        if salvar_mapas_por_obra and pasta_mapas is not None:
            nome_seguro = sanitizar_nome_arquivo(Path(registro.nome_arquivo).stem)
            nome_csv = f"{idx_obra:03d}_{nome_seguro}_mapas_abundancia_vetorizados.csv"
            caminho_csv_mapa = pasta_mapas / nome_csv
            report(
                min(0.98, frac_base + 0.30 / max(total_obras, 1) * 0.55),
                f"Salvando mapas vetorizados de {nome_breve} em CSV...",
            )
            caminho_salvo = salvar_mapa_vetorizado_por_obra_csv(
                caminho_csv=caminho_csv_mapa,
                W_obra=W_obra,
                y_b=y_b,
                x_b=x_b,
                n_componentes=n_componentes,
            )
            arquivos_mapas_salvos.append(str(caminho_salvo))
            linhas_manifesto_mapas.append({
                "Indice_Obra": idx_obra,
                "Nome_Arquivo": Path(registro.nome_arquivo).name,
                "Caminho_Arquivo_Original": registro.nome_arquivo,
                "Dataset_HDF5": registro.dataset_path,
                "Y_pos_binning": int(y_b),
                "X_pos_binning": int(x_b),
                "ROI_Canal_Inicial": int(roi_inicio),
                "ROI_Canal_Final_Inclusivo": int(roi_fim) - 1,
                "Eixo_Espectral_Exportado": rotulo_eixo_exportacao,
                "N_Componentes": int(n_componentes),
                "Metodo_Fatoracao": metodologia,
                "Escala_Mapa": (
                    "Score SVD"
                    if metodo_eh_svd(metodologia)
                    else (
                        "W* compensado por área(H), com área(H*)=1 e sem x100"
                        if padronizar_escala_componentes
                        else "Abundância relativa bruta sem x100"
                    )
                ),
                "Arquivo_CSV_Mapas": str(caminho_salvo),
            })

        if armazenar_mapa_quando_unico and total_obras == 1:
            nome_obra_visualizacao = registro.nome_arquivo
            mapa_abundancia_visualizacao = W_obra.reshape(y_b, x_b, n_componentes).astype(np.float32, copy=False)

        del cubo_roi, cubo_binnado, X_roi, X_tratado, W_obra
        gc.collect()
        frac_fim = 0.65 + 0.30 * (idx_obra / max(total_obras, 1))
        report(frac_fim, f"Etapa 2/2: obra {idx_obra}/{total_obras} projetada com sucesso.")

    if salvar_mapas_por_obra and pasta_mapas is not None and linhas_manifesto_mapas:
        caminho_manifesto = pasta_mapas / f"{sanitizar_nome_arquivo(prefixo_mapas_por_obra)}_manifesto_mapas_por_obra.csv"
        pd.DataFrame(linhas_manifesto_mapas).to_csv(caminho_manifesto, index=False, sep=",", decimal=".")
        caminho_manifesto_mapas = str(caminho_manifesto)

    contribuicao_relativa_componentes = None
    if soma_w_componentes_global is not None and soma_w_total_global > 0:
        contribuicao_relativa_componentes = (100.0 * soma_w_componentes_global / soma_w_total_global).astype(np.float64)

    report(1.0, "Processamento concluído.")
    return (
        H_global,
        dict_medias_por_obra,
        nome_obra_visualizacao,
        mapa_abundancia_visualizacao,
        arquivos_mapas_salvos,
        caminho_manifesto_mapas,
        contribuicao_relativa_componentes,
        init_nmf_efetivo,
        alerta_convergencia,
        reconstruction_err_treino,
        n_iter_treino,
        metricas_reconstrucao_obras,
        fatores_escala_h,
    )



def _validar_h_global_eixo_roi(H_global: np.ndarray, eixo_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valida e padroniza H_global e eixo_roi para exportação espectral restrita à ROI."""
    H = np.asarray(H_global, dtype=np.float64)
    eixo = np.asarray(eixo_roi, dtype=np.float64)

    if H.ndim != 2:
        raise ValueError(f"H_global deve ser 2D. Shape encontrado: {H.shape}")

    if eixo.ndim != 1:
        raise ValueError(f"eixo_roi deve ser 1D. Shape encontrado: {eixo.shape}")

    if H.shape[1] != eixo.shape[0]:
        raise ValueError(
            "Inconsistência entre H_global e eixo_roi: a exportação dos espectros deve conter "
            "exatamente os canais/energias da ROI processada."
        )

    return H, eixo


def _eixo_exportacao_para_csv(eixo_roi: np.ndarray, rotulo_eixo: str) -> tuple[str, np.ndarray]:
    """
    Define o cabeçalho e os valores da primeira coluna dos CSVs espectrais individuais.

    Regras:
    - sem calibração ativa: channel,intensidade
    - com calibração ativa: energia (keV),intensidade
    """
    eixo = np.asarray(eixo_roi, dtype=np.float64)
    if rotulo_eixo.startswith("Energia"):
        return "energia (keV)", eixo

    if np.all(np.isfinite(eixo)) and np.allclose(eixo, np.rint(eixo)):
        return "channel", np.rint(eixo).astype(np.int64)
    return "channel", eixo


def exportar_componentes_espectrais_individuais(
    caminho_saida_prefixo: str,
    H_global: np.ndarray,
    eixo_roi: np.ndarray,
    rotulo_eixo: str,
    rotulos_componentes: list[str] | None = None,
    prefixo_arquivo: str | None = None,
) -> list[Path]:
    """
    Exporta uma componente espectral por arquivo CSV, em formato de espectro único.

    Formatos obrigatórios:
    - channel,intensidade
    - energia (keV),intensidade

    Cada arquivo contém somente duas colunas, sem índice, sem metadados e sem linhas extras.
    O nome da componente fica no nome do arquivo: Prefixo_H00.csv, Prefixo_H01.csv, etc.
    """
    prefixo = Path(caminho_saida_prefixo)
    pasta_saida = prefixo.parent
    base = sanitizar_nome_arquivo(prefixo_arquivo or prefixo.name)
    H, eixo = _validar_h_global_eixo_roi(H_global, eixo_roi)
    nome_eixo, eixo_exportado = _eixo_exportacao_para_csv(eixo, rotulo_eixo)

    caminhos: list[Path] = []
    for idx in range(H.shape[0]):
        sufixo_componente = f"H{idx:02d}"
        if rotulos_componentes is not None and idx < len(rotulos_componentes):
            rotulo = sanitizar_nome_arquivo(rotulos_componentes[idx])
            if rotulo:
                sufixo_componente = f"{sufixo_componente}_{rotulo}"

        caminho_csv = pasta_saida / f"{base}-{sufixo_componente}.csv"
        df = pd.DataFrame({
            nome_eixo: eixo_exportado,
            "intensidade": H[idx, :],
        })
        df.to_csv(caminho_csv, index=False, sep=",", decimal=".")
        caminhos.append(caminho_csv)

    return caminhos




def exportar_h_global_consolidado_csv(
    caminho_saida_prefixo: str,
    H_global: np.ndarray,
    eixo_roi: np.ndarray,
    rotulo_eixo: str,
    prefixo_arquivo: str | None = None,
) -> Path:
    """
    Exporta a matriz H consolidada em um único CSV.

    Formato:
        energia (keV),H00,H01,H02,...

    ou, quando não houver calibração energética:
        channel,H00,H01,H02,...

    Esta exportação é complementar aos espectros H individuais e não altera
    a fatoração, o W, a KL ou qualquer métrica do processamento.
    """
    prefixo = Path(caminho_saida_prefixo)
    pasta_saida = prefixo.parent
    base = sanitizar_nome_arquivo(prefixo_arquivo or prefixo.name).upper()

    H, eixo = _validar_h_global_eixo_roi(H_global, eixo_roi)
    nome_eixo, eixo_exportado = _eixo_exportacao_para_csv(eixo, rotulo_eixo)

    dados: dict[str, np.ndarray] = {nome_eixo: eixo_exportado}
    for idx in range(H.shape[0]):
        dados[f"H{idx:02d}"] = H[idx, :]

    df = pd.DataFrame(dados)
    caminho_csv = pasta_saida / f"{base}_H_consolidado.csv"
    df.to_csv(caminho_csv, index=False, sep=",", decimal=".")
    return caminho_csv


def exportar_contribuicao_relativa_componentes_csv(
    caminho_saida_prefixo: str,
    contribuicao_relativa_componentes: np.ndarray | None,
    metodo_fatoracao: str = "",
) -> Path | None:
    """Exporta a contribuição relativa interna de cada componente NMF/NMFKL."""
    if contribuicao_relativa_componentes is None or metodo_eh_svd(metodo_fatoracao):
        return None

    contrib = np.asarray(contribuicao_relativa_componentes, dtype=np.float64).ravel()
    prefixo = Path(caminho_saida_prefixo)
    df = pd.DataFrame({
        "componente": [f"H{i:02d}" for i in range(contrib.shape[0])],
        "contribuicao_relativa_interna_percentual": contrib,
        "menor_que_0_5_percentual": ["sim" if v < 0.5 else "não" for v in contrib],
    })
    caminho = prefixo.parent / f"{prefixo.name}_contribuicao_relativa_componentes.csv"
    df.to_csv(caminho, index=False, sep=",", decimal=".")
    return caminho


def exportar_resultados_lote(
    caminho_saida_prefixo: str,
    dict_medias: dict[str, np.ndarray],
    abundancias_em_percentual: bool = False,
    metodo_fatoracao: str = "",
    contribuicao_relativa_componentes: np.ndarray | None = None,
) -> tuple[Path, Path | None]:
    """
    Exporta apenas a planilha condensada com uma linha por obra pronta para PCA.

    A paleta global H não é mais exportada em formato consolidado channel,H00,H01,...
    porque a exportação espectral passa a ser feita em arquivos individuais de duas colunas.
    """
    prefixo = Path(caminho_saida_prefixo)

    linhas_tabela_pca = []
    for caminho_completo, vetor_componentes in dict_medias.items():
        path_obj = Path(caminho_completo)
        registro: dict[str, object] = {
            "Caminho_Arquivo": caminho_completo,
            "Nome_Obra": path_obj.stem,
            "Artista_Label": path_obj.parent.name if path_obj.parent.name != "" else "Desconhecido",
        }
        for idx, valor_medio in enumerate(vetor_componentes):
            if metodo_eh_svd(metodo_fatoracao):
                nome_coluna = f"SVD_Score_Medio_{idx+1}"
            else:
                nome_coluna = f"Comp_Media_{idx+1}"
            registro[nome_coluna] = float(valor_medio)
        linhas_tabela_pca.append(registro)

    df_tabela_pca = pd.DataFrame(linhas_tabela_pca)
    caminho_pca = prefixo.parent / f"{prefixo.name}_dados_consolidados_para_PCA.csv"
    df_tabela_pca.to_csv(caminho_pca, index=False, sep=",", decimal=".")

    caminho_contrib = exportar_contribuicao_relativa_componentes_csv(
        caminho_saida_prefixo=str(prefixo),
        contribuicao_relativa_componentes=contribuicao_relativa_componentes,
        metodo_fatoracao=metodo_fatoracao,
    )

    return caminho_pca, caminho_contrib


def exportar_espectros_e_mapas_csv(
    caminho_saida_prefixo: str,
    H_global: np.ndarray,
    eixo_roi: np.ndarray,
    rotulo_eixo: str,
    mapas_abundancia: np.ndarray | None,
    rotulo_x_mapa: str,
    rotulo_y_mapa: str,
    abundancias_em_percentual: bool = False,
    prefixo_curto_obra: str | None = None,
) -> tuple[list[Path], Path | None, Path | None]:
    """
    Exporta arquivos CSV compatíveis com o leitor externo:
    1) espectros individuais: um CSV por componente, com duas colunas;
    2) mapas vetorizados: row, column, NNMA_Image_00, ... (quando disponíveis em memória).
    """
    prefixo = Path(caminho_saida_prefixo)

    prefixo_curto = sanitizar_nome_arquivo(prefixo_curto_obra or prefixo.name).upper()

    caminhos_espectros = exportar_componentes_espectrais_individuais(
        caminho_saida_prefixo=str(prefixo),
        H_global=H_global,
        eixo_roi=eixo_roi,
        rotulo_eixo=rotulo_eixo,
        prefixo_arquivo=prefixo_curto,
    )

    caminho_mapa_vet = None
    pasta_mapas = None
    if mapas_abundancia is not None:
        mapas = np.asarray(mapas_abundancia, dtype=np.float64)
        y_idx, x_idx = np.indices(mapas.shape[:2])
        dados_vet: dict[str, np.ndarray] = {
            "row": y_idx.ravel().astype(np.int64, copy=False),
            "column": x_idx.ravel().astype(np.int64, copy=False),
        }
        for i in range(mapas.shape[2]):
            dados_vet[f"NNMA_Image_{i:02d}"] = mapas[:, :, i].ravel()
        df_vet = pd.DataFrame(dados_vet)
        caminho_mapa_vet = prefixo.parent / f"W-{prefixo_curto}.csv"
        df_vet.to_csv(caminho_mapa_vet, index=False, sep=",", decimal=".")

    return caminhos_espectros, caminho_mapa_vet, pasta_mapas


def exportar_metricas_reconstrucao_csv(
    pasta_saida: str | Path,
    prefixo_curto: str,
    resultado: ResultadoLote,
) -> Path | None:
    metricas = resultado.metricas_reconstrucao_obras or []
    if not metricas:
        return None
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    linhas = []
    for item in metricas:
        linha = dict(item)
        linha["k"] = int(resultado.n_componentes)
        linha["reconstruction_err_treino_amostra"] = resultado.reconstruction_err_treino
        linha["n_iter_treino_global"] = resultado.n_iter_treino
        if resultado.padronizar_escala_componentes:
            linha["observacao"] = (
                "H exportado com área discreta unitária e W compensado pelo fator de área; "
                "o produto WH e a KL permanecem invariantes."
            )
        else:
            linha["observacao"] = (
                "W exportado em escala bruta, KL da obra completa "
                "foi recalculada com X preprocessado, W bruto e H."
            )
        linhas.append(linha)
    caminho = pasta / f"{prefixo_curto}_metricas_reconstrucao_KL.csv"
    pd.DataFrame(linhas).to_csv(caminho, index=False, sep=",", decimal=".")
    return caminho


def exportar_fatores_escala_componentes_csv(
    pasta_saida: str | Path,
    prefixo_curto: str,
    resultado: ResultadoLote,
) -> Path | None:
    """Registra os fatores a_k usados em H*=H/a_k e W*=W*a_k."""
    if not resultado.padronizar_escala_componentes or resultado.fatores_escala_h is None:
        return None

    fatores = np.asarray(resultado.fatores_escala_h, dtype=np.float64).ravel()
    if fatores.shape[0] != resultado.H_global.shape[0]:
        raise ValueError("Número de fatores de escala incompatível com o número de componentes H.")

    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{prefixo_curto}_fatores_escala_componentes.csv"
    pd.DataFrame({
        "componente": [f"H{i:02d}" for i in range(fatores.shape[0])],
        "eixo_de_integracao": [resultado.rotulo_eixo] * fatores.shape[0],
        "area_H_antes": fatores,
        "area_H_depois": np.ones(fatores.shape[0], dtype=np.float64),
        "operacao_H": ["H* = H / area_H_antes"] * fatores.shape[0],
        "operacao_W": ["W* = W * area_H_antes"] * fatores.shape[0],
    }).to_csv(caminho, index=False, sep=",", decimal=".")
    return caminho


def salvar_manifesto_processamento(
    pasta_saida: str | Path,
    resultado: ResultadoLote,
    nome_pasta_exportacao: str,
    prefixo_curto: str,
    caminhos_espectros: list[Path] | None = None,
    caminho_h_consolidado: Path | None = None,
    caminho_mapa_vet: Path | None = None,
    caminho_fatores_escala: Path | None = None,
) -> tuple[Path, Path]:
    """
    Salva um manifesto metodológico junto aos CSVs exportados.

    O objetivo é tornar rastreável a rodada que gerou os espectros H e mapas W:
    parâmetros de ROI, normalização, fatoração, orientação do cubo, semente,
    inicialização e lista de arquivos usados.
    """
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    caminho_txt = pasta / "manifesto_processamento.txt"
    caminho_csv = pasta / "manifesto_processamento.csv"

    espectros_txt = "\n".join(str(Path(c)) for c in (caminhos_espectros or []))
    arquivos_txt = "\n".join(str(c) for c in resultado.nomes_obras)

    texto = (
        f"Aplicação: {APP_NAME}\n"
        f"Versão: {APP_VERSION}\n"
        f"Data/Hora da exportação: {timestamp}\n"
        f"Pasta de exportação: {nome_pasta_exportacao}\n"
        f"Prefixo curto de exportação: {prefixo_curto}\n\n"
        "=== Parâmetros da rodada ===\n"
        f"{resultado.info_textual}\n"
        "=== Observações metodológicas ===\n"
        "1. Os espectros H são cargas/componentes espectrais da fatoração, não espectros XRF brutos.\n"
        "2. Os mapas W/score maps são descritores relativos do modelo; não são porcentagens químicas absolutas.\n"
        "3. A normalização por Soma ou Área transforma a análise em comparação de perfis relativos dentro da ROI.\n"
        "4. A exportação H individual preserva apenas a ROI espectral efetivamente processada.\n\n"
        "5. Quando área(H)=1 está ativa, a integração usa o eixo espectral exportado e a escala retirada de cada H é transferida exatamente ao W correspondente; WH não muda.\n\n"
        "=== Arquivos HDF5 processados ===\n"
        f"{arquivos_txt}\n\n"
        "=== Espectros H individuais exportados ===\n"
        f"{espectros_txt or '(nenhum)'}\n\n"
        "=== Matriz H consolidada ===\n"
        f"{str(caminho_h_consolidado) if caminho_h_consolidado is not None else '(não exportada)'}\n\n"
        "=== Mapa W vetorizado ===\n"
        f"{str(caminho_mapa_vet) if caminho_mapa_vet is not None else '(não exportado)'}\n\n"
        "=== Fatores de escala das componentes ===\n"
        f"{str(caminho_fatores_escala) if caminho_fatores_escala is not None else '(não aplicados)'}\n"
    )
    caminho_txt.write_text(texto, encoding="utf-8")

    linhas = [
        ("app", APP_NAME),
        ("versao", APP_VERSION),
        ("data_hora_exportacao", timestamp),
        ("pasta_exportacao", nome_pasta_exportacao),
        ("prefixo_curto", prefixo_curto),
        ("metodo_fatoracao", resultado.metodo_fatoracao),
        ("n_componentes", resultado.n_componentes),
        ("normalizacao", resultado.metodo_norm),
        ("binning", resultado.fator_binning),
        ("binning_espectral", resultado.fator_binning_espectral),
        ("rotulo_eixo", resultado.rotulo_eixo),
        ("orientacao_cubo", resultado.orientacao_cubo),
        ("modo_semente", resultado.modo_semente),
        ("random_state_usado", resultado.random_state_usado),
        ("init_nmf_modo", resultado.init_nmf_modo),
        ("init_nmf_efetivo", resultado.init_nmf_efetivo),
        ("n_arquivos_hdf5", len(resultado.nomes_obras)),
        ("n_espectros_h_exportados", len(caminhos_espectros or [])),
        ("h_consolidado_csv", str(caminho_h_consolidado) if caminho_h_consolidado is not None else ""),
        ("mapa_w_vetorizado", str(caminho_mapa_vet) if caminho_mapa_vet is not None else ""),
        ("padronizar_escala_componentes", "sim" if resultado.padronizar_escala_componentes else "não"),
        ("convencao_escala_componentes", "H*=H/área(H) no eixo exportado; W*=W×área(H)" if resultado.padronizar_escala_componentes else "escala livre da NMF"),
        ("fatores_escala_componentes_csv", str(caminho_fatores_escala) if caminho_fatores_escala is not None else ""),
        ("escala_w", resultado.rotulo_abundancia),
        ("w_exportado_bruto_sem_x100", "não" if resultado.padronizar_escala_componentes else "sim"),
        ("reconstruction_err_treino_amostra", resultado.reconstruction_err_treino if resultado.reconstruction_err_treino is not None else ""),
        ("n_iter_treino_global", resultado.n_iter_treino if resultado.n_iter_treino is not None else ""),
    ]
    pd.DataFrame(linhas, columns=["parametro", "valor"]).to_csv(caminho_csv, index=False, sep=",", decimal=".")
    return caminho_txt, caminho_csv


# ============================================================
# Aplicação principal
# ============================================================

class AppMAXRF(ctk.CTk):
    PADX = 6
    WRAP = 270

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("1520x920")
        self.minsize(1260, 760)

        self.arquivos_h5: list[str] = []
        self.dataset_path = DEFAULT_DATASET_PATH
        self.resultado_lote: ResultadoLote | None = None
        self.executando = False
        self.ultimo_relatorio_erro: Path | None = None
        self.componente_atual_idx = 0
        self._colorbar = None
        self.var_usar_calibracao = ctk.BooleanVar(value=True)
        self.var_salvar_mapas_lote = ctk.BooleanVar(value=False)
        self.var_calcular_kl_completo = ctk.BooleanVar(value=False)
        self.var_padronizar_escala_componentes = ctk.BooleanVar(value=False)
        self._modo_roi_anterior = "Energia"
        self._ultima_semente_usada: int | None = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._criar_painel_lateral()
        self._criar_painel_plot()

    # --------------------------------------------------------
    # Construção da GUI
    # --------------------------------------------------------
    def _criar_painel_lateral(self):
        self.sidebar = ctk.CTkScrollableFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        row = 0

        titulo = ctk.CTkLabel(
            self.sidebar,
            text="Júlia 1",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        titulo.grid(row=row, column=0, padx=self.PADX, pady=(8, 4), sticky="w")
        row += 1

        self._secao_label("1. Carregamento de dados", row)
        row += 1

        self.btn_selecionar = ctk.CTkButton(
            self.sidebar,
            text="Selecionar arquivos .h5",
            command=self.selecionar_arquivos,
            height=28,
        )
        self.btn_selecionar.grid(row=row, column=0, padx=self.PADX, pady=(2, 3), sticky="ew")
        row += 1

        self.lbl_arquivo = ctk.CTkLabel(
            self.sidebar,
            text="Nenhum arquivo selecionado",
            justify="left",
            wraplength=self.WRAP,
        )
        self.lbl_arquivo.grid(row=row, column=0, padx=self.PADX, pady=(0, 3), sticky="w")
        row += 1

        self.entry_offset = self._entry_labeled(row, "Offset da calibração (opcional)", DEFAULT_ENERGIA_OFFSET)
        row += 2

        self.entry_gain = self._entry_labeled(row, "Gain da calibração (opcional)", DEFAULT_ENERGIA_GAIN)
        row += 2

        self.check_usar_calibracao = ctk.CTkCheckBox(
            self.sidebar,
            text="Usar calibração energética na visualização/exportação",
            variable=self.var_usar_calibracao,
            onvalue=True,
            offvalue=False,
        )
        self.check_usar_calibracao.grid(row=row, column=0, padx=self.PADX, pady=(2, 4), sticky="w")
        row += 1

        self._secao_label("2. Processamento espacial e ROI", row)
        row += 1

        self.combo_binning = self._combo_labeled(row, "Binning espacial", ["1", "2", "4", "8"], "2")
        row += 2

        self.combo_binning_espectral = self._combo_labeled(
            row, "Binning Canais/Energia", ["1", "2", "4", "8"], "1"
        )
        row += 2

        self.combo_modo_roi = self._combo_labeled(row, "ROI espectral baseada em", ["Canal", "Energia"], "Energia")
        self.combo_modo_roi.configure(command=self._ao_mudar_modo_roi)
        row += 2

        self.entry_roi_ini = self._entry_labeled(row, "ROI inicial (canal ou energia)", "1.8")
        row += 2

        self.entry_roi_fim = self._entry_labeled(row, "ROI final (canal ou energia)", "16")
        row += 2

        self.combo_norm = self._combo_labeled(row, "Método de normalização", ["Nenhum", "Soma", "Área"], "Área")
        row += 2

        self.combo_orientacao_cubo = self._combo_labeled(
            row,
            "Orientação do cubo HDF5",
            ORIENTACOES_CUBO_GUI,
            ORIENTACAO_AUTO_SEGURA,
        )
        row += 2

        self._secao_label("3. Configuração dos algoritmos", row)
        row += 1

        self.combo_metodo = self._combo_labeled(
            row,
            "Metodologia de fatoração",
            ["NMF-KL", "NMF-Euclidiana", "MiniBatchNMF-Fast", "TruncatedSVD"],
            "NMF-KL",
        )
        row += 2

        self.check_padronizar_escala_componentes = ctk.CTkCheckBox(
            self.sidebar,
            text="Fixar escala NMF: área(H)=1 no eixo e compensar W",
            variable=self.var_padronizar_escala_componentes,
            onvalue=True,
            offvalue=False,
        )
        self.check_padronizar_escala_componentes.grid(
            row=row,
            column=0,
            padx=self.PADX,
            pady=(2, 4),
            sticky="w",
        )
        row += 1

        self.frame_k = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_k.grid(row=row, column=0, padx=self.PADX, pady=(0, 0), sticky="ew")
        self.frame_k.grid_columnconfigure(1, weight=1)
        self.frame_k.grid_columnconfigure(3, weight=1)

        self.lbl_k = ctk.CTkLabel(self.frame_k, text="k: de")
        self.lbl_k.grid(row=0, column=0, padx=(0, 4), pady=(0, 0), sticky="w")

        self.entry_k_de = ctk.CTkEntry(self.frame_k, height=27, width=58)
        self.entry_k_de.grid(row=0, column=1, padx=(0, 8), pady=(0, 0), sticky="ew")
        self.entry_k_de.insert(0, "8")

        self.lbl_k_ate = ctk.CTkLabel(self.frame_k, text="até")
        self.lbl_k_ate.grid(row=0, column=2, padx=(0, 4), pady=(0, 0), sticky="w")

        self.entry_k_ate = ctk.CTkEntry(self.frame_k, height=27, width=58)
        self.entry_k_ate.grid(row=0, column=3, padx=(0, 0), pady=(0, 0), sticky="ew")

        # Mantém compatibilidade interna com trechos antigos que esperavam self.entry_k.
        self.entry_k = self.entry_k_de
        row += 1

        self.entry_pixels_amostra = self._entry_labeled(row, "Pixels amostrados por obra", "2000")
        row += 2


        self.entry_max_iter = self._entry_labeled(row, "Máximo de iterações", "1000")
        row += 2

        self.entry_tol = self._entry_labeled(row, "EPS / Tolerância", "1e-4")
        row += 2

        self.check_calcular_kl_completo = ctk.CTkCheckBox(
            self.sidebar,
            text="Calcular KL completa após projetar W (lento; opcional)",
            variable=self.var_calcular_kl_completo,
            onvalue=True,
            offvalue=False,
        )
        self.check_calcular_kl_completo.grid(row=row, column=0, padx=self.PADX, pady=(2, 4), sticky="w")
        row += 1

        self.combo_modo_semente = self._combo_labeled(
            row,
            "Modo de semente",
            [
                "Reprodutível (semente fixa)",
                "Aleatória nova a cada execução",
                "Usar semente informada",
            ],
            "Reprodutível (semente fixa)",
        )
        row += 2

        self.entry_seed = self._entry_labeled(row, "Semente informada/fixa", "42")
        row += 2

        self.combo_init_nmf = self._combo_labeled(
            row,
            "Inicialização NMF",
            [
                "Auto estável (NNDSVDa quando possível)",
                "Aleatória (random)",
            ],
            "Auto estável (NNDSVDa quando possível)",
        )
        row += 2

        self._secao_label("4. Execução e exportação", row)
        row += 1

        self.btn_executar = ctk.CTkButton(
            self.sidebar,
            text="EXECUTAR FATORAÇÃO",
            height=32,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.iniciar_fatoracao,
        )
        self.btn_executar.grid(row=row, column=0, padx=self.PADX, pady=(6, 4), sticky="ew")
        row += 1

        self.btn_exportar = ctk.CTkButton(
            self.sidebar,
            text="EXPORTAR CSVs",
            command=self.exportar_resultados,
            fg_color="#2E8B57",
            hover_color="#246B45",
            height=28,
        )
        self.btn_exportar.grid(row=row, column=0, padx=self.PADX, pady=(0, 3), sticky="ew")
        row += 1

        self.btn_zerar = ctk.CTkButton(
            self.sidebar,
            text="ZERAR / LIMPAR MEMÓRIA",
            command=self.zerar_memoria_resultados,
            fg_color="#8B4513",
            hover_color="#6E3610",
            height=28,
        )
        self.btn_zerar.grid(row=row, column=0, padx=self.PADX, pady=(0, 5), sticky="ew")
        row += 1

        self.progress = ctk.CTkProgressBar(self.sidebar, mode="determinate")
        self.progress.set(0)
        self.progress.grid(row=row, column=0, padx=self.PADX, pady=(0, 3), sticky="ew")
        row += 1

        self.lbl_status = ctk.CTkLabel(
            self.sidebar,
            text="Status: aguardando",
            justify="left",
            wraplength=self.WRAP,
        )
        self.lbl_status.grid(row=row, column=0, padx=self.PADX, pady=(0, 4), sticky="w")
        row += 1

        self.txt_info = ctk.CTkTextbox(self.sidebar, height=92)
        self.txt_info.grid(row=row, column=0, padx=self.PADX, pady=(0, 10), sticky="nsew")
        self.txt_info.insert("1.0", "Informações do processamento em lote aparecerão aqui.\n")
        self.txt_info.configure(state="disabled")

        self.sidebar.grid_rowconfigure(row, weight=1)

    def _criar_painel_plot(self):
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 12), pady=10)
        self.plot_frame.grid_rowconfigure(2, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        self.lbl_plot = ctk.CTkLabel(
            self.plot_frame,
            text="Visualização das componentes e mapas de abundância",
            font=ctk.CTkFont(size=19, weight="bold"),
        )
        self.lbl_plot.grid(row=0, column=0, padx=12, pady=(8, 0), sticky="w")

        self.frame_plot_controls = ctk.CTkFrame(self.plot_frame)
        self.frame_plot_controls.grid(row=1, column=0, sticky="ew", padx=10, pady=(6, 2))
        for col in range(8):
            self.frame_plot_controls.grid_columnconfigure(col, weight=1)

        self.lbl_modo_plot = ctk.CTkLabel(self.frame_plot_controls, text="Visualização")
        self.lbl_modo_plot.grid(row=0, column=0, padx=(8, 4), pady=(6, 2), sticky="w")
        self.combo_visualizacao = ctk.CTkOptionMenu(
            self.frame_plot_controls,
            values=[
                "Componente espectral + mapa",
                "Espectro da componente",
                "Mapa de abundância",
                "Todos os espectros",
            ],
            command=lambda _v: self._atualizar_plot_interativo(),
            height=28,
        )
        self.combo_visualizacao.grid(row=0, column=1, padx=(0, 8), pady=(6, 2), sticky="ew")
        self.combo_visualizacao.set("Componente espectral + mapa")

        self.lbl_componente = ctk.CTkLabel(self.frame_plot_controls, text="Componente")
        self.lbl_componente.grid(row=0, column=2, padx=(8, 4), pady=(6, 2), sticky="w")
        self.combo_componente_plot = ctk.CTkOptionMenu(
            self.frame_plot_controls,
            values=["Componente 00"],
            command=self._ao_mudar_componente_plot,
            height=28,
        )
        self.combo_componente_plot.grid(row=0, column=3, padx=(0, 8), pady=(6, 2), sticky="ew")
        self.combo_componente_plot.set("Componente 00")

        self.lbl_cmap = ctk.CTkLabel(self.frame_plot_controls, text="Mapa de cores")
        self.lbl_cmap.grid(row=0, column=4, padx=(8, 4), pady=(6, 2), sticky="w")
        self.combo_cmap = ctk.CTkOptionMenu(
            self.frame_plot_controls,
            values=["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "hot", "temperature", "jet", "gray"],
            command=lambda _v: self._atualizar_plot_interativo(),
            height=28,
        )
        self.combo_cmap.grid(row=0, column=5, padx=(0, 8), pady=(6, 2), sticky="ew")
        self.combo_cmap.set("temperature")

        self.lbl_interp = ctk.CTkLabel(self.frame_plot_controls, text="Interpolação")
        self.lbl_interp.grid(row=0, column=6, padx=(8, 4), pady=(6, 2), sticky="w")
        self.combo_interp = ctk.CTkOptionMenu(
            self.frame_plot_controls,
            values=["nearest", "bilinear", "bicubic", "lanczos", "none"],
            command=lambda _v: self._atualizar_plot_interativo(),
            height=28,
        )
        self.combo_interp.grid(row=0, column=7, padx=(0, 8), pady=(6, 2), sticky="ew")
        self.combo_interp.set("nearest")

        self.lbl_x_espectral = ctk.CTkLabel(self.frame_plot_controls, text="Eixo X espectral")
        self.lbl_x_espectral.grid(row=1, column=0, padx=(8, 4), pady=(2, 2), sticky="w")
        self.entry_x_espectral = ctk.CTkEntry(self.frame_plot_controls, height=28)
        self.entry_x_espectral.grid(row=1, column=1, padx=(0, 8), pady=(2, 2), sticky="ew")
        self.entry_x_espectral.insert(0, "Energia (keV)")

        self.lbl_y_espectral = ctk.CTkLabel(self.frame_plot_controls, text="Eixo Y espectral")
        self.lbl_y_espectral.grid(row=1, column=2, padx=(8, 4), pady=(2, 2), sticky="w")
        self.entry_y_espectral = ctk.CTkEntry(self.frame_plot_controls, height=28)
        self.entry_y_espectral.grid(row=1, column=3, padx=(0, 8), pady=(2, 2), sticky="ew")
        self.entry_y_espectral.insert(0, "Carga espectral")

        self.lbl_x_mapa = ctk.CTkLabel(self.frame_plot_controls, text="Eixo X mapa")
        self.lbl_x_mapa.grid(row=1, column=4, padx=(8, 4), pady=(2, 2), sticky="w")
        self.entry_x_mapa = ctk.CTkEntry(self.frame_plot_controls, height=28)
        self.entry_x_mapa.grid(row=1, column=5, padx=(0, 8), pady=(2, 2), sticky="ew")
        self.entry_x_mapa.insert(0, "X-pixel")

        self.lbl_y_mapa = ctk.CTkLabel(self.frame_plot_controls, text="Eixo Y mapa")
        self.lbl_y_mapa.grid(row=1, column=6, padx=(8, 4), pady=(2, 2), sticky="w")
        self.entry_y_mapa = ctk.CTkEntry(self.frame_plot_controls, height=28)
        self.entry_y_mapa.grid(row=1, column=7, padx=(0, 8), pady=(2, 2), sticky="ew")
        self.entry_y_mapa.insert(0, "Y-pixel")

        self.lbl_escala_y = ctk.CTkLabel(self.frame_plot_controls, text="Escala espectral")
        self.lbl_escala_y.grid(row=2, column=0, padx=(8, 4), pady=(2, 6), sticky="w")
        self.combo_escala_y = ctk.CTkOptionMenu(
            self.frame_plot_controls,
            values=["Linear", "Log"],
            command=lambda _v: self._atualizar_plot_interativo(),
            height=28,
        )
        self.combo_escala_y.grid(row=2, column=1, padx=(0, 8), pady=(2, 6), sticky="ew")
        self.combo_escala_y.set("Linear")

        self.lbl_contrib_atual = ctk.CTkLabel(
            self.frame_plot_controls,
            text="Contribuição relativa interna: --",
            anchor="w",
        )
        self.lbl_contrib_atual.grid(row=2, column=2, columnspan=4, padx=(8, 8), pady=(2, 6), sticky="ew")

        self.btn_aplicar_rotulos = ctk.CTkButton(
            self.frame_plot_controls,
            text="Aplicar rótulos/estilo",
            command=self._atualizar_plot_interativo,
            height=28,
        )
        self.btn_aplicar_rotulos.grid(row=2, column=6, columnspan=2, padx=(0, 8), pady=(2, 6), sticky="ew")

        for entry in [self.entry_x_espectral, self.entry_y_espectral, self.entry_x_mapa, self.entry_y_mapa]:
            entry.bind("<Return>", lambda _e: self._atualizar_plot_interativo())

        self.fig = Figure(figsize=(11, 7), dpi=100)
        ax = self.fig.add_subplot(111)
        ax.set_title("Aguardando processamento")
        ax.set_xlabel("Canal / Energia")
        ax.set_ylabel("Carga espectral")
        ax.grid(True, alpha=0.25)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=10, pady=(4, 0))

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

    # --------------------------------------------------------
    # Helpers de GUI
    # --------------------------------------------------------
    def _secao_label(self, texto: str, row: int):
        lbl = ctk.CTkLabel(self.sidebar, text=texto, font=ctk.CTkFont(size=13, weight="bold"))
        lbl.grid(row=row, column=0, padx=self.PADX, pady=(4, 0), sticky="w")
        return lbl

    def _entry_labeled(self, row: int, texto: str, valor_inicial: str):
        lbl = ctk.CTkLabel(self.sidebar, text=texto)
        lbl.grid(row=row, column=0, padx=self.PADX, pady=(0, 0), sticky="w")
        entry = ctk.CTkEntry(self.sidebar, height=27)
        entry.grid(row=row + 1, column=0, padx=self.PADX, pady=(0, 0), sticky="ew")
        if valor_inicial:
            entry.insert(0, valor_inicial)
        return entry

    def _combo_labeled(self, row: int, texto: str, valores: list[str], valor_padrao: str):
        lbl = ctk.CTkLabel(self.sidebar, text=texto)
        lbl.grid(row=row, column=0, padx=self.PADX, pady=(0, 0), sticky="w")
        combo = ctk.CTkOptionMenu(self.sidebar, values=valores, height=27)
        combo.grid(row=row + 1, column=0, padx=self.PADX, pady=(0, 0), sticky="ew")
        combo.set(valor_padrao)
        return combo

    def _atualizar_status(self, texto: str):
        self.lbl_status.configure(text=f"Status: {texto}")

    def _escrever_info(self, texto: str):
        self.txt_info.configure(state="normal")
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", texto)
        self.txt_info.configure(state="disabled")

    def _atualizar_progresso(self, fracao: float, texto: str | None = None):
        self.progress.set(max(0.0, min(1.0, float(fracao))))
        if texto:
            self._atualizar_status(texto)

    def _reportar_progresso_threadsafe(self, fracao: float, texto: str):
        self.after(0, lambda: self._atualizar_progresso(fracao, texto))

    def _ao_mudar_componente_plot(self, valor: str):
        try:
            self.componente_atual_idx = max(0, int(valor.split()[-1]))
        except Exception:
            self.componente_atual_idx = 0
        self._atualizar_plot_interativo()

    def _formatar_valor_roi(self, valor: float, casas: int = 6) -> str:
        """Formata valores de ROI sem zeros desnecessários, preservando precisão suficiente."""
        texto = f"{float(valor):.{casas}f}"
        texto = texto.rstrip("0").rstrip(".")
        return texto or "0"

    def _ler_offset_gain(self) -> tuple[float, float]:
        """Lê offset/gain atuais da interface para conversões Canal <-> Energia."""
        offset = float(self.entry_offset.get().strip())
        gain = float(self.entry_gain.get().strip())
        if gain == 0:
            raise ValueError("O gain não pode ser zero.")
        return offset, gain

    def _canal_para_energia(self, canal: float) -> float:
        offset, gain = self._ler_offset_gain()
        return offset + gain * float(canal)

    def _energia_para_canal(self, energia: float) -> int:
        offset, gain = self._ler_offset_gain()
        return int(round((float(energia) - offset) / gain))

    def _ao_mudar_modo_roi(self, valor: str):
        """
        Converte os valores exibidos na ROI ao alternar entre Canal e Energia.

        Exemplo com os padrões atuais:
        canal 270  -> E = offset + gain * 270
        canal 1430 -> E = offset + gain * 1430

        Assim, ao clicar em Energia, a interface mostra a energia equivalente
        aos canais realmente preenchidos, em vez de usar limites fixos.
        """
        try:
            valor = (valor or "").strip()
            modo_anterior = getattr(self, "_modo_roi_anterior", "Canal")
            v_ini = float(self.entry_roi_ini.get().strip().replace(",", "."))
            v_fim = float(self.entry_roi_fim.get().strip().replace(",", "."))

            if valor == "Energia" and modo_anterior != "Energia":
                novo_ini = self._canal_para_energia(v_ini)
                novo_fim = self._canal_para_energia(v_fim)
                self.var_usar_calibracao.set(True)
                self.entry_roi_ini.delete(0, "end")
                self.entry_roi_fim.delete(0, "end")
                self.entry_roi_ini.insert(0, self._formatar_valor_roi(novo_ini, casas=6))
                self.entry_roi_fim.insert(0, self._formatar_valor_roi(novo_fim, casas=6))

            elif valor == "Canal" and modo_anterior != "Canal":
                novo_ini = self._energia_para_canal(v_ini)
                novo_fim = self._energia_para_canal(v_fim)
                self.entry_roi_ini.delete(0, "end")
                self.entry_roi_fim.delete(0, "end")
                self.entry_roi_ini.insert(0, str(novo_ini))
                self.entry_roi_fim.insert(0, str(novo_fim))

            self._modo_roi_anterior = valor or modo_anterior
        except Exception as exc:
            # Mantém o valor digitado pelo usuário e informa discretamente no status.
            try:
                self._atualizar_status(f"não foi possível converter ROI Canal/Energia: {exc}")
            except Exception:
                pass

    def _rotulo_x_espectral(self) -> str:
        valor = self.entry_x_espectral.get().strip()
        if valor:
            return valor
        if self.resultado_lote is not None:
            return self.resultado_lote.rotulo_eixo
        return "Canal / Energia"

    def _rotulo_y_espectral(self) -> str:
        return self.entry_y_espectral.get().strip() or "Carga espectral"

    def _rotulo_x_mapa(self) -> str:
        return self.entry_x_mapa.get().strip() or "X (pós-binning)"

    def _rotulo_y_mapa(self) -> str:
        return self.entry_y_mapa.get().strip() or "Y (pós-binning)"

    def _ler_intervalo_k(self) -> tuple[int, int]:
        """Lê o intervalo k: de [ ] até [ ]. Se o campo até estiver vazio, roda um único k."""
        texto_k_ini = self.entry_k_de.get().strip() if hasattr(self, "entry_k_de") else self.entry_k.get().strip()
        texto_k_fim = self.entry_k_ate.get().strip() if hasattr(self, "entry_k_ate") else ""

        def _parse_k_inteiro(texto: str, nome_campo: str) -> int:
            texto_limpo = texto.strip()
            if not texto_limpo:
                raise ValueError(f"Informe o {nome_campo}.")
            try:
                valor_float = float(texto_limpo.replace(",", "."))
            except Exception:
                raise ValueError(f"O {nome_campo} deve ser um número inteiro.")
            if not valor_float.is_integer():
                raise ValueError(f"O {nome_campo} deve ser inteiro; não use valor decimal.")
            return int(valor_float)

        if not texto_k_ini:
            raise ValueError("Informe o k inicial no campo 'k: de'.")

        k_ini = _parse_k_inteiro(texto_k_ini, "k inicial")

        if texto_k_fim == "":
            k_fim = k_ini
        else:
            k_fim = _parse_k_inteiro(texto_k_fim, "k final")

        if k_ini <= 0:
            raise ValueError("O k inicial deve ser maior que zero.")
        if k_fim <= 0:
            raise ValueError("O k final deve ser maior que zero.")
        if k_fim < k_ini:
            raise ValueError("O k final deve ser maior ou igual ao k inicial.")

        return k_ini, k_fim

    def _configurar_controles_plot(self, resultado: ResultadoLote):
        opcoes = [f"Componente {i:02d}" for i in range(resultado.H_global.shape[0])]
        self.combo_componente_plot.configure(values=opcoes)
        self.combo_componente_plot.set(opcoes[0])
        self.componente_atual_idx = 0
        self.combo_visualizacao.set("Componente espectral + mapa")

        self.entry_x_espectral.delete(0, "end")
        self.entry_x_espectral.insert(0, resultado.rotulo_eixo)
        if not self.entry_y_espectral.get().strip():
            self.entry_y_espectral.insert(0, "Carga espectral")
        if not self.entry_x_mapa.get().strip():
            self.entry_x_mapa.insert(0, "X (pós-binning)")
        if not self.entry_y_mapa.get().strip():
            self.entry_y_mapa.insert(0, "Y (pós-binning)")

    def _registrar_erro(self, exc: BaseException, contexto: str) -> Path:
        caminho = gerar_relatorio_erro(exc, contexto)
        self.ultimo_relatorio_erro = caminho
        return caminho

    # --------------------------------------------------------
    # Eventos
    # --------------------------------------------------------
    def selecionar_arquivos(self):
        caminhos = filedialog.askopenfilenames(
            title="Selecione os arquivos HDF5",
            filetypes=[("Arquivos HDF5", "*.h5 *.hdf5"), ("Todos os arquivos", "*.*")],
        )
        if not caminhos:
            return

        try:
            candidatos = listar_datasets_hdf5_3d(caminhos[0])
            if not candidatos:
                raise ValueError(
                    "O primeiro arquivo não contém datasets 3D numéricos compatíveis com um cubo MA-XRF."
                )

            caminho_dataset = escolher_dataset_preferencial(candidatos)
            if not caminho_dataset:
                dialogo = DialogoEscolhaDataset(self, candidatos)
                self.wait_window(dialogo)
                caminho_dataset = dialogo.resultado
                if not caminho_dataset:
                    return

        except Exception as exc:
            caminho = self._registrar_erro(exc, "Falha ao inspecionar o HDF5 e listar datasets 3D candidatos.")
            messagebox.showerror(
                "Erro ao ler HDF5",
                "Não foi possível identificar os datasets do arquivo selecionado.\n\n"
                f"Relatório salvo em:\n{caminho}",
            )
            return

        self.arquivos_h5 = list(caminhos)
        self.dataset_path = caminho_dataset.strip() or DEFAULT_DATASET_PATH

        nomes = [Path(c).name for c in self.arquivos_h5]
        preview = "\n".join(nomes[:4])
        restante = ""
        if len(nomes) > 4:
            restante = f"\n... e mais {len(nomes) - 4} arquivo(s)"

        self.lbl_arquivo.configure(
            text=(
                f"Arquivos selecionados: {len(self.arquivos_h5)}\n"
                f"Dataset detectado/escolhido (prioridade: cube): {self.dataset_path}\n\n"
                f"{preview}{restante}"
            )
        )
        self._atualizar_status("arquivos selecionados; dataset do cubo definido")

    def abrir_pasta_relatorios(self):
        pasta = pasta_base_relatorios()
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(pasta))
            elif sys.platform == "darwin":
                os.system(f'open "{pasta}"')
            else:
                os.system(f'xdg-open "{pasta}"')
        except Exception as exc:
            caminho = self._registrar_erro(exc, "Falha ao abrir a pasta de relatórios.")
            messagebox.showerror("Erro", f"Não foi possível abrir a pasta de relatórios.\n\nRelatório salvo em:\n{caminho}")

    def _resolver_random_state_e_modo(self) -> tuple[int | None, str]:
        """
        Resolve a semente usada em NumPy/scikit-learn.

        Modos disponíveis:
        - Reprodutível: usa a semente informada no campo, padrão 42.
        - Aleatória nova: gera uma semente inteira nova e a registra no relatório.
        - Usar semente informada: igual ao modo reprodutível, mas deixa explícito
          que o número digitado foi escolhido pelo usuário.
        """
        modo = self.combo_modo_semente.get().strip() if hasattr(self, "combo_modo_semente") else "Reprodutível (semente fixa)"
        texto_seed = self.entry_seed.get().strip() if hasattr(self, "entry_seed") else "42"

        if modo == "Aleatória nova a cada execução":
            seed = int.from_bytes(os.urandom(4), byteorder="little", signed=False)
            # Atualiza a interface para que a execução possa ser repetida exatamente,
            # caso o usuário copie/mantenha este valor depois.
            try:
                self.entry_seed.delete(0, "end")
                self.entry_seed.insert(0, str(seed))
            except Exception:
                pass
            return seed, modo

        try:
            seed = int(float(texto_seed))
        except Exception:
            raise ValueError("A semente deve ser um número inteiro. Use, por exemplo, 42 ou escolha 'Aleatória nova a cada execução'.")

        if seed < 0:
            raise ValueError("A semente deve ser um inteiro maior ou igual a zero.")
        return seed, modo

    def zerar_memoria_resultados(self):
        """
        Remove resultados e objetos gráficos da execução anterior sem apagar os
        parâmetros nem a lista de arquivos. Isto ajuda a garantir uma nova rodada
        limpa e libera memória ocupada por H, W e mapas locais.
        """
        if self.executando:
            messagebox.showwarning("Processamento em andamento", "Aguarde o processamento terminar antes de zerar a memória.")
            return

        self.resultado_lote = None
        self.componente_atual_idx = 0
        self._ultima_semente_usada = None

        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_title("Aguardando processamento")
            ax.set_xlabel("Canal / Energia")
            ax.set_ylabel("Carga espectral")
            ax.grid(True, alpha=0.25)
            self._colorbar = None
            self.canvas.draw_idle()
        except Exception:
            pass

        try:
            self.combo_componente_plot.configure(values=["Componente 00"])
            self.combo_componente_plot.set("Componente 00")
        except Exception:
            pass

        self.progress.set(0)
        try:
            self.lbl_plot.configure(text="Visualização das componentes e mapas de abundância")
        except Exception:
            pass

        self._escrever_info(
            "Resultados anteriores removidos da memória.\n\n"
            "A lista de arquivos e os parâmetros foram preservados.\n"
            "Para obter uma solução diferente, use 'Aleatória nova a cada execução' "
            "e, preferencialmente, 'Inicialização NMF = Aleatória (random)'.\n"
            "Para tese/reprodutibilidade, use semente fixa e registre o valor.\n"
        )
        gc.collect()
        self._atualizar_status("memória de resultados zerada")

    def iniciar_fatoracao(self):
        if self.executando:
            messagebox.showwarning("Processamento em andamento", "Já existe um processamento em execução.")
            return

        if not self.arquivos_h5:
            messagebox.showwarning("Arquivos ausentes", "Selecione um ou mais arquivos .h5 antes de executar.")
            return

        try:
            k_ini, k_fim = self._ler_intervalo_k()
        except Exception as exc:
            messagebox.showerror("Intervalo de k inválido", str(exc))
            return

        self._lista_k_execucao = list(range(k_ini, k_fim + 1))
        self._pasta_raiz_varredura_k = None

        if len(self._lista_k_execucao) > 1:
            pasta_raiz = filedialog.askdirectory(
                title="Escolha a pasta raiz onde serão salvas as rodadas de k"
            )
            if not pasta_raiz:
                return
            self._pasta_raiz_varredura_k = pasta_raiz

        self.executando = True
        self.btn_executar.configure(state="disabled")
        self.btn_exportar.configure(state="disabled")
        if hasattr(self, "btn_zerar"):
            self.btn_zerar.configure(state="disabled")
        self.progress.set(0)

        if len(self._lista_k_execucao) > 1:
            self._atualizar_status(f"iniciando varredura de k = {k_ini} até {k_fim}...")
        else:
            self._atualizar_status("iniciando processamento em lote...")

        thread = threading.Thread(target=self._worker_fatoracao, daemon=True)
        thread.start()

    def _worker_fatoracao(self):
        try:
            lista_k = getattr(self, "_lista_k_execucao", None) or [self._ler_intervalo_k()[0]]
            pasta_raiz_varredura = getattr(self, "_pasta_raiz_varredura_k", None)

            if len(lista_k) > 1:
                resultado_final, pasta_raiz, caminho_resumo = self._executar_varredura_k(lista_k, pasta_raiz_varredura)
                self.after(0, lambda: self._finalizar_sucesso_varredura(resultado_final, pasta_raiz, caminho_resumo, lista_k))
            else:
                resultado = self._executar_pipeline_lote(k_override=lista_k[0])
                self.after(0, lambda: self._finalizar_sucesso(resultado))
        except Exception as exc:
            caminho = self._registrar_erro(exc, "Erro durante o pipeline de leitura, pré-processamento ou fatoração em lote.")
            erro_texto = f"{exc}\n\nRelatório salvo em:\n{caminho}"
            self.after(0, lambda: self._finalizar_erro(erro_texto))

    def _finalizar_sucesso(self, resultado: ResultadoLote):
        self.resultado_lote = resultado
        self._configurar_controles_plot(resultado)
        self._atualizar_plot_interativo()
        self._escrever_info(resultado.info_textual)
        titulo_visualizacao = "Visualização das componentes e mapas de abundância"
        if resultado.alerta_convergencia:
            self.lbl_plot.configure(text=f"{titulo_visualizacao} — {resultado.alerta_convergencia}")
            self._atualizar_status(f"concluído com alerta: {resultado.alerta_convergencia}")
        else:
            self.lbl_plot.configure(text=titulo_visualizacao)
            if resultado.pasta_mapas_por_obra:
                self._atualizar_status(f"concluído; mapas salvos em {resultado.pasta_mapas_por_obra}")
            else:
                self._atualizar_status("fatoração em lote concluída com sucesso")
        self.progress.set(1.0)
        self.executando = False
        self.btn_executar.configure(state="normal")
        self.btn_exportar.configure(state="normal")
        if hasattr(self, "btn_zerar"):
            self.btn_zerar.configure(state="normal")

    def _finalizar_erro(self, erro: str):
        self.progress.set(0)
        self.executando = False
        self.btn_executar.configure(state="normal")
        self.btn_exportar.configure(state="normal")
        if hasattr(self, "btn_zerar"):
            self.btn_zerar.configure(state="normal")
        self._atualizar_status("erro no processamento")
        self._escrever_info("Falha no processamento em lote.\n\n" + erro)
        messagebox.showerror("Erro no processamento", erro)

    # --------------------------------------------------------
    # Pipeline principal em lote
    # --------------------------------------------------------
    def _executar_pipeline_lote(self, k_override: int | None = None, progress_callback=None) -> ResultadoLote:
        dataset_path = self.dataset_path.strip() or DEFAULT_DATASET_PATH
        offset_txt = self.entry_offset.get().strip()
        gain_txt = self.entry_gain.get().strip()
        fator_binning = int(self.combo_binning.get())
        fator_binning_espectral = int(self.combo_binning_espectral.get())
        modo_roi = self.combo_modo_roi.get()
        roi_ini = self.entry_roi_ini.get().strip()
        roi_fim = self.entry_roi_fim.get().strip()
        metodo_norm = self.combo_norm.get()
        metodo_fatoracao = self.combo_metodo.get()
        if k_override is None:
            k, _k_fim_ignorado = self._ler_intervalo_k()
        else:
            k = int(k_override)
        progress_callback = progress_callback or self._reportar_progresso_threadsafe
        pixels_amostra_por_obra = int(self.entry_pixels_amostra.get().strip())
        max_iter = int(self.entry_max_iter.get().strip())
        tol = float(self.entry_tol.get().strip())
        orientacao_cubo = normalizar_orientacao_cubo(self.combo_orientacao_cubo.get())
        calcular_kl_completo = bool(self.var_calcular_kl_completo.get())
        padronizar_escala_componentes = bool(self.var_padronizar_escala_componentes.get())
        random_state_usado, modo_semente = self._resolver_random_state_e_modo()
        init_nmf_modo = self.combo_init_nmf.get().strip()
        self._ultima_semente_usada = random_state_usado
        # Versão otimizada: não salva mapas automaticamente durante lote múltiplo.
        # Os mapas são exportados pelo botão de exportação quando há uma única obra em memória.
        salvar_mapas_lote = False

        if k <= 0:
            raise ValueError("O número de componentes (k) deve ser maior que zero.")
        if pixels_amostra_por_obra <= 0:
            raise ValueError("O número de pixels amostrados por obra deve ser maior que zero.")
        if max_iter <= 0:
            raise ValueError("O máximo de iterações deve ser maior que zero.")
        if tol <= 0:
            raise ValueError("A tolerância EPS deve ser maior que zero.")

        # Usa o primeiro arquivo como referência para interpretar a ROI sem carregar o cubo completo.
        dataset_path_ref = resolver_dataset_para_arquivo(self.arquivos_h5[0], dataset_path)
        with h5py.File(self.arquivos_h5[0], "r") as h5f_ref:
            if dataset_path_ref not in h5f_ref:
                raise KeyError(f"O dataset '{dataset_path_ref}' não foi encontrado no primeiro arquivo do lote.")
            n_canais_ref = numero_canais_dataset(h5f_ref[dataset_path_ref], orientacao_cubo)

        calibracao_ativa = bool(self.var_usar_calibracao.get()) or modo_roi == "Energia"
        eixo_calibrado = tentar_construir_eixo_calibrado(n_canais_ref, offset_txt, gain_txt) if calibracao_ativa else None
        origem_eixo = "Canal"
        if eixo_calibrado is not None:
            origem_eixo = f"Calibração linear (offset={offset_txt}, gain={gain_txt})"

        idx_ini, idx_fim_inclusivo, eixo_roi, rotulo_eixo = interpretar_roi(
            modo_roi=modo_roi,
            valor_inicial=roi_ini,
            valor_final=roi_fim,
            n_canais=n_canais_ref,
            eixo_energia=eixo_calibrado if modo_roi == "Energia" else None,
        )
        idx_fim_exclusivo = idx_fim_inclusivo + 1

        # Mesmo quando a ROI é definida em canais, se houver offset/gain válidos,
        # a visualização e a exportação passam a usar o eixo calibrado em energia.
        if calibracao_ativa and eixo_calibrado is not None and modo_roi == "Canal":
            eixo_roi = eixo_calibrado[idx_ini:idx_fim_exclusivo].astype(np.float64)
            rotulo_eixo = "Energia (keV)"

        canais_descartados = len(eixo_roi) % fator_binning_espectral
        eixo_roi = aplicar_binning_eixo(eixo_roi, fator_binning_espectral)
        canal_ar_index = None

        pasta_mapas_por_obra: Path | None = None
        if salvar_mapas_lote:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pasta_base = Path(self.arquivos_h5[0]).resolve().parent
            pasta_mapas_por_obra = pasta_base / f"mapas_por_obra_{metodo_fatoracao}_{k}comp_{timestamp}"
            pasta_mapas_por_obra.mkdir(parents=True, exist_ok=True)

        (
            H_global,
            dict_medias,
            obra_visualizacao_nome,
            mapas_abundancia,
            arquivos_mapas_por_obra,
            manifesto_mapas_por_obra,
            contribuicao_relativa_componentes,
            init_nmf_efetivo,
            alerta_convergencia,
            reconstruction_err_treino,
            n_iter_treino,
            metricas_reconstrucao_obras,
            fatores_escala_h,
        ) = executar_fatoracao_global_lote(
            lista_caminhos_h5=self.arquivos_h5,
            dataset_path=dataset_path,
            fator_binning=fator_binning,
            fator_binning_espectral=fator_binning_espectral,
            roi_inicio=idx_ini,
            roi_fim=idx_fim_exclusivo,
            metodo_norm=metodo_norm,
            canal_ar_index=canal_ar_index,
            metodologia=metodo_fatoracao,
            n_componentes=k,
            max_iter=max_iter,
            eps=tol,
            pixels_amostra_por_obra=pixels_amostra_por_obra,
            progress_callback=progress_callback,
            armazenar_mapa_quando_unico=True,
            salvar_mapas_por_obra=salvar_mapas_lote,
            pasta_mapas_por_obra=pasta_mapas_por_obra,
            prefixo_mapas_por_obra=f"{metodo_fatoracao}_{k}componentes",
            rotulo_eixo_exportacao=rotulo_eixo,
            eixo_espectral_escala=eixo_roi,
            orientacao_cubo=orientacao_cubo,
            random_state=random_state_usado,
            init_nmf_modo=init_nmf_modo,
            calcular_kl_reconstrucao_completa=calcular_kl_completo,
            padronizar_escala_componentes=padronizar_escala_componentes,
        )


        lista_nomes = [Path(c).name for c in self.arquivos_h5]
        preview = "\n".join(f"- {nome}" for nome in lista_nomes[:8])
        if len(lista_nomes) > 8:
            preview += f"\n- ... e mais {len(lista_nomes) - 8} arquivo(s)"

        abundancias_em_percentual = False
        if metodo_eh_svd(metodo_fatoracao):
            rotulo_abundancia = "Score SVD"
        elif padronizar_escala_componentes:
            rotulo_abundancia = "W* compensado por área(H), com área(H*)=1; sem reescala ×100"
        else:
            rotulo_abundancia = "Abundância relativa (W bruto; sem reescala ×100)"
        max_iter_projecao = min(max_iter, 200) if metodo_fatoracao in ("NMF-KL", "NMF-Euclidiana", "MiniBatchNMF-Fast") else "não se aplica"

        linhas_contribuicao = ""
        if contribuicao_relativa_componentes is not None:
            linhas_contribuicao = "Contribuição relativa interna das componentes:\n"
            for i, valor in enumerate(contribuicao_relativa_componentes):
                aviso = "  < 0,5%" if float(valor) < 0.5 else ""
                linhas_contribuicao += f"- H{i:02d}: {float(valor):.4f}%{aviso}\n"
            linhas_contribuicao += (
                "Observação: estes valores são indicadores relativos internos da decomposição, "
                "não porcentagens físico-químicas absolutas.\n"
            )

        info = (
            f"Aplicação: {APP_NAME} v{APP_VERSION}\n"
            f"Arquivos no lote: {len(self.arquivos_h5)}\n"
            f"Dataset do cubo selecionado: {dataset_path}\n"
            f"Dataset usado no primeiro arquivo: {dataset_path_ref}\n"
            f"Orientação espectral do cubo: {orientacao_cubo}\n"
            f"Calibração energética ativa: {'sim' if calibracao_ativa else 'não'}\n"
            f"Origem do eixo espectral: {origem_eixo}\n"
            f"ROI ({modo_roi}): {roi_ini} -> {roi_fim}\n"
            f"Índices efetivos da ROI: {idx_ini} -> {idx_fim_inclusivo}\n"
            f"Binning espacial: {fator_binning}\n"
            f"Binning Canais/Energia: {fator_binning_espectral} (soma após ROI; eixo médio)\n"
            f"Canais finais descartados por bloco incompleto: {canais_descartados}\n"
            f"Normalização: {metodo_norm}\n"
            f"Subtração automática de background: não\n"
            f"Convenção de escala das componentes: "
            f"{'área(H*)=1 no eixo exportado e W*=W×área(H)' if padronizar_escala_componentes else 'escala livre da NMF'}\n"
            f"Escala dos mapas W: {rotulo_abundancia}\n"
            f"Fatoração: {metodo_fatoracao}\n"
            f"k = {k}\n"
            f"Pixels amostrados por obra = {pixels_amostra_por_obra}\n"
            f"max_iter treino global = {max_iter}\n"
            f"max_iter projeção W por obra = {max_iter_projecao}\n"
            f"tol = {tol}\n"
            f"Calcular KL completa da reconstrução = {'sim' if calcular_kl_completo else 'não'}\n"
            f"Modo de semente = {modo_semente}\n"
            f"random_state usado = {random_state_usado}\n"
            f"Inicialização NMF solicitada = {init_nmf_modo}\n"
            f"Inicialização NMF efetiva = {init_nmf_efetivo}\n"
            f"Shape de H_global = {H_global.shape}\n"
            f"Obras consolidadas = {len(dict_medias)}\n"
            f"Mapas locais disponíveis = {'sim' if mapas_abundancia is not None else 'não'}\n"
            f"Salvar mapas por obra durante lote múltiplo = não\n"
            f"Exportação de mapas = disponível somente quando há uma única obra em memória\n"
            f"{linhas_contribuicao}\n"
            f"Arquivos processados:\n{preview}\n"
        )

        return ResultadoLote(
            H_global=H_global,
            dict_medias=dict_medias,
            eixo_roi=eixo_roi,
            rotulo_eixo=rotulo_eixo,
            metodo_fatoracao=metodo_fatoracao,
            info_textual=info,
            nomes_obras=[str(c) for c in self.arquivos_h5],
            obra_visualizacao_nome=obra_visualizacao_nome,
            mapas_abundancia=mapas_abundancia,
            abundancias_em_percentual=abundancias_em_percentual,
            rotulo_abundancia=rotulo_abundancia,
            contribuicao_relativa_componentes=contribuicao_relativa_componentes,
            fator_binning=fator_binning,
            fator_binning_espectral=fator_binning_espectral,
            metodo_norm=metodo_norm,
            n_componentes=k,
            orientacao_cubo=orientacao_cubo,
            modo_semente=modo_semente,
            random_state_usado=random_state_usado,
            init_nmf_modo=init_nmf_modo,
            init_nmf_efetivo=init_nmf_efetivo,
            alerta_convergencia=alerta_convergencia,
            pasta_mapas_por_obra=str(pasta_mapas_por_obra) if pasta_mapas_por_obra else None,
            arquivos_mapas_por_obra=arquivos_mapas_por_obra,
            manifesto_mapas_por_obra=manifesto_mapas_por_obra,
            reconstruction_err_treino=reconstruction_err_treino,
            n_iter_treino=n_iter_treino,
            metricas_reconstrucao_obras=metricas_reconstrucao_obras,
            padronizar_escala_componentes=padronizar_escala_componentes,
            fatores_escala_h=fatores_escala_h,
        )

    # --------------------------------------------------------
    # Plotagem
    # --------------------------------------------------------
    def _atualizar_plot_interativo(self):
        if self.resultado_lote is None:
            return

        resultado = self.resultado_lote
        modo = self.combo_visualizacao.get()
        idx = max(0, min(self.componente_atual_idx, resultado.H_global.shape[0] - 1))
        x = resultado.eixo_roi
        y = resultado.H_global[idx]
        cmap = resolver_colormap(self.combo_cmap.get())
        interp = self.combo_interp.get()
        rot_x_esp = self._rotulo_x_espectral()
        rot_y_esp = self._rotulo_y_espectral()
        rot_x_map = self._rotulo_x_mapa()
        rot_y_map = self._rotulo_y_mapa()
        nome_obra = Path(resultado.obra_visualizacao_nome).name if resultado.obra_visualizacao_nome else "obra"
        texto_contrib = ""
        if resultado.contribuicao_relativa_componentes is not None and idx < len(resultado.contribuicao_relativa_componentes):
            valor_contrib = float(resultado.contribuicao_relativa_componentes[idx])
            texto_contrib = f" | contribuição relativa interna: {valor_contrib:.4f}%"

        escala_y = self.combo_escala_y.get() if hasattr(self, "combo_escala_y") else "Linear"
        usar_log = escala_y.strip().lower().startswith("log")

        def _y_espectral_plot(vetor):
            arr = np.asarray(vetor, dtype=np.float64)
            if usar_log:
                return np.where(arr > 0, arr, np.nan)
            return arr

        def _aplicar_escala_espectral(ax):
            if usar_log:
                ax.set_yscale("log")

        self.fig.clear()
        self._colorbar = None

        def _plotar_mapa(ax):
            if resultado.mapas_abundancia is None:
                ax.text(
                    0.5,
                    0.5,
                    "Mapa de abundância indisponível para este lote.\n"
                    "Os mapas são mantidos automaticamente quando há apenas 1 cubo no processamento.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
                ax.set_title("Mapa de abundância indisponível")
                ax.set_axis_off()
                return None

            mapa = resultado.mapas_abundancia[:, :, idx]
            im = ax.imshow(
                mapa,
                aspect="auto",
                interpolation=interp,
                cmap=cmap,
                origin="lower",
            )
            titulo_mapa = "Mapa de score SVD" if metodo_eh_svd(resultado.metodo_fatoracao) else "Mapa de abundância"
            ax.set_title(f"{titulo_mapa} - Comp. {idx:02d} - {nome_obra}")
            if resultado.alerta_convergencia:
                ax.text(
                    0.5,
                    0.98,
                    resultado.alerta_convergencia,
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
                )
            ax.set_xlabel(rot_x_map)
            ax.set_ylabel(rot_y_map)
            ax.grid(False)
            return im

        if modo == "Componente espectral + mapa":
            ax1 = self.fig.add_subplot(1, 2, 1)
            ax2 = self.fig.add_subplot(1, 2, 2)
            ax1.plot(x, _y_espectral_plot(y), linewidth=1.8)
            titulo_comp = "Componente SVD" if metodo_eh_svd(resultado.metodo_fatoracao) else "Espectro puro - Componente"
            ax1.set_title(f"{titulo_comp} {idx:02d}{texto_contrib}")
            ax1.set_xlabel(rot_x_esp)
            ax1.set_ylabel(rot_y_esp)
            _aplicar_escala_espectral(ax1)
            ax1.grid(True, alpha=0.25)
            im = _plotar_mapa(ax2)
            if im is not None:
                self._colorbar = self.fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                self._colorbar.set_label(resultado.rotulo_abundancia)
            self.fig.subplots_adjust(wspace=0.28)

        elif modo == "Espectro da componente":
            ax = self.fig.add_subplot(111)
            ax.plot(x, _y_espectral_plot(y), linewidth=1.8)
            titulo_comp = "Componente SVD" if metodo_eh_svd(resultado.metodo_fatoracao) else "Espectro puro - Componente"
            ax.set_title(f"{titulo_comp} {idx:02d}{texto_contrib}")
            ax.set_xlabel(rot_x_esp)
            ax.set_ylabel(rot_y_esp)
            _aplicar_escala_espectral(ax)
            ax.grid(True, alpha=0.25)

        elif modo == "Todos os espectros":
            ax = self.fig.add_subplot(111)
            for i in range(resultado.H_global.shape[0]):
                label_i = f"Comp. {i:02d}"
                if resultado.contribuicao_relativa_componentes is not None and i < len(resultado.contribuicao_relativa_componentes):
                    label_i += f" ({float(resultado.contribuicao_relativa_componentes[i]):.2f}%)"
                ax.plot(x, _y_espectral_plot(resultado.H_global[i]), linewidth=1.2, label=label_i)
            ax.set_title(f"Paleta global completa - {resultado.metodo_fatoracao}")
            ax.set_xlabel(rot_x_esp)
            ax.set_ylabel(rot_y_esp)
            _aplicar_escala_espectral(ax)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8, ncol=2)

        elif modo == "Mapa de abundância":
            ax = self.fig.add_subplot(111)
            im = _plotar_mapa(ax)
            if im is not None:
                self._colorbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                self._colorbar.set_label(resultado.rotulo_abundancia)
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Modo de visualização inválido.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # --------------------------------------------------------
    # Exportação
    # --------------------------------------------------------
    def _exportar_resultado_em_pasta_base(self, resultado: ResultadoLote, pasta_base: str | Path) -> dict[str, object]:
        """Exporta um ResultadoLote dentro de uma pasta-base, sem abrir diálogos."""
        if len(resultado.nomes_obras) == 1:
            nome_obra = Path(resultado.nomes_obras[0]).stem
        else:
            nome_obra = f"Lote_{len(resultado.nomes_obras)}obras"

        nome_pasta = nome_pasta_exportacao_obra(
            nome_obra=nome_obra,
            n_componentes=resultado.n_componentes or resultado.H_global.shape[0],
            fator_binning=resultado.fator_binning or 1,
            fator_binning_espectral=resultado.fator_binning_espectral,
            metodo_norm=resultado.metodo_norm or "",
            padronizar_escala_componentes=resultado.padronizar_escala_componentes,
        )
        pasta_saida = Path(pasta_base) / nome_pasta
        pasta_saida.mkdir(parents=True, exist_ok=True)

        prefixo_curto = prefixo_tres_letras_obra(nome_obra)
        prefixo_saida = pasta_saida / prefixo_curto

        caminhos_espectros, caminho_mapa_vet, _pasta_mapas = exportar_espectros_e_mapas_csv(
            caminho_saida_prefixo=str(prefixo_saida),
            H_global=resultado.H_global,
            eixo_roi=resultado.eixo_roi,
            rotulo_eixo=resultado.rotulo_eixo,
            mapas_abundancia=resultado.mapas_abundancia,
            rotulo_x_mapa=self._rotulo_x_mapa(),
            rotulo_y_mapa=self._rotulo_y_mapa(),
            abundancias_em_percentual=resultado.abundancias_em_percentual,
            prefixo_curto_obra=prefixo_curto,
        )

        caminho_h_consolidado = exportar_h_global_consolidado_csv(
            caminho_saida_prefixo=str(prefixo_saida),
            H_global=resultado.H_global,
            eixo_roi=resultado.eixo_roi,
            rotulo_eixo=resultado.rotulo_eixo,
            prefixo_arquivo=prefixo_curto,
        )

        caminho_metricas_reconstrucao = exportar_metricas_reconstrucao_csv(
            pasta_saida=pasta_saida,
            prefixo_curto=prefixo_curto,
            resultado=resultado,
        )

        caminho_fatores_escala = exportar_fatores_escala_componentes_csv(
            pasta_saida=pasta_saida,
            prefixo_curto=prefixo_curto,
            resultado=resultado,
        )

        caminho_manifesto_txt, caminho_manifesto_csv = salvar_manifesto_processamento(
            pasta_saida=pasta_saida,
            resultado=resultado,
            nome_pasta_exportacao=nome_pasta,
            prefixo_curto=prefixo_curto,
            caminhos_espectros=caminhos_espectros,
            caminho_h_consolidado=caminho_h_consolidado,
            caminho_mapa_vet=caminho_mapa_vet,
            caminho_fatores_escala=caminho_fatores_escala,
        )

        return {
            "nome_obra": nome_obra,
            "nome_pasta": nome_pasta,
            "pasta_saida": pasta_saida,
            "prefixo_curto": prefixo_curto,
            "caminhos_espectros": caminhos_espectros,
            "caminho_mapa_vet": caminho_mapa_vet,
            "caminho_h_consolidado": caminho_h_consolidado,
            "caminho_metricas_reconstrucao": caminho_metricas_reconstrucao,
            "caminho_fatores_escala": caminho_fatores_escala,
            "caminho_manifesto_txt": caminho_manifesto_txt,
            "caminho_manifesto_csv": caminho_manifesto_csv,
        }

    def _executar_varredura_k(self, lista_k: list[int], pasta_raiz_varredura: str | None) -> tuple[ResultadoLote, Path, Path]:
        """Executa k inicial -> k final, exportando cada rodada e liberando memória entre elas."""
        if not pasta_raiz_varredura:
            raise ValueError("Para varredura de k, informe uma pasta raiz de exportação.")

        pasta_raiz = Path(pasta_raiz_varredura)
        pasta_raiz.mkdir(parents=True, exist_ok=True)

        total = len(lista_k)
        linhas_resumo: list[dict[str, object]] = []
        resultado_final: ResultadoLote | None = None

        for idx, k_atual in enumerate(lista_k, start=1):
            def progresso_varredura(fracao: float, texto: str, *, _idx=idx, _k=k_atual):
                fracao_global = ((_idx - 1) + max(0.0, min(1.0, float(fracao)))) / max(total, 1)
                self._reportar_progresso_threadsafe(
                    fracao_global,
                    f"varredura k={_k} ({_idx}/{total}) — {texto}",
                )

            progresso_varredura(0.0, "iniciando rodada")
            resultado = self._executar_pipeline_lote(k_override=k_atual, progress_callback=progresso_varredura)

            progresso_varredura(0.985, "exportando CSVs da rodada")
            info_export = self._exportar_resultado_em_pasta_base(resultado, pasta_raiz)

            linha_resumo: dict[str, object] = {
                "k": int(k_atual),
                "pasta_saida": str(info_export["pasta_saida"]),
                "normalizacao": resultado.metodo_norm,
                "binning": int(resultado.fator_binning),
                "binning_espectral": int(resultado.fator_binning_espectral),
                "metodo_fatoracao": resultado.metodo_fatoracao,
                "random_state_usado": resultado.random_state_usado,
                "init_nmf_efetivo": resultado.init_nmf_efetivo,
                "reconstruction_err_treino_amostra": resultado.reconstruction_err_treino if resultado.reconstruction_err_treino is not None else "",
                "n_iter_treino_global": resultado.n_iter_treino if resultado.n_iter_treino is not None else "",
                "alerta_convergencia": resultado.alerta_convergencia or "",
                "h_consolidado_csv": str(info_export["caminho_h_consolidado"]),
                "w_vetorizado_csv": str(info_export["caminho_mapa_vet"] or ""),
                "metricas_reconstrucao_csv": str(info_export["caminho_metricas_reconstrucao"] or ""),
                "escala_componentes_area_h_unitaria": "sim" if resultado.padronizar_escala_componentes else "não",
                "fatores_escala_componentes_csv": str(info_export["caminho_fatores_escala"] or ""),
            }

            if resultado.metricas_reconstrucao_obras:
                # Para lote de uma única obra, estas colunas já alimentam diretamente o gráfico elbow.
                primeira_metrica = resultado.metricas_reconstrucao_obras[0]
                for chave in ("kl_total", "kl_medio_por_pixel", "kl_medio_por_entrada", "erro_relativo_L1"):
                    linha_resumo[chave] = primeira_metrica.get(chave, "")

            linhas_resumo.append(linha_resumo)

            if resultado_final is not None:
                del resultado_final
                gc.collect()
            resultado_final = resultado

            progresso_varredura(1.0, "rodada exportada e memória liberada")

        if resultado_final is None:
            raise RuntimeError("A varredura de k terminou sem produzir resultados.")

        caminho_resumo = pasta_raiz / "resumo_varredura_k.csv"
        df_resumo = pd.DataFrame(linhas_resumo)

        if "kl_total" in df_resumo.columns:
            df_resumo["delta_KL"] = df_resumo["kl_total"].shift(1) - df_resumo["kl_total"]
            df_resumo["ganho_pct"] = 100.0 * df_resumo["delta_KL"] / df_resumo["kl_total"].shift(1)

        df_resumo.to_csv(caminho_resumo, index=False, sep=",", decimal=".")
        return resultado_final, pasta_raiz, caminho_resumo

    def _finalizar_sucesso_varredura(self, resultado: ResultadoLote, pasta_raiz: Path, caminho_resumo: Path, lista_k: list[int]):
        self._finalizar_sucesso(resultado)
        self._atualizar_status(f"varredura k concluída; resultados em {pasta_raiz}")
        self._escrever_info(
            "Varredura automática de k concluída.\n\n"
            f"Intervalo executado: k = {min(lista_k)} até {max(lista_k)}\n"
            f"Pasta raiz de exportação: {pasta_raiz}\n"
            f"Resumo da varredura: {caminho_resumo}\n\n"
            "A visualização atual corresponde à última rodada executada.\n\n"
            "=== Última rodada ===\n"
            f"{resultado.info_textual}"
        )
        messagebox.showinfo(
            "Varredura de k concluída",
            "Todas as rodadas foram executadas e exportadas.\n\n"
            f"Pasta raiz:\n{pasta_raiz}\n\n"
            f"Resumo:\n{caminho_resumo}",
        )

    def exportar_resultados(self):
        if self.resultado_lote is None:
            messagebox.showwarning("Sem resultados", "Execute a fatoração antes de exportar.")
            return

        pasta_base = filedialog.askdirectory(
            title="Escolha a pasta onde será criada a pasta de exportação"
        )
        if not pasta_base:
            return

        try:
            info_export = self._exportar_resultado_em_pasta_base(self.resultado_lote, pasta_base)

            caminhos_espectros = info_export["caminhos_espectros"]
            caminho_mapa_vet = info_export["caminho_mapa_vet"]
            caminho_h_consolidado = info_export["caminho_h_consolidado"]
            caminho_metricas_reconstrucao = info_export["caminho_metricas_reconstrucao"]
            caminho_fatores_escala = info_export["caminho_fatores_escala"]
            caminho_manifesto_txt = info_export["caminho_manifesto_txt"]
            caminho_manifesto_csv = info_export["caminho_manifesto_csv"]
            pasta_saida = info_export["pasta_saida"]

            lista_espectros = "\n".join(str(c) for c in caminhos_espectros[:12])
            if len(caminhos_espectros) > 12:
                lista_espectros += f"\n... e mais {len(caminhos_espectros) - 12} arquivo(s)"

            msg = (
                "Os CSVs foram salvos com sucesso.\n\n"
                f"Pasta de exportação:\n{pasta_saida}\n\n"
                "Espectros H individuais:\n"
                f"{lista_espectros}"
            )
            msg += f"\n\nMatriz H consolidada:\n{caminho_h_consolidado}"

            if caminho_mapa_vet is not None:
                msg += f"\n\nMapa W vetorizado:\n{caminho_mapa_vet}"
            else:
                msg += "\n\nMapa W não exportado porque há mais de uma obra no lote ou o mapa não está em memória."

            if caminho_metricas_reconstrucao is not None:
                msg += f"\n\nMétricas de reconstrução KL:\n{caminho_metricas_reconstrucao}"

            if caminho_fatores_escala is not None:
                msg += f"\n\nFatores de escala das componentes:\n{caminho_fatores_escala}"

            msg += (
                f"\n\nManifesto metodológico:\n{caminho_manifesto_txt}"
                f"\n{caminho_manifesto_csv}"
            )

            messagebox.showinfo("Exportação concluída", msg)
        except Exception as exc:
            caminho = self._registrar_erro(exc, "Erro durante a exportação dos arquivos CSV.")
            messagebox.showerror("Erro ao exportar", f"Falha ao salvar os arquivos.\n\nRelatório salvo em:\n{caminho}")

# ============================================================
# Execução principal com captura de erro fatal
# ============================================================

def executar_app():
    app = AppMAXRF()
    app.mainloop()


if __name__ == "__main__":
    try:
        executar_app()
    except Exception as exc:  # pragma: no cover
        caminho = gerar_relatorio_erro(exc, "Erro fatal na inicialização da aplicação.")
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Erro fatal",
                "A aplicação encontrou um erro fatal.\n\n"
                f"Relatório salvo em:\n{caminho}",
            )
            root.destroy()
        except Exception:
            print("Erro fatal ao iniciar a aplicação.")
            print(f"Relatório salvo em: {caminho}")
