#!/usr/bin/env python3
"""
Costurador de Dados XRF — Aplicação Desktop (vBeta)
=========================================================
Costura inteligente de mapas MA‑XRF em HDF5.

Principais recursos:
- Registro automático para estimar sobreposição e deslocamentos (dx/dy).
- Normalização robusta na região de overlap (fator por mediana/quantis).
- Overlap gravado uma única vez: mantém apenas o mapa selecionado na sobreposição.
- Recorte "lossy" (opcional/automático) para remover bordas vazias do deslocamento:
    * Costura vertical  → recorte em X (mantém altura).
    * Costura horizontal → recorte em Y (mantém largura).
- Saída HDF5 chunked + compressão GZIP (nível configurável) + shuffle.

Tipo de dado de saída (OUTPUT_DTYPE_MODE):
- "auto": float32 quando houver normalização (scale != 1) ou quando a entrada já for float;
          caso contrário, uint32 (compacto/rápido).
- "float32": sempre float32 (preserva valores fracionários).
- "uint32": sempre uint32 (arredonda após escala; pode perder frações).

Compatibilidade: PyMCA (/cube, /x_axis, /y_axis, /energy).

Dependências:
    pip install PyQt5 h5py numpy scipy

Uso:
    python xrf_stitcher.py
"""

import sys
import os

# Evita erros de lock do HDF5 no Windows (ex.: WinError 33 / unable to lock file)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
from scipy.signal import fftconvolve
from scipy.ndimage import shift as ndi_shift

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QGroupBox,
    QProgressBar, QTextEdit, QTabWidget, QFrame, QRadioButton,
    QButtonGroup, QScrollArea, QSizePolicy, QMessageBox,
    QStackedWidget, QSplitter, QLineEdit, QCheckBox, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE ESTILO
# ─────────────────────────────────────────────────────────────────────────────

STYLE = """
QMainWindow, QWidget {
    background-color: #f0f4f8;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    color: #1a2a3a;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d0dce8;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: bold;
    font-size: 13px;
    color: #1a3a5c;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #1a3a5c;
}
QPushButton {
    background-color: #1a3a5c;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #2e5c8a;
}
QPushButton:pressed {
    background-color: #0f2540;
}
QPushButton:disabled {
    background-color: #b0bec5;
    color: #eceff1;
}
QPushButton#secondary {
    background-color: #ffffff;
    color: #1a3a5c;
    border: 1.5px solid #1a3a5c;
}
QPushButton#secondary:hover {
    background-color: #e8f0f8;
}
QPushButton#success {
    background-color: #1b7a4e;
}
QPushButton#success:hover {
    background-color: #25a36a;
}
QPushButton#danger {
    background-color: #c0392b;
}
QLabel#title {
    font-size: 22px;
    font-weight: 700;
    color: #1a3a5c;
}
QLabel#subtitle {
    font-size: 13px;
    color: #5a7a9a;
}
QLabel#section {
    font-size: 14px;
    font-weight: 600;
    color: #1a3a5c;
}
QLabel#metric_value {
    font-size: 20px;
    font-weight: 700;
    color: #2e7d9e;
}
QLabel#metric_label {
    font-size: 11px;
    color: #7a9ab0;
}
QLabel#file_path {
    background-color: #eef4fb;
    border: 1px solid #c0d8f0;
    border-radius: 5px;
    padding: 6px 10px;
    color: #1a3a5c;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
QProgressBar {
    border: none;
    border-radius: 5px;
    background-color: #dce8f5;
    height: 10px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2e7d9e;
    border-radius: 5px;
}
QTextEdit {
    background-color: #0d1b2a;
    color: #a8d8ea;
    border: 1px solid #2a4a6a;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}
QComboBox {
    background-color: #ffffff;
    border: 1.5px solid #c0d8f0;
    border-radius: 5px;
    padding: 6px 10px;
    min-width: 160px;
}
QComboBox:focus {
    border-color: #2e7d9e;
}
QRadioButton {
    font-size: 13px;
    color: #1a2a3a;
    spacing: 8px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
}
QFrame#card {
    background-color: #ffffff;
    border: 1px solid #d0dce8;
    border-radius: 8px;
}
QFrame#separator {
    background-color: #d0dce8;
    max-height: 1px;
}
QTabWidget::pane {
    border: 1px solid #d0dce8;
    border-radius: 6px;
    background: white;
}
QTabBar::tab {
    background: #e8f0f8;
    color: #5a7a9a;
    padding: 8px 18px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: #1a3a5c;
    color: white;
}
QLabel#stepPillBase {
    border-radius: 14px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 600;
    background: #eaf1f8;
    color: #5a7a9a;
    border: 1px solid #d0dce8;
}
QLabel#stepPillDone {
    border-radius: 14px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 700;
    background: #e8f5e9;
    color: #1b7a4e;
    border: 1px solid #bfe2ca;
}
QLabel#stepPillActive {
    border-radius: 14px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 700;
    background: #1a3a5c;
    color: white;
    border: 1px solid #1a3a5c;
}
QFrame#statusCard {
    background-color: #ffffff;
    border: 1px solid #d0dce8;
    border-radius: 10px;
}
QLabel#statusIcon {
    font-size: 22px;
    font-weight: 700;
    color: #1a3a5c;
    min-width: 28px;
}
QLabel#statusTitle {
    font-size: 13px;
    font-weight: 700;
    color: #1a3a5c;
}
QLabel#statusHint {
    font-size: 12px;
    color: #5a7a9a;
}
QLabel#softHint {
    font-size: 12px;
    color: #6f8aa3;
}
"""

# ── CONFIGURAÇÕES DE PERFORMANCE ──
# Chunks dinâmicos baseados no tamanho do arquivo
CHUNK_ROWS = 32  # Aumentado de 8 para 32 (melhor performance para arquivos grandes)
CHUNK_ROWS_ROI = 64  # chunks maiores para mapas de ROI (registro/normalização)
CHUNK_COLS = 512  # Colunas por chunk
CHUNK_ENERGY = 256  # Canais de energia por chunk

# Compressão adaptativa: GZIP nível 2 para velocidade (ainda comprime bem)
# Para máxima compressão, alterar para 4 (mais lento)
COMPRESSION_LEVEL = 4  # Nível 4: ~10x compressão (máxima)
# Tipo do dataset de saída (importante para normalização):
#   - "auto": usa float32 quando houver normalização (scale != 1) ou quando a entrada já for float;
#             caso contrário, usa uint32 (compacto/rápido).
#   - "float32": sempre float32 (preserva escala; arquivos tendem a ficar maiores)
#   - "uint32": sempre uint32 (arredonda após escala; perde parte fracionária)
OUTPUT_DTYPE_MODE = "uint32"

# Se o recorte "lossy" remover mais do que esse percentual, emite um aviso no log.
LOSSY_CROP_WARN_THRESHOLD = 0.10  # 10%


# ── ROIs PARA REGISTRO GUIADO POR ELEMENTOS ──
ELEMENT_ROIS_KEV = {
    # Leves / médios (linhas Kα)
    "Ca": (3.55, 3.85),  # Ca Kα ~ 3.69 keV
    "Ti": (4.35, 4.70),  # Ti Kα ~ 4.51 keV
    "Cr": (5.25, 5.55),  # Cr Kα ~ 5.41 keV
    "Mn": (5.75, 6.05),  # Mn Kα ~ 5.90 keV
    "Fe": (6.25, 6.55),  # Fe Kα ~ 6.40 keV
    "Co": (6.78, 7.08),  # Co Kα ~ 6.93 keV
    "Ni": (7.32, 7.64),  # Ni Kα ~ 7.48 keV
    "Zn": (8.45, 8.80),  # Zn Kα ~ 8.64 keV

    # Pesados (linhas L; com 40 kV normalmente são as mais relevantes)
    # Ba Lα (~4.47) se sobrepõe a Ti Kα; por isso usamos Ba Lβ (~4.83) para ficar mais discriminante.
    "Ba": (4.72, 4.95),  # Ba Lβ ~ 4.83 keV
    "Pb": (10.35, 10.75),  # Pb Lα ~ 10.55 keV
}



# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA DE PROCESSAMENTO (thread separada)
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    analysis_done = pyqtSignal(dict)
    stitch_done = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, mode, map1_path, map2_path, direction,
                 selected_map=None, output_hdf5=None, output_report=None,
                 map1_name=None, map2_name=None,
                 reg_method="total", norm_method="total", elements_selected=None,
                 overlap_search="global", edge_start=10, edge_max=300, edge_step=1, edge_dx_max=300):
        super().__init__()
        self.mode = mode
        self.map1_path = map1_path
        self.map2_path = map2_path
        self.direction = direction
        self.selected_map = selected_map
        self.output_hdf5 = output_hdf5
        self.output_report = output_report
        self.map1_name = map1_name or Path(map1_path).name
        self.map2_name = map2_name or Path(map2_path).name
        self.reg_method = reg_method
        self.norm_method = norm_method
        self.elements_selected = list(elements_selected) if elements_selected else []
        self.overlap_search = overlap_search or "global"
        self.edge_start = int(edge_start)
        self.edge_max = int(edge_max)
        self.edge_step = int(edge_step)
        self.edge_dx_max = int(edge_dx_max)

    def run(self):
        try:
            if self.mode == "analyze":
                self._analyze()
            else:
                self._stitch()
        except Exception as e:
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")

    # ── Análise ──────────────────────────────────────────────────────────────

    def _analyze(self):
        self.progress.emit(5, "Lendo dimensões dos arquivos...")
        self.log.emit(f"[INFO] Mapa 1: {self.map1_name}")
        self.log.emit(f"[INFO] Mapa 2: {self.map2_name}")

        shape1 = self._cube_shape(self.map1_path)
        shape2 = self._cube_shape(self.map2_path)
        self.log.emit(f"[INFO] Shape mapa 1: {shape1[0]}×{shape1[1]}×{shape1[2]}")
        self.log.emit(f"[INFO] Shape mapa 2: {shape2[0]}×{shape2[1]}×{shape2[2]}")

        if self.reg_method == "elements":
            self.progress.emit(15, "Construindo imagem de registro (ROIs selecionadas)...")
            self.log.emit("[INFO] Imagem de registro: Elementos (seleção manual)")
            self.log.emit("[INFO] Elementos marcados: " + (", ".join(self.elements_selected) if self.elements_selected else "Fe, Ca"))
        else:
            self.progress.emit(15, "Construindo imagem de registro (soma Z)...")
            self.log.emit("[INFO] Imagem de registro: Total (soma sobre canais Z)")

        # ── Overlap: método global vs. por borda ────────────────────────────
        if self.overlap_search == "edge" and self.direction == "vertical":
            self.progress.emit(30, "Construindo faixas de borda para registro...")
            map1_edge_strip, map2_edge_strip, reginfo = self._registration_vertical_strips(
                self.map1_path, self.map2_path, self.reg_method,
                elements=self.elements_selected, max_lines=self.edge_max
            )
            self.progress.emit(45, "Detectando sobreposição por varredura (borda)...")
            overlap_info = self._detect_overlap_edge_vertical(
                map1_edge_strip, map2_edge_strip, ny2_total=shape2[0],
                start_lines=self.edge_start,
                max_lines=reginfo.get("max_lines", self.edge_max),
                step=self.edge_step,
                dx_max=self.edge_dx_max,
            )
            self.log.emit(f"[INFO] Sobreposição detectada (borda): {overlap_info['overlap_pixels']} linhas")
            self.log.emit(f"[INFO] dx detectado (borda): {overlap_info.get('shift_dx', 0)} px | score={overlap_info['best_correlation']:.4f}")
        elif self.overlap_search == "edge" and self.direction == "horizontal":
            self.progress.emit(30, "Construindo faixas de borda horizontais...")
            left_strip, right_strip, reginfo = self._registration_horizontal_strips(
                self.map1_path, self.map2_path, self.reg_method,
                elements=self.elements_selected, max_cols=self.edge_max
            )
            self.progress.emit(45, "Detectando sobreposicao por varredura (borda H)...")
            overlap_info = self._detect_overlap_edge_horizontal(
                left_strip, right_strip, nx1_total=shape1[1],
                start_cols=self.edge_start,
                max_cols=reginfo.get("max_cols", self.edge_max),
                step=self.edge_step,
                dy_max=self.edge_dx_max,
            )
            self.log.emit(f"[INFO] Sobreposicao detectada (borda H): {overlap_info['overlap_pixels']} colunas")
            self.log.emit(f"[INFO] dy detectado (borda H): {overlap_info.get('shift_dy', 0)} px | score={overlap_info['best_correlation']:.4f}")
        else:
            img1, img2, reginfo = self._registration_pair(self.map1_path, self.map2_path, self.reg_method, elements=self.elements_selected)

            self.progress.emit(40, "Detectando sobreposição por correlação cruzada...")
            self.log.emit("[INFO] Detectando sobreposição por correlação cruzada...")
            overlap_info = self._detect_overlap(img1, img2, self.direction)
            self.log.emit(f"[INFO] Sobreposição detectada: {overlap_info['overlap_pixels']} pixels")
            self.log.emit(f"[INFO] Correlação de imagem: {overlap_info['best_correlation']:.4f}")

        self.progress.emit(65, "Analisando qualidade espectral...")
        self.log.emit("[INFO] Analisando qualidade na região de sobreposição...")
        quality = self._analyze_quality(overlap_info)
        self.log.emit(f"[INFO] SNR mapa 1: {quality['map1_snr']:.4f}")
        self.log.emit(f"[INFO] SNR mapa 2: {quality['map2_snr']:.4f}")
        self.log.emit(f"[INFO] Correlação espectral: {quality.get('spectral_correlation', 'N/A')}")
        self.log.emit(f"[SUGESTÃO] Manter: {'Mapa 1' if quality['suggested_map']=='map1' else 'Mapa 2'}")
        self.log.emit(f"[SUGESTÃO] Motivo: {quality['suggestion_reason']}")

        self.progress.emit(100, "Análise concluída.")
        self.analysis_done.emit({
            "cube1_shape": list(shape1),
            "cube2_shape": list(shape2),
            "overlap": overlap_info,
            "quality": quality,
            "registration": reginfo,
            "search": {
                "mode": self.overlap_search,
                "edge_start": self.edge_start,
                "edge_max": self.edge_max,
                "edge_step": self.edge_step,
                "edge_dx_max": self.edge_dx_max,
            },
        })

    # ── Costura ───────────────────────────────────────────────────────────────

    def _stitch(self):
        self.progress.emit(5, "Iniciando costura...")
        self.log.emit(f"[INFO] Mapa selecionado para sobreposição: {'Mapa 1' if self.selected_map=='map1' else 'Mapa 2'}")

        shape1 = self._cube_shape(self.map1_path)
        shape2 = self._cube_shape(self.map2_path)

        if self.overlap_search == "edge" and self.direction == "vertical":
            map1_edge_strip, map2_edge_strip, reginfo = self._registration_vertical_strips(
                self.map1_path, self.map2_path, self.reg_method,
                elements=self.elements_selected, max_lines=self.edge_max
            )
            overlap_info = self._detect_overlap_edge_vertical(
                map1_edge_strip, map2_edge_strip, ny2_total=shape2[0],
                start_lines=self.edge_start,
                max_lines=reginfo.get("max_lines", self.edge_max),
                step=self.edge_step,
                dx_max=self.edge_dx_max,
            )
        elif self.overlap_search == "edge" and self.direction == "horizontal":
            left_strip, right_strip, reginfo = self._registration_horizontal_strips(
                self.map1_path, self.map2_path, self.reg_method,
                elements=self.elements_selected, max_cols=self.edge_max
            )
            overlap_info = self._detect_overlap_edge_horizontal(
                left_strip, right_strip, nx1_total=shape1[1],
                start_cols=self.edge_start,
                max_cols=reginfo.get("max_cols", self.edge_max),
                step=self.edge_step,
                dy_max=self.edge_dx_max,
            )
        else:
            img1, img2, reginfo = self._registration_pair(self.map1_path, self.map2_path, self.reg_method, elements=self.elements_selected)
            overlap_info = self._detect_overlap(img1, img2, self.direction)

        overlap = overlap_info["overlap_pixels"]

        self.progress.emit(18, "Preparando normalização e costura em chunks...")
        self.log.emit(f"[INFO] Sobreposição: {overlap} pixels — iniciando costura em chunks de {CHUNK_ROWS} linhas...")

        stitched_shape, scale_info = self._stitch_and_save(overlap_info)

        self.progress.emit(88, "Gerando relatório científico...")
        self.log.emit("[INFO] Gerando relatório HTML...")
        quality = self._analyze_quality(overlap_info)
        self._generate_report(shape1, shape2, overlap_info, quality, stitched_shape, scale_info)

        self.progress.emit(100, "Processamento concluído!")
        self.log.emit(f"[OK] HDF5 salvo em: {self.output_hdf5}")
        self.log.emit(f"[OK] Relatório salvo em: {self.output_report}")
        self.stitch_done.emit({
            "stitched_shape": list(stitched_shape),
            "output_hdf5": self.output_hdf5,
            "output_report": self.output_report,
        })

    # ── Helpers de processamento ──────────────────────────────────────────────

    def _cube_shape(self, path):
        with h5py.File(path, "r") as f:
            return tuple(f["cube"].shape)

    def _integrated_image(self, path):
        with h5py.File(path, "r") as f:
            ds = f["cube"]
            ny, nx, nz = ds.shape
            img = np.zeros((ny, nx), dtype=np.float64)
            for r0 in range(0, ny, CHUNK_ROWS):
                r1 = min(r0 + CHUNK_ROWS, ny)
                img[r0:r1, :] = ds[r0:r1, :, :].astype(np.float64).sum(axis=2)
        return img


    def _find_energy_axis(self, f, nz):
        """Tenta localizar um eixo de energia (keV) no HDF5.
        Retorna np.ndarray (nz,) ou None.

        Heurística:
          - procura datasets comuns (energy, energy_axis, etc.)
          - se não achar, procura qualquer dataset 1D de tamanho nz com valores crescentes e max < 80.
        """
        # candidatos diretos (raiz)
        direct = ["energy", "energies", "energy_axis", "energy_keV", "keV", "E"]
        for k in direct:
            if k in f and isinstance(f[k], h5py.Dataset):
                arr = np.array(f[k][...]).astype(np.float64, copy=False).ravel()
                if arr.size == nz and np.isfinite(arr).all() and arr.max() < 80:
                    return arr

        # busca leve por datasets 1D com tamanho nz
        found = None

        def visitor(name, obj):
            nonlocal found
            if found is not None:
                return
            if isinstance(obj, h5py.Dataset):
                if obj.ndim == 1 and obj.shape[0] == nz and np.issubdtype(obj.dtype, np.number):
                    try:
                        arr = np.array(obj[...]).astype(np.float64, copy=False).ravel()
                        if np.isfinite(arr).all() and arr.max() < 80:
                            # preferir nomes que contenham "energy"
                            if "energy" in name.lower() or "kev" in name.lower():
                                found = arr
                            else:
                                # guarda como fallback, mas continua procurando algo mais explícito
                                found = arr
                    except Exception:
                        pass

        try:
            f.visititems(visitor)
        except Exception:
            return None

        # valida monotonicidade (não precisa ser perfeita)
        if found is not None:
            if (np.diff(found) >= -1e-6).mean() > 0.9:
                return found
        return None

    @staticmethod
    def _texture_score(img2d):
        """Score simples de 'textura' (estrutura espacial) para registro.
        Maior = mais bordas/variação útil para correlação.
        """
        a = img2d.astype(np.float64, copy=False)
        if a.size == 0:
            return 0.0
        # gradientes (evita custos de filtros)
        gy = np.diff(a, axis=0)
        gx = np.diff(a, axis=1)
        return float(np.nanvar(gy) + np.nanvar(gx))

    def _roi_map(self, path, e_lo, e_hi):
        """Mapa 2D integrando o cubo em uma janela de energia [e_lo, e_hi] em keV.
        Retorna np.ndarray (Y,X) float64 ou None se não houver eixo de energia em keV.
        """
        with h5py.File(path, "r") as f:
            ds = f["cube"]
            ny, nx, nz = ds.shape
            E = self._find_energy_axis(f, nz)
            if E is None:
                return None

            idx = np.where((E >= e_lo) & (E <= e_hi))[0]
            if idx.size == 0:
                # tenta janela um pouco mais larga
                pad = 0.15
                idx = np.where((E >= e_lo - pad) & (E <= e_hi + pad))[0]
            if idx.size == 0:
                return None

            i0 = int(idx.min())
            i1 = int(idx.max()) + 1

            out = np.zeros((ny, nx), dtype=np.float64)
            for r0 in range(0, ny, CHUNK_ROWS_ROI):
                r1 = min(r0 + CHUNK_ROWS_ROI, ny)
                block = ds[r0:r1, :, i0:i1].astype(np.float64)
                out[r0:r1, :] = block.sum(axis=2)
            return out


# ── Registro por borda (faixas) ─────────────────────────────────────────

    def _integrated_y_strip(self, path, y0, y1):
        """Soma em Z apenas para um intervalo de linhas [y0,y1) -> (y1-y0, X)"""
        with h5py.File(path, "r") as f:
            ds = f["cube"]
            ny, nx, _ = ds.shape
            y0 = max(0, int(y0)); y1 = min(ny, int(y1))
            h = max(0, y1 - y0)
            out = np.zeros((h, nx), dtype=np.float64)
            for r0 in range(y0, y1, CHUNK_ROWS_ROI):
                r1 = min(r0 + CHUNK_ROWS_ROI, y1)
                blk = ds[r0:r1, :, :].astype(np.float64)
                out[r0 - y0:r1 - y0, :] = blk.sum(axis=2)
            return out

    def _roi_map_y_strip(self, path, y0, y1, e_lo, e_hi):
        """Mapa ROI (keV) apenas para linhas [y0,y1) -> (y1-y0, X). Retorna None se não houver eixo keV."""
        with h5py.File(path, "r") as f:
            ds = f["cube"]
            ny, nx, nz = ds.shape
            y0 = max(0, int(y0)); y1 = min(ny, int(y1))
            h = max(0, y1 - y0)
            if h == 0:
                return np.zeros((0, nx), dtype=np.float64)

            E = self._find_energy_axis(f, nz)
            if E is None:
                return None

            idx = np.where((E >= e_lo) & (E <= e_hi))[0]
            if idx.size == 0:
                pad = 0.15
                idx = np.where((E >= e_lo - pad) & (E <= e_hi + pad))[0]
            if idx.size == 0:
                return None

            i0 = int(idx.min())
            i1 = int(idx.max()) + 1

            out = np.zeros((h, nx), dtype=np.float64)
            for r0 in range(y0, y1, CHUNK_ROWS_ROI):
                r1 = min(r0 + CHUNK_ROWS_ROI, y1)
                blk = ds[r0:r1, :, i0:i1].astype(np.float64)
                out[r0 - y0:r1 - y0, :] = blk.sum(axis=2)
            return out

    def _registration_vertical_strips(self, map1_path, map2_path, method, elements=None, max_lines=300):
        """Constrói faixas (strips) usadas no registro vertical por borda.

        Observação de manutenção:
          - Os nomes internos foram mantidos neutros (map1/map2), sem assumir "topo" ou "base".
          - A geometria original da costura NÃO foi alterada nesta revisão.
          - Para preservar o comportamento já validado, esta rotina continua usando a faixa
            inicial do Mapa 1 e a faixa final do Mapa 2 como regiões candidatas de overlap.

        Retorna:
          map1_edge_strip: (m, X), faixa extraída do Mapa 1
          map2_edge_strip: (m, X), faixa extraída do Mapa 2
          info: dict
        """
        ny1, _, _ = self._cube_shape(map1_path)
        ny2, _, _ = self._cube_shape(map2_path)
        m = int(max_lines)
        m = max(5, min(m, ny1, ny2))

        y1a, y1b = 0, m
        y2a, y2b = ny2 - m, ny2

        if method != "elements":
            a = self._integrated_y_strip(map1_path, y1a, y1b)
            b = self._integrated_y_strip(map2_path,  y2a, y2b)
            return a, b, {"method": "total_edge", "elements": [], "max_lines": m}

        candidates = list(elements) if elements else ["Fe", "Ca"]
        roi1, roi2, used = {}, {}, []

        for el in candidates:
            if el not in ELEMENT_ROIS_KEV:
                continue
            lo, hi = ELEMENT_ROIS_KEV[el]
            a = self._roi_map_y_strip(map1_path, y1a, y1b, lo, hi)
            b = self._roi_map_y_strip(map2_path,  y2a, y2b, lo, hi)
            if a is None or b is None:
                continue
            roi1[el] = a
            roi2[el] = b
            used.append(el)

        if not used:
            a = self._integrated_y_strip(map1_path, y1a, y1b)
            b = self._integrated_y_strip(map2_path,  y2a, y2b)
            return a, b, {"method": "total_edge_fallback", "elements": [], "max_lines": m}

        def zscore(img):
            s = img.std()
            return (img - img.mean()) / (s + 1e-12) if s > 1e-10 else (img - img.mean())

        tex = {el: self._texture_score(roi1[el]) + self._texture_score(roi2[el]) for el in used}
        wsum = sum(tex.values()) + 1e-12
        weights = {el: tex[el] / wsum for el in used}

        img1 = np.zeros_like(next(iter(roi1.values())), dtype=np.float64)
        img2 = np.zeros_like(next(iter(roi2.values())), dtype=np.float64)
        for el in used:
            ww = weights[el]
            img1 += ww * zscore(roi1[el])
            img2 += ww * zscore(roi2[el])

        desc = [f"{el}(w={weights[el]:.2f})" for el in used]
        return img1, img2, {"method": "elements_edge", "elements": desc, "max_lines": m}

    def _best_dx_from_profiles(self, A, B, dx_max):
        a = A.sum(axis=0).astype(np.float64)
        b = B.sum(axis=0).astype(np.float64)
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        corr = fftconvolve(a, b[::-1], mode="full")
        lags = np.arange(corr.size) - (b.size - 1)
        if dx_max and dx_max > 0:
            mask = (lags >= -dx_max) & (lags <= dx_max)
            corr = corr[mask]
            lags = lags[mask]
        return int(lags[int(np.argmax(corr))])

    def _score_strips_ncc(self, A, B, dx):
        h = min(A.shape[0], B.shape[0])
        w = min(A.shape[1], B.shape[1])
        if h < 5 or w < 20:
            return -np.inf
        A = A[:h, :w]
        B = B[:h, :w]
        dx = int(dx)

        if dx >= 0:
            if dx >= w - 5:
                return -np.inf
            A2 = A[:, dx:]
            B2 = B[:, :w - dx]
        else:
            if -dx >= w - 5:
                return -np.inf
            A2 = A[:, :w + dx]
            B2 = B[:, -dx:]

        A2 = (A2 - A2.mean()) / (A2.std() + 1e-12)
        B2 = (B2 - B2.mean()) / (B2.std() + 1e-12)
        return float(np.mean(A2 * B2))

    def _detect_overlap_edge_vertical(self, map1_edge_strip, map2_edge_strip, ny2_total,
                                     start_lines=10, max_lines=300, step=1, dx_max=300):
        m = min(int(max_lines), map1_edge_strip.shape[0], map2_edge_strip.shape[0])
        start = max(5, int(start_lines))
        step = max(1, int(step))
        dx_max = max(0, int(dx_max))

        if start > m:
            start = max(5, m // 2)

        best = {"score": -np.inf, "overlap": start, "dx": 0, "raw_score": -np.inf}

        w = min(map1_edge_strip.shape[1], map2_edge_strip.shape[1])
        map1_edge_strip = map1_edge_strip[:m, :w]
        map2_edge_strip = map2_edge_strip[:m, :w]

        for o in range(start, m + 1, step):
            A = map1_edge_strip[:o, :]
            B = map2_edge_strip[m - o:m, :]

            dx0 = self._best_dx_from_profiles(A, B, dx_max)
            candidates = [dx0 - 2, dx0 - 1, dx0, dx0 + 1, dx0 + 2]
            best_local = (-np.inf, dx0)
            for dxx in candidates:
                if dx_max and abs(dxx) > dx_max:
                    continue
                sc = self._score_strips_ncc(A, B, dxx)
                if sc > best_local[0]:
                    best_local = (sc, dxx)

            score = float(best_local[0])
            dx = int(best_local[1])
            score_adj = score + 0.004 * np.log1p(o)

            if score_adj > best["score"]:
                best = {"score": score_adj, "overlap": o, "dx": dx, "raw_score": score}

        overlap = int(best["overlap"])
        dx = int(best["dx"])
        offset = - (int(ny2_total) - overlap)

        self.log.emit(f"[INFO] Registro por borda: overlap={overlap} linhas | dx={dx} px | score={best['raw_score']:.4f}")
        return {
            "overlap_pixels": overlap,
            "best_correlation": float(best["raw_score"]),
            "offset": int(offset),
            "shift_dx": int(dx),
            "shift_dy": 0,
            "method": "edge_strips",
            "edge_start": int(start_lines),
            "edge_max": int(max_lines),
            "edge_step": int(step),
            "edge_dx_max": int(dx_max),
        }

    def _robust_scale_overlap_strips(self, map1_edge_strip, map2_edge_strip, overlap_info):
        overlap = int(overlap_info["overlap_pixels"])
        dx = int(overlap_info.get("shift_dx", 0))
        if overlap <= 0:
            return 1.0

        m = min(map1_edge_strip.shape[0], map2_edge_strip.shape[0])
        o = min(overlap, m)
        w = min(map1_edge_strip.shape[1], map2_edge_strip.shape[1])
        A = map1_edge_strip[:o, :w].astype(np.float64, copy=False)
        B = map2_edge_strip[m - o:m, :w].astype(np.float64, copy=False)

        if dx >= 0:
            if dx >= w - 5:
                return 1.0
            A2 = A[:, dx:]
            B2 = B[:, :w - dx]
        else:
            if -dx >= w - 5:
                return 1.0
            A2 = A[:, :w + dx]
            B2 = B[:, -dx:]

        eps = 1e-12
        pos = (A2 > 0) & (B2 > 0)
        if pos.sum() < 200:
            return float(np.median(A2[A2 > 0]) / (np.median(B2[B2 > 0]) + eps))

        pA10, pA90 = np.percentile(A2[pos], [10, 90])
        pB10, pB90 = np.percentile(B2[pos], [10, 90])
        mask = pos & (A2 >= pA10) & (A2 <= pA90) & (B2 >= pB10) & (B2 <= pB90)
        if mask.sum() < 200:
            mask = pos
        return float(np.median(A2[mask]) / (np.median(B2[mask]) + eps))

    # ── Equivalentes horizontais dos métodos de borda ───────────────────────────────────────

    def _integrated_x_strip(self, path, x0, x1):
        """Imagem integrada (soma Z) de uma faixa de colunas [x0:x1]."""
        with h5py.File(path, "r") as f:
            ds = f["cube"]
            ny, nx, nz = ds.shape
            x0 = max(0, x0); x1 = min(nx, x1)
            out = np.zeros((ny, x1 - x0), dtype=np.float64)
            for r0 in range(0, ny, CHUNK_ROWS_ROI):
                r1 = min(r0 + CHUNK_ROWS_ROI, ny)
                out[r0:r1, :] = ds[r0:r1, x0:x1, :].astype(np.float64).sum(axis=2)
        return out

    def _roi_map_x_strip(self, path, x0, x1, lo_kev, hi_kev):
        """Mapa ROI (keV) apenas para colunas [x0:x1] -> (Y, x1-x0). Retorna None se não houver eixo keV."""
        with h5py.File(path, "r") as f:
            ds = f["cube"]
            ny, nx, nz = ds.shape
            x0 = max(0, int(x0)); x1 = min(nx, int(x1))
            w = max(0, x1 - x0)
            if w == 0:
                return np.zeros((ny, 0), dtype=np.float64)

            E = self._find_energy_axis(f, nz)
            if E is None:
                return None

            idx = np.where((E >= lo_kev) & (E <= hi_kev))[0]
            if idx.size == 0:
                pad = 0.15
                idx = np.where((E >= lo_kev - pad) & (E <= hi_kev + pad))[0]
            if idx.size == 0:
                return None

            i0 = int(idx.min())
            i1 = int(idx.max()) + 1

            out = np.zeros((ny, w), dtype=np.float64)
            for r0 in range(0, ny, CHUNK_ROWS_ROI):
                r1 = min(r0 + CHUNK_ROWS_ROI, ny)
                blk = ds[r0:r1, x0:x1, i0:i1].astype(np.float64)
                out[r0:r1, :] = blk.sum(axis=2)
            return out

    def _registration_horizontal_strips(self, left_path, right_path, method,
                                        elements=None, max_cols=300):
        """Constrói faixas (strips) usadas no registro horizontal por borda.

        Convenção do app para costura horizontal:
          - Mapa 1 = ESQUERDA -> usamos as ÚLTIMAS colunas (borda direita do mapa1)
          - Mapa 2 = DIREITA  -> usamos as PRIMEIRAS colunas (borda esquerda do mapa2)

        Retorna:
          left_strip : (Y, m) a partir da borda direita do mapa1
          right_strip: (Y, m) a partir da borda esquerda do mapa2
          info: dict
        """
        _, nx1, _ = self._cube_shape(left_path)
        _, nx2, _ = self._cube_shape(right_path)
        m = int(max_cols)
        m = max(5, min(m, nx1, nx2))

        x1a, x1b = nx1 - m, nx1
        x2a, x2b = 0, m

        if method != "elements":
            a = self._integrated_x_strip(left_path, x1a, x1b)
            b = self._integrated_x_strip(right_path, x2a, x2b)
            return a, b, {"method": "total_edge_h", "elements": [], "max_cols": m}

        candidates = list(elements) if elements else ["Fe", "Ca"]
        roi1, roi2, used = {}, {}, []

        for el in candidates:
            if el not in ELEMENT_ROIS_KEV:
                continue
            lo, hi = ELEMENT_ROIS_KEV[el]
            a = self._roi_map_x_strip(left_path, x1a, x1b, lo, hi)
            b = self._roi_map_x_strip(right_path, x2a, x2b, lo, hi)
            if a is None or b is None:
                continue
            roi1[el] = a
            roi2[el] = b
            used.append(el)

        if not used:
            a = self._integrated_x_strip(left_path, x1a, x1b)
            b = self._integrated_x_strip(right_path, x2a, x2b)
            return a, b, {"method": "total_edge_h_fallback", "elements": [], "max_cols": m}

        def zscore(img):
            s = img.std()
            return (img - img.mean()) / (s + 1e-12) if s > 1e-10 else (img - img.mean())

        tex = {el: self._texture_score(roi1[el]) + self._texture_score(roi2[el]) for el in used}
        wsum = sum(tex.values()) + 1e-12
        weights = {el: tex[el] / wsum for el in used}

        img1 = np.zeros_like(next(iter(roi1.values())), dtype=np.float64)
        img2 = np.zeros_like(next(iter(roi2.values())), dtype=np.float64)
        for el in used:
            ww = weights[el]
            img1 += ww * zscore(roi1[el])
            img2 += ww * zscore(roi2[el])

        desc = [f"{el}(w={weights[el]:.2f})" for el in used]
        return img1, img2, {"method": "elements_edge_h", "elements": desc, "max_cols": m}

    def _best_dy_from_profiles(self, A, B, dy_max):
        """Correlacao em Y para estimar shift vertical residual (costura horizontal)."""
        a = A.sum(axis=1).astype(np.float64)
        b = B.sum(axis=1).astype(np.float64)
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        corr = fftconvolve(a, b[::-1], mode="full")
        lags = np.arange(corr.size) - (b.size - 1)
        if dy_max and dy_max > 0:
            mask = (lags >= -dy_max) & (lags <= dy_max)
            corr = corr[mask]; lags = lags[mask]
        return int(lags[int(np.argmax(corr))])

    def _score_strips_ncc_horizontal(self, A, B, dy):
        """NCC entre strips com shift em Y (costura horizontal)."""
        h = min(A.shape[0], B.shape[0])
        w = min(A.shape[1], B.shape[1])
        if h < 20 or w < 5:
            return -np.inf
        A = A[:h, :w]; B = B[:h, :w]
        dy = int(dy)
        if dy >= 0:
            if dy >= h - 5:
                return -np.inf
            A2 = A[dy:, :]; B2 = B[:h - dy, :]
        else:
            if -dy >= h - 5:
                return -np.inf
            A2 = A[:h + dy, :]; B2 = B[-dy:, :]
        A2 = (A2 - A2.mean()) / (A2.std() + 1e-12)
        B2 = (B2 - B2.mean()) / (B2.std() + 1e-12)
        return float(np.mean(A2 * B2))

    def _detect_overlap_edge_horizontal(self, left_strip, right_strip, nx1_total,
                                        start_cols=10, max_cols=300, step=1, dy_max=300):
        """Equivalente horizontal da detecção vertical por borda.

        Convenção fixa:
          - Mapa 1 = esquerda
          - Mapa 2 = direita

        Portanto, o offset principal do mapa2 é calculado no sistema do mapa1 como:
            offset = largura_mapa1 - overlap
        """
        m = min(int(max_cols), left_strip.shape[1], right_strip.shape[1])
        start = max(5, int(start_cols))
        step = max(1, int(step))
        dy_max = max(0, int(dy_max))

        if start > m:
            start = max(5, m // 2)

        best = {"score": -np.inf, "overlap": start, "dy": 0, "raw_score": -np.inf}
        h = min(left_strip.shape[0], right_strip.shape[0])
        left_strip = left_strip[:h, :m]
        right_strip = right_strip[:h, :m]

        for o in range(start, m + 1, step):
            A = left_strip[:, m - o:]
            B = right_strip[:, :o]

            dy0 = self._best_dy_from_profiles(A, B, dy_max)
            candidates = [dy0 - 2, dy0 - 1, dy0, dy0 + 1, dy0 + 2]
            best_local = (-np.inf, dy0)
            for dyy in candidates:
                if dy_max and abs(dyy) > dy_max:
                    continue
                sc = self._score_strips_ncc_horizontal(A, B, dyy)
                if sc > best_local[0]:
                    best_local = (sc, dyy)

            score = float(best_local[0])
            dy = int(best_local[1])
            score_adj = score + 0.004 * np.log1p(o)
            if score_adj > best["score"]:
                best = {"score": score_adj, "overlap": o, "dy": dy, "raw_score": score}

        overlap = int(best["overlap"])
        dy = int(best["dy"])
        offset = int(nx1_total) - overlap

        self.log.emit(
            f"[INFO] Registro por borda (H): overlap={overlap} cols | dy={dy} px | offset x={offset} px | score={best['raw_score']:.4f}"
        )
        return {
            "overlap_pixels": overlap,
            "best_correlation": float(best["raw_score"]),
            "offset": int(offset),
            "shift_dy": int(dy),
            "shift_dx": 0,
            "method": "edge_strips_h",
            "edge_start": int(start_cols),
            "edge_max": int(max_cols),
            "edge_step": int(step),
            "edge_dx_max": int(dy_max),
        }

    def _robust_scale_overlap_strips_horizontal(self, left_strip, right_strip, overlap_info):
        """Equivalente horizontal de _robust_scale_overlap_strips."""
        overlap = int(overlap_info["overlap_pixels"])
        dy = int(overlap_info.get("shift_dy", 0))
        if overlap <= 0:
            return 1.0
        h = min(left_strip.shape[0], right_strip.shape[0])
        o = min(overlap, left_strip.shape[1], right_strip.shape[1])
        A = left_strip[:h, left_strip.shape[1] - o:].astype(np.float64, copy=False)
        B = right_strip[:h, :o].astype(np.float64, copy=False)
        if dy >= 0:
            if dy >= h - 5:
                return 1.0
            A2 = A[dy:, :]; B2 = B[:h - dy, :]
        else:
            if -dy >= h - 5:
                return 1.0
            A2 = A[:h + dy, :]; B2 = B[-dy:, :]
        eps = 1e-12
        pos = (A2 > 0) & (B2 > 0)
        if pos.sum() < 200:
            return float(np.median(A2[A2 > 0]) / (np.median(B2[B2 > 0]) + eps))
        pA10, pA90 = np.percentile(A2[pos], [10, 90])
        pB10, pB90 = np.percentile(B2[pos], [10, 90])
        mask = pos & (A2 >= pA10) & (A2 <= pA90) & (B2 >= pB10) & (B2 <= pB90)
        if mask.sum() < 200:
            mask = pos
        return float(np.median(A2[mask]) / (np.median(B2[mask]) + eps))


    def _registration_pair(self, path1, path2, method, elements=None):
        """Gera as imagens 2D usadas para registrar (detectar overlap + dy/dx).

        method:
          - 'total'    : soma em Z (rápido)
          - 'elements' : combinação automática de ROIs (Fe+Ca) se houver eixo de energia (keV)

        Retorna (img1, img2, info_dict).
        """
        if method != "elements":
            return self._integrated_image(path1), self._integrated_image(path2), {"method": "total", "elements": []}

        # tentar construir mapas de ROI para elementos candidatos
        candidates = list(elements) if elements else ["Fe", "Ca"]
        roi_maps_1 = {}
        roi_maps_2 = {}
        used = []

        for el in candidates:
            if el not in ELEMENT_ROIS_KEV:
                continue
            lo, hi = ELEMENT_ROIS_KEV[el]
            m1 = self._roi_map(path1, lo, hi)
            m2 = self._roi_map(path2, lo, hi)
            if m1 is None or m2 is None:
                continue

            tex = min(self._texture_score(m1), self._texture_score(m2))
            # intensidade robusta (se o elemento quase não aparece, não ajuda)
            med1 = float(np.median(m1[m1 > 0])) if np.any(m1 > 0) else 0.0
            med2 = float(np.median(m2[m2 > 0])) if np.any(m2 > 0) else 0.0
            inten = min(med1, med2)

            # peso: textura × presença do elemento (log para não explodir)
            w = tex * np.log1p(inten)

            if w <= 0 or not np.isfinite(w):
                continue

            roi_maps_1[el] = m1
            roi_maps_2[el] = m2
            used.append((el, w, tex, inten))

        if not used:
            self.log.emit("[WARN] Registro por elementos solicitado, mas não foi possível construir ROIs (sem eixo keV?). Usando soma total em Z.")
            return self._integrated_image(path1), self._integrated_image(path2), {"method": "total_fallback", "elements": []}

        # normalizar pesos
        used.sort(key=lambda t: t[1], reverse=True)
        # usar apenas os elementos mais informativos (evita somar mapas 'chapados')
        max_e = min(4, len(used))
        used = used[:max_e]
        wsum = sum(u[1] for u in used) + 1e-12

        def zscore(img):
            img = img.astype(np.float64, copy=False)
            s = img.std()
            return (img - img.mean()) / (s + 1e-12)

        img1 = np.zeros_like(next(iter(roi_maps_1.values())), dtype=np.float64)
        img2 = np.zeros_like(next(iter(roi_maps_2.values())), dtype=np.float64)

        desc = []
        for el, w, tex, inten in used:
            a = zscore(roi_maps_1[el])
            b = zscore(roi_maps_2[el])
            ww = w / wsum
            img1 += ww * a
            img2 += ww * b
            desc.append({"element": el, "weight": float(ww), "texture": float(tex), "presence": float(inten)})

        self.log.emit("[INFO] Registro guiado por elementos — usados:")
        for d in desc:
            self.log.emit(f"       - {d['element']}: peso={d['weight']:.2f} | textura={d['texture']:.2e} | presença~{d['presence']:.2f}")

        return img1, img2, {"method": "elements", "elements": desc}

    @staticmethod
    def _xcorr2_shift(a, b):
        """Estima o shift (dy, dx) para alinhar b com a via correlação cruzada 2D (FFT).
        Retorna (dy, dx, peak_corr).
        """
        a0 = (a - a.mean()) / (a.std() + 1e-12)
        b0 = (b - b.mean()) / (b.std() + 1e-12)
        corr = fftconvolve(a0, b0[::-1, ::-1], mode="full")
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        dy = int(y - (b.shape[0] - 1))
        dx = int(x - (b.shape[1] - 1))
        return dy, dx, float(corr[y, x])

    def _detect_overlap(self, img1, img2, direction):
        """Detecta sobreposição entre img1 e img2 por correlação.

        Convenção:
          offset = posição do INÍCIO do mapa2 no sistema de coordenadas do mapa1.
          vertical: offset em Y (pode ser <0 ou >0)
          horizontal: offset em X (pode ser <0 ou >0)

        Após encontrar o offset principal (1D), estima um ajuste 2D residual (shift_dy, shift_dx)
        na região de overlap via correlação cruzada 2D (FFT). O sinal do shift é validado
        escolhendo o que maximiza a correlação após aplicar o deslocamento.
        """
        def zscore(img):
            std = img.std()
            return (img - img.mean()) / std if std > 1e-10 else img - img.mean()

        A = zscore(img1)
        B = zscore(img2)

        size1 = A.shape[0] if direction == "vertical" else A.shape[1]
        size2 = B.shape[0] if direction == "vertical" else B.shape[1]

        # Agora permitimos offsets positivos e negativos:
        #   offset < 0: mapa2 começa acima/à esquerda de mapa1
        #   offset > 0: mapa2 começa abaixo/à direita de mapa1
        best = {"corr": -np.inf, "offset": 0, "overlap": 0, "sl1": None, "sl2": None}

        if direction == "vertical":
            offset_iter = range(-size2 + 1, size1)  # [-size2+1, size1-1]
        else:
            # Convenção horizontal fixa do app: Mapa 1 = esquerda, Mapa 2 = direita.
            # Assim, o início do mapa2 no sistema do mapa1 deve ficar à direita (offset >= 0).
            offset_iter = range(0, size1)

        for offset in offset_iter:
            if direction == "vertical":
                y1a = max(0, offset)
                y1b = min(size1, offset + size2)
                ov = y1b - y1a
                if ov < 8:
                    continue
                y2a = max(0, -offset)
                y2b = y2a + ov
                sl1 = A[y1a:y1b, :]
                sl2 = B[y2a:y2b, :]
            else:
                x1a = max(0, offset)
                x1b = min(size1, offset + size2)
                ov = x1b - x1a
                if ov < 8:
                    continue
                x2a = max(0, -offset)
                x2b = x2a + ov
                sl1 = A[:, x1a:x1b]
                sl2 = B[:, x2a:x2b]

            min_r = min(sl1.shape[0], sl2.shape[0])
            min_c = min(sl1.shape[1], sl2.shape[1])
            sl1c = sl1[:min_r, :min_c]
            sl2c = sl2[:min_r, :min_c]
            corr = float(np.mean(sl1c * sl2c))

            if corr > best["corr"]:
                best = {"corr": corr, "offset": offset, "overlap": ov, "sl1": sl1c, "sl2": sl2c}

        # Shift 2D residual via FFT no overlap (downsample p/ velocidade)
        shift_dy, shift_dx, corr2d = 0, 0, 0.0
        if best["sl1"] is not None and best["sl2"] is not None:
            s1d = best["sl1"][::2, ::2]
            s2d = best["sl2"][::2, ::2]
            if s1d.shape[0] >= 6 and s1d.shape[1] >= 6:
                dy2, dx2, corr2d = self._xcorr2_shift(s1d, s2d)
                # Converter para pixels reais
                cand = (dy2 * 2, dx2 * 2)

                # Validar sinal do shift: escolhe (dy,dx) ou (-dy,-dx) que dá maior correlação
                def score(dy, dx):
                    s2s = ndi_shift(best["sl2"].astype(np.float64),
                                   shift=(dy, dx), order=0,
                                   mode="constant", cval=0.0)
                    return float(np.mean(best["sl1"] * s2s))

                s_pos = score(cand[0], cand[1])
                s_neg = score(-cand[0], -cand[1])
                if s_neg > s_pos:
                    shift_dy, shift_dx = -cand[0], -cand[1]
                else:
                    shift_dy, shift_dx = cand[0], cand[1]

        where = ""
        if direction == "vertical":
            where = "ACIMA" if best["offset"] < 0 else ("ABAIXO" if best["offset"] > 0 else "ALINHADO")
        else:
            where = "À ESQUERDA" if best["offset"] < 0 else ("À DIREITA" if best["offset"] > 0 else "ALINHADO")

        self.log.emit(f"[INFO] Offset principal: {best['offset']} px ({where}) | Overlap real: {best['overlap']} px")
        self.log.emit(f"[INFO] Shift 2D residual: dy={shift_dy}, dx={shift_dx} (corr2D={corr2d:.3f})")

        return {
            "overlap_pixels": int(best["overlap"]),
            "best_correlation": float(best["corr"]),
            "offset": int(best["offset"]),
            "shift_dy": int(shift_dy),
            "shift_dx": int(shift_dx),
            "best_correlation_2d": float(corr2d),
        }

    def _analyze_quality(self, overlap_info):
        overlap = overlap_info["overlap_pixels"]

        def snr(path):
            with h5py.File(path, "r") as f:
                ds = f["cube"]
                ny = ds.shape[0]
                s, sq, n = 0.0, 0.0, 0
                for r0 in range(0, ny, CHUNK_ROWS):
                    r1 = min(r0 + CHUNK_ROWS, ny)
                    c = ds[r0:r1].astype(np.float64)
                    s += c.sum(); sq += (c**2).sum(); n += c.size
            mean = s / n
            std = np.sqrt(sq / n - mean**2)
            return float(mean), float(std), float(mean / std if std > 1e-10 else 0)

        mean1, std1, snr1 = snr(self.map1_path)
        mean2, std2, snr2 = snr(self.map2_path)

        spectral_corr = None
        intensity_ratio = None

        if overlap > 0:
            with h5py.File(self.map1_path, "r") as f1, h5py.File(self.map2_path, "r") as f2:
                ds1, ds2 = f1["cube"], f2["cube"]
                ny1, ny2 = ds1.shape[0], ds2.shape[0]
                spec1 = np.zeros(ds1.shape[2], dtype=np.float64)
                spec2 = np.zeros(ds2.shape[2], dtype=np.float64)
                n1 = n2 = 0
                if self.direction == "vertical":
                    for r0 in range(0, overlap, CHUNK_ROWS):
                        r1 = min(r0 + CHUNK_ROWS, overlap)
                        c = ds1[r0:r1, :, :].astype(np.float64)
                        spec1 += c.sum(axis=(0, 1)); n1 += c.shape[0] * c.shape[1]
                    for r0 in range(ny2 - overlap, ny2, CHUNK_ROWS):
                        r1 = min(r0 + CHUNK_ROWS, ny2)
                        c = ds2[r0:r1, :, :].astype(np.float64)
                        spec2 += c.sum(axis=(0, 1)); n2 += c.shape[0] * c.shape[1]
                else:
                    nx1, nx2 = ds1.shape[1], ds2.shape[1]
                    for r0 in range(0, ds1.shape[0], CHUNK_ROWS):
                        r1 = min(r0 + CHUNK_ROWS, ds1.shape[0])
                        c = ds1[r0:r1, :overlap, :].astype(np.float64)
                        spec1 += c.sum(axis=(0, 1)); n1 += c.shape[0] * c.shape[1]
                    for r0 in range(0, ds2.shape[0], CHUNK_ROWS):
                        r1 = min(r0 + CHUNK_ROWS, ds2.shape[0])
                        c = ds2[r0:r1, nx2-overlap:, :].astype(np.float64)
                        spec2 += c.sum(axis=(0, 1)); n2 += c.shape[0] * c.shape[1]
            spec1 /= max(n1, 1); spec2 /= max(n2, 1)
            spectral_corr = float(np.corrcoef(spec1 / (spec1.max() + 1e-10), spec2 / (spec2.max() + 1e-10))[0, 1])
            intensity_ratio = float(np.mean(spec2) / (np.mean(spec1) + 1e-10))

        suggested = "map1" if snr1 >= snr2 else "map2"
        reason = (
            f"Mapa 1 apresenta maior razão sinal-ruído (SNR={snr1:.3f}) vs Mapa 2 (SNR={snr2:.3f})"
            if snr1 >= snr2 else
            f"Mapa 2 apresenta maior razão sinal-ruído (SNR={snr2:.3f}) vs Mapa 1 (SNR={snr1:.3f})"
        )
        return {
            "overlap_pixels": int(overlap),
            "spectral_correlation": spectral_corr,
            "intensity_ratio": intensity_ratio,
            "map1_snr": snr1, "map2_snr": snr2,
            "map1_mean_intensity": mean1, "map2_mean_intensity": mean2,
            "map1_std": std1, "map2_std": std2,
            "suggested_map": suggested,
            "suggestion_reason": reason,
            "no_overlap": overlap <= 0,
        }

    def _robust_scale_overlap(self, img1, img2, overlap_info):
        """Retorna fator para reescalar o Mapa2 na escala do Mapa1 (map2_to_map1),
        usando a região de overlap já alinhada (offset + shift 2D).
        """
        overlap = int(overlap_info["overlap_pixels"])
        offset  = int(overlap_info["offset"])
        shift_dy = int(overlap_info.get("shift_dy", 0))
        shift_dx = int(overlap_info.get("shift_dx", 0))
        direction = self.direction

        if overlap <= 0:
            self.log.emit("[WARN] overlap <= 0 — fator de normalização = 1.0")
            return 1.0

        # Recortes do overlap (sem assumir offset <= 0)
        if direction == "vertical":
            y1a = max(0, offset)
            y1b = y1a + overlap
            y2a = max(0, -offset)
            y2b = y2a + overlap
            A = img1[y1a:y1b, :].astype(np.float64, copy=False)
            B = img2[y2a:y2b, :].astype(np.float64, copy=False)
        else:
            x1a = max(0, offset)
            x1b = x1a + overlap
            x2a = max(0, -offset)
            x2b = x2a + overlap
            A = img1[:, x1a:x1b].astype(np.float64, copy=False)
            B = img2[:, x2a:x2b].astype(np.float64, copy=False)

        # Aplicar shift 2D residual ao B (sem mudar A)
        if shift_dy != 0 or shift_dx != 0:
            B = ndi_shift(B, shift=(shift_dy, shift_dx), order=1, mode="constant", cval=0.0)

        # Máscara robusta: remove zeros e outliers (p10–p90)
        eps = 1e-12
        pos = (A > 0) & (B > 0)
        if pos.sum() < 200:
            self.log.emit("[WARN] Overlap insuficiente para máscara robusta — usando mediana simples (A/B)")
            return float(np.median(A[A > 0]) / (np.median(B[B > 0]) + eps))

        pA10, pA90 = np.percentile(A[pos], [10, 90])
        pB10, pB90 = np.percentile(B[pos], [10, 90])
        mask = pos & (A >= pA10) & (A <= pA90) & (B >= pB10) & (B <= pB90)
        if mask.sum() < 200:
            mask = pos

        factor = float(np.median(A[mask]) / (np.median(B[mask]) + eps))  # map2_to_map1
        self.log.emit(f"[INFO] Fator map2→map1 (mediana overlap alinhado): {factor:.4f} (mask={mask.sum()})")
        return factor


    def _robust_scale_overlap_cube(self, overlap_info, max_samples=200_000, stride=2):
        """Estimativa robusta do fator multiplicativo usando o OVERLAP REAL do cubo (não só strips/imagens).

        Retorna fator f tal que: map2_scaled = map2 * f  (para casar com map1) no overlap.
        Estratégia (rápida e robusta):
          - usa a região de interseção (overlap) em coordenadas do mosaico
          - amostra pixels em grade (stride) para reduzir custo
          - usa soma espectral (total counts) para evitar viés por elemento
          - calcula razões por pixel e usa mediana (com corte por quantis)
        """
        overlap = int(overlap_info.get("overlap_pixels", 0) or 0)
        if overlap <= 0:
            return 1.0

        shift_dx = int(overlap_info.get("shift_dx", 0) or 0)
        shift_dy = int(overlap_info.get("shift_dy", 0) or 0)
        offset = int(overlap_info.get("offset", 0) or 0)

        with h5py.File(self.map1_path, "r") as f1, h5py.File(self.map2_path, "r") as f2:
            ds1, ds2 = f1["cube"], f2["cube"]
            ny1, nx1, nz1 = ds1.shape
            ny2, nx2, nz2 = ds2.shape
            if nz1 != nz2:
                return 1.0

            # posições no mosaico (mesma convenção da costura)
            x1, y1 = 0, 0
            if self.direction == "vertical":
                y2, x2 = offset, 0
            else:
                x2, y2 = offset, 0
            x2 += shift_dx
            y2 += shift_dy

            # bbox do mosaico (para coordenadas >=0)
            y_min = min(y1, y2); x_min = min(x1, x2)
            y1_off, x1_off = y1 - y_min, x1 - x_min
            y2_off, x2_off = y2 - y_min, x2 - x_min

            rect1 = (y1_off, y1_off + ny1, x1_off, x1_off + nx1)
            rect2 = (y2_off, y2_off + ny2, x2_off, x2_off + nx2)

            # interseção (overlap) em coords do mosaico
            oy0 = max(rect1[0], rect2[0])
            oy1 = min(rect1[1], rect2[1])
            ox0 = max(rect1[2], rect2[2])
            ox1 = min(rect1[3], rect2[3])

            if oy1 <= oy0 or ox1 <= ox0:
                return 1.0

            # mapeia para coords dos datasets
            y1a0 = oy0 - y1_off; y1a1 = oy1 - y1_off
            x1a0 = ox0 - x1_off; x1a1 = ox1 - x1_off
            y2a0 = oy0 - y2_off; y2a1 = oy1 - y2_off
            x2a0 = ox0 - x2_off; x2a1 = ox1 - x2_off

            # amostragem em grade
            ys = np.arange(0, y1a1 - y1a0, stride, dtype=np.int32)
            xs = np.arange(0, x1a1 - x1a0, stride, dtype=np.int32)
            if ys.size == 0 or xs.size == 0:
                return 1.0

            # limita total de amostras
            grid = ys.size * xs.size
            if grid > max_samples:
                # aumenta stride efetivo
                step = int(np.ceil(np.sqrt(grid / max_samples)))
                ys = ys[::step]
                xs = xs[::step]

            ratios = []
            # lê por blocos de linhas para manter RAM baixa
            for yy in ys:
                y1i = int(y1a0 + yy)
                y2i = int(y2a0 + yy)
                # lê uma linha completa do overlap (em X) e depois subamostra em xs
                a = ds1[y1i, x1a0:x1a1, :].astype(np.float64).sum(axis=1)
                b = ds2[y2i, x2a0:x2a1, :].astype(np.float64).sum(axis=1)
                a = a[xs]; b = b[xs]
                mask = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
                if np.any(mask):
                    ratios.append(a[mask] / b[mask])

            if not ratios:
                return 1.0

            r = np.concatenate(ratios).astype(np.float64, copy=False)
            r = r[np.isfinite(r)]
            if r.size < 50:
                return float(np.median(r)) if r.size else 1.0

            # corta extremos (evita hotspots e ruído)
            q1, q99 = np.quantile(r, [0.01, 0.99])
            r = r[(r >= q1) & (r <= q99)]
            if r.size == 0:
                return 1.0

            f = float(np.median(r))
            if not np.isfinite(f) or f <= 0:
                return 1.0
            return f


    def _compute_gain_profile_vertical(self, img1, img2, overlap_info, smooth_sigma=7.0):
        """Calcula perfil de ganho g(x) para corrigir degrau de intensidade na costura vertical.
        
        Retorna um vetor 1D com tamanho = largura de img2.
        g(x) = razão de intensidade por coluna na região de overlap.
        """
        overlap = int(overlap_info.get("overlap_pixels", 0))
        offset = int(overlap_info.get("offset", 0))
        shift_dx = int(overlap_info.get("shift_dx", 0))
        
        nx2 = img2.shape[1]
        g = np.ones(nx2, dtype=np.float32)
        
        if overlap <= 0:
            return g
        
        # Extrair região de overlap
        y1a = max(0, offset)
        y1b = min(img1.shape[0], y1a + overlap)
        y2a = max(0, -offset)
        y2b = min(img2.shape[0], y2a + overlap)
        
        if y1b - y1a < 5 or y2b - y2a < 5:
            return g
        
        A = img1[y1a:y1b, :].astype(np.float64)
        B = img2[y2a:y2b, :].astype(np.float64)
        
        # Alinhar em X via shift_dx
        w = min(A.shape[1], B.shape[1])
        if w < 10:
            return g
        
        A = A[:, :w]
        B = B[:, :w]
        
        if shift_dx >= 0:
            if shift_dx >= w - 5:
                return g
            A2 = A[:, shift_dx:]
            B2 = B[:, :w - shift_dx]
            x_start = 0
        else:
            if -shift_dx >= w - 5:
                return g
            A2 = A[:, :w + shift_dx]
            B2 = B[:, -shift_dx:]
            x_start = -shift_dx
        
        # Calcular ganho por coluna
        h, ww = A2.shape
        if ww < 10 or h < 5:
            return g
        
        gx = np.full(ww, np.nan, dtype=np.float64)
        eps = 1e-12
        
        for j in range(ww):
            a = A2[:, j]
            b = B2[:, j]
            
            # Pixels válidos (ambos > 0)
            mask = (a > 0) & (b > 0)
            if mask.sum() < 20:
                continue
            
            # Razão a/b
            r = a[mask] / (b[mask] + eps)
            
            # Remove outliers (p10-p90)
            p10, p90 = np.percentile(r, [10, 90])
            m = (r >= p10) & (r <= p90)
            if m.sum() < 10:
                m = np.ones_like(r, dtype=bool)
            
            r_clean = r[m]
            
            # Ganho ponderado por SNR (a+b)
            weights = (a[mask][m] + b[mask][m]).astype(np.float64)
            w_sum = weights.sum()
            
            if w_sum > 0:
                gx[j] = float((weights * r_clean).sum() / w_sum)
            else:
                gx[j] = float(np.median(r_clean))
        
        # Interpolar NaNs
        valid = np.isfinite(gx) & (gx > 0)
        if valid.sum() < max(10, ww // 10):
            # Fallback: ganho escalar
            all_valid = gx[valid]
            if len(all_valid) > 0:
                g[:] = np.float32(np.median(all_valid))
            return g
        
        # Interpolar valores inválidos
        idx = np.arange(ww)
        gx_interp = gx.copy()
        gx_interp[~valid] = np.interp(idx[~valid], idx[valid], gx[valid])
        
        # Suavizar
        try:
            from scipy.ndimage import gaussian_filter1d
            gx_smooth = gaussian_filter1d(gx_interp.astype(np.float64), sigma=float(smooth_sigma), mode='nearest')
        except:
            gx_smooth = gx_interp
        
        # Limitar ganhos extremos
        gx_smooth = np.clip(gx_smooth, 0.2, 5.0)
        
        # Colocar no vetor de saída
        x0 = int(x_start)
        x1 = min(nx2, x0 + ww)
        g[x0:x1] = gx_smooth[:(x1 - x0)].astype(np.float32)
        
        # Preencher resto com ganho global
        global_gain = float(np.median(g[x0:x1]))
        g[:x0] = np.float32(global_gain)
        g[x1:] = np.float32(global_gain)
        
        return g


    def _stitch_and_save(self, overlap_info):
        """Costura os dois mapas em um único HDF5 usando mosaico 2D (bounding box),
        suportando offset positivo/negativo e shift residual (dy, dx).

        Regras:
          - A posição do mapa2 relativa ao mapa1 é dada por:
              vertical:   y2 = offset, x2 = 0
              horizontal: x2 = offset, y2 = 0
            e somamos o shift 2D residual: (y2 += shift_dy, x2 += shift_dx)

          - Normalização:
              fator_base = map2_to_map1 (para colocar map2 na escala do map1)
              se selected_map == "map1": escala map2 por fator_base (map1 intacto)
              se selected_map == "map2": escala map1 por 1/fator_base (map2 intacto)

          - Sobreposição:
              escrevemos primeiro o mapa NÃO selecionado e depois o selecionado,
              assim o selecionado "vence" em pixels que se sobrepõem (sem blending).
        """
        overlap  = int(overlap_info["overlap_pixels"])
        offset   = int(overlap_info["offset"])
        shift_dx = int(overlap_info.get("shift_dx", 0))
        shift_dy = int(overlap_info.get("shift_dy", 0))

        self.progress.emit(22, "Calculando fator de normalização (overlap alinhado)...")

        if self.overlap_search == "edge" and self.direction == "vertical":
            # Estimar fator usando apenas strips (rápido e consistente com o registro por borda)
            if self.norm_method == "elements":
                self.log.emit("[INFO] Normalização: overlap por elementos (seleção manual) — strips")
                map1_n, map2_n, _ = self._registration_vertical_strips(
                    self.map1_path, self.map2_path, "elements",
                    elements=self.elements_selected, max_lines=max(self.edge_max, overlap)
                )
            else:
                self.log.emit("[INFO] Normalização: total (soma em Z) — strips")
                map1_n, map2_n, _ = self._registration_vertical_strips(
                    self.map1_path, self.map2_path, "total",
                    elements=self.elements_selected, max_lines=max(self.edge_max, overlap)
                )
            factor_map2_to_map1 = self._robust_scale_overlap_strips(map1_n, map2_n, overlap_info)
            self.log.emit(f"[INFO] Fator map2→map1 (strips): {factor_map2_to_map1:.4f}")
        elif self.overlap_search == "edge" and self.direction == "horizontal":
            if self.norm_method == "elements":
                self.log.emit("[INFO] Normalizacao: overlap por elementos - strips H")
                left_n, right_n, _ = self._registration_horizontal_strips(
                    self.map1_path, self.map2_path, "elements",
                    elements=self.elements_selected, max_cols=max(self.edge_max, overlap)
                )
            else:
                self.log.emit("[INFO] Normalizacao: total (soma em Z) - strips H")
                left_n, right_n, _ = self._registration_horizontal_strips(
                    self.map1_path, self.map2_path, "total",
                    elements=self.elements_selected, max_cols=max(self.edge_max, overlap)
                )
            factor_map2_to_map1 = self._robust_scale_overlap_strips_horizontal(left_n, right_n, overlap_info)
            self.log.emit(f"[INFO] Fator map2->map1 (strips H): {factor_map2_to_map1:.4f}")
        else:
            # Imagem usada para estimar o fator de escala (modo global)
            if self.norm_method == "elements":
                self.log.emit("[INFO] Normalização: overlap por elementos (seleção manual)")
                img1n, img2n, _ = self._registration_pair(self.map1_path, self.map2_path, "elements", elements=self.elements_selected)
            else:
                self.log.emit("[INFO] Normalização: total (soma em Z)")
                img1n = self._integrated_image(self.map1_path)
                img2n = self._integrated_image(self.map2_path)
            factor_map2_to_map1 = self._robust_scale_overlap(img1n, img2n, overlap_info)


        # ── Refinamento do fator usando o overlap REAL do cubo ───────────────────
        # (mais fiel para corrigir o "degrau" quando houve variação de distância/fluxo entre as aquisições)
        try:
            f_cube = self._robust_scale_overlap_cube(overlap_info)
            if np.isfinite(f_cube) and f_cube > 0:
                self.log.emit(f"[INFO] Fator map2→map1 (OVERLAP do cubo): {f_cube:.4f}")
                # se divergir muito, avisar e preferir o do cubo (é o que interessa p/ NMF/PCA)
                if abs(np.log(f_cube / (factor_map2_to_map1 if factor_map2_to_map1 else 1.0))) > np.log(1.15):
                    self.log.emit("[WARN] Fator por strips/imagem divergiu >15% do fator por overlap do cubo; usando o do cubo.")
                factor_map2_to_map1 = f_cube
        except Exception as _e:
            self.log.emit(f"[WARN] Não foi possível refinar fator pelo overlap do cubo: {_e}")

        self.progress.emit(30, "Definindo escalas e preparando o mosaico...")

        # Escalas conforme mapa escolhido
        scale1, scale2 = 1.0, 1.0
        if self.selected_map == "map1":
            scale2 = factor_map2_to_map1
            self.log.emit(f"[INFO] Referência (sem escala): MAPA1 | Escala aplicada ao MAPA2: {scale2:.4f}")
        else:
            # manter mapa2 intacto e reescalar mapa1
            scale1 = 1.0 / (factor_map2_to_map1 if factor_map2_to_map1 != 0 else 1.0)
            self.log.emit(f"[INFO] Referência (sem escala): MAPA2 | Escala aplicada ao MAPA1: {scale1:.4f}")

        with h5py.File(self.map1_path, "r") as f1, h5py.File(self.map2_path, "r") as f2:
            ds1, ds2 = f1["cube"], f2["cube"]
            ny1, nx1, nz1 = ds1.shape
            ny2, nx2, nz2 = ds2.shape
            if nz1 != nz2:
                raise ValueError(f"Canais espectrais diferentes: mapa1={nz1}, mapa2={nz2}")

            # Posições relativas (map1 em 0,0)
            x1, y1 = 0, 0
            if self.direction == "vertical":
                y2 = offset
                x2 = 0
            else:
                x2 = offset
                y2 = 0

            # Aplicar shift residual como deslocamento do tile (inteiro, sem interpolar o cubo)
            x2 += shift_dx
            y2 += shift_dy

            # Bounding box do mosaico
            y_min = min(y1, y2)
            x_min = min(x1, x2)
            y_max = max(y1 + ny1, y2 + ny2)
            x_max = max(x1 + nx1, x2 + nx2)

            total_rows = y_max - y_min
            total_cols = x_max - x_min

            # Offsets no mosaico (>=0)
            y1_off, x1_off = y1 - y_min, x1 - x_min
            y2_off, x2_off = y2 - y_min, x2 - x_min

            self.log.emit(f"[INFO] Mosaic: {total_rows}×{total_cols}×{nz1}")
            self.log.emit(f"[INFO] Posição mapa1: (y={y1_off}, x={x1_off}) | mapa2: (y={y2_off}, x={x2_off})")
            self.log.emit(f"[INFO] offset={offset} | shift(dy,dx)=({shift_dy},{shift_dx}) | overlap={overlap}")

            # ── RECORTE LOSSY PARA ELIMINAR BORDAS VAZIAS (aceita perda de dados) ──
            # Regra solicitada:
            #   - Costura vertical  -> recorta em X (laterais), mantém altura
            #   - Costura horizontal-> recorta em Y (topo/base), mantém largura
            # O recorte usa a INTERSEÇÃO das áreas cobertas por map1 e map2 na dimensão correspondente.
            pre_total_rows, pre_total_cols = total_rows, total_cols
            crop_x0 = crop_x1 = None
            crop_y0 = crop_y1 = None

            if self.direction == "vertical":
                crop_x0 = max(x1_off, x2_off)
                crop_x1 = min(x1_off + nx1, x2_off + nx2)
                if crop_x1 <= crop_x0:
                    raise ValueError(
                        f"Recorte em X inviável: não há interseção em X após shift_dx={shift_dx}. "
                        f"x1=[{x1_off},{x1_off+nx1}) x2=[{x2_off},{x2_off+nx2})"
                    )
                if crop_x0 != 0 or crop_x1 != total_cols:
                    self.log.emit(f"[INFO] Recorte lossy em X: x=[{crop_x0}:{crop_x1}] (remove {crop_x0} px à esquerda e {total_cols-crop_x1} px à direita)")
                x1_off -= crop_x0
                x2_off -= crop_x0
                total_cols = crop_x1 - crop_x0
            else:  # horizontal
                crop_y0 = max(y1_off, y2_off)
                crop_y1 = min(y1_off + ny1, y2_off + ny2)
                if crop_y1 <= crop_y0:
                    raise ValueError(
                        f"Recorte em Y inviável: não há interseção em Y após shift_dy={shift_dy}. "
                        f"y1=[{y1_off},{y1_off+ny1}) y2=[{y2_off},{y2_off+ny2})"
                    )
                if crop_y0 != 0 or crop_y1 != total_rows:
                    self.log.emit(f"[INFO] Recorte lossy em Y: y=[{crop_y0}:{crop_y1}] (remove {crop_y0} px acima e {total_rows-crop_y1} px abaixo)")
                y1_off -= crop_y0
                y2_off -= crop_y0
                total_rows = crop_y1 - crop_y0



            # Percentual removido (informativo; recorte com perda proposital)
            if pre_total_cols > 0 and pre_total_rows > 0:
                kept = (total_rows * total_cols) / (pre_total_rows * pre_total_cols)
                removed_pct = max(0.0, 1.0 - kept) * 100.0
                if removed_pct > 0:
                    self.log.emit(f"[INFO] Recorte lossy removeu ~{removed_pct:.2f}% da área do mosaico.")
                    if removed_pct / 100.0 >= LOSSY_CROP_WARN_THRESHOLD:
                        self.log.emit("[WARN] Recorte lossy significativo (perda proposital). Se não quiser perder dados, desative o recorte.")

            self.log.emit(f"[INFO] Mosaic (após recorte): {total_rows}×{total_cols}×{nz1}")

            # Segurança: saída não pode sobrescrever entradas
            out_path = Path(self.output_hdf5)
            in1 = Path(self.map1_path).resolve()
            in2 = Path(self.map2_path).resolve()
            try:
                out_res = out_path.resolve()
            except Exception:
                out_res = out_path

            if out_res == in1 or out_res == in2:
                raise ValueError("O arquivo de saída HDF5 não pode ser o mesmo que um dos arquivos de entrada.")

            self.progress.emit(36, "Preparando arquivo HDF5 de saída...")

            # Se já existir, tenta remover (evita WinError 33 / lock violation)
            if out_path.exists():
                try:
                    out_path.unlink()
                except PermissionError as e:
                    raise RuntimeError(
                        "Não foi possível criar o HDF5 de saída porque ele está em uso/aberto por outro programa "
                        "(PyMCA, HDFView, Explorer com painel de pré-visualização, etc.).\n"
                        "Feche o arquivo e tente novamente, ou escolha outro nome/pasta."
                    ) from e

            try:
                fout = h5py.File(self.output_hdf5, "w")
            except OSError as e:
                raise RuntimeError(
                    "Erro ao criar o arquivo HDF5 de saída. Em Windows isso costuma ocorrer por lock do arquivo "
                    "(WinError 33) ou por pasta sincronizada/restrita.\n"
                    "Sugestões: feche o arquivo em outros programas, desative o painel de pré-visualização do Explorer, "
                    "ou escolha um caminho de saída diferente."
                ) from e

            with fout as fout:
                # ── VERSÃO OTIMIZADA: GZIP + chunks maiores + escrita sem duplicação no overlap ──
                # Compressão: GZIP (nível configurável; foco em velocidade)
                # Tipo de dado: automático (OUTPUT_DTYPE_MODE)
                # Chunks: dinâmicos baseados no tamanho do mosaico
                
                # ── Saída: dtype e compressão ───────────────────────────────────────────
                if OUTPUT_DTYPE_MODE == "float32":
                    out_dtype = np.float32
                elif OUTPUT_DTYPE_MODE == "uint32":
                    out_dtype = np.uint32
                else:  # auto
                    need_scale = (abs(scale1 - 1.0) > 1e-9) or (abs(scale2 - 1.0) > 1e-9)
                    if need_scale or np.issubdtype(ds1.dtype, np.floating) or np.issubdtype(ds2.dtype, np.floating):
                        out_dtype = np.float32
                    else:
                        out_dtype = np.uint32

                fill_value = 0.0 if out_dtype == np.float32 else 0

                self.log.emit(
                    f"[INFO] Criando dataset com compressão GZIP (nível {COMPRESSION_LEVEL}) + "
                    f"dtype={np.dtype(out_dtype).name} (modo={OUTPUT_DTYPE_MODE})"
                )
                self.log.emit(f"[INFO] Chunks (PyMCA/NMF): 1x1x{nz1} (pixel spectra)")
                self.progress.emit(40, "Criando dataset HDF5 de saída...")
                # Chunks para máxima performance no PyMCA/NMF:
                # chunk por pixel (1×1×Z) => leitura rápida de espectros individuais e boa compressão.
                chunk_rows = 1
                chunk_cols = 1
                chunk_energy = nz1

                out_ds = fout.create_dataset(
                    "cube", shape=(total_rows, total_cols, nz1),
                    dtype=out_dtype,
                    compression="gzip",
                    compression_opts=COMPRESSION_LEVEL,
                    chunks=(chunk_rows, chunk_cols, chunk_energy),
                    shuffle=True,  # melhora compressão em inteiros (gzip)
                    fillvalue=fill_value,
                )

                write_progress = {"done": 0, "total": max(1, int(total_rows) * int(total_cols)), "last_pct": 44}

                def _area(rect):
                    if rect is None:
                        return 0
                    y0, y1, x0, x1 = rect
                    return max(0, int(y1) - int(y0)) * max(0, int(x1) - int(x0))

                def _emit_write_progress(force=False):
                    total = max(1, int(write_progress.get("total", 1)))
                    done = min(total, int(write_progress.get("done", 0)))
                    pct = 52 + int(28 * min(1.0, done / total))
                    if force or pct > write_progress.get("last_pct", -1):
                        write_progress["last_pct"] = pct
                        self.progress.emit(pct, f"Gravando HDF5 em chunks... {done:,}/{total:,} pixels".replace(",", "."))

                def _clip_rect(rect):
                    y0, y1, x0, x1 = rect
                    y0 = max(0, int(y0))
                    x0 = max(0, int(x0))
                    y1 = min(total_rows, int(y1))
                    x1 = min(total_cols, int(x1))
                    if y1 <= y0 or x1 <= x0:
                        return None
                    return (y0, y1, x0, x1)

                def _rect_intersection(a, b):
                    ay0, ay1, ax0, ax1 = a
                    by0, by1, bx0, bx1 = b
                    y0 = max(ay0, by0)
                    y1 = min(ay1, by1)
                    x0 = max(ax0, bx0)
                    x1 = min(ax1, bx1)
                    if y1 <= y0 or x1 <= x0:
                        return None
                    return (y0, y1, x0, x1)

                def _rect_diff(rect, ov):
                    """Retorna lista de retângulos (y0,y1,x0,x1) = rect \\ ov (tudo em coordenadas do mosaico)."""
                    if rect is None:
                        return []
                    if ov is None:
                        return [rect]
                    y0, y1, x0, x1 = rect
                    oy0, oy1, ox0, ox1 = ov

                    # sem interseção
                    if oy1 <= y0 or oy0 >= y1 or ox1 <= x0 or ox0 >= x1:
                        return [rect]

                    out = []
                    # acima
                    if y0 < oy0:
                        out.append((y0, oy0, x0, x1))
                    # abaixo
                    if oy1 < y1:
                        out.append((oy1, y1, x0, x1))
                    # esquerda e direita no miolo em Y
                    yy0 = max(y0, oy0)
                    yy1 = min(y1, oy1)
                    if x0 < ox0:
                        out.append((yy0, yy1, x0, ox0))
                    if ox1 < x1:
                        out.append((yy0, yy1, ox1, x1))
                    return [r for r in out if (r[1] > r[0] and r[3] > r[2])]

                def paste_rect(ds_in, y_off, x_off, scale, rect, label):
                    """Cola uma sub-região retangular do tile no mosaico, escrevendo em chunks (Y,X,Z)."""
                    if rect is None:
                        return
                    out_y0, out_y1, out_x0, out_x1 = rect

                    # origem no tile
                    in_y0 = out_y0 - y_off
                    in_x0 = out_x0 - x_off

                    rows = out_y1 - out_y0
                    cols = out_x1 - out_x0

                    progress_step = max(1000, int(write_progress.get("total", 1)) // 80)
                    local_written = 0

                    for yy in range(0, rows, chunk_rows):
                        y_len = min(chunk_rows, rows - yy)
                        y0_in = in_y0 + yy
                        y1_in = y0_in + y_len
                        y0_out = out_y0 + yy
                        y1_out = y0_out + y_len

                        for xx in range(0, cols, chunk_cols):
                            x_len = min(chunk_cols, cols - xx)
                            x0_in = in_x0 + xx
                            x1_in = x0_in + x_len
                            x0_out = out_x0 + xx
                            x1_out = x0_out + x_len

                            for zz in range(0, nz1, chunk_energy):
                                z_len = min(chunk_energy, nz1 - zz)
                                z0 = zz
                                z1 = zz + z_len

                                try:
                                    slab = ds_in[y0_in:y1_in, x0_in:x1_in, z0:z1]
                                except Exception as e:
                                    raise RuntimeError(
                                        
                                        f"Erro ao ler dados HDF5 ({label}) y=[{y0_in}:{y1_in}] x=[{x0_in}:{x1_in}] z=[{z0}:{z1}]"
                                    ) from e

                                if out_dtype == np.float32:
                                    # Preserva normalização (valores fracionários). Ideal para PyMCA/NMF/PCA.
                                    slab_f = slab.astype(np.float32, copy=False)
                                    if scale != 1.0:
                                        slab_f = slab_f * np.float32(scale)
                                    out_ds[y0_out:y1_out, x0_out:x1_out, z0:z1] = slab_f
                                else:
                                    # Compacto (uint32): arredonda após escala (perde parte fracionária).
                                    if scale != 1.0:
                                        slab_f = slab.astype(np.float32, copy=False)
                                        slab_f = np.rint(slab_f * np.float32(scale))
                                        np.clip(slab_f, 0, np.iinfo(np.uint32).max, out=slab_f)
                                        slab_u = slab_f.astype(np.uint32, copy=False)
                                    else:
                                        slab_u = slab.astype(np.uint32, copy=False)

                                    out_ds[y0_out:y1_out, x0_out:x1_out, z0:z1] = slab_u

                            local_written += int(y_len) * int(x_len)
                            if local_written >= progress_step:
                                write_progress["done"] += local_written
                                local_written = 0
                                _emit_write_progress()

                    if local_written:
                        write_progress["done"] += local_written
                        _emit_write_progress()

                # Retângulos dos tiles no mosaico (podem ter offsets negativos após recorte)
                rect1_raw = (y1_off, y1_off + ny1, x1_off, x1_off + nx1)
                rect2_raw = (y2_off, y2_off + ny2, x2_off, x2_off + nx2)
                rect1 = _clip_rect(rect1_raw)
                rect2 = _clip_rect(rect2_raw)

                if rect1 is None or rect2 is None:
                    raise ValueError("Após recorte, um dos tiles ficou completamente fora do mosaico.")


                ov = _rect_intersection(rect1, rect2)
                if ov is None:
                    write_progress["total"] = max(1, _area(rect1) + _area(rect2))
                else:
                    write_progress["total"] = max(
                        1,
                        sum(_area(r) for r in _rect_diff(rect1, ov))
                        + sum(_area(r) for r in _rect_diff(rect2, ov))
                        + _area(ov)
                    )
                self.progress.emit(44, "Preparando cálculo da seam e escrita do mosaico...")

                def _read_intensity_line(ds_in, y_idx, x0_in, x1_in, scale):
                    a = ds_in[y_idx, x0_in:x1_in, :].astype(np.float32).sum(axis=1)
                    if abs(scale - 1.0) > 1e-9:
                        a *= np.float32(scale)
                    return a

                def _read_intensity_col(ds_in, y0_in, y1_in, x_idx, scale):
                    a = ds_in[y0_in:y1_in, x_idx, :].astype(np.float32).sum(axis=1)
                    if abs(scale - 1.0) > 1e-9:
                        a *= np.float32(scale)
                    return a

                def _read_scaled_rect(ds_in, y0_in, y1_in, x0_in, x1_in, scale):
                    a = ds_in[y0_in:y1_in, x0_in:x1_in, :].astype(np.float32)
                    if abs(scale - 1.0) > 1e-9:
                        a *= np.float32(scale)
                    return a

                def _smooth_path_1d(path, win=9):
                    win = int(max(3, win))
                    if win % 2 == 0:
                        win += 1
                    half = win // 2
                    out = path.copy()
                    n = len(path)
                    for ii in range(n):
                        aa = max(0, ii - half)
                        bb = min(n, ii + half + 1)
                        out[ii] = int(np.median(path[aa:bb]))
                    return out

                def _compute_overlap_cost_matrix():
                    if ov is None:
                        return None, None

                    oy0, oy1, ox0, ox1 = ov

                    if self.direction == "vertical":
                        top_is_map1 = rect1[0] <= rect2[0]
                        if top_is_map1:
                            top_ds, top_y_off, top_x_off, top_scale = ds1, y1_off, x1_off, scale1
                            bot_ds, bot_y_off, bot_x_off, bot_scale = ds2, y2_off, x2_off, scale2
                        else:
                            top_ds, top_y_off, top_x_off, top_scale = ds2, y2_off, x2_off, scale2
                            bot_ds, bot_y_off, bot_x_off, bot_scale = ds1, y1_off, x1_off, scale1

                        A = _read_scaled_rect(top_ds, oy0 - top_y_off, oy1 - top_y_off, ox0 - top_x_off, ox1 - top_x_off, top_scale).sum(axis=2)
                        B = _read_scaled_rect(bot_ds, oy0 - bot_y_off, oy1 - bot_y_off, ox0 - bot_x_off, ox1 - bot_x_off, bot_scale).sum(axis=2)
                        # robust local mismatch metric; low counts are naturally down-weighted by denominator
                        C = np.abs(A - B) / np.sqrt(A + B + 1.0)
                        return C.astype(np.float32), {
                            "orientation": "vertical",
                            "top_is_map1": bool(top_is_map1),
                            "oy0": int(oy0), "oy1": int(oy1), "ox0": int(ox0), "ox1": int(ox1)
                        }
                    else:
                        left_is_map1 = rect1[2] <= rect2[2]
                        if left_is_map1:
                            left_ds, left_y_off, left_x_off, left_scale = ds1, y1_off, x1_off, scale1
                            right_ds, right_y_off, right_x_off, right_scale = ds2, y2_off, x2_off, scale2
                        else:
                            left_ds, left_y_off, left_x_off, left_scale = ds2, y2_off, x2_off, scale2
                            right_ds, right_y_off, right_x_off, right_scale = ds1, y1_off, x1_off, scale1

                        A = _read_scaled_rect(left_ds, oy0 - left_y_off, oy1 - left_y_off, ox0 - left_x_off, ox1 - left_x_off, left_scale).sum(axis=2)
                        B = _read_scaled_rect(right_ds, oy0 - right_y_off, oy1 - right_y_off, ox0 - right_x_off, ox1 - right_x_off, right_scale).sum(axis=2)
                        C = np.abs(A - B) / np.sqrt(A + B + 1.0)
                        return C.astype(np.float32), {
                            "orientation": "horizontal",
                            "left_is_map1": bool(left_is_map1),
                            "oy0": int(oy0), "oy1": int(oy1), "ox0": int(ox0), "ox1": int(ox1)
                        }

                def _find_optimal_seam_path(C, seam_ctx):
                    # Dynamic-programming path with continuity penalty; no spectral mixing.
                    if C is None:
                        return None
                    penalty = 0.20

                    if seam_ctx["orientation"] == "vertical":
                        H, W = C.shape  # y within overlap, x within overlap
                        dp = np.empty((H, W), dtype=np.float32)
                        prev = np.zeros((H, W), dtype=np.int16)
                        dp[:, 0] = C[:, 0]
                        prev[:, 0] = -1
                        seam_progress_step = max(1, W // 10)
                        for x in range(1, W):
                            if x % seam_progress_step == 0:
                                self.progress.emit(49 + min(2, int(2 * x / max(1, W - 1))), "Calculando caminho ótimo da seam vertical...")
                            for y in range(H):
                                best_cost = dp[y, x - 1]
                                best_py = y
                                if y > 0:
                                    c = dp[y - 1, x - 1] + penalty
                                    if c < best_cost:
                                        best_cost = c; best_py = y - 1
                                if y < H - 1:
                                    c = dp[y + 1, x - 1] + penalty
                                    if c < best_cost:
                                        best_cost = c; best_py = y + 1
                                dp[y, x] = C[y, x] + best_cost
                                prev[y, x] = best_py
                        y = int(np.argmin(dp[:, W - 1]))
                        path = np.zeros(W, dtype=np.int32)
                        path[W - 1] = y
                        for x in range(W - 1, 0, -1):
                            y = int(prev[y, x])
                            path[x - 1] = y
                        path = _smooth_path_1d(path, win=9)
                        return path
                    else:
                        H, W = C.shape  # y within overlap, x within overlap; path is x(y)
                        dp = np.empty((H, W), dtype=np.float32)
                        prev = np.zeros((H, W), dtype=np.int16)
                        dp[0, :] = C[0, :]
                        prev[0, :] = -1
                        seam_progress_step = max(1, H // 10)
                        for y in range(1, H):
                            if y % seam_progress_step == 0:
                                self.progress.emit(49 + min(2, int(2 * y / max(1, H - 1))), "Calculando caminho ótimo da seam horizontal...")
                            for x in range(W):
                                best_cost = dp[y - 1, x]
                                best_px = x
                                if x > 0:
                                    c = dp[y - 1, x - 1] + penalty
                                    if c < best_cost:
                                        best_cost = c; best_px = x - 1
                                if x < W - 1:
                                    c = dp[y - 1, x + 1] + penalty
                                    if c < best_cost:
                                        best_cost = c; best_px = x + 1
                                dp[y, x] = C[y, x] + best_cost
                                prev[y, x] = best_px
                        x = int(np.argmin(dp[H - 1, :]))
                        path = np.zeros(H, dtype=np.int32)
                        path[H - 1] = x
                        for y in range(H - 1, 0, -1):
                            x = int(prev[y, x])
                            path[y - 1] = x
                        path = _smooth_path_1d(path, win=9)
                        return path

                if ov is None:
                    self.progress.emit(52, "Iniciando gravação do mosaico sem overlap real...")
                    paste_rect(ds1, y1_off, x1_off, scale1, rect1, "map1")
                    paste_rect(ds2, y2_off, x2_off, scale2, rect2, "map2")
                    seam_ctx = None
                    seam_path = None
                    self.log.emit("[INFO] Sem overlap real após recorte — mapas gravados integralmente.")
                else:
                    self.progress.emit(46, "Calculando matriz de custo da sobreposição...")
                    C, seam_ctx = _compute_overlap_cost_matrix()
                    self.progress.emit(49, "Calculando caminho ótimo da seam...")
                    seam_path = _find_optimal_seam_path(C, seam_ctx)
                    self.progress.emit(52, "Iniciando gravação do mosaico sem duplicar o overlap...")

                    oy0, oy1, ox0, ox1 = ov

                    # 1) Escreve TUDO fora do overlap (sem duplicidade)
                    for r in _rect_diff(rect1, ov):
                        paste_rect(ds1, y1_off, x1_off, scale1, r, "map1")
                    for r in _rect_diff(rect2, ov):
                        paste_rect(ds2, y2_off, x2_off, scale2, r, "map2")

                    # 2) Escreve overlap usando a seam ótima; cada pixel entra uma única vez.
                    if seam_ctx["orientation"] == "vertical":
                        top_is_map1 = seam_ctx["top_is_map1"]
                        if top_is_map1:
                            top_ds, top_y_off, top_x_off, top_scale = ds1, y1_off, x1_off, scale1
                            bot_ds, bot_y_off, bot_x_off, bot_scale = ds2, y2_off, x2_off, scale2
                        else:
                            top_ds, top_y_off, top_x_off, top_scale = ds2, y2_off, x2_off, scale2
                            bot_ds, bot_y_off, bot_x_off, bot_scale = ds1, y1_off, x1_off, scale1

                        for ix, gx in enumerate(range(ox0, ox1)):
                            local_y = int(seam_path[ix])  # within overlap
                            seam_y = oy0 + local_y
                            # top part
                            if seam_y > oy0:
                                r_top = (oy0, seam_y, gx, gx + 1)
                                paste_rect(top_ds, top_y_off, top_x_off, top_scale, r_top, "top_overlap")
                            # bottom part
                            if seam_y < oy1:
                                r_bot = (seam_y, oy1, gx, gx + 1)
                                paste_rect(bot_ds, bot_y_off, bot_x_off, bot_scale, r_bot, "bottom_overlap")

                        self.log.emit(
                            f"[INFO] Seam ótima (vertical) calculada no overlap: y local variável em X; "
                            f"faixa y=[{oy0},{oy1}), x=[{ox0},{ox1})"
                        )

                    else:
                        left_is_map1 = seam_ctx["left_is_map1"]
                        if left_is_map1:
                            left_ds, left_y_off, left_x_off, left_scale = ds1, y1_off, x1_off, scale1
                            right_ds, right_y_off, right_x_off, right_scale = ds2, y2_off, x2_off, scale2
                        else:
                            left_ds, left_y_off, left_x_off, left_scale = ds2, y2_off, x2_off, scale2
                            right_ds, right_y_off, right_x_off, right_scale = ds1, y1_off, x1_off, scale1

                        for iy, gy in enumerate(range(oy0, oy1)):
                            local_x = int(seam_path[iy])  # within overlap
                            seam_x = ox0 + local_x
                            if seam_x > ox0:
                                r_left = (gy, gy + 1, ox0, seam_x)
                                paste_rect(left_ds, left_y_off, left_x_off, left_scale, r_left, "left_overlap")
                            if seam_x < ox1:
                                r_right = (gy, gy + 1, seam_x, ox1)
                                paste_rect(right_ds, right_y_off, right_x_off, right_scale, r_right, "right_overlap")

                        self.log.emit(
                            f"[INFO] Seam ótima (horizontal) calculada no overlap: x local variável em Y; "
                            f"faixa y=[{oy0},{oy1}), x=[{ox0},{ox1})"
                        )

                _emit_write_progress(force=True)
                self.progress.emit(82, "Salvando metadados e eixos do HDF5...")

                # Salva também a seam no metadata para inspeção posterior
                try:
                    if seam_path is not None and seam_ctx is not None:
                        if seam_ctx["orientation"] == "vertical":
                            fout.create_dataset("optimal_seam_y_by_x", data=(seam_path.astype(np.int32) + seam_ctx["oy0"]))
                            fout["optimal_seam_y_by_x"].attrs["x_start"] = int(seam_ctx["ox0"])
                        else:
                            fout.create_dataset("optimal_seam_x_by_y", data=(seam_path.astype(np.int32) + seam_ctx["ox0"]))
                            fout["optimal_seam_x_by_y"].attrs["y_start"] = int(seam_ctx["oy0"])
                except Exception as _se:
                    self.log.emit(f"[WARN] Não foi possível salvar seam path: {_se}")

                fout.create_dataset("x_axis", data=np.arange(total_cols, dtype=np.float32))
                fout.create_dataset("y_axis", data=np.arange(total_rows, dtype=np.float32))
                fout.create_dataset("energy", data=np.arange(nz1, dtype=np.float32))
                meta = fout.create_group("metadata")
                meta.attrs["source_map1"] = self.map1_name
                meta.attrs["source_map2"] = self.map2_name
                meta.attrs["stitch_direction"] = self.direction
                meta.attrs["registration_method"] = self.reg_method
                meta.attrs["normalization_method"] = self.norm_method

                meta.attrs["overlap_write_mode"] = "optimal_seam_path"
                # recorte aplicado (lossy) para remover bordas vazias
                if crop_x0 is not None:
                    meta.attrs["crop_x0"] = int(crop_x0)
                    meta.attrs["crop_x1"] = int(crop_x1)
                if crop_y0 is not None:
                    meta.attrs["crop_y0"] = int(crop_y0)
                    meta.attrs["crop_y1"] = int(crop_y1)
                meta.attrs["elements_selected"] = ",".join(self.elements_selected) if self.elements_selected else ""
                meta.attrs["selected_map"] = self.selected_map
                meta.attrs["offset"] = str(offset)
                meta.attrs["shift_dx"] = str(shift_dx)
                meta.attrs["shift_dy"] = str(shift_dy)
                meta.attrs["overlap_pixels"] = str(overlap)
                meta.attrs["factor_map2_to_map1"] = str(round(float(factor_map2_to_map1), 8))
                meta.attrs["scale_map1"] = str(round(float(scale1), 8))
                meta.attrs["scale_map2"] = str(round(float(scale2), 8))
                meta.attrs["output_dtype"] = np.dtype(out_dtype).name
                meta.attrs["output_dtype_mode"] = OUTPUT_DTYPE_MODE
                meta.attrs["compression"] = "gzip"
                meta.attrs["compression_level"] = int(COMPRESSION_LEVEL)
                meta.attrs["shuffle"] = True
                meta.attrs["processing_date"] = datetime.now().isoformat()
                meta.attrs["tool"] = "XRF Data Stitcher - vBeta"

        scale_info = {
            "factor_map2_to_map1": float(factor_map2_to_map1),
            "scale_map1": float(scale1),
            "scale_map2": float(scale2),
            "registration_method": self.reg_method,
            "normalization_method": self.norm_method,
            "output_dtype": np.dtype(out_dtype).name,
            "dtype_mode": OUTPUT_DTYPE_MODE,
            "compression": "gzip",
            "compression_level": int(COMPRESSION_LEVEL),
            "shuffle": True,
        }
        return (total_rows, total_cols, nz1), scale_info

    def _generate_report(self, shape1, shape2, overlap_info, quality, stitched_shape, scale_info):
        overlap = overlap_info["overlap_pixels"]
        selected_label = "Mapa 1" if self.selected_map == "map1" else "Mapa 2"
        selected_name = self.map1_name if self.selected_map == "map1" else self.map2_name
        discarded_name = self.map2_name if self.selected_map == "map1" else self.map1_name
        direction_pt = "Vertical" if self.direction == "vertical" else "Horizontal"
        reg_raw = str(scale_info.get("registration_method", "total"))
        norm_raw = str(scale_info.get("normalization_method", "total"))
        els = ", ".join(self.elements_selected) if self.elements_selected else "Fe, Ca"
        reg_method_pt = f"Elementos (ROIs: {els})" if reg_raw.startswith("elements") else "Total (soma em Z)"
        norm_method_pt = f"Overlap por elementos (ROIs: {els})" if norm_raw.startswith("elements") else "Total (soma em Z)"
        factor_map2_to_map1 = float(scale_info.get("factor_map2_to_map1", 1.0))
        scale_map1 = float(scale_info.get("scale_map1", 1.0))
        scale_map2 = float(scale_info.get("scale_map2", 1.0))
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        sc = f"{quality['spectral_correlation']:.4f}" if quality.get("spectral_correlation") else "N/A"
        ir = f"{quality['intensity_ratio']:.4f}" if quality.get("intensity_ratio") else "N/A"

        # ── Info da saída (para o relatório) ─────────────────────────────────────
        out_dtype_name = str(scale_info.get("output_dtype", "N/A"))
        dtype_mode = str(scale_info.get("dtype_mode", OUTPUT_DTYPE_MODE))
        comp = str(scale_info.get("compression", "gzip"))
        comp_lvl = scale_info.get("compression_level", COMPRESSION_LEVEL)
        shuf = bool(scale_info.get("shuffle", True))
        if comp.lower() == "gzip":
            compression_desc = f"GZIP nível {int(comp_lvl)}" + (" + shuffle" if shuf else "")
        else:
            compression_desc = comp


        map1_role = "Topo" if self.direction == "vertical" else "Esquerda"
        map2_role = "Base" if self.direction == "vertical" else "Direita"

        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><title>XRF Data Stitcher - vBeta</title>
<style>
:root{{--p:#1a3a5c;--a:#2e7d9e;--bg:#f8f9fa;--c:#fff;--b:#dee2e6;--t:#212529;--m:#6c757d;--s:#198754}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:var(--bg);color:var(--t);line-height:1.6}}
.wrap{{max-width:960px;margin:0 auto;padding:2rem}}
header{{background:var(--p);color:#fff;padding:2rem;border-radius:8px;margin-bottom:2rem}}
header h1{{font-size:1.6rem;font-weight:700}}
header p{{opacity:.85;margin-top:.5rem;font-size:.95rem}}
.badge{{display:inline-block;background:rgba(255,255,255,.15);color:#fff;padding:.2rem .7rem;border-radius:20px;font-size:.8rem;margin-top:.5rem;margin-right:.4rem}}
section{{background:var(--c);border:1px solid var(--b);border-radius:8px;padding:1.5rem;margin-bottom:1.5rem}}
section h2{{font-size:1.1rem;color:var(--p);border-bottom:2px solid var(--a);padding-bottom:.5rem;margin-bottom:1rem}}
table{{width:100%;border-collapse:collapse;font-size:.9rem}}
th{{background:var(--p);color:#fff;padding:.6rem 1rem;text-align:left}}
td{{padding:.55rem 1rem;border-bottom:1px solid var(--b)}}
tr:last-child td{{border-bottom:none}}
tr:nth-child(even) td{{background:#f1f5f9}}
.hl{{background:#e8f4f8!important;font-weight:600;color:var(--p)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-top:1rem}}
.card{{background:var(--bg);border:1px solid var(--b);border-radius:6px;padding:1rem;text-align:center}}
.card .v{{font-size:1.6rem;font-weight:700;color:var(--a)}}
.card .l{{font-size:.8rem;color:var(--m);margin-top:.25rem}}
.sug{{background:#e8f5e9;border-left:4px solid var(--s);padding:1rem 1.2rem;border-radius:4px;margin-top:1rem}}
.sug strong{{color:var(--s)}}
.meth{{background:#e3f2fd;border-left:4px solid var(--a);padding:1rem 1.2rem;border-radius:4px;margin-top:1rem;font-size:.92rem}}
footer{{text-align:center;color:var(--m);font-size:.8rem;margin-top:2rem;padding-top:1rem;border-top:1px solid var(--b)}}
code{{background:#f1f5f9;padding:.1rem .4rem;border-radius:3px;font-family:monospace;font-size:.88rem}}
p{{margin-bottom:.5rem}}
</style></head>
<body><div class="wrap">
<header>
  <h1>Relatório de Costura de Mapas MA-XRF</h1>
  <p>Costura inteligente de dados de MA-XRF</p>
  <span class="badge">{now}</span>
  <span class="badge">XRF Data Stitcher - vBeta</span>
</header>

<section><h2>1. Arquivos de Entrada</h2>
<table>
  <tr><th>Parâmetro</th><th>Mapa 1 ({map1_role})</th><th>Mapa 2 ({map2_role})</th></tr>
  <tr><td>Nome</td><td>{self.map1_name}</td><td>{self.map2_name}</td></tr>
  <tr><td>Dimensão Y</td><td>{shape1[0]} px</td><td>{shape2[0]} px</td></tr>
  <tr><td>Dimensão X</td><td>{shape1[1]} px</td><td>{shape2[1]} px</td></tr>
  <tr><td>Canais espectrais</td><td>{shape1[2]}</td><td>{shape2[2]}</td></tr>
  <tr><td>Total de pontos</td><td>{shape1[0]*shape1[1]:,}</td><td>{shape2[0]*shape2[1]:,}</td></tr>
</table></section>

<section><h2>2. Detecção de Sobreposição</h2>
<p>Sobreposição detectada por correlação cruzada da <b>imagem de registro</b> (método: <b>{reg_method_pt}</b>), normalizada por z-score.</p>
<div class="grid">
  <div class="card"><div class="v">{overlap}</div><div class="l">Pixels de sobreposição</div></div>
  <div class="card"><div class="v">{overlap_info['best_correlation']:.4f}</div><div class="l">Correlação de imagem</div></div>
  <div class="card"><div class="v">{direction_pt}</div><div class="l">Direção de costura</div></div>
</div></section>

<section><h2>3. Análise de Qualidade</h2>
<table>
  <tr><th>Métrica</th><th>Mapa 1</th><th>Mapa 2</th></tr>
  <tr><td>Razão Sinal-Ruído (SNR)</td><td>{quality['map1_snr']:.4f}</td><td>{quality['map2_snr']:.4f}</td></tr>
  <tr><td>Intensidade média</td><td>{quality['map1_mean_intensity']:.4f}</td><td>{quality['map2_mean_intensity']:.4f}</td></tr>
  <tr><td>Desvio padrão</td><td>{quality['map1_std']:.4f}</td><td>{quality['map2_std']:.4f}</td></tr>
</table>
<table style="margin-top:1rem">
  <tr><th>Métrica Global</th><th>Valor</th></tr>
  <tr><td>Correlação espectral na sobreposição</td><td>{sc}</td></tr>
  <tr><td>Razão de intensidade (Mapa2/Mapa1)</td><td>{ir}</td></tr>
</table>
<div class="sug"><strong>Sugestão:</strong> Manter <strong>{selected_label} ({selected_name})</strong> na sobreposição.<br>
<em>{quality['suggestion_reason']}</em></div></section>

<section><h2>4. Normalização</h2>
<div class="meth"><strong>Normalização por fator robusto na sobreposição</strong><br><br>
Método usado para estimar o fator: <b>{norm_method_pt}</b>.<br><br>

O software estima um fator <code>f = mediana(A)/mediana(B)</code> na região de sobreposição (após aplicar o shift residual),
onde <code>A</code> é o mapa 1 e <code>B</code> é o mapa 2, medidos na imagem escolhida para normalização.
O fator estimado foi: <b>{factor_map2_to_map1:.6g}</b> (map2 → escala do map1).<br><br>

Para preservar a integridade espectral, o mapa escolhido pelo usuário na região de sobreposição é mantido sem reamostragem.
A equalização de intensidade é aplicada apenas por multiplicação escalar:<br>
<ul style="margin-left:18px; margin-top:.6rem; line-height:1.7">
  <li><code>Mapa 1' = Mapa 1 × {scale_map1:.6g}</code></li>
  <li><code>Mapa 2' = Mapa 2 × {scale_map2:.6g}</code></li>
</ul>
</div></section>

<section><h2>5. Resultado Final</h2>
<p>A região de sobreposição de <strong>{discarded_name}</strong> foi descartada. <strong>{selected_name}</strong> foi mantido na sobreposição.</p>
<table style="margin-top:1rem">
  <tr><th>Parâmetro</th><th>Valor</th></tr>
  <tr><td>Direção de costura</td><td>{direction_pt}</td></tr>
  <tr><td>Método de registro</td><td>{reg_method_pt}</td></tr>
  <tr><td>Método de normalização</td><td>{norm_method_pt}</td></tr>
  <tr><td>Mapa mantido</td><td>{selected_label} ({selected_name})</td></tr>
  <tr><td>Pixels de sobreposição removidos</td><td>{overlap}</td></tr>
  <tr class="hl"><td>Dimensão final</td><td>{stitched_shape[0]} × {stitched_shape[1]} × {stitched_shape[2]}</td></tr>
  <tr class="hl"><td>Total de pontos</td><td>{stitched_shape[0]*stitched_shape[1]:,}</td></tr>
  <tr><td>Formato</td><td>HDF5 (compatível com PyMca)</td></tr>
  <tr><td>Tipo de dado</td><td>{out_dtype_name} (modo: {dtype_mode})</td></tr>
  <tr><td>Compressão</td><td>{compression_desc}</td></tr>
</table></section>

<section><h2>6. Estrutura do HDF5 de Saída</h2>
<table>
  <tr><th>Dataset</th><th>Dimensão</th><th>Descrição</th></tr>
  <tr><td><code>/cube</code></td><td>{stitched_shape[0]}×{stitched_shape[1]}×{stitched_shape[2]}</td><td>Dados costurados (dtype={out_dtype_name})</td></tr>
  <tr><td><code>/x_axis</code></td><td>{stitched_shape[1]}</td><td>Posições em X (pixels)</td></tr>
  <tr><td><code>/y_axis</code></td><td>{stitched_shape[0]}</td><td>Posições em Y (pixels)</td></tr>
  <tr><td><code>/energy</code></td><td>{stitched_shape[2]}</td><td>Canais de energia</td></tr>
  <tr><td><code>/metadata</code></td><td>—</td><td>Atributos do processamento</td></tr>
</table></section>

<section><h2>7. Justificativa Metodológica</h2>
<p>A <strong>detecção de sobreposição por correlação cruzada</strong> é uma técnica robusta de processamento de sinais que maximiza a similaridade estrutural entre regiões candidatas à sobreposição, sem depender de informações de posicionamento externas.</p>
<p>A <strong>normalização por reescalamento global de mediana</strong> equaliza diferenças instrumentais entre quadrantes adquiridos em sessões distintas, sem destruir o contraste espacial intra-mapa. A mediana é preferida à média por ser robusta a outliers (pixels com contagem anormalmente alta ou baixa). Os dados são mantidos em escala de contagens absolutas (float32), preservando a estrutura espectral necessária para NMF e PCA.</p>
<p>A <strong>seleção do mapa com maior SNR</strong> evita artefatos de blending que comprometeriam a integridade espectral, fundamental para autenticação de obras de arte por MA-XRF.</p>
</section>

<footer><p>XRF Data Stitcher - vBeta — Costura inteligente de dados de MA-XRF</p>
<p style="margin-top:.3rem">{now}</p></footer>
</div></body></html>"""

        with open(self.output_report, "w", encoding="utf-8") as f:
            f.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# WIDGETS AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

class FileSelector(QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.title_label = QLabel(label)
        self.title_label.setObjectName("section")
        layout.addWidget(self.title_label)

        row = QHBoxLayout()
        self.path_label = QLabel("Nenhum arquivo selecionado")
        self.path_label.setObjectName("file_path")
        self.path_label.setWordWrap(True)
        self.path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row.addWidget(self.path_label)

        btn = QPushButton("Selecionar...")
        btn.setObjectName("secondary")
        btn.setFixedWidth(110)
        btn.clicked.connect(self._browse)
        row.addWidget(btn)
        layout.addLayout(row)

        self._path = ""

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar arquivo HDF5", "",
            "Arquivos HDF5 (*.h5 *.hdf5);;Todos os arquivos (*)"
        )
        if path:
            self._path = path
            self.path_label.setText(path)

    def set_label(self, text):
        self.title_label.setText(text)

    @property
    def path(self):
        return self._path


class MetricCard(QFrame):
    def __init__(self, value, label, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedHeight(80)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        v = QLabel(str(value))
        v.setObjectName("metric_value")
        v.setAlignment(Qt.AlignCenter)
        l = QLabel(label)
        l.setObjectName("metric_label")
        l.setAlignment(Qt.AlignCenter)
        layout.addWidget(v)
        layout.addWidget(l)


# ─────────────────────────────────────────────────────────────────────────────
# JANELA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRF Data Stitcher - vBeta - Costura inteligente de dados de MA-XRF")
        self.setMinimumSize(900, 700)
        self.resize(1000, 760)
        self.setStyleSheet(STYLE)

        self._analysis_result = None
        self._thread = None
        self._spinner_frames = ["\\", "|", "/", "|"]
        self._spinner_index = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._stage_order = ["Arquivos", "Análise", "Saída", "Costura"]

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet("background-color: #1a3a5c;")
        header.setFixedHeight(64)
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(24, 0, 24, 0)
        title = QLabel("XRF Data Stitcher - vBeta")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: 700;")
        subtitle = QLabel("Costura inteligente de dados de MA-XRF")
        subtitle.setStyleSheet("color: rgba(255,255,255,0.65); font-size: 12px;")
        hlay.addWidget(title)
        hlay.addSpacing(16)
        hlay.addWidget(subtitle)
        hlay.addStretch()
        root.addWidget(header)

        # Conteúdo com abas
        content = QWidget()
        content.setContentsMargins(0, 0, 0, 0)
        clay = QVBoxLayout(content)
        clay.setContentsMargins(20, 16, 20, 16)

        self.tabs = QTabWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._build_stitch_tab())
        self.tabs.addTab(scroll, "  Nova Costura  ")
        self.tabs.addTab(self._build_about_tab(), "  Sobre  ")
        clay.addWidget(self.tabs)
        root.addWidget(content)

    # ── Aba principal ─────────────────────────────────────────────────────────

    def _build_stitch_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)

        stage_card = QFrame()
        stage_card.setObjectName("card")
        stage_layout = QVBoxLayout(stage_card)
        stage_layout.setContentsMargins(14, 12, 14, 12)
        stage_layout.setSpacing(8)

        stage_title = QLabel("Fluxo de processamento")
        stage_title.setObjectName("section")
        stage_layout.addWidget(stage_title)

        stage_row = QHBoxLayout()
        stage_row.setSpacing(8)
        self._step_labels = {}
        for idx, name in enumerate(self._stage_order, start=1):
            lbl = QLabel(f"{idx}. {name}")
            lbl.setObjectName("stepPillBase")
            lbl.setAlignment(Qt.AlignCenter)
            stage_row.addWidget(lbl)
            self._step_labels[name] = lbl
        stage_layout.addLayout(stage_row)

        self.flow_hint = QLabel("Selecione os dois arquivos e depois execute a análise automática.")
        self.flow_hint.setObjectName("softHint")
        stage_layout.addWidget(self.flow_hint)
        layout.addWidget(stage_card)

        # ── Seção 1: Arquivos ─────────────────────────────────────────────────
        grp_files = QGroupBox("1. Seleção de Arquivos HDF5")
        files_layout = QVBoxLayout(grp_files)
        files_layout.setSpacing(10)

        self.map1_sel = FileSelector("Mapa 1 — Topo (parte superior da obra)")
        self.map2_sel = FileSelector("Mapa 2 — Base (parte inferior da obra)")
        files_layout.addWidget(self.map1_sel)
        files_layout.addWidget(self.map2_sel)
        layout.addWidget(grp_files)

        # ── Seção 2: Configuração ─────────────────────────────────────────────
        grp_cfg = QGroupBox("2. Configuração de Costura")
        cfg_layout = QHBoxLayout(grp_cfg)

        dir_lbl = QLabel("Direção:")
        dir_lbl.setObjectName("section")
        cfg_layout.addWidget(dir_lbl)

        self.radio_v = QRadioButton("Vertical (um mapa acima do outro)")
        self.radio_h = QRadioButton("Horizontal (um mapa ao lado do outro)")
        self.radio_v.setChecked(True)
        self.dir_group = QButtonGroup()
        self.dir_group.addButton(self.radio_v)
        self.dir_group.addButton(self.radio_h)
        self.radio_v.toggled.connect(self._update_map_role_labels)
        self.radio_h.toggled.connect(self._update_map_role_labels)
        cfg_layout.addWidget(self.radio_v)
        cfg_layout.addSpacing(20)
        cfg_layout.addWidget(self.radio_h)
        cfg_layout.addStretch()

        # ── Seção 2.1: Parâmetros de Busca por Borda ────────────────────────
        adv_row = QHBoxLayout()
        self.chk_advanced = QCheckBox("Mostrar parâmetros avançados de busca")
        self.chk_advanced.setChecked(False)
        adv_row.addWidget(self.chk_advanced)
        adv_row.addStretch()
        layout.addLayout(adv_row)

        grp_search = QGroupBox("2.1 Parâmetros de Busca de Sobreposição (por faixas de borda)")
        search_layout = QHBoxLayout(grp_search)

        def _mk_int_edit(val):
            le = QLineEdit(str(val))
            le.setFixedWidth(70)
            le.setAlignment(Qt.AlignCenter)
            return le

        search_layout.addWidget(QLabel("Linhas iniciais:"))
        self.edge_start_edit = _mk_int_edit(10)
        search_layout.addWidget(self.edge_start_edit)

        search_layout.addSpacing(8)
        search_layout.addWidget(QLabel("Máx.:"))
        self.edge_max_edit = _mk_int_edit(300)
        search_layout.addWidget(self.edge_max_edit)

        search_layout.addSpacing(8)
        search_layout.addWidget(QLabel("Passo:"))
        self.edge_step_edit = _mk_int_edit(1)
        search_layout.addWidget(self.edge_step_edit)

        search_layout.addSpacing(8)
        search_layout.addWidget(QLabel("dx máx.:"))
        self.edge_dxmax_edit = _mk_int_edit(300)
        search_layout.addWidget(self.edge_dxmax_edit)

        search_layout.addStretch()
        grp_search.setVisible(False)
        self.chk_advanced.toggled.connect(grp_search.setVisible)
        layout.addWidget(grp_cfg)
        layout.addWidget(grp_search)

        # ── Botão Analisar ────────────────────────────────────────────────────
        self.btn_analyze = QPushButton("Analisar Sobreposição e Qualidade")
        self.btn_analyze.setFixedHeight(40)
        self.btn_analyze.clicked.connect(self._start_analysis)
        layout.addWidget(self.btn_analyze)

        # ── Seção 3: Resultados da análise ────────────────────────────────────
        self.grp_results = QGroupBox("3. Resultados da Análise")
        self.grp_results.setVisible(False)
        res_layout = QVBoxLayout(self.grp_results)

        # Painel de métricas compacto
        self.lbl_metrics = QLabel()
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setStyleSheet(
            "background:#eef4fb; border-left:4px solid #2e7d9e; "
            "padding:10px 14px; border-radius:4px; color:#1a3a5c; font-size:13px; font-family:'Consolas','Courier New',monospace;"
        )
        res_layout.addWidget(self.lbl_metrics)

        # Sugestão
        self.lbl_suggestion = QLabel()
        self.lbl_suggestion.setWordWrap(True)
        self.lbl_suggestion.setStyleSheet(
            "background:#e8f5e9; border-left:4px solid #1b7a4e; "
            "padding:10px 14px; border-radius:4px; color:#1a3a5c; font-size:13px;"
        )
        res_layout.addWidget(self.lbl_suggestion)

        # Seleção manual
        sel_row = QHBoxLayout()
        sel_lbl = QLabel("Mapa a manter na sobreposição:")
        sel_lbl.setObjectName("section")
        sel_row.addWidget(sel_lbl)
        self.combo_map = QComboBox()
        self.combo_map.addItem("Mapa 1 (sugerido pelo sistema)", "map1")
        self.combo_map.addItem("Mapa 2", "map2")
        sel_row.addWidget(self.combo_map)
        sel_row.addStretch()
        res_layout.addLayout(sel_row)

        layout.addWidget(self.grp_results)

        # ── Seção 4: Saída ────────────────────────────────────────────────────
        self.grp_output = QGroupBox("4. Arquivos de Saída")
        self.grp_output.setVisible(False)
        out_layout = QVBoxLayout(self.grp_output)

        # HDF5
        hdf_row = QHBoxLayout()
        hdf_lbl = QLabel("HDF5 de saída:")
        hdf_lbl.setObjectName("section")
        hdf_lbl.setFixedWidth(130)
        hdf_row.addWidget(hdf_lbl)
        self.hdf5_path = QLabel("Não definido")
        self.hdf5_path.setObjectName("file_path")
        self.hdf5_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        hdf_row.addWidget(self.hdf5_path)
        btn_hdf = QPushButton("Definir...")
        btn_hdf.setObjectName("secondary")
        btn_hdf.setFixedWidth(90)
        btn_hdf.clicked.connect(self._choose_hdf5_output)
        hdf_row.addWidget(btn_hdf)
        out_layout.addLayout(hdf_row)

        # Relatório (automático, mesmo nome e pasta do HDF5)
        rep_row = QHBoxLayout()
        rep_lbl = QLabel("Relatório HTML:")
        rep_lbl.setObjectName("section")
        rep_lbl.setFixedWidth(130)
        rep_row.addWidget(rep_lbl)
        self.report_path = QLabel("Será salvo automaticamente com o mesmo nome do HDF5")
        self.report_path.setObjectName("file_path")
        self.report_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rep_row.addWidget(self.report_path)
        out_layout.addLayout(rep_row)

        layout.addWidget(self.grp_output)

        # ── Botão Costurar ────────────────────────────────────────────────────
        self.btn_stitch = QPushButton("Normalizar e Costurar")
        self.btn_stitch.setObjectName("success")
        self.btn_stitch.setFixedHeight(42)
        self.btn_stitch.setVisible(False)
        self.btn_stitch.clicked.connect(self._start_stitch)
        layout.addWidget(self.btn_stitch)

        # ── Progresso / Status ───────────────────────────────────────────────
        self.status_card = QFrame()
        self.status_card.setObjectName("statusCard")
        status_layout = QHBoxLayout(self.status_card)
        status_layout.setContentsMargins(14, 12, 14, 12)
        status_layout.setSpacing(12)

        self.status_icon = QLabel("•")
        self.status_icon.setObjectName("statusIcon")
        self.status_icon.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_icon)

        status_text_layout = QVBoxLayout()
        status_text_layout.setSpacing(2)
        self.status_title = QLabel("Pronto para iniciar")
        self.status_title.setObjectName("statusTitle")
        self.status_hint = QLabel("Escolha os arquivos, revise a direção e execute a análise.")
        self.status_hint.setObjectName("statusHint")
        self.status_hint.setWordWrap(True)
        status_text_layout.addWidget(self.status_title)
        status_text_layout.addWidget(self.status_hint)
        status_layout.addLayout(status_text_layout, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #5a7a9a; font-size: 12px;")
        self.progress_label.setVisible(False)
        layout.addWidget(self.status_card)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)

        # ── Log interno (sem janela de informações visível) ───────────────────
        self._log_lines = []

        # ── Botões de resultado ───────────────────────────────────────────────
        self.result_row = QHBoxLayout()
        self.btn_open_hdf5 = QPushButton("Abrir pasta do HDF5")
        self.btn_open_hdf5.setObjectName("secondary")
        self.btn_open_hdf5.setVisible(False)
        self.btn_open_hdf5.clicked.connect(self._open_hdf5_folder)
        self.btn_open_report = QPushButton("Abrir Relatório no Navegador")
        self.btn_open_report.setObjectName("secondary")
        self.btn_open_report.setVisible(False)
        self.btn_open_report.clicked.connect(self._open_report)
        self.btn_new = QPushButton("Nova Costura")
        self.btn_new.setVisible(False)
        self.btn_new.clicked.connect(self._reset)
        self.result_row.addWidget(self.btn_open_hdf5)
        self.result_row.addWidget(self.btn_open_report)
        self.result_row.addStretch()
        self.result_row.addWidget(self.btn_new)
        layout.addLayout(self.result_row)

        self._set_stage("Arquivos", "Selecione os quadrantes e mantenha a direção de costura correta.")
        self._set_status("Pronto para iniciar", "As mudanças visuais não alteram o algoritmo nem a velocidade da costura.", busy=False, icon="•")
        layout.addStretch()
        return w

    # ── Aba Sobre ─────────────────────────────────────────────────────────────

    def _build_about_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)

        grp = QGroupBox("Sobre o XRF Data Stitcher")
        g_lay = QVBoxLayout(grp)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setStyleSheet("background:#fff; color:#1a2a3a; border:none; font-size:13px;")
        txt.setHtml("""
<h3 style="color:#1a3a5c">XRF Data Stitcher - vBeta</h3>
<p>Ferramenta para costura inteligente de mapas de fluorescência de raios-X por mapeamento (MA-XRF)
obtidos por varreduras de MA-XRF em quadrantes.</p>

<h4 style="color:#1a3a5c; margin-top:12px">Fluxo de Processamento</h4>
<ol style="margin-left:16px; line-height:1.8">
  <li><b>Seleção dos arquivos HDF5</b> — vertical: Mapa 1 (topo) e Mapa 2 (base); horizontal: Mapa 1 (esquerda) e Mapa 2 (direita)</li>
  <li><b>Detecção automática de sobreposição</b> — correlação cruzada das imagens integradas (soma sobre Z)</li>
  <li><b>Análise de qualidade</b> — SNR, correlação espectral e razão de intensidade na região sobreposta</li>
  <li><b>Seleção do mapa</b> — sugestão automática (maior SNR) com opção de sobrescrever</li>
  <li><b>Normalização por contagem total</b> — I_norm = I / Σ_z I</li>
  <li><b>Costura em chunks</b> — processamento eficiente em memória (CHUNK_ROWS linhas por vez)</li>
  <li><b>Saída HDF5</b> — compatível com PyMca (/cube, /x_axis, /y_axis, /energy, /metadata)</li>
  <li><b>Relatório HTML</b> — com métricas, justificativas metodológicas e parâmetros do processamento</li>
</ol>

<h4 style="color:#1a3a5c; margin-top:12px">Dependências</h4>
<p style="font-family:monospace; background:#f1f5f9; padding:8px; border-radius:4px;">
pip install PyQt5 h5py numpy scipy
</p>

<h4 style="color:#1a3a5c; margin-top:12px">Estrutura do HDF5 de Saída</h4>
<ul style="margin-left:16px; line-height:1.8; font-family:monospace;">
  <li>/cube — (Y × X × Z) uint32 por padrão, compressão GZIP-4; pode ser configurado para float32</li>
  <li>/x_axis — posições em X (pixels)</li>
  <li>/y_axis — posições em Y (pixels)</li>
  <li>/energy — índice de canais de energia</li>
  <li>/metadata — atributos do processamento</li>
</ul>

<h4 style="color:#1a3a5c; margin-top:12px">Costura em Cascata</h4>
<p>Para obras com mais de 2 quadrantes, basta usar o HDF5 gerado como entrada para uma nova costura,
repetindo o processo quantas vezes forem necessárias.</p>
        """)
        g_lay.addWidget(txt)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    # ── Ações ─────────────────────────────────────────────────────────────────

    def _set_stage(self, active_name, hint=None):
        active_idx = self._stage_order.index(active_name) if active_name in self._stage_order else -1
        for idx, name in enumerate(self._stage_order):
            lbl = self._step_labels[name]
            if idx < active_idx:
                lbl.setObjectName("stepPillDone")
            elif idx == active_idx:
                lbl.setObjectName("stepPillActive")
            else:
                lbl.setObjectName("stepPillBase")
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)
            lbl.update()
        if hint is not None:
            self.flow_hint.setText(hint)

    def _tick_spinner(self):
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self.status_icon.setText(self._spinner_frames[self._spinner_index])

    def _update_map_role_labels(self):
        if self.radio_h.isChecked():
            self.map1_sel.set_label("Mapa 1 — Esquerda (parte esquerda da obra)")
            self.map2_sel.set_label("Mapa 2 — Direita (parte direita da obra)")
            self.flow_hint.setText("Na costura horizontal, carregue o Mapa 1 à esquerda e o Mapa 2 à direita antes da análise automática.")
        else:
            self.map1_sel.set_label("Mapa 1 — Topo (parte superior da obra)")
            self.map2_sel.set_label("Mapa 2 — Base (parte inferior da obra)")
            self.flow_hint.setText("Na costura vertical, carregue o Mapa 1 no topo e o Mapa 2 na base antes da análise automática.")

    def _set_status(self, title, hint, busy=False, icon=None):
        self.status_title.setText(title)
        self.status_hint.setText(hint)
        if busy:
            self._spinner_index = 0
            self.status_icon.setText(self._spinner_frames[self._spinner_index])
            if not self._spinner_timer.isActive():
                self._spinner_timer.start(140)
        else:
            if self._spinner_timer.isActive():
                self._spinner_timer.stop()
            self.status_icon.setText(icon or "✓")

    def _selected_elements(self):
        """Método mantido por compatibilidade. Registro e normalização sempre por total."""
        return []


    def _start_analysis(self):
        if not self.map1_sel.path or not self.map2_sel.path:
            QMessageBox.warning(self, "Arquivos ausentes", "Selecione os dois arquivos HDF5 antes de analisar.")
            return

        direction = "vertical" if self.radio_v.isChecked() else "horizontal"
        self.btn_analyze.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self._log_lines = []
        self.grp_results.setVisible(False)
        self.grp_output.setVisible(False)
        self.btn_stitch.setVisible(False)
        self._set_stage("Análise", "Analisando automaticamente a sobreposição e a qualidade dos quadrantes.")
        self._set_status("Processando análise", "Estimando overlap, deslocamentos e métricas de qualidade.", busy=True)

        self._thread = ProcessingThread(
            mode="analyze",
            map1_path=self.map1_sel.path,
            map2_path=self.map2_sel.path,
            direction=direction,
            reg_method="total",
            norm_method="total",
            elements_selected=[],
            overlap_search="edge",
            edge_start=int(self.edge_start_edit.text() or 10),
            edge_max=int(self.edge_max_edit.text() or 300),
            edge_step=int(self.edge_step_edit.text() or 1),
            edge_dx_max=int(self.edge_dxmax_edit.text() or 300),
        )
        self._thread.log.connect(self._append_log)
        self._thread.progress.connect(self._update_progress)
        self._thread.analysis_done.connect(self._on_analysis_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_analysis_done(self, result):
        self._analysis_result = result
        q = result["quality"]
        ov = result["overlap"]

        sc = q.get("spectral_correlation")
        sc_str = f"{sc:.4f}" if sc else "N/A"
        shift_dx = ov.get('shift_dx', 0)
        shift_dy = ov.get('shift_dy', 0)
        self.lbl_metrics.setText(
            f"<b>Sobreposição:</b> {ov['overlap_pixels']} px &nbsp;&nbsp; "
            f"<b>Score NCC:</b> {ov['best_correlation']:.4f} &nbsp;&nbsp; "
            f"<b>dx:</b> {shift_dx} px &nbsp;&nbsp; <b>dy:</b> {shift_dy} px<br>"
            f"<b>SNR Mapa 1:</b> {q['map1_snr']:.3f} &nbsp;&nbsp; "
            f"<b>SNR Mapa 2:</b> {q['map2_snr']:.3f} &nbsp;&nbsp; "
            f"<b>Corr. espectral:</b> {sc_str}"
        )

        sug = "Mapa 1" if q["suggested_map"] == "map1" else "Mapa 2"
        self.lbl_suggestion.setText(
            f"<b>Sugestão do sistema:</b> Manter <b>{sug}</b> na região de sobreposição.<br>"
            f"<i>{q['suggestion_reason']}</i>"
        )

        # Combo
        self.combo_map.clear()
        if q["suggested_map"] == "map1":
            self.combo_map.addItem("Mapa 1 (sugerido pelo sistema)", "map1")
            self.combo_map.addItem("Mapa 2", "map2")
        else:
            self.combo_map.addItem("Mapa 2 (sugerido pelo sistema)", "map2")
            self.combo_map.addItem("Mapa 1", "map1")

        self.grp_results.setVisible(True)
        self.grp_output.setVisible(True)
        self.btn_stitch.setVisible(True)
        self.btn_analyze.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._set_stage("Saída", "Revise a sugestão do sistema e confirme o arquivo final antes da costura.")
        self._set_status("Análise concluída", "Sobreposição detectada e caminho de saída pronto para a costura.", busy=False, icon="✓")

        # Sugerir caminhos de saída
        base = Path(self.map1_sel.path).parent
        stem = Path(self.map1_sel.path).stem
        self._hdf5_out = str(base / f"{stem}_stitched.h5")
        self.hdf5_path.setText(self._hdf5_out)
        self._sync_report_output()

    def _start_stitch(self):
        if not hasattr(self, "_hdf5_out"):
            QMessageBox.warning(self, "Saída não definida", "Defina o arquivo HDF5 de saída antes de costurar.")
            return

        self._sync_report_output()

        direction = "vertical" if self.radio_v.isChecked() else "horizontal"
        selected_map = self.combo_map.currentData()

        self.btn_stitch.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.btn_open_hdf5.setVisible(False)
        self.btn_open_report.setVisible(False)
        self.btn_new.setVisible(False)
        self._set_stage("Costura", "Aplicando a normalização escolhida e executando a costura final.")
        self._set_status("Costurando dados", "Processamento em andamento com a mesma lógica de costura já validada.", busy=True)

        self._thread = ProcessingThread(
            mode="stitch",
            map1_path=self.map1_sel.path,
            map2_path=self.map2_sel.path,
            direction=direction,
            selected_map=selected_map,
            output_hdf5=self._hdf5_out,
            output_report=self._report_out,
            map1_name=Path(self.map1_sel.path).name,
            map2_name=Path(self.map2_sel.path).name,
            reg_method="total",
            norm_method="total",
            elements_selected=[],
            overlap_search="edge",
            edge_start=int(self.edge_start_edit.text() or 10),
            edge_max=int(self.edge_max_edit.text() or 300),
            edge_step=int(self.edge_step_edit.text() or 1),
            edge_dx_max=int(self.edge_dxmax_edit.text() or 300),
        )
        self._thread.log.connect(self._append_log)
        self._thread.progress.connect(self._update_progress)
        self._thread.stitch_done.connect(self._on_stitch_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_stitch_done(self, result):
        self.btn_stitch.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._set_stage("Costura", "Processo finalizado com sucesso.")
        self._set_status("Costura concluída", "HDF5 e relatório salvos na mesma pasta.", busy=False, icon="✓")
        self.btn_open_hdf5.setVisible(True)
        self.btn_open_report.setVisible(True)
        self.btn_new.setVisible(True)
        self._stitch_result = result

        shape = result["stitched_shape"]
        QMessageBox.information(
            self, "Costura concluída!",
            f"Processamento finalizado com sucesso.\n\n"
            f"Cubo costurado: {shape[0]} × {shape[1]} × {shape[2]}\n"
            f"HDF5: {result['output_hdf5']}\n"
            f"Relatório: {result['output_report']}"
        )

    def _on_error(self, msg):
        self.btn_analyze.setEnabled(True)
        self.btn_stitch.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._set_status("Erro no processamento", "O algoritmo foi interrompido antes de concluir. Revise os arquivos e tente novamente.", busy=False, icon="!")
        self._append_log(f"[ERRO] {msg}")
        QMessageBox.critical(self, "Erro no processamento", msg[:500])

    def _update_progress(self, value, label):
        self.progress_bar.setValue(value)
        self.progress_label.setText(label)
        if value < 100:
            self.status_hint.setText(label)

    def _append_log(self, text):
        self._log_lines.append(text)

    def _sync_report_output(self):
        hdf5_out = getattr(self, "_hdf5_out", "")
        if not hdf5_out:
            return
        report_out = str(Path(hdf5_out).with_suffix(".html"))
        self._report_out = report_out
        self.report_path.setText(report_out)

    def _choose_hdf5_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar HDF5 costurado", "", "Arquivos HDF5 (*.h5 *.hdf5)"
        )
        if path:
            self._hdf5_out = path
            self.hdf5_path.setText(path)
            self._sync_report_output()

    def _open_hdf5_folder(self):
        if hasattr(self, "_stitch_result"):
            folder = str(Path(self._stitch_result["output_hdf5"]).parent)
            import subprocess, platform
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", folder])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])

    def _open_report(self):
        if hasattr(self, "_stitch_result"):
            import webbrowser
            webbrowser.open(f"file://{self._stitch_result['output_report']}")

    def _reset(self):
        self.map1_sel.path_label.setText("Nenhum arquivo selecionado")
        self.map1_sel._path = ""
        self.map2_sel.path_label.setText("Nenhum arquivo selecionado")
        self.map2_sel._path = ""
        self.grp_results.setVisible(False)
        self.grp_output.setVisible(False)
        self.btn_stitch.setVisible(False)
        self.btn_open_hdf5.setVisible(False)
        self.btn_open_report.setVisible(False)
        self.btn_new.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._log_lines = []
        self._analysis_result = None
        self._set_stage("Arquivos", "Selecione novos quadrantes para iniciar outra costura.")
        self._set_status("Pronto para iniciar", "Escolha os arquivos, revise a direção e execute a análise.", busy=False, icon="•")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("XRF Data Stitcher - vBeta")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
