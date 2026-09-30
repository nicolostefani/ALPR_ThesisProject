"""
******************************************************************+
RILEVAMENTO E LETTURA TARGHE 
******************************************************************+
Il seguente codice permette di leggere le targhe da video, utilizzando un approccio a 
pipeline che combina il rilevamento delle targhe con ASRED, la segmentazione dei bordi 
e l'OCR con DeepPlate.
La segmentazione dei bordi è stata implementata utilizzando il metodo Canny, la chiusura morfologica
e la ricerca dei contorni; il quadrilatero preciso della targa viene ricostruito dalla fila di 
caratteri rilevati, sfruttando le proporzioni della targa italiana.

PIPELINE:
    video -> [1] ASRED -> [2] ritagli -> [3] SEGMENTAZIONE A BORDI -> [4] DeepPlate -> [5] Consolidamento su finestra di fotogrammi -> risultati

REQUISITI PER L'ESECUZIONE DEL CODICE:
    - Python 3 con i pacchetti opencv-python (cv2) e numpy installati.
    - ASRED: asred.exe + lo script createHash.py
    - DeepPlate: deepPlate_thesis.exe
    - Solo per la modalita' gridsearch: un file di annotazioni in formato CVAT XML
      (ANNOTATIONS_XML) con i box veri delle targhe, per calcolare l'IoU da massimizzare.

INPUT ATTESI
    - BASE_VIDEO_FOLDER: una cartella con i video .mp4 da elaborare

MODALITA' DI ESECUZIONE:
    python Pipeline.py           esegue la catena su tutti i video
    python Pipeline.py gridsearch         cerca i parametri che massimizzano l'IoU

ELENCO DEI PARAMETRI PRINCIPALI:
    - BASE_OUTPUT_SEGMENTATION  cartella dove finiscono i risultati di una corsa
    - ESEGUI_ASRED              False per riusare data.json/ritagli gia' prodotti, senza
                                rilanciare ASRED 
    - ESEGUI_OCR                False per saltare DeepPlate e valutare solo la segmentazione
    - RIESTRAI_RITAGLI          True per rifare sempre frame e ritagli, anche se gia' presenti
    - FINESTRA_FRAME            quanti fotogrammi consolidare per ogni lettura OCR
"""
import argparse
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ================================================================================================
# CONFIGURAZIONE 
# ================================================================================================
BASE_VIDEO_FOLDER = r"E:\DatasetNuovo\Video4k"
BASE_ASRED_FOLDER = r"E:\DatasetNuovo\pipeline4k\Asred"
BASE_FRAMES_FOLDER = r"E:\DatasetNuovo\pipeline4k\Frame_video"

BASE_OUTPUT_SEGMENTATION = r"E:\DatasetNuovo\TESI\output" # Cartella <output> dei risultati della pipeline di segmentazione

# Percorso agli strumenti esterni: ASRED e DeepPlate. 
STRUMENTI_FOLDER = r"C:\Users\root\Desktop\programmi"
SCRIPT_HASH_PATH = os.path.join(STRUMENTI_FOLDER, "createHash.py")
ASRED_FOLDER = os.path.join(STRUMENTI_FOLDER, "asred")
ASRED_EXE_PATH = os.path.join(ASRED_FOLDER, "asred.exe")
DEEPPLATE_EXE_PATH = os.path.join(STRUMENTI_FOLDER, "deepPlate_thesis", "deepPlate_thesis.exe")

ANNOTATIONS_XML = r"E:\DatasetNuovo\annotations.xml" # File di annotazioni in formato CVAT XML per la grid search
GRIDSEARCH_OUTPUT = r"E:\DatasetNuovo\gridsearch_bordi" # Cartella dove salvare i risultati della grid search
CAMPIONE_GRIDSEARCH_N = 800    # numero massimo di ritagli da usare come campione nella grid search
PASSATE_GRIDSEARCH = 2         # quante passate di coordinate-descent fare sulla griglia

ESTENSIONI_VIDEO_VALIDE = ('.mp4',)


MARGINE_RITAGLIO = 0.25 # margine attorno al box ASRED

ESEGUI_ASRED = True    # False = salta ASRED e riusa data.json/ritagli gia' presenti su disco
ESEGUI_OCR = True      # False = salta DeepPlate, utile per valutare solo la segmentazione
RIESTRAI_RITAGLI = False  # True = rifa' sempre frame e ritagli, anche se gia' presenti su disco

FINESTRA_FRAME = 10 # Fotogrammi per consolidare ogni lettura OCR

# Parametri del segmentatore (include valori prodotti della grid search)
PARAMETRI = dict(
    scala=0, # attiva l'ingrandimento adattivo
    altezza_target_px=90, # altezza target della targa dopo l'ingrandimento
    raffina_caratteri=True, # restringe il quadrilatero attorno alla baseline dei caratteri, invece che al contorno chiuso
    solo_rettangoli=True, # usa solo i rettangoli ruotati e non approxPolyDP
    candidati_da_caratteri=True,
    # Proporzioni della targa italiana standard
    rapp_altezza_targa=1.72,    
    rapp_larghezza_targa=1.24,   
    rapp_larg_da_carattere=11.82,  
    rapp_passo_carattere=1.409,  
    angolo_da_baseline=True, # calcola l'angolo della targa dalla baseline dei caratteri, invece che dal contorno chiuso
    max_angolo_baseline=25.0,      # oltre: non e' inclinazione, e' un allineamento sbagliato
    # Se le due stime di larghezza discordano di piu' di cosi', vince quella dal glifo.
    max_discordanza_larghezza=0.35,
    passi_raffinazione=2, # numero di passate per la ricostruzione della targa, per affinare la baseline dei caratteri
    max_lato_vs_seed=1.35, # il lato lungo ricostruito non può eccedere troppo le dimensioni trovate dal detector
    clahe_clip=5.0, # limite di clipper per il filtro CLAHE, necessario per fare emergere i bordi dei caratteri in condizioni di illuminazione difficile
    usa_bilateral=False,         # non attivo in quanto non porta benefici ma ne allunga i tempi di esecuzione
    canny_sigmas=(0.20, 0.33, 0.50), # tre soglie di Canny
    close_w_ratio=0.12, # larghezza del kernel di chiusura
    close_h_ratio=0.03, # altezza del kernel di chiusura
    close_iter=2, 
    min_aspect=1.05,             # dai percentili delle targhe annotate
    max_aspect=5.00,
    min_area_vs_seed=0.08,
    max_area_vs_seed=1.30, # limite diverso da quello proposto dalla grid search (2.20) per evitare ritagli troppo grandi
    min_solidita=0.50,
    peso_aspect=0.80, peso_solid=0.20, peso_area=0.40, peso_bordo=0.00, # pesi per la scelta del candidato migliore
    aspect_atteso=2.70,
    peso_testo=1.40, min_punteggio_testo=0.22,
    # filtri sui colori caldi
    warm_frac_hard=0.18,
    bright_warm_frac_hard=0.22,
)

# La griglia di parametri da esplorare nella grid search. I valori sono stati scelti per
# mantenere la pipeline stabile ma ampi da permettere di trovare un ottimo locale. (la ricerca 
# parte dal valore centrale di ogni lista)
GRIGLIA = {
    "scala":             [4, 5, 6],
    "close_iter":        [2, 3, 4],
    "close_w_ratio":     [0.12, 0.20, 0.30, 0.40],
    "close_h_ratio":     [0.02, 0.03, 0.05],
    "clahe_clip":        [3.0, 5.0, 8.0],
    "min_area_vs_seed":  [0.08, 0.15, 0.30],
    "max_area_vs_seed":  [1.2, 1.6, 2.2],
    "min_solidita":      [0.35, 0.50, 0.70],
    "peso_aspect":       [0.80, 1.20, 1.80],
    "peso_area":         [0.15, 0.25, 0.40],
    "peso_bordo":        [0.00, 0.03],
    "peso_testo":        [0.90, 1.40, 2.00],
}

# Alfabeto targhe italiane: 22 lettere, escluse I, O, Q, U
LETTERE_VALIDE = set("ABCDEFGHJKLMNPRSTVWXYZ")
CIFRE_VALIDE = set("0123456789")
FORMATO_TARGA_IT = "LLDDDLL"


# ================================================================================================
# ASRED, FRAME E RITAGLI
# ================================================================================================
def filtra_json_targhe(file_input, file_output):
    """Funzione che filtra il JSON prodotto da ASRED tenendo solo le rilevazioni con
    object_class == 'license_plate', sia fra le detection che fra i tracking."""
    try:
        with open(file_input, 'r', encoding='utf-8') as f:
            dati = json.load(f)
        nuovi = [{"idx_frame": fr.get("idx_frame"),
                  "detection": [d for d in fr.get('detection', [])
                                if d.get('object_class') == 'license_plate'],
                  "tracking": [t for t in fr.get('tracking', [])
                               if t.get('object_class') == 'license_plate']}
                 for fr in dati.get('frames', [])]
        with open(file_output, 'w', encoding='utf-8') as f:
            json.dump({"frames": nuovi}, f, indent=4)
        logger.info("  [+] Filtraggio JSON completato.")
    except Exception as e:
        logger.error(f"  [-] Errore nel filtraggio JSON: {e}")


def estrai_frame_e_ritagli(video_path, json_path, crop_dir, frames_dir,
                           margin_ratio=None):
    """Funzione che salva i frame interi e i ritagli espansi del 25% attorno al box ASRED,
    e assegna a ogni ritaglio il track_id della targa a cui appartiene.

    Il margine serve a lasciare, attorno alla targa, un po' di sfondo da cui i bordi si stacchino;
    il box originale del detector viene memorizzato in coordinate locali e usato come riferimento
    per scegliere il contorno giusto.
    """

    if margin_ratio is None:
        margin_ratio = MARGINE_RITAGLIO
    logger.info(f"  [*] Estrazione frame e ritagli (margine {margin_ratio:g})...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dati = json.load(f)
    except FileNotFoundError:
        logger.error(f"  [-] JSON non trovato: {json_path}")
        return {}
    per_frame = {fr.get("idx_frame"): fr.get("detection", []) for fr in dati.get("frames", [])}
    
    # Associazione per sovrapposizione degli ID di tracking delle targhe per fotogramma
    per_track = {fr.get("idx_frame"): [t for t in fr.get("tracking", [])
                                       if t.get("object_class") == "license_plate"]
                 for fr in dati.get("frames", [])}

    def _iou_box(a, b):
        # Funzione che calcola l'IoU fra due box (x1, y1, x2, y2), usata per associare
        # detection e tracking e per agganciare le targhe orfane ai fotogrammi vicini.
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / u if u > 0 else 0.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"  [-] Impossibile aprire il video: {video_path}")
        return {}
    offsets, idx = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:05d}.jpg"), frame)
        for det_i, det in enumerate(per_frame.get(idx) or []):
            bb = det.get("bounding_box")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = (int(float(v)) for v in bb)
            H, W = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            bw, bh = x2 - x1, y2 - y1
            mx, my = int(bw * margin_ratio), int(bh * margin_ratio)
            ex1, ey1 = max(0, x1 - mx), max(0, y1 - my)
            ex2, ey2 = min(W, x2 + mx), min(H, y2 + my)
            nome = f"frame_{idx:05d}_crop_{det_i}.jpg"
            cv2.imwrite(os.path.join(crop_dir, nome), frame[ey1:ey2, ex1:ex2])
            # La detection prende lo stesso IDE della traccia dello stesso fogramma con IoU > 0.35 altrimenti resta -1
            tid, best = -1, 0.35
            for tk in per_track.get(idx) or []:
                tb = tk.get("bounding_box")
                if tb and len(tb) == 4:
                    v = _iou_box([x1, y1, x2, y2], [float(z) for z in tb])
                    if v > best:
                        best, tid = v, int(tk.get("id", -1))
            offsets[nome] = {"frame_idx": str(idx), "offset_x": ex1, "offset_y": ey1,
                             "asred_local_rect": [x1 - ex1, y1 - ey1, bw, bh],
                             "track_id": tid, "abs_box": [x1, y1, bw, bh]}
        idx += 1
    cap.release()
    # Assegnare un nuovo id progressivo a quelle orfane
    orfani = [k for k, v in offsets.items() if v["track_id"] < 0]
    if orfani:
        per_f = defaultdict(list)
        for k, v in offsets.items():
            per_f[int(v["frame_idx"])].append(k)
        nuovo = 9_000_000 # ID progressivo per le targhe orfane, per distinguerle da quelle tracciate
        for k in sorted(orfani, key=lambda n: int(offsets[n]["frame_idx"])):
            fi = int(offsets[k]["frame_idx"])
            ab = offsets[k]["abs_box"]
            b1 = [ab[0], ab[1], ab[0] + ab[2], ab[1] + ab[3]]
            trovato = -1
            for dd in (-1, -2, -3, 1, 2, 3):
                for altro in per_f.get(fi + dd, []):
                    tv = offsets[altro]["track_id"]
                    if tv < 0:
                        continue
                    ob = offsets[altro]["abs_box"]
                    if _iou_box(b1, [ob[0], ob[1], ob[0]+ob[2], ob[1]+ob[3]]) >= 0.30:
                        trovato = tv
                        break
                if trovato >= 0:
                    break
            offsets[k]["track_id"] = trovato if trovato >= 0 else nuovo
            if trovato < 0:
                nuovo += 1
    n_tr = len({v["track_id"] for v in offsets.values()})
    logger.info(f"  [+] {len(offsets)} ritagli estratti, {n_tr} tracce di targa.")
    return offsets


# ================================================================================================
# SEGMENTAZIONE A BORDI
# ================================================================================================
class SegmentatoreBordi:
    """
    Classe che trova il quadrilatero della targa dentro ogni ritaglio, partendo dal box di ASRED e ne esegue l'OCR con DeepPlate. 
    """
    def __init__(self, bb_output_dir: str, rectified_dir: Optional[str] = None,
                 deepplate_exe: Optional[str] = None, dp_output_dir: Optional[str] = None,
                 deepplate_country: str = "it", deepplate_timeout: int = 60,
                 scala: int = 3, clahe_clip: float = 2.0, usa_bilateral: bool = True,
                 canny_sigmas: Tuple[float, ...] = (0.20, 0.33, 0.50),
                 close_w_ratio: float = 0.25, close_h_ratio: float = 0.10, close_iter: int = 2,
                 min_aspect: float = 1.05, max_aspect: float = 5.0,
                 min_area_vs_seed: float = 0.20, max_area_vs_seed: float = 1.60,
                 min_solidita: float = 0.55, min_area_px: int = 60,
                 peso_aspect: float = 0.30, peso_solid: float = 0.20,
                 peso_area: float = 0.25, peso_bordo: float = 0.25,
                 peso_testo: float = 0.60, min_punteggio_testo: float = 0.22,
                 aspect_atteso: float = 2.70,
                 altezza_target_px: int = 90,
                 raffina_caratteri: bool = False,
                 solo_rettangoli: bool = True,
                 candidati_da_caratteri: bool = True,
                 rapp_altezza_targa: float = 1.72,
                 rapp_larghezza_targa: float = 1.24,
                 rapp_larg_da_carattere: float = 11.82,
                 rapp_passo_carattere: float = 1.409,
                 angolo_da_baseline: bool = True,
                 max_angolo_baseline: float = 25.0,
                 max_discordanza_larghezza: float = 0.35,
                 passi_raffinazione: int = 2,
                 max_lato_vs_seed: float = 1.35,
                 warm_frac_hard: float = 0.18, bright_warm_frac_hard: float = 0.22,
                 finestra_frame: int = FINESTRA_FRAME):
        """Funzione che inizializza il segmentatore con tutti i parametri del metodo, crea le
        cartelle di output necessarie e prepara il contatore diagnostico degli scarti."""
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        for d in (self.dp_output_dir, self.rectified_dir):
            if d:
                os.makedirs(d, exist_ok=True)
        # Il contatore è un dizionario, con chiave il motivo dello scarto e valore il numero di ritagli scartati per quel motivo
        self.diag_scarti = defaultdict(int)

    @staticmethod
    def _clip_rect(rect, shape):
        """Funzione che ritaglia un rettangolo (x, y, w, h) dentro i confini di un'immagine."""
        H, W = shape[:2]
        x, y, w, h = (int(v) for v in rect)
        x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
        return x, y, max(1, min(w, W - x)), max(1, min(h, H - y))

    @staticmethod
    def _ordina(pts: np.ndarray) -> np.ndarray:
        """Funzione che ordina i 4 vertici di un quadrilatero in senso TL, TR, BR, BL."""
        r = np.zeros((4, 2), np.float32)
        s, d = pts.sum(axis=1), np.diff(pts, axis=1)
        r[0], r[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        r[1], r[3] = pts[np.argmin(d)], pts[np.argmax(d)]
        return r

    @staticmethod
    def _rect_to_quad(rect) -> np.ndarray:
        """Funzione che converte un rettangolo (x, y, w, h) in un quadrilatero di 4 punti."""
        x, y, w, h = rect
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)

    def _valido(self, quad) -> bool:
        """Funzione che verifica se un quadrilatero e' valido: 4 vertici distinti, convesso e
        con area di almeno self.min_area_px."""
        if quad is None:
            return False
        p = np.asarray(quad, np.float32)
        if p.shape != (4, 2) or not np.all(np.isfinite(p)):
            return False
        for i in range(4):
            if float(np.linalg.norm(p[(i + 1) % 4] - p[i])) < 3.0:
                return False
        c = p.reshape(-1, 1, 2)
        if abs(float(cv2.contourArea(c))) < self.min_area_px:
            return False
        return bool(cv2.isContourConvex(c.astype(np.int32)))

    @staticmethod
    def _metriche(quad) -> Dict[str, float]:
        """Funzione che calcola aspect, solidita' e angolo di un quadrilatero, misurati sul
        rettangolo ruotato che lo racchiude."""
        (_, _), (w, h), ang = cv2.minAreaRect(quad.astype(np.float32))
        if w < 1 or h < 1:
            return {"aspect": 0.0, "solidita": 0.0, "angolo": 0.0}
        area = abs(float(cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32))))
        if w < h:
            w, h, ang = h, w, ang + 90.0
        while ang > 45:
            ang -= 90
        while ang < -45:
            ang += 90
        return {"aspect": w / h, "solidita": area / (w * h), "angolo": float(ang)}

    # preprocessing con ingrandimento 
    def _fattore_scala(self, seed):
        """Funzione che decide di quanto ingrandire il ritaglio.

        Con scala > 0 il fattore e' fisso. Con scala = 0 diventa adattivo: si sceglie
        il fattore che porta l'altezza della targa vicino ad altezza_target_px.
        Lasciare il valore impostato a 0 per far si che il fattore di scala si adatti automaticamente alla dimensione della targa.
        """
        if self.scala and self.scala > 0:
            return max(1, int(self.scala))
        h = max(1, int(seed[3]))
        return int(np.clip(round(self.altezza_target_px / h), 1, 10))

    def _prepara(self, image, seed):
        """Funzione che ingrandisce il ritaglio, lo converte in grigi, equalizza localmente con CLAHE e
        attenua il rumore con il filtro bilaterale.
        
        Ritorna l'immagine ingrandita, la versione in grigio, il box seed scalato e il fattore di scala usato.
        """
        s = self._fattore_scala(seed)
        grande = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(grande, cv2.COLOR_BGR2GRAY)
        if self.clahe_clip > 0:
            gray = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8)).apply(gray)
        if self.usa_bilateral:
            gray = cv2.bilateralFilter(gray, 7, 60, 60)
        seed_s = tuple(int(v * s) for v in seed)
        return grande, gray, seed_s, s

    # Canny a soglie multiple
    def _mappe_bordi(self, gray, seed_s) -> List[np.ndarray]:
        """Funzione che calcola le mappe dei bordi di Canny a tre soglie diverse, calcolate
        dentro il box ASRED."""
        x, y, w, h = self._clip_rect(seed_s, gray.shape)
        dentro = gray[y:y + h, x:x + w]
        # la mediana si calcola sui soli pixel della targa, non sul ritaglio che include il
        # margine del 25% con carrozzeria e sfondo
        med = float(np.median(dentro)) if dentro.size else float(np.median(gray))
        mappe = []
        for sigma in self.canny_sigmas:
            lo = int(max(0, (1.0 - sigma) * med))
            hi = int(min(255, (1.0 + sigma) * med))
            if hi <= lo:
                hi = lo + 1
            mappe.append(cv2.Canny(gray, lo, hi))
        return mappe

    # Chiusura orizzontale proporzionale 
    def _chiudi(self, edges, plate_h):
        """Funzione che chiude morfologicamente i bordi con un kernel orizzontale proporzionale
        all'altezza della targa, per fondere i bordi dei singoli caratteri in un'unica macchia."""
        # close_iter = 0 indica nessuna chiusura. 
        if self.close_iter <= 0:
            return edges
        kw = max(1, int(plate_h * self.close_w_ratio) | 1)
        kh = max(1, int(plate_h * self.close_h_ratio) | 1)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=int(self.close_iter))

    # Contorno scelto per sovrapposizione col box ASRED 
    def _contorno_migliore(self, chiuso, seed_s):
        """Funzione che sceglie, tra i contorni esterni della maschera chiusa, quello il cui
        rettangolo circoscritto copre la frazione maggiore del box di ASRED, invece di quello
        con l'area maggiore.
        Scarta i contorni con area minore di min_area_px e quelli che non intersecano il box.
        Restituisce None se nessun contorno e' valido."""
        contorni, _ = cv2.findContours(chiuso, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contorni:
            return None
        x, y, w, h = seed_s
        box_seed = (x, y, x + w, y + h)
        migliore, best = None, 0.0
        for c in contorni:
            if cv2.contourArea(c) < self.min_area_px:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            ix1, iy1 = max(box_seed[0], bx), max(box_seed[1], by)
            ix2, iy2 = min(box_seed[2], bx + bw), min(box_seed[3], by + bh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                continue
            # Punteggio = area di intersezione / area del box di ASRED. 
            punteggio = inter / float(w * h)
            if punteggio > best:
                best, migliore = punteggio, c
        return migliore

    # Candidati quadrilatero dal contorno 
    def _quads_da_contorno(self, contorno) -> List[Tuple[np.ndarray, str]]:
        """Funzione che genera i candidati quadrilatero a partire da un contorno.
        Con solo_rettangoli=True (default) l'unico candidato e' il rettangolo ruotato di
        minAreaRect.

        Se solo_rettangoli=False, si prova anche a generare un quadrilatero con approxPolyDP, con tre valori di epsilon diversi. 
        Il primo che produce un quadrilatero valido viene restituito.
        """
        out = []
        try:
            box = cv2.boxPoints(cv2.minAreaRect(contorno)).astype(np.float32)
            q = self._ordina(box)
            if self._valido(q):
                out.append((q, "minarea"))
        except cv2.error:
            pass
        if self.solo_rettangoli:
            return out
        hull = cv2.convexHull(contorno)
        peri = cv2.arcLength(hull, True)
        if peri > 0:
            for coef in (0.02, 0.04, 0.06):
                ap = cv2.approxPolyDP(hull, coef * peri, True)
                if len(ap) == 4:
                    q = self._ordina(ap.reshape(4, 2).astype(np.float32))
                    if self._valido(q):
                        out.append((q, "approx"))
                    break
        return out

    def _rettangolizza(self, quad):
        """Funzione che riporta un quadrilatero qualsiasi al rettangolo ruotato che lo racchiude."""
        if not self.solo_rettangoli:
            return quad
        try:
            box = cv2.boxPoints(cv2.minAreaRect(np.asarray(quad, np.float32))).astype(np.float32)
            return self._ordina(box)
        except cv2.error:
            return quad

    # Candidati quadrilatero dai soli caratteri
    def _quad_da_caratteri_diretto(self, gray_s, seed_s):
        """Funzione che costruisce un quadrilatero partendo solo dai caratteri, senza usare Canny o la chiusura morfologica. 
        """
        x, y, w, h = self._clip_rect(seed_s, gray_s.shape)
        if w < 16 or h < 8:
            return None
        # Il quadrilatero di partenza e' il box di ASRED, dopo la prima passata viene sostituito dal quadrilatero ricostruito sui caratteri.
        q = self._rect_to_quad((x, y, w, h)).astype(np.float32)
        # Dopo la prima passata, il box di ASRED viene sostituito dal quadrilatero ricostruito sui caratteri. La seconda passata riparte da
        # un quadrilatero già ruotato.
        buono = None
        for _ in range(max(1, self.passi_raffinazione)):
            qr = self._raffina_su_caratteri(gray_s, q, seed_s)
            if qr is None or not self._valido(qr):
                break
            q = buono = qr
        if buono is None:
            self.diag_scarti['candidato diretto: caratteri non trovati'] += 1
            return None
        return buono

    # Filtro sui colori caldi
    def _frazioni_calde(self, image, quad, seed) -> Dict[str, float]:
        """Funzione che misura la frazione di pixel caldi (rosso e ambra) dentro il quadrilatero 
        e il box di ASRED, per scartare i ritagli FP."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv, np.array([0, 70, 60], np.uint8), np.array([10, 255, 255], np.uint8))
        r2 = cv2.inRange(hsv, np.array([168, 70, 60], np.uint8), np.array([179, 255, 255], np.uint8))
        amb = cv2.inRange(hsv, np.array([11, 90, 60], np.uint8), np.array([38, 255, 255], np.uint8))
        caldo = cv2.bitwise_or(cv2.bitwise_or(r1, r2), amb)
        poly = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(poly, [quad.astype(np.int32)], 255)
        n = max(1, cv2.countNonZero(poly))
        wp = cv2.countNonZero(cv2.bitwise_and(caldo, poly)) / n
        x, y, w, h = self._clip_rect(seed, image.shape)
        roi = caldo[y:y + h, x:x + w]
        wb = (cv2.countNonZero(roi) / max(1, roi.size)) if roi.size else 0.0
        bgr = image.astype(np.int16)
        rb = bgr[:, :, 2] - bgr[:, :, 0]
        bright = ((image.max(axis=2) >= 185) & (rb >= 22)).astype(np.uint8) * 255
        bp = cv2.countNonZero(cv2.bitwise_and(bright, poly)) / n
        broi = bright[y:y + h, x:x + w]
        bb = (cv2.countNonZero(broi) / max(1, broi.size)) if broi.size else 0.0
        return {"warm": float(max(wp, wb)), "bright_warm": float(max(bp, bb))}

    # Punteggio di quanto il quadrilatero somiglia a una riga di testo
    def _punteggio_testo(self, gray_s, quad_s) -> Dict[str, float]:
        """Funzione che misura quanto il contenuto del quadrilatero somiglia a una riga di testo.

        Questa funzione viene aggiunta perchè i filtri sui colori caldi non bastano a scartare i falsi positivi. 
        Una targa è una riga di caratteri, individuabili da blob della stessa altezza, allineati sulla stessa base e distribuiti in orizzontale.

        Punteggio (fra 0 e 1) = 0.35 * n_blob + 0.30 * allineamento delle basi + 0.20 * uniformità delle altezze + 0.15 * estensione orizzioanle
        Con meno di 3 blob il punteggio è 0.
        """
        vuoto = {"punteggio": 0.0, "n_char": 0}
        rett = self.rectify(gray_s, quad_s)
        if rett is None or rett.size == 0:
            return vuoto
        if rett.ndim == 3:
            rett = cv2.cvtColor(rett, cv2.COLOR_BGR2GRAY)
        if rett.shape[0] < 8 or rett.shape[1] < 16:
            return vuoto
        try:
            sc = 40.0 / rett.shape[0]
            rett = cv2.resize(rett, (max(16, int(rett.shape[1] * sc)), 40), interpolation=cv2.INTER_CUBIC)
            # binarizzazione locale
            bw = cv2.adaptiveThreshold(rett, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 8)
        except cv2.error:
            return vuoto
        n_lab, _, st_, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        H, W = rett.shape[:2]
        comp = []
        for i in range(1, n_lab):
            bwid, bhei = st_[i, cv2.CC_STAT_WIDTH], st_[i, cv2.CC_STAT_HEIGHT]
            if not (0.25 * H <= bhei <= 0.95 * H):     # un carattere occupa buona parte dell'altezza
                continue
            if bwid > bhei * 1.6 or bwid < 1:          
                continue
            comp.append((st_[i, cv2.CC_STAT_LEFT] + bwid / 2.0, bhei,
                         st_[i, cv2.CC_STAT_TOP] + bhei))
        if len(comp) < 3:
            return {"punteggio": 0.0, "n_char": len(comp)}
        alt = np.array([c[1] for c in comp], float)
        basi = np.array([c[2] for c in comp], float)
        cx = np.array([c[0] for c in comp], float)
        unif = max(0.0, 1.0 - float(np.std(alt) / max(1e-6, np.mean(alt))) / 0.35)
        allin = max(0.0, 1.0 - float(np.std(basi) / max(1e-6, np.mean(alt))) / 0.30)
        est = min(1.0, float((cx.max() - cx.min()) / max(1e-6, W)) / 0.6)
        n_sc = min(1.0, (len(comp) - 2) / 5.0)
        return {"punteggio": float(0.35 * n_sc + 0.30 * allin + 0.20 * unif + 0.15 * est),
                "n_char": len(comp)}

    # Raffinamento del quadrilatero a partire dai soli caratteri 
    def _raffina_su_caratteri(self, gray_s, quad_s, seed_s=None):
        """Funzione che ricostruisce il quadrilatero della targa a partire dalla fila di caratteri.

        Il contorno trovato con i bordi spesso include anche portatarga o paraurti. I
        caratteri invece si trovano in modo abbastanza affidabile, e le misure della targa
        italiana sono fisse. Quindi, misurati i caratteri, la targa si puo' ricostruire con le proporzioni:
            altezza targa   = 1.72 x altezza carattere      (110/64)
            larghezza targa = 1.24 x larghezza della fila   (520/416)
        Usare la larghezza della fila, e non un margine fisso, permette di includere anche
        le bande blu laterali, che DeepPlate si aspetta di vedere.

        Le misure si fanno sull'immagine rettificata.

        Restituisce None se i caratteri non si trovano: in quel caso il quadrilatero resta com'è."""
        rett = self.rectify(gray_s, quad_s)
        if rett is None or rett.size == 0 or rett.shape[0] < 8 or rett.shape[1] < 16:
            return None
        if rett.ndim == 3:
            rett = cv2.cvtColor(rett, cv2.COLOR_BGR2GRAY)
        h0, w0 = rett.shape[:2]
        try:
            sc = 40.0 / h0
            r40 = cv2.resize(rett, (max(16, int(w0 * sc)), 40), interpolation=cv2.INTER_CUBIC)
            bw = cv2.adaptiveThreshold(r40, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 8)
        except cv2.error:
            return None
        n_lab, _, st_, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        H, W = r40.shape[:2]
        # Tengo tutte le macchie che hanno la forma di un carattere, senza chiedere che
        # siano alte quanto la striscia: se il quadrilatero di partenza e' troppo alto
        # i caratteri ne occupano solo una piccola parte
        grezzi = []
        for i in range(1, n_lab):
            bwid, bhei = st_[i, cv2.CC_STAT_WIDTH], st_[i, cv2.CC_STAT_HEIGHT]
            if bhei < 4 or bhei > 0.95 * H or bwid < 1 or bwid > bhei * 1.6:
                continue
            grezzi.append((st_[i, cv2.CC_STAT_LEFT], st_[i, cv2.CC_STAT_TOP], bwid, bhei))
        if len(grezzi) < 3:
            return None

        # Filtro prima per altezza e poi per posizione della base i caratteri.
        med_h = float(np.median([c[3] for c in grezzi]))
        comp = [c for c in grezzi if 0.70 * med_h <= c[3] <= 1.40 * med_h]
        if len(comp) < 3:
            return None
        med_b = float(np.median([c[1] + c[3] for c in comp]))
        comp = [c for c in comp if abs((c[1] + c[3]) - med_b) <= 0.40 * med_h]
        if len(comp) < 3:
            return None

        # Rettangolo che contiene tutti i caratteri trovati
        x0 = min(c[0] for c in comp); x1 = max(c[0] + c[2] for c in comp)
        y0 = min(c[1] for c in comp); y1 = max(c[1] + c[3] for c in comp)

        # Inclinazione, retta che passa per le basi dei caratteri. La targa trovata, senza l'inclinazione avrebbe 
        # l'orientamento del quadrilatero di partenza. 
        m_base = 0.0
        if self.angolo_da_baseline and len(comp) >= 3:
            xs = np.array([c[0] + c[2] / 2.0 for c in comp], float)
            ys = np.array([c[1] + c[3] for c in comp], float)   # il PIEDE di ogni carattere
            if float(xs.max() - xs.min()) > 1e-6:
                try:
                    m_base = float(np.polyfit(xs, ys, 1)[0])
                except (np.linalg.LinAlgError, ValueError):
                    m_base = 0.0
        # Se l'inclinazione è eccessiva significa che i blob sono allineati mali, viene scartato
        lim = math.tan(math.radians(self.max_angolo_baseline))
        if abs(m_base) > lim:
            self.diag_scarti['baseline troppo inclinata: ignorata'] += 1
            m_base = 0.0
        # L'altezza si prende dalla mediana dei caratteri, un blob spurio
        # piu' alto degli altri allungherebbe la targa ricostruita, mentre la mediana lo ignora.
        hc = float(np.median([c[3] for c in comp]))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

        alt = hc * self.rapp_altezza_targa

        # Larghezza: la stimo in due modi
        #  (a) dalla larghezza della fila di caratteri, riportata a 7 caratteri  -> 1.24 x fila
        #  (b) dalla larghezza di un singolo carattere                           -> 11.82 x carattere
        # La (a) e' piu' precisa, la (b) e' piu' grossolana ma non dipende da quanti caratteri
        # sono stati trovati.
        wc = float(np.median([c[2] for c in comp]))

        # Stimo la larghezza che avrebbe la fila completa di 7 caratteri.
        # Dalla distanza fra il primo e l'ultimo carattere trovato ricavo quanti passi
        # ci sono in mezzo, poi riporto la fila a 6 passi. Se mancano caratteri al centro
        # la fila risulta gia' completa e non viene allungata.
        passo = max(1e-6, wc * self.rapp_passo_carattere)
        n_int = int(np.clip(round((x1 - x0 - wc) / passo), 1, 6))
        campata_piena = (x1 - x0 - wc) * (6.0 / n_int) + wc

        larg_campata = campata_piena * self.rapp_larghezza_targa
        larg_carattere = wc * self.rapp_larg_da_carattere
        if n_int < 6:
            self.diag_scarti[f'ricostruzione: campata su {n_int+1} posizioni di 7'] += 1

        # La stima per glifo resta come seconda scelta
        disc = (abs(larg_campata - larg_carattere) / larg_carattere
                if larg_carattere > 0 else 0.0)
        if disc > self.max_discordanza_larghezza:
            self.diag_scarti['ricostruzione: campata e glifo discordi'] += 1
            ordine = [larg_carattere, larg_campata]
        else:
            ordine = [larg_campata, larg_carattere]

        if alt < 4:
            return None

        # Trasformazione inversa della rettifica: serve per riportare il rettangolo
        # ricostruito nelle coordinate dell'immagine
        dst = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], np.float32)
        try:
            M = cv2.getPerspectiveTransform(dst, quad_s.astype(np.float32))
        except cv2.error:
            return None

        # Il centro verticale si prende sulla retta di base all'ascissa del centro, non
        # dall'inviluppo dei blob
        q_base = float(np.median([c[1] + c[3] - m_base * (c[0] + c[2] / 2.0) for c in comp]))
        cy = (m_base * cx + q_base) - hc / 2.0
        ct, st_ = math.cos(math.atan(m_base)), math.sin(math.atan(m_base))
        R_base = np.array([[ct, -st_], [st_, ct]], np.float32)

        # provo le due larghezze nell'ordine scelto
        for larg in ordine:
            if larg < 8:
                continue
            # Rettangolo della targa centrato in (cx, cy) e ruotato, riportato prima alla
            # scala del rettificato (/sc) e poi all'immagine (M)
            base = np.array([[-larg / 2, -alt / 2], [larg / 2, -alt / 2],
                             [larg / 2, alt / 2], [-larg / 2, alt / 2]], np.float32)
            ang = ((base @ R_base.T) + np.array([cx, cy], np.float32)) / sc
            try:
                nuovo = cv2.perspectiveTransform(ang.reshape(-1, 1, 2), M).reshape(4, 2)
            except cv2.error:
                continue
            # Scarto il risultato se il lato lungo e' molto piu' grande della larghezza
            # del box di ASRED
            if seed_s is not None and self.max_lato_vs_seed > 0:
                (_c, (rw, rh), _a) = cv2.minAreaRect(nuovo.astype(np.float32))
                if max(rw, rh) > seed_s[2] * self.max_lato_vs_seed:
                    self.diag_scarti['ricostruzione oltre il box di ASRED'] += 1
                    continue
            return self._ordina(nuovo.astype(np.float32))
        return None

    # Calcolo del punteggio di un quadrilatero candidato
    def _punteggio(self, gray_s, quad_s, seed_s, chiuso) -> float:
        """Funzione che assegna un punteggio a un candidato quadrilatero, combinando aspect,
        solidita', area, aderenza ai bordi e punteggio del testo."""
        m = self._metriche(quad_s)
        if not (self.min_aspect <= m["aspect"] <= self.max_aspect):
            # Aspect ratio fuori dai limiti
            self.diag_scarti['aspect troppo grande (contorno allungato)'
                             if m["aspect"] > self.max_aspect
                             else 'aspect troppo piccolo (frammento)'] += 1
            return -1.0
        ar = max(0.0, 1.0 - abs(m["aspect"] - self.aspect_atteso) / self.aspect_atteso)
        sol = min(1.0, max(0.0, m["solidita"]))
        if sol < self.min_solidita:
            self.diag_scarti['solidita troppo bassa'] += 1
            return -1.0
        area = abs(float(cv2.contourArea(quad_s.reshape(-1, 1, 2).astype(np.float32))))
        sa = float(seed_s[2] * seed_s[3])
        rap = area / sa if sa > 0 else 0.0
        if not (self.min_area_vs_seed <= rap <= self.max_area_vs_seed):
            self.diag_scarti['area troppo grande' if rap > self.max_area_vs_seed
                             else 'area troppo piccola'] += 1
            return -1.0
        area_score = max(0.0, 1.0 - abs(rap - 0.90) / 0.90)
        # Il contorno chiuso è prodotto dalla chiusura morfologica.
        bordo = np.zeros(chiuso.shape, np.uint8)
        cv2.polylines(bordo, [quad_s.astype(np.int32)], True, 255, 2)
        n_bordo = max(1, cv2.countNonZero(bordo))
        aderenza = cv2.countNonZero(cv2.bitwise_and(bordo, chiuso)) / n_bordo
        testo = self._punteggio_testo(gray_s, quad_s)["punteggio"]
        return (self.peso_aspect * ar + self.peso_solid * sol
                + self.peso_area * area_score + self.peso_bordo * aderenza
                + self.peso_testo * testo)

    # ---------- segmentazione di un ritaglio ----------
    def segment(self, image, seed_rect) -> Dict[str, Any]:
        """Funzione che segmenta un singolo ritaglio: genera i candidati quadrilatero, li valuta
        e restituisce il migliore, oppure lo scarta se nessuno supera i vincoli."""
        seed = self._clip_rect(seed_rect, image.shape)
        grande, gray_s, seed_s, s = self._prepara(image, seed)
        plate_h = max(4, seed_s[3])

        migliore, punteggio_migliore, metodo = None, -1.0, "nessuno"
        contorni_trovati = candidati_generati = 0
        for i, edges in enumerate(self._mappe_bordi(gray_s, seed_s)):
            chiuso = self._chiudi(edges, plate_h)
            cont = self._contorno_migliore(chiuso, seed_s)
            if cont is None:
                continue
            contorni_trovati += 1
            for q, tipo in self._quads_da_contorno(cont):
                candidati_generati += 1
                if self.raffina_caratteri:
                    # Raffinazione iterativa: il primo quadrilatero e' costruito sul contorno, ma
                    # ingloba sempre qualcosa oltre alla targa. La raffinazione sui
                    # caratteri restituisce un quadrilatero piu' aderente alla targa,
                    # e si itera per catturare anche l'inclinazione della targa, che il primo quadrilatero non ha.
                    for _ in range(max(1, self.passi_raffinazione)):
                        qr = self._raffina_su_caratteri(gray_s, q, seed_s)
                        if qr is None or not self._valido(qr):
                            break
                        q, tipo = qr, (tipo if tipo.endswith("R") else tipo + "R")
                q = self._rettangolizza(q)
                p = self._punteggio(gray_s, q, seed_s, chiuso)
                if p > punteggio_migliore:
                    punteggio_migliore, migliore, metodo = p, q, f"{tipo}_s{i}"

        # Candidato costruito sui soli caratteri, senza chiusura
        if self.candidati_da_caratteri:
            qd = self._quad_da_caratteri_diretto(gray_s, seed_s)
            if qd is not None:
                candidati_generati += 1
                qd = self._rettangolizza(qd)
                vuoto = np.zeros(gray_s.shape, np.uint8)
                p = self._punteggio(gray_s, qd, seed_s, vuoto)
                if p > punteggio_migliore:
                    punteggio_migliore, migliore, metodo = p, qd, "caratteri"

        if migliore is None or punteggio_migliore < 0:
            if contorni_trovati == 0:
                self.diag_scarti['nessun contorno da findContours'] += 1
            elif candidati_generati == 0:
                self.diag_scarti['contorno trovato, nessun quadrilatero valido'] += 1
            else:
                self.diag_scarti['candidati generati ma tutti esclusi'] += 1
            # scarto per nessun contorno
            quad = self._rect_to_quad(seed)
            return {"status": "rejected", "points": quad, "method": "nessun_contorno",
                    "confidence": 0.0, "aspect_ratio": 0.0, "rotation_deg": 0.0,
                    "solidita": 0.0, "warm_frac": 0.0, "punteggio_testo": 0.0, "n_char": 0,
                    "hard_reject": "nessun_contorno"}

        quad = (migliore / s).astype(np.float32)
        quad[:, 0] = np.clip(quad[:, 0], 0, image.shape[1] - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, image.shape[0] - 1)
        if not self._valido(quad):
            # scarto per geometria invalida
            quad = self._rect_to_quad(seed)
            return {"status": "rejected", "points": quad, "method": "geometria_invalida",
                    "confidence": 0.0, "aspect_ratio": 0.0, "rotation_deg": 0.0,
                    "solidita": 0.0, "warm_frac": 0.0, "punteggio_testo": 0.0, "n_char": 0,
                    "hard_reject": "geometria_invalida"}
        m = self._metriche(quad)
        caldi = self._frazioni_calde(image, quad, seed)
        # il controllo sui caratteri si misura sull'immagine ingrandita, dove i glifi sono
        # risolvibili anche per le targhe piccole
        quad_s = (quad * s).astype(np.float32)
        testo = self._punteggio_testo(gray_s, quad_s)

        hard = None
        if caldi["warm"] > self.warm_frac_hard:
            hard = "colore_caldo"
        elif caldi["bright_warm"] > self.bright_warm_frac_hard:
            hard = "luce_calda"
        elif testo["punteggio"] < self.min_punteggio_testo:
            # NESSUNA FILA DI CARATTERI
            hard = "nessun_testo"

        return {"status": "rejected" if hard else "ok", "points": quad, "method": metodo,
                "confidence": float(min(1.0, max(0.0, punteggio_migliore))),
                "aspect_ratio": float(m["aspect"]), "rotation_deg": float(m["angolo"]),
                "solidita": float(m["solidita"]), "warm_frac": float(caldi["warm"]),
                "punteggio_testo": float(testo["punteggio"]), "n_char": int(testo["n_char"]),
                "hard_reject": hard}

    # Rettifica usata dal punteggio testo, dalla ricostruzione sui caratteri e per salvare le immagini rettificate
    def rectify(self, image, quad):
        """Funzione che rettifica un quadrilatero in un'immagine rettangolare."""
        tl, tr, br, bl = quad.astype(np.float32)
        w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
        if w < 8 or h < 4:
            return None
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        try:
            return cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad.astype(np.float32), dst), (w, h))
        except cv2.error:
            return None

    # Overlay elementi per analisi
    def overlay(self, image, pts, seed, testo, colore, larghezza=320):
        """Funzione che disegna il box di ASRED, il poligono trovato e una riga di testo
        diagnostico sotto il ritaglio."""
        H, W = image.shape[:2]
        s = max(1, int(round(larghezza / max(1, W))))
        canvas = cv2.resize(image, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
        banda = 30
        out = np.full((canvas.shape[0] + banda, canvas.shape[1], 3), 30, np.uint8)
        out[:canvas.shape[0]] = canvas
        th = max(1, s // 2)
        x, y, w, h = seed
        cv2.rectangle(out, (x * s, y * s), ((x + w) * s, (y + h) * s), (0, 200, 255), th)
        cv2.polylines(out, [(pts * s).astype(np.int32)], True, colore, th)
        cv2.putText(out, testo[:56], (4, canvas.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, colore, 1, cv2.LINE_AA)
        return out

    # OCR 
    def deepplate(self, image_path, quad) -> Dict[str, Any]:
        """Funzione che esegue DeepPlate sul ritaglio, legge la tabella di probabilita' che
        produce e restituisce il testo decodificato."""
        vuoto = {"ok": False, "status": "ocr_disabilitato", "text": None, "text_raw": None,
                 "conf_media": 0.0, "tabella": None}
        if not self.deepplate_exe:
            return vuoto
        flat = [str(int(round(v))) for p in quad for v in p]
        out_dir = self.dp_output_dir or self.bb_output_dir
        cmd = [self.deepplate_exe, "--input_path", image_path, "--output_dir", out_dir,
               "--country", self.deepplate_country, "--pos"] + flat
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                           timeout=self.deepplate_timeout)
        except subprocess.CalledProcessError:
            return {**vuoto, "status": "errore_deepplate"}
        except FileNotFoundError:
            return {**vuoto, "status": "exe_non_trovato"}
        except subprocess.TimeoutExpired:
            return {**vuoto, "status": "timeout"}
        stem = Path(image_path).stem
        tab = Path(out_dir) / "tables" / f"{stem}.txt"
        if not tab.is_file():
            alt = list(Path(out_dir).rglob(f"{stem}.txt"))
            tab = alt[0] if alt else None
        if tab is None:
            return {**vuoto, "ok": True, "status": "tabella_mancante"}
        pos = leggi_tabella(tab)
        if not pos:
            return {**vuoto, "ok": True, "status": "tabella_illeggibile", "tabella": str(tab)}
        v, g = decodifica(pos, True), decodifica(pos, False)
        return {"ok": True, "status": "ok" if v["testo"] else "nessun_testo",
                "text": v["testo"], "text_raw": g["testo"],
                "conf_media": round(v["conf_media"], 4), "tabella": str(tab)}

    # Elaborazione di una cartella 
    def process_folder(self, input_folder, crop_offsets, output_json_path, extensions=None):
        """Funzione che elabora tutti i ritagli di una cartella."""
        extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        files = sorted({p for e in extensions for p in Path(input_folder).glob(f"*{e}")})
        if not files:
            return
        boxes, rejected, record = {}, [], []
        n_ok = n_rej = n_ocr = 0
        diag_h, diag_s, motivi = [], [], defaultdict(int)
        logger.info(f"  [*] Segmentazione di {len(files)} ritagli...")
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            info = crop_offsets.get(f.name)
            seed = tuple(info["asred_local_rect"]) if info and "asred_local_rect" in info \
                else (0, 0, img.shape[1], img.shape[0])
            seed = self._clip_rect(seed, img.shape)
            diag_h.append(seed[3])
            diag_s.append(self._fattore_scala(seed))
            r = self.segment(img, seed)
            if r["status"] == "rejected":
                n_rej += 1
                motivi[r["hard_reject"] or "?"] += 1
                txt = (f"[SCARTATA] {r['hard_reject']} testo={r.get('punteggio_testo',0):.2f} "
                       f"nch={r.get('n_char',0)}")
                cv2.imwrite(os.path.join(self.bb_output_dir, f"{f.stem}_SCARTATA.jpg"),
                            self.overlay(img, r["points"], seed, txt, (0, 0, 255)))
                poly = frame_i = None
                if info is not None:
                    poly = (r["points"] + np.array([info["offset_x"], info["offset_y"]])).astype(np.int32).tolist()
                    frame_i = info["frame_idx"]
                rejected.append({"crop_name": f.name, "frame_idx": frame_i, "polygon": poly,
                                 "hard_reject": r["hard_reject"], "neg_signals": [],
                                 "warm_frac": round(r["warm_frac"], 3),
                                 "punteggio_testo": round(r.get("punteggio_testo", 0), 3),
                                 "n_char": r.get("n_char", 0),
                                 "aspect_ratio": round(r["aspect_ratio"], 2)})
                continue
            n_ok += 1
            pts = r["points"]
            txt = (f"[OK] conf={r['confidence']:.2f} ar={r['aspect_ratio']:.2f} "
                   f"rot={r['rotation_deg']:.1f} testo={r.get('punteggio_testo',0):.2f}")
            cv2.imwrite(os.path.join(self.bb_output_dir, f"{f.stem}_annotated.jpg"),
                        self.overlay(img, pts, seed, txt, (0, 255, 0)))
            rect_name = None
            if self.rectified_dir:
                rt = self.rectify(img, pts)
                if rt is not None and rt.size:
                    rect_name = f"{f.stem}_rectified.jpg"
                    cv2.imwrite(os.path.join(self.rectified_dir, rect_name), rt)
            ocr = self.deepplate(str(f), pts)
            if ocr["ok"]:
                n_ocr += 1
            if info is not None:
                record.append({"info": info, "ocr": ocr, "crop": f.name,
                               "track": info.get("track_id", -1),
                               "frame": int(info["frame_idx"]),
                               "area": abs(float(cv2.contourArea(pts.reshape(-1, 1, 2))))})
                abs_pts = (pts + np.array([info["offset_x"], info["offset_y"]])).astype(np.int32)
                x, y, w, h = cv2.boundingRect(abs_pts)
                boxes.setdefault(info["frame_idx"], []).append({
                    "crop_name": f.name, "bounding_box": [int(x), int(y), int(x + w), int(y + h)],
                    "track_id": info.get("track_id", -1),
                    "polygon": abs_pts.tolist(), "rectified_image": rect_name,
                    "method": r["method"], "confidence": round(r["confidence"], 4),
                    "aspect_ratio": round(r["aspect_ratio"], 3),
                    "rotation_deg": round(r["rotation_deg"], 2),
                    "solidita": round(r["solidita"], 3), "recovered": False,
                    "punteggio_testo": round(r.get("punteggio_testo", 0), 3),
                    "n_char": r.get("n_char", 0),
                    "ocr_text": ocr["text"], "ocr_text_raw": ocr.get("text_raw"),
                    "ocr_conf_media": ocr.get("conf_media", 0.0),
                    "ocr_table": ocr.get("tabella"), "ocr_status": ocr["status"]})
        # Per ogni detection si aggregano le tabelle dei fotogrammi vicini della stessa traccia,
        # fino a FINESTRA_FRAME. 
        finestre = {}
        blocchi = []
        # Con finestra_frame = 1 la finestra e' il solo fotogramma corrente: la lettura non
        # cambia, ma la likelihood viene comunque calcolata dalla sua tabella.
        if self.finestra_frame >= 1 and record:
            per_traccia = defaultdict(list)
            for r in record:
                per_traccia[r["track"]].append(r)
            for tid, gruppo in per_traccia.items():
                gruppo.sort(key=lambda r: r["frame"])
                tabelle = [leggi_tabella(r["ocr"].get("tabella")) if r["ocr"].get("tabella") else None
                           for r in gruppo]
                pesi = [max(0.05, float(r["ocr"].get("conf_media") or 0.3)) * math.sqrt(max(1.0, r["area"]))
                        for r in gruppo]
                N = self.finestra_frame
                # finestra scorrevole centrata su ogni detection
                for i, r in enumerate(gruppo):
                    lo = max(0, i - N // 2)
                    hi = min(len(gruppo), lo + N)
                    lo = max(0, hi - N)
                    tb = [t for t in tabelle[lo:hi] if t]
                    pz = [pesi[j] for j in range(lo, hi) if tabelle[j]]
                    if not tb:
                        continue
                    cons = consolida_tabelle(tb, pesi=pz)
                    if cons and cons["testo"]:
                        finestre[r["crop"]] = cons
                # blocchi non sovrapposti per il riepilogo
                for lo in (range(0, len(gruppo), N) if N > 1 else ()):
                    fette = gruppo[lo:lo + N]
                    tb = [t for t in tabelle[lo:lo + N] if t]
                    pz = [pesi[lo + j] for j in range(len(fette)) if tabelle[lo + j]]
                    if not tb:
                        continue
                    cons = consolida_tabelle(tb, pesi=pz)
                    if cons and cons["testo"]:
                        blocchi.append({"track": tid, "f0": fette[0]["frame"], "f1": fette[-1]["frame"],
                                        "targa": cons["testo"], "likelihood": cons["likelihood"],
                                        "conf": cons["conf_media"], "n": cons["n_fotogrammi"]})
            n_cambiate = sum(1 for r in record
                             if r["crop"] in finestre and r["ocr"].get("text")
                             and finestre[r["crop"]]["testo"] != r["ocr"]["text"])
            if finestre and self.finestra_frame > 1:
                logger.info(f"  [*] Consolidamento su finestra di {self.finestra_frame} fotogrammi: "
                            f"{len(finestre)} letture, {n_cambiate} diverse dal singolo fotogramma")
            elif finestre:
                logger.info(f"  [*] Modalita' per-fotogramma: {len(finestre)} letture, "
                            f"nessun consolidamento")

        logger.info(f"  -> OK: {n_ok} | Scartate: {n_rej} | OCR riusciti: {n_ocr} | Totale: {len(files)}")
        if diag_h:
            logger.info(f"     altezza targa nei ritagli: mediana {np.median(diag_h):.0f} px "
                        f"(min {min(diag_h)}, max {max(diag_h)})  |  ingrandimento usato: "
                        f"mediana {np.median(diag_s):.0f}x  ->  targa a "
                        f"{np.median(diag_h)*np.median(diag_s):.0f} px")
        if self.diag_scarti:
            tot_d = sum(self.diag_scarti.values())
            logger.info("     perche' i candidati vengono esclusi (piu' cause per ritaglio):")
            for k, v in sorted(self.diag_scarti.items(), key=lambda kv: -kv[1]):
                logger.info(f"        {v:6d} ({v/tot_d:5.1%})  {k}")
        if motivi:
            logger.info("     motivi degli scarti: "
                        + ", ".join(f"{m} {c}" for m, c in
                                    sorted(motivi.items(), key=lambda kv: -kv[1])))
        if n_ok == 0 and n_rej:
            logger.warning("     NESSUN RITAGLIO CONSERVATO.")
        # la lettura consolidata diventa quella principale; quella del singolo fotogramma resta
        # accanto, cosi' il guadagno e' misurabile
        for voci in boxes.values():
            for v in voci:
                c = finestre.get(v["crop_name"])
                v["ocr_text_frame"] = v.get("ocr_text")
                if c:
                    v["ocr_text"] = c["testo"]
                    v["ocr_conf_media"] = round(c["conf_media"], 4)
                    v["ocr_likelihood"] = round(c["likelihood"], 6)
                    v["ocr_frame_finestra"] = c["n_fotogrammi"]
                else:
                    v["ocr_likelihood"] = 0.0
                    v["ocr_frame_finestra"] = 0

        base = Path(output_json_path).parent
        with open(output_json_path, 'w', encoding='utf-8') as fh:
            json.dump(boxes, fh, indent=4)
        with open(base / "rejected_log.json", 'w', encoding='utf-8') as fh:
            json.dump(rejected, fh, indent=4)
        with open(base / "risultati.csv", 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh, delimiter=';')
            w.writerow(["frame", "targa", "likelihood", "conf_ocr", "traccia", "frame_finestra",
                        "targa_singolo_frame", "x1", "y1", "x2", "y2",
                        "conf_segm", "metodo", "rotazione", "ritaglio", "poligono"])
            for fi in sorted(boxes, key=lambda k: int(k)):
                for v in boxes[fi]:
                    x1, y1, x2, y2 = v["bounding_box"]
                    w.writerow([fi, v.get("ocr_text") or "", v.get("ocr_likelihood", 0.0),
                                v.get("ocr_conf_media", 0.0), v.get("track_id", -1),
                                v.get("ocr_frame_finestra", 0), v.get("ocr_text_frame") or "",
                                x1, y1, x2, y2, v["confidence"], v["method"],
                                v["rotation_deg"], v["crop_name"],
                                " ".join(f"{a},{b}" for a, b in v["polygon"])])
        n_t = sum(1 for vv in boxes.values() for v in vv if v.get("ocr_text"))
        logger.info(f"  -> risultati.csv: {sum(len(v) for v in boxes.values())} rilevazioni, "
                    f"{n_t} con targa letta")
        if blocchi:
            with open(base / "targhe_finestre.csv", 'w', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh, delimiter=';')
                w.writerow(["traccia", "frame_da", "frame_a", "n_fotogrammi", "targa",
                            "likelihood", "conf_media"])
                for b in sorted(blocchi, key=lambda z: (z["track"], z["f0"])):
                    w.writerow([b["track"], b["f0"], b["f1"], b["n"], b["targa"],
                                round(b["likelihood"], 6), round(b["conf"], 4)])
            logger.info(f"  -> targhe_finestre.csv: {len(blocchi)} finestre da max "
                        f"{self.finestra_frame} fotogrammi")


# ================================================================================================
# TABELLE PRODOTTRE DA DEEPPLATE
# ================================================================================================
def leggi_tabella(path):
    """Funzione che legge la tabella di probabilita' scritta da DeepPlate per un ritaglio e la
    riordina per posizione, scartando la copia duplicata del blocco."""
    try:
        testo = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
    except OSError:
        return None
    righe = [r.strip() for r in testo.splitlines() if r.strip()]
    coppie = []
    for i in range(0, len(righe) - 1, 2):
        sim = [s.strip() for s in righe[i].split(';')]
        try:
            pr = [float(x) for x in righe[i + 1].split(';')]
        except ValueError:
            continue
        if sim and len(sim) == len(pr):
            coppie.append((sim, pr))
    if not coppie:
        return None
    n = len(coppie[0][0])
    blocco, prec = [], float('inf')
    for sim, pr in coppie:
        if len(sim) != n:
            continue
        if pr[0] > prec + 1e-9:      # la colonna riparte: inizio della copia duplicata
            break
        blocco.append((sim, pr))
        prec = pr[0]
    pos = [[] for _ in range(n)]
    for sim, pr in blocco:
        for p in range(n):
            pos[p].append((sim[p], pr[p]))
    for p in range(n):
        pos[p].sort(key=lambda t: -t[1])
    return pos or None


def aggrega(tabelle, pesi=None, vincolata=True, _solo_posizioni=False):
    """Funzione che fonde le tabelle di piu' fotogrammi della stessa targa in un'unica lettura."""
    tabelle = [t for t in tabelle if t]
    if not tabelle:
        return None
    pesi = list(pesi or [])[:len(tabelle)]
    while len(pesi) < len(tabelle):
        pesi.append(1.0)
    n_pos = max(len(t) for t in tabelle)
    accum = [dict() for _ in range(n_pos)]
    peso_tot = [0.0] * n_pos
    for tab, w in zip(tabelle, pesi):
        if w <= 0:
            continue
        for pp in range(min(n_pos, len(tab))):
            peso_tot[pp] += w
            for c, pr in tab[pp]:
                accum[pp][c] = accum[pp].get(c, 0.0) + w * pr
    posizioni = [sorted(((c, v / (peso_tot[pp] or 1.0)) for c, v in accum[pp].items()),
                        key=lambda t: -t[1]) for pp in range(n_pos)]
    if _solo_posizioni:                # usato da aggrega_voti come ripiego per una posizione
        return posizioni
    ris = decodifica(posizioni, vincolata=vincolata)
    ris['n_fotogrammi'] = len(tabelle)
    return ris


def aggrega_voti(tabelle, pesi=None, vincolata=True):
    """
    La seguente funzione fonde in un'unica lettura le tabelle di probabilità prodotte da
    DeepPlate per piu' fotogrammi, con un voto pesato. Al contrario di aggrega(), per ogni
    posizione si considera solo il primo candidato di ogni fotogramma, che vota con peso
    pari a (peso del fotogramma x probabilità con cui è stato espresso); i voti si dividono poi
    per la somma dei pesi dei fotogrammi. Il risultato non e' una distribuzione di
    probabilità (la somma per posizione puo' essere minore di 1).

    La funzione restituisce un dizionario contenente la targa decodificata (testo), la
    confidenza media e minima (conf_media, conf_min), la likelihood e il numero di fotogrammi
    considerati (n_fotogrammi)
    """
    tabelle = [t for t in tabelle if t]
    if not tabelle:
        return None
    pesi = list(pesi or [])[:len(tabelle)]
    while len(pesi) < len(tabelle):
        pesi.append(1.0)

    n_pos = max(len(t) for t in tabelle)
    voti = [dict() for _ in range(n_pos)]
    peso_tot = [0.0] * n_pos
    for tab, w in zip(tabelle, pesi):
        if w <= 0:
            continue
        for pp in range(min(n_pos, len(tab))):
            if not tab[pp]:
                continue
            # leggi_tabella ordina gia' per probabilita' decrescente: il primo è l'argmax
            c, pr = tab[pp][0]
            # Calcolo del voto pesato come somma dei voti pesati di ciascun fotogramma, normalizzato per il peso totale
            voti[pp][c] = voti[pp].get(c, 0.0) + w * float(pr) 
            peso_tot[pp] += w

    posizioni = [sorted(((c, v / (peso_tot[pp] or 1.0)) for c, v in voti[pp].items()),
                        key=lambda t: -t[1]) for pp in range(n_pos)]

    # In modalità vincolata, se per una posizione nessun carattere candidato rientra
    # nel set valido per il suo formato, viene sostituito con la
    # distribuzione media non vincolata di quella posizione.
    if vincolata:
        media = None
        for pp in range(n_pos):
            t = FORMATO_TARGA_IT[pp] if pp < len(FORMATO_TARGA_IT) else None
            if t is None:
                continue
            ok = LETTERE_VALIDE if t == 'L' else CIFRE_VALIDE
            if any(c in ok for c, _ in posizioni[pp]):
                continue
            if media is None:                      # calcolata una volta sola se necessario
                media = aggrega(tabelle, pesi=pesi, vincolata=False, _solo_posizioni=True)
            if media and pp < len(media):
                posizioni[pp] = media[pp]

    ris = decodifica(posizioni, vincolata=vincolata)
    ris['n_fotogrammi'] = len(tabelle)
    return ris


def consolida_tabelle(tabelle, pesi=None, vincolata=True):
    """Funzione che consolida le tabelle di piu' fotogrammi della stessa targa in un'unica lettura."""
    return aggrega_voti(tabelle, pesi=pesi, vincolata=vincolata)


def decodifica(pos, vincolata=True, formato=FORMATO_TARGA_IT):
    """Funzione che decodifica le distribuzioni di probabilita' in una stringa targa.
    vincolata=True sceglie, per ogni posizione, il candidato piu' probabile combatibile con
    LL-DDD-LL.
    """
    testo, probs = [], []
    for p, cand in enumerate(pos):
        if not cand:
            testo.append('_')
            probs.append(0.0)
            continue
        sc = None
        if vincolata and p < len(formato):
            t = formato[p]
            ok = LETTERE_VALIDE if t == 'L' else CIFRE_VALIDE
            sc = next(((c, pr) for c, pr in cand if c in ok), None)
        if sc is None:
            sc = cand[0]
        testo.append(sc[0])
        probs.append(float(sc[1]))
    like = 1.0
    for x in probs:
        like *= x
    return {'testo': ''.join(testo).replace('_', ''),
            'conf_media': (sum(probs) / len(probs)) if probs else 0.0,
            'conf_min': min(probs) if probs else 0.0,
            'likelihood': like}


# ================================================================================================
# GRID SEARCH
# ================================================================================================
def rotated_box_to_quad(xtl, ytl, xbr, ybr, rot):
    """Funzione che converte un box ruotato (xtl, ytl, xbr, ybr, rotazione) in un quadrilatero
    di 4 punti."""
    cx, cy = (xtl + xbr) / 2.0, (ytl + ybr) / 2.0
    w, h = xbr - xtl, ybr - ytl
    a = math.radians(rot or 0.0)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[cx + dx * ca - dy * sa, cy + dx * sa + dy * ca]
                     for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))],
                    np.float32)


def parse_cvat(xml_path):
    """Funzione che legge le annotazioni CVAT in XML e restituisce, per ogni video e fotogramma,
    i quadrilateri annotati a mano."""
    root = ET.parse(xml_path).getroot()
    tasks = {}
    for t in root.iter('task'):
        src = (t.findtext('source') or '') or (t.findtext('name') or '')
        tasks[int(t.findtext('id'))] = {'video': Path(src).stem, 'size': int(t.findtext('size') or 0)}
    off, cum = {}, 0
    for tid in sorted(tasks):
        off[tid] = cum
        cum += tasks[tid]['size']
    gt = defaultdict(lambda: defaultdict(list))
    for track in root.iter('track'):
        tid = int(track.get('task_id'))
        if tid not in tasks:
            continue
        for box in track.iter('box'):
            if box.get('outside') == '1': # Indica che la targa non è visibile nel fotogramma
                continue
            fl = int(box.get('frame')) - off[tid]
            if not (0 <= fl < tasks[tid]['size']): 
                continue
            gt[tasks[tid]['video']][fl].append(rotated_box_to_quad(
                float(box.get('xtl')), float(box.get('ytl')), float(box.get('xbr')),
                float(box.get('ybr')), float(box.get('rotation', 0) or 0)))
    return gt


def poly_iou(p1, p2) -> float:
    """Funzione che calcola l'IoU (intersection over union) fra due poligoni convessi."""
    a = np.asarray(p1, np.float32).reshape(-1, 2)
    b = np.asarray(p2, np.float32).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    aa, ab = abs(float(cv2.contourArea(a))), abs(float(cv2.contourArea(b)))
    if aa <= 0 or ab <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(cv2.convexHull(a), cv2.convexHull(b))
    un = aa + ab - float(inter)
    return float(inter) / un if un > 0 else 0.0


_CAMPIONE = None


def _init_worker(c):
    """Funzione che inizializza il campione condiviso in ogni processo worker della grid search."""
    global _CAMPIONE
    _CAMPIONE = c


def valuta_combo(params):
    """
    Funzione che valuta una combinazione di parametri sul campione, misurando l'IoU medio.
    """
    seg = SegmentatoreBordi(bb_output_dir=".", deepplate_exe=None, **params)
    ious, scartate = [], 0
    for v in _CAMPIONE:
        img = cv2.imread(v["path"])
        if img is None:
            continue
        r = seg.segment(img, seg._clip_rect(tuple(v["seed"]), img.shape))
        if r["status"] == "rejected":
            ious.append(0.0)
            scartate += 1
            continue
        q = r["points"] + np.array(v["offset"], np.float32)
        ious.append(poly_iou(q, np.array(v["gt"], np.float32)))
    if not ious:
        return {"params": params, "iou_media": 0.0, "cop05": 0.0, "n": 0, "scartate": 0.0}
    a = np.array(ious)
    return {"params": params, "n": len(a), "iou_media": float(a.mean()),
            "cop05": float((a >= 0.5).mean()), "cop07": float((a >= 0.7).mean()),
            "scartate": float(scartate / len(a))}


def costruisci_campione(annotations, base_asred, n_max, seed=20260804):
    """Funzione che costruisce il campione di ritagli con la relativa ground
    truth, da usare nella grid search."""
    gt = parse_cvat(annotations)
    campione = []
    for cart in sorted(d for d in Path(base_asred).iterdir() if d.is_dir()):
        chiave = next((k for k in gt if k.lower() == cart.name.lower()), None)
        f_off = cart / "crop_offsets.json"
        crop_dir = cart / f"{cart.name}_asred_frame"
        if chiave is None or not f_off.is_file() or not crop_dir.is_dir():
            continue
        for nome, info in json.load(open(f_off, encoding="utf-8")).items():
            p = crop_dir / nome
            sr = info.get("asred_local_rect")
            if not p.is_file() or sr is None:
                continue
            quads = gt[chiave].get(int(info["frame_idx"]))
            if not quads:
                continue
            ox, oy = int(info["offset_x"]), int(info["offset_y"])
            asred = np.array([[ox + sr[0], oy + sr[1]], [ox + sr[0] + sr[2], oy + sr[1]],
                              [ox + sr[0] + sr[2], oy + sr[1] + sr[3]],
                              [ox + sr[0], oy + sr[1] + sr[3]]], np.float32)
            mig, best = None, 0.0
            for q in quads:
                v = poly_iou(asred, q)
                if v > best:
                    mig, best = q, v
            # solo i ritagli che ASRED ha messo davvero su una targa
            if mig is None or best < 0.2:
                continue
            campione.append({"path": str(p), "offset": (ox, oy), "seed": list(sr),
                             "gt": mig.tolist(), "video": cart.name})
    rng = random.Random(seed)
    rng.shuffle(campione)
    if n_max and len(campione) > n_max:
        per_video = defaultdict(list)
        for c in campione:
            per_video[c["video"]].append(c)
        quota = max(1, n_max // max(1, len(per_video)))
        rid = [x for l in per_video.values() for x in l[:quota]]
        rng.shuffle(rid)
        campione = rid[:n_max]
    return campione


def esegui(combos, campione, processi):
    """Funzione che valuta una lista di combinazioni di parametri, in sequenza o in parallelo su
    piu' processi, stampando l'IoU di ognuna."""
    ris = []
    t0 = time.perf_counter()
    if processi == 1:
        _init_worker(campione)
        for i, c in enumerate(combos, 1):
            ris.append(valuta_combo(c))
            print(f"    {i}/{len(combos)}  IoU={ris[-1]['iou_media']:.4f}  "
                  f"scartate={ris[-1].get('scartate',0):.0%}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=processi, initializer=_init_worker,
                                 initargs=(campione,)) as ex:
            for i, r in enumerate(ex.map(valuta_combo, combos), 1):
                ris.append(r)
                print(f"    {i}/{len(combos)}  IoU={r['iou_media']:.4f}  "
                      f"scartate={r.get('scartate',0):.0%}", flush=True)
    print(f"    ({time.perf_counter()-t0:.1f} s)")
    return ris


def modalita_gridsearch():
    """Funzione che esegue la grid search dei parametri del segmentatore, per coordinate-descent,
    cercando di massimizzare l'IoU medio rispetto alle annotazioni vere."""
    if not Path(ANNOTATIONS_XML).is_file():
        print(f"annotations.xml non trovato: {ANNOTATIONS_XML}")
        return 1
    processi = os.cpu_count() or 1
    out = Path(GRIDSEARCH_OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    base_par = {k: v[len(v) // 2] for k, v in GRIGLIA.items()}
    print("Costruzione del campione...")
    campione = costruisci_campione(ANNOTATIONS_XML, BASE_ASRED_FOLDER, CAMPIONE_GRIDSEARCH_N)
    if not campione:
        print(f"Nessun ritaglio utilizzabile in {BASE_ASRED_FOLDER}")
        print("  serve <video>/crop_offsets.json e <video>/<video>_asred_frame")
        return 1
    print(f"  {len(campione)} ritagli da {len({c['video'] for c in campione})} video")
    storia = []
    corrente = dict(base_par)
    for passata in range(1, PASSATE_GRIDSEARCH + 1):
        print(f"\n  --- passata {passata}/{PASSATE_GRIDSEARCH} ---")
        for nome, valori in GRIGLIA.items():
            combos = [{**corrente, nome: v} for v in valori]
            print(f"  {nome}: {valori}")
            ris = esegui(combos, campione, processi)
            storia.extend(ris)
            mig = max(ris, key=lambda r: r["iou_media"])
            att = next((r for r in ris if r["params"][nome] == corrente[nome]), None)
            if att and mig["iou_media"] > att["iou_media"] + 1e-4:
                print(f"    -> {nome}: {corrente[nome]} -> {mig['params'][nome]} "
                      f"(IoU {att['iou_media']:.4f} -> {mig['iou_media']:.4f})")
                corrente[nome] = mig["params"][nome]
            else:
                print(f"    -> {nome}: resta {corrente[nome]}")
    trovati = [r for r in storia if r["params"] == corrente]
    ris_mig = trovati[0] if trovati else esegui([corrente], campione, 1)[0]
    partenza = next((r for r in storia if r["params"] == base_par), None)
    print(f"\n{'='*80}\n RISULTATO\n{'='*80}")
    if partenza:
        print(f"  Partenza : IoU {partenza['iou_media']:.4f}  IoU>=0.5 {partenza['cop05']:.1%}"
              f"  scartate {partenza.get('scartate',0):.1%}")
    print(f"  Migliore : IoU {ris_mig['iou_media']:.4f}  IoU>=0.5 {ris_mig['cop05']:.1%}"
          f"  scartate {ris_mig.get('scartate',0):.1%}")

    print("\n  Da incollare in PARAMETRI, in testa al file:")
    for k in GRIGLIA:
        segna = "   <- cambiato" if partenza and corrente[k] != base_par[k] else ""
        print(f"      {k}={corrente[k]!r},{segna}")
    print(f"\n{'='*80}\n SENSIBILITA'\n{'='*80}")
    righe = []
    for nome in GRIGLIA:
        pv = defaultdict(list)
        for r in storia:
            pv[r["params"][nome]].append(r["iou_media"])
        if len(pv) > 1:
            med = {v: float(np.mean(x)) for v, x in pv.items()}
            righe.append((max(med.values()) - min(med.values()), nome, min(med.values()), max(med.values())))
    for esc, nome, lo, hi in sorted(righe, reverse=True):
        print(f"  {nome:20s} {lo:8.4f} {hi:8.4f} escursione {esc:8.4f}")
    json.dump({"migliore": corrente, "metriche": ris_mig, "partenza": base_par,
               "n_campione": len(campione)},
              open(out / "gridsearch_bordi.json", "w", encoding="utf-8"), indent=2)
    print(f"\nRisultati -> {out/'gridsearch_bordi.json'}")
    return 0


# ================================================================================================
# PIPELINE
# ================================================================================================
def modalita_pipeline():
    """Funzione che esegue la catena completa (ASRED + ritagli + segmentazione + OCR + consolidamento) su tutti
    i video mp4 trovati in BASE_VIDEO_FOLDER."""
    logger.info(f"  [*] Margine ritaglio: {MARGINE_RITAGLIO:g}")
    for d in (BASE_ASRED_FOLDER, BASE_FRAMES_FOLDER, BASE_OUTPUT_SEGMENTATION):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(BASE_VIDEO_FOLDER):
        logger.error(f"Cartella video inesistente: {BASE_VIDEO_FOLDER}")
        return 1
    video = [f for f in os.listdir(BASE_VIDEO_FOLDER) if f.lower().endswith(ESTENSIONI_VIDEO_VALIDE)]
    logger.info(f"Trovati {len(video)} video.\n")
    for fv in video:
        path = os.path.join(BASE_VIDEO_FOLDER, fv)
        nome = os.path.splitext(fv)[0]
        logger.info("=" * 50)
        logger.info(f"--- {fv} ---")
        logger.info("=" * 50)
        out_asred = os.path.join(BASE_ASRED_FOLDER, nome)
        out_frames = os.path.join(BASE_FRAMES_FOLDER, nome)
        crop_dir = os.path.join(out_asred, f"{nome}_asred_frame")
        base_out = Path(BASE_OUTPUT_SEGMENTATION) / nome
        bb, rect, dp = base_out / "boundingbox", base_out / "rectified", base_out / "deepplate"
        for d in (out_asred, out_frames, crop_dir):
            os.makedirs(d, exist_ok=True)
        for d in (bb, rect, dp):
            d.mkdir(parents=True, exist_ok=True)

        if ESEGUI_ASRED:
            try:
                # Calcolo dell'hash del video per ASRED
                res = subprocess.run(["python", SCRIPT_HASH_PATH, path], check=True,
                                     capture_output=True, text=True)
                h = res.stdout.strip()
            except subprocess.CalledProcessError:
                logger.error("  [-] Errore nel calcolo dell'hash.")
                continue
            cmd = [ASRED_EXE_PATH, "--input-path", path, "--output-dir", out_asred,
                   "--hash", h, "--mode", "all", "--device", "cuda", "--trc-max-age", "120"]
            try:
                logger.info("  [*] ASRED in corso...")
                subprocess.run(cmd, cwd=ASRED_FOLDER, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"  [-] Errore ASRED: {e}")
                continue

        j_in = os.path.join(out_asred, "data.json")
        j_filt = os.path.join(out_asred, "data_filtered.json")
        f_off = os.path.join(out_asred, "crop_offsets.json")
        offsets = {}
        # Se i ritagli esistono gia' e RIESTRAI_RITAGLI e' False, vengono riusati anche se ASRED e'
        # stato appena rieseguito: in quel caso il nuovo data.json viene ignorato. Dopo aver
        # rieseguito ASRED con impostazioni diverse bisogna quindi mettere RIESTRAI_RITAGLI = True.
        # Altrimenti si filtra data.json e si estraggono frame e ritagli.
        gia_pronti = (os.path.exists(f_off) and os.path.isdir(crop_dir)
                      and any(Path(crop_dir).glob("*.jpg")))
        if gia_pronti and not RIESTRAI_RITAGLI:
            offsets = json.load(open(f_off, encoding='utf-8'))
            logger.info(f"  [*] Riuso di {len(offsets)} ritagli gia' estratti "
                        f"(RIESTRAI_RITAGLI = True per rifarli)")
        elif os.path.exists(j_in):
            filtra_json_targhe(j_in, j_filt)
            offsets = estrai_frame_e_ritagli(path, j_filt, crop_dir, out_frames)
            with open(f_off, 'w', encoding='utf-8') as fh:
                json.dump(offsets, fh, indent=2)
        else:
            logger.error(f"  [-] Manca sia data.json sia crop_offsets.json in {out_asred}")
            continue

        seg = SegmentatoreBordi(bb_output_dir=str(bb), rectified_dir=str(rect),
                                deepplate_exe=(DEEPPLATE_EXE_PATH if ESEGUI_OCR else None),
                                dp_output_dir=str(dp), finestra_frame=FINESTRA_FRAME, **PARAMETRI)
        seg.process_folder(crop_dir, offsets, str(base_out / "segmenter_boxes.json"))
        print()
    logger.info("Elaborazione completata.")
    return 0


def main():
    """Funzione che legge la modalita' da riga di comando (pipeline o gridsearch) e lancia 
    la funzione corrispondente."""
    ap = argparse.ArgumentParser(description="Targhe: segmentazione a bordi, OCR, grid search")
    ap.add_argument("modalita", nargs="?", default="pipeline",
                    choices=["pipeline", "gridsearch"],
                    help="pipeline = esegue la catena sui video; "
                         "gridsearch = cerca i parametri che massimizzano l'IoU")
    args = ap.parse_args()

    return {"pipeline": modalita_pipeline, "gridsearch": modalita_gridsearch}[args.modalita]()
 

if __name__ == "__main__":
    sys.exit(main())
