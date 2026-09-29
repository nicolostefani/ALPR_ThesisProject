"""
******************************************************************
VALIDAZIONE PIPELINE DI LOCALIZZAZIONE E OCR SU TARGHE ITALIANE
******************************************************************
Il presente script permette di effettuare la valutazione dei risultati ottenuti dall'esecuzione
della pipeline di ALPR (Pipeline.py). Per funzionare necessita dei dati prodotti dalla pipeline
stessa (JSON contenente le predizioni e tabelle di probabilità di DeepPlate) e delle annotazioni
della ground truth in formato CVAT (XML).

Il consolidamento temporale e' lo stesso della pipeline: voto
pesato sul primo candidato di ogni fotogramma, con peso w = max(0.05, conf) x sqrt(area), e
decodifica vincolata al formato LL-DDD-LL. Le funzioni di lettura, decodifica e
consolidamento delle tabelle sono copiate identiche da Pipeline.py, quindi lo script e'
autonomo e non ha bisogno di Pipeline.py per essere eseguito.

SEZIONI DEL REPORT:
    A. Localizzazione: TP, FP, FN, precision, recall e F1 a piu' soglie di IoU, per distinguere
       le targhe non trovate da quelle ritagliate male.
    B. Costo dei filtri: quante targhe coperte da ASRED vengono perse dai filtri anti-falsi-positivi.
    C. OCR per rilevazione: lettura del singolo fotogramma contro lettura consolidata sulla
       finestra di fotogrammi della pipeline.
    D. OCR per traccia: consolidamento sull'intera traccia fisica di ogni targa.
    E. Confidenza di DeepPlate: se e' utilizzabile come filtro.
    F. Targhe lette contro targhe annotate, indipendentemente dalla localizzazione.
    G. Matrice di confusione dei caratteri e accuratezza per posizione.
    Alla fine viene generato il report PDF (evaluation_report.pdf, accanto alla cartella di output).
"""
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Matplotlib non e' un requisito della pipeline, ma serve per generare il report PDF.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    HA_MATPLOTLIB = True
except ImportError:
    HA_MATPLOTLIB = False


# ================================================================================================
# CONFIGURAZIONE
# ================================================================================================
BASE_OUTPUT_SEGMENTATION = r"E:\DatasetNuovo\TESI\output"  # cartella <output> della pipeline, con una sottocartella per video
ANNOTATIONS_XML = r"E:\DatasetNuovo\annotations.xml"          # file CVAT con le annotazioni della ground truth
SOGLIA_IOU = 0.5                 # soglia di IoU per considerare corretta una localizzazione
SOGLIE_EXTRA = [0.3, 0.5, 0.7]   # soglie aggiuntive della sezione A, per capire se il problema e' trovare la targa o ritagliarla bene

# Alfabeto delle targhe italiane: 22 lettere, escluse I, O, Q, U.
LETTERE_VALIDE = set("ABCDEFGHJKLMNPRSTVWXYZ")
CIFRE_VALIDE = set("0123456789")
FORMATO_TARGA_IT = "LLDDDLL"
VUOTO = "_"  # simbolo di posizione vuota nelle tabelle di DeepPlate


# ================================================================================================
# TABELLE DI DEEPPLATE E CONSOLIDAMENTO 
# ================================================================================================
def leggi_tabella(path):
    """Funzione che legge la tabella di probabilita' scritta da DeepPlate per un ritaglio e la
    riordina per posizione, scartando la copia duplicata del blocco.
    Il file alterna una riga di simboli e una riga di probabilita' (una colonna per posizione).
    Restituisce, per ogni posizione, la lista di coppie (carattere, probabilita') in ordine di
    probabilita' decrescente, oppure None se il file non si legge."""
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


def decodifica(pos, vincolata=True, formato=FORMATO_TARGA_IT):
    """Funzione che decodifica le distribuzioni di probabilita' in una stringa targa (Alg. 3.6).
    vincolata=True sceglie, per ogni posizione, il candidato piu' probabile compatibile con
    LL-DDD-LL; se nessuno e' compatibile si prende il piu' probabile in assoluto. Le posizioni
    senza candidati vengono omesse dal testo. Restituisce il testo, la confidenza media e minima
    e la likelihood (prodotto delle probabilita' scelte).
    """
    testo, probs = [], []
    for p, cand in enumerate(pos):
        if not cand:
            testo.append(VUOTO)
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
    return {'testo': ''.join(testo).replace(VUOTO, ''),
            'conf_media': (sum(probs) / len(probs)) if probs else 0.0,
            'conf_min': min(probs) if probs else 0.0,
            'likelihood': like}


def _media_distribuzioni(tabelle, pesi):
    """Funzione che fa, per ogni posizione, la media pesata delle distribuzioni complete (tutti i
    candidati) delle tabelle. Serve solo da ripiego in consolida_tabelle, per le posizioni in cui
    nessun carattere votato rispetta il formato."""
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
    return [sorted(((c, v / (peso_tot[pp] or 1.0)) for c, v in accum[pp].items()),
                   key=lambda t: -t[1]) for pp in range(n_pos)]


def consolida_tabelle(tabelle, pesi=None, vincolata=True):
    """
    Funzione che consolida le tabelle di piu' fotogrammi della stessa targa in un'unica lettura
    con il voto pesato.
    Per ogni posizione vota solo il primo candidato di ogni fotogramma, con peso pari a
    (peso del fotogramma x probabilita' del candidato); i voti si dividono poi per la somma dei
    pesi dei fotogrammi e la lettura finale si decodifica con il vincolo LL-DDD-LL.

    Restituisce testo, conf_media, conf_min, likelihood e n_fotogrammi, oppure None se non ci
    sono tabelle.
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
            # leggi_tabella ordina gia' per probabilita' decrescente: il primo e' l'argmax
            c, pr = tab[pp][0]
            # Voto pesato del fotogramma
            voti[pp][c] = voti[pp].get(c, 0.0) + w * float(pr)
            peso_tot[pp] += w

    posizioni = [sorted(((c, v / (peso_tot[pp] or 1.0)) for c, v in voti[pp].items()),
                        key=lambda t: -t[1]) for pp in range(n_pos)]

    # In modalita' vincolata, se per una posizione nessun carattere votato rientra nel set valido
    # per il suo formato, i voti di quella posizione vengono sostituiti con la media delle
    # distribuzioni complete, che contiene anche i candidati successivi al primo.
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
                media = _media_distribuzioni(tabelle, pesi)
            if pp < len(media):
                posizioni[pp] = media[pp]

    ris = decodifica(posizioni, vincolata=vincolata)
    ris['n_fotogrammi'] = len(tabelle)
    return ris


# ================================================================================================
# PESI E DISTANZE
# ================================================================================================
def _ammesso(c, tipo):
    """Funzione che verifica se il carattere e' ammesso in una posizione del formato LL-DDD-LL."""
    if tipo == 'L':
        return c in LETTERE_VALIDE
    if tipo == 'D':
        return c in CIFRE_VALIDE
    return c != VUOTO


def peso_fotogramma(posizioni, quad):
    """Funzione che calcola il peso di un fotogramma nel consolidamento, come nella pipeline:
    w = max(0.05, conf) x sqrt(area), dove conf e' la confidenza media della lettura vincolata
    del singolo fotogramma e area e' l'area del poligono segmentato.

    La confidenza si ricalcola dalla tabella perche' nel JSON la pipeline sostituisce
    ocr_conf_media con quella della lettura consolidata.
    """
    conf = decodifica(posizioni, vincolata=True)['conf_media']
    area = abs(float(cv2.contourArea(np.asarray(quad, np.float32).reshape(-1, 1, 2))))
    return max(0.05, float(conf or 0.3)) * math.sqrt(max(1.0, area))


def levenshtein(a, b):
    """Numero minimo di inserimenti, cancellazioni e sostituzioni per passare da a a b."""
    a, b = a or '', b or ''
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prec = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prec[j] + 1, cur[j - 1] + 1, prec[j - 1] + (ca != cb)))
        prec = cur
    return prec[-1]


def similarita(a, b):
    """Similarita' fra due stringhe, fra 0 e 1: 1 meno la distanza di Levenshtein divisa per la
    lunghezza della stringa piu' lunga (1 = stringhe uguali)."""
    n = max(len(a or ''), len(b or ''))
    return 1.0 if n == 0 else 1.0 - levenshtein(a, b) / n


# ================================================================================================
# GROUND TRUTH CVAT
# ================================================================================================
def rotated_box_to_quad(xtl, ytl, xbr, ybr, rotation_deg):
    """Funzione che converte un box ruotato di CVAT (rotazione in gradi attorno al centro) in un
    quadrilatero di 4 punti, cosi' da poter calcolare l'IoU con i poligoni predetti dalla pipeline."""
    cx, cy = (xtl + xbr) / 2.0, (ytl + ybr) / 2.0
    w, h = xbr - xtl, ybr - ytl
    a = math.radians(rotation_deg or 0.0)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[cx + dx * ca - dy * sa, cy + dx * sa + dy * ca]
                     for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))],
                    dtype=np.float32)


def parse_cvat(xml_path):
    """Funzione che legge l'export CVAT e restituisce due valori:
    - gt: {video: {fotogramma: [box]}}, dove ogni box contiene il quadrilatero, la targa e
      l'identificativo della traccia fisica;
    - problemi: lista delle anomalie trovate durante la lettura (tracce senza task, fotogrammi
      fuori dal video, eccetera)."""
    root = ET.parse(xml_path).getroot()
    tasks = {}
    for t in root.iter('task'):
        tid = int(t.findtext('id'))
        src = (t.findtext('source') or '') or (t.findtext('name') or '')
        tasks[tid] = {'video': Path(src).stem, 'size': int(t.findtext('size') or 0)}
    # Nell'export di progetto i fotogrammi sono numerati di seguito su tutti i task: per ogni
    # task si calcola l'offset da sottrarre per tornare al fotogramma del singolo video
    offset, cum = {}, 0
    for tid in sorted(tasks):
        offset[tid] = cum
        cum += tasks[tid]['size']
    gt, problemi = defaultdict(lambda: defaultdict(list)), []
    for track in root.iter('track'):
        tid = int(track.get('task_id'))
        if tid not in tasks:
            problemi.append(f"track {track.get('id')}: task_id {tid} assente nel meta")
            continue
        info, off = tasks[tid], offset[tid]
        targa = None
        for att in track.iter('attribute'):  
            if att.get('name') == 'Number' and (att.text or '').strip():
                targa = re.sub(r'[^A-Z0-9]', '', att.text.strip().upper())
                break
        for box in track.iter('box'):
            if box.get('outside') == '1':     # targa non visibile in questo fotogramma: nessuna ground truth
                continue
            f_loc = int(box.get('frame')) - off
            if not (0 <= f_loc < info['size']):
                problemi.append(f"{info['video']}: frame locale {f_loc} fuori da [0,{info['size']})")
                continue
            gt[info['video']][f_loc].append({
                'quad': rotated_box_to_quad(float(box.get('xtl')), float(box.get('ytl')),
                                            float(box.get('xbr')), float(box.get('ybr')),
                                            float(box.get('rotation', 0) or 0)),
                'targa': targa, 'track': f"{tid}_{track.get('id')}"})
    return gt, problemi


# ================================================================================================
# LETTURA DEGLI OUTPUT DELLA PIPELINE
# ================================================================================================
def poly_iou(p1, p2):
    """IoU fra due poligoni convessi, calcolata sull'intersezione vera e non sui box allineati
    agli assi."""
    a = np.asarray(p1, np.float32).reshape(-1, 2)
    b = np.asarray(p2, np.float32).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    aa, ab = abs(float(cv2.contourArea(a))), abs(float(cv2.contourArea(b)))  # aree dei due poligoni
    if aa <= 0 or ab <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(cv2.convexHull(a), cv2.convexHull(b))
    unione = aa + ab - float(inter)
    return float(inter) / unione if unione > 0 else 0.0


def carica_predizioni(cartella):
    """
    Funzione che carica i risultati della pipeline per un singolo video.

    Legge le rilevazioni tenute e quelle scartate dai filtri. Per ogni rilevazione tiene:
    - ocr: lettura principale, cioe' quella consolidata dalla pipeline sulla finestra di fotogrammi;
    - ocr_frame: lettura del singolo fotogramma, con vincolo LL-DDD-LL;
    - ocr_raw: lettura del singolo fotogramma, senza vincolo;
    - tabella: percorso della tabella di probabilita' di DeepPlate.
    Se nel JSON manca la lettura ma la tabella e' su disco, la lettura viene ricostruita dalla tabella.

    Restituisce: rilevazioni tenute, rilevazioni scartate, numero di tabelle non trovate,
    conteggio degli stati OCR e numero di letture ricostruite.
    """
    f = Path(cartella) / "segmenter_boxes.json"
    if not f.is_file():
        return None, None, 0, defaultdict(int), 0
    tabelle_mancanti = 0
    stati_ocr = defaultdict(int)
    ricostruite = 0        # letture ricavate dalle tabelle perche' assenti nel JSON
    tenute = defaultdict(list)
    for frame_str, voci in json.load(open(f, encoding='utf-8')).items():
        for v in voci:
            if v.get("polygon"):
                q = np.array(v["polygon"], np.float32)
            elif v.get("bounding_box"):
                x1, y1, x2, y2 = v["bounding_box"]
                q = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            else:
                continue
            # Conto gli stati dell'OCR per la diagnostica
            stati_ocr[v.get("ocr_status") or "campo_assente"] += 1
            # Se il percorso della tabella nel JSON manca o non esiste piu',
            # la cerco in <cartella>/deepplate/tables e poi in tutte le sottocartelle di deepplate
            tab = v.get("ocr_table")
            if not tab or not Path(tab).is_file():
                stem = Path(v.get('crop_name', '')).stem
                alt = Path(cartella) / "deepplate" / "tables" / f"{stem}.txt"
                if alt.is_file():
                    tab = str(alt)
                else:
                    trovati = list((Path(cartella) / "deepplate").rglob(f"{stem}.txt")) if stem else []
                    tab = str(trovati[0]) if trovati else None
                    if tab is None:
                        tabelle_mancanti += 1
          
            testo = (v.get("ocr_text") or "").upper() or None
            testo_raw = (v.get("ocr_text_raw") or "").upper() or None
            conf = v.get("ocr_conf_media", 0.0)
            if testo is None and tab:
                pos = leggi_tabella(tab)
                if pos:
                    d_vinc = decodifica(pos, vincolata=True)
                    testo = d_vinc['testo'] or None
                    conf = round(d_vinc['conf_media'], 4)
                    if testo_raw is None:
                        testo_raw = decodifica(pos, vincolata=False)['testo'] or None
                    ricostruite += 1
            # Lettura del singolo fotogramma: se il campo manca o e' vuoto uso la lettura principale
            testo_frame = (v.get("ocr_text_frame") or "").upper() or None
            tenute[int(frame_str)].append({
                'quad': q, 'ocr': testo, 'ocr_raw': testo_raw,
                'ocr_frame': testo_frame if testo_frame is not None else testo,
                'n_finestra': v.get("ocr_frame_finestra", 0),
                'conf_ocr': conf, 'tabella': tab, 'crop': v.get("crop_name", "")})
    # Rilevazioni scartate dai filtri, con il motivo dello scarto
    scartate = defaultdict(list)
    fr = Path(cartella) / "rejected_log.json"
    if fr.is_file():
        try:
            for v in json.load(open(fr, encoding='utf-8')):
                if v.get("polygon") and v.get("frame_idx") is not None:
                    scartate[int(v["frame_idx"])].append({
                        'quad': np.array(v["polygon"], np.float32),
                        'motivo': v.get("hard_reject") or "|".join(v.get("neg_signals", []))})
        except (json.JSONDecodeError, ValueError):
            pass
    return tenute, scartate, tabelle_mancanti, stati_ocr, ricostruite


def accoppia(gts, preds, soglia):
    """Funzione che accoppia ogni box di ground truth alla predizione con IoU migliore, senza
    riusare due volte lo stesso ground truth o la stessa predizione.

    Si calcolano tutte le coppie possibili, si ordinano per IoU decrescente e si accettano una alla volta scartando
    quelle che userebbero un box gia' assegnato. Per un singolo fotogramma il numero di targhe e'
    basso, quindi la soluzione ottima coincide quasi sempre con quella greedy.

    Restituisce la lista delle coppie accettate (indice ground truth, indice predizione, IoU) e i
    due insiemi degli indici gia' assegnati, che servono a calcolare FN e FP per differenza."""
    coppie = sorted(((poly_iou(g['quad'], p['quad']), gi, pi)
                     for gi, g in enumerate(gts) for pi, p in enumerate(preds)), key=lambda t: -t[0])
    ug, up, out = set(), set(), []
    for iou, gi, pi in coppie:
        if iou < soglia or gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
        out.append((gi, pi, iou))
    return out, ug, up


# ================================================================================================
# VALUTAZIONE
# ================================================================================================
def valuta_video(gt_video, tenute, scartate, soglia):
    """
    Funzione che calcola le metriche di un singolo video in un unico passaggio sui fotogrammi:
    localizzazione (TP, FP, FN, IoU), costo dei filtri, letture OCR per rilevazione e, per ogni
    traccia fisica, le tabelle e i pesi che servono al consolidamento sull'intera traccia.
    Restituisce un dizionario con tutte le metriche raccolte.
    """
    st = {'tp': 0, 'fp': 0, 'fn': 0, 'iou_tp': [],
          'multi': {s: {'tp': 0, 'fp': 0, 'fn': 0} for s in SOGLIE_EXTRA},
          'gt_coperti': 0, 'gt_persi': 0, 'motivi': defaultdict(int),
          'ocr_tot': 0, 'ocr_letti': 0, 'ocr_esatti': 0, 'ocr_esatti_raw': 0,
          'ocr_esatti_frame': 0, 'lev_frame': [], 'n_fin': [],
          'lev': [], 'sim': [], 'conf_ok': [], 'conf_ko': [],
          'h_trovate': [], 'h_mancate': [], 'n_scartate': 0, 'area_ratio': [],
          'tracce': defaultdict(lambda: {'targa': None, 'tabelle': [], 'pesi': [], 'letture': [],
                                         'lev_migliore': None, 'n_gt': 0, 'n_trovati': 0,
                                         'iou': [], 'iou_ogni': []})}
    for f, boxes in gt_video.items():
        for b in boxes:
            st['tracce'][b['track']]['n_gt'] += 1
            st['tracce'][b['track']]['targa'] = b['targa']
    for f in set(gt_video) | set(tenute):
        gts, preds = list(gt_video.get(f, [])), list(tenute.get(f, []))
        coppie, ug, up = accoppia(gts, preds, soglia)
        st['tp'] += len(coppie)
        # Rapporto fra area predetta e area annotata: > 1 se il ritaglio predetto e' piu' grande
        # di quello annotato, < 1 se e' piu' piccolo
        for gi, pi, _ in coppie:
            area_gt = abs(float(cv2.contourArea(gts[gi]['quad'])))
            if area_gt > 0:
                st['area_ratio'].append(abs(float(cv2.contourArea(preds[pi]['quad']))) / area_gt)
        st['fn'] += len(gts) - len(ug)
        st['fp'] += len(preds) - len(up)
        # Accoppio le GT anche con le rilevazioni scartate dai filtri: le GT coperte qui ma non
        # dalle predizioni tenute sono targhe perse per colpa dei filtri, e ne registro il motivo
        _, cop, _ = accoppia(gts, preds + list(scartate.get(f, [])), soglia)
        st['gt_coperti'] += len(cop)
        persi = cop - ug
        st['gt_persi'] += len(persi)
        for gi in persi:
            best = max(((poly_iou(gts[gi]['quad'], s['quad']), s['motivo'])
                        for s in scartate.get(f, [])), default=(0, '?'))
            st['motivi'][best[1]] += 1
        for s in SOGLIE_EXTRA:
            c2, ug2, up2 = accoppia(gts, preds, s)
            st['multi'][s]['tp'] += len(c2)
            st['multi'][s]['fn'] += len(gts) - len(ug2)
            st['multi'][s]['fp'] += len(preds) - len(up2)
        st['n_scartate'] += len(scartate.get(f, []))
        for gi, gbox in enumerate(gts):
            # IoU migliore di ogni GT con le predizioni tenute
            st['tracce'][gbox['track']]['iou_ogni'].append(
                max((poly_iou(gbox['quad'], p['quad']) for p in preds), default=0.0))
            # Altezza della targa annotata in pixel (media dei due lati corti), per confrontare
            # le dimensioni delle targhe trovate e di quelle mancate
            q = gbox['quad']
            h_px = min((float(np.linalg.norm(q[3] - q[0])) + float(np.linalg.norm(q[2] - q[1]))) / 2.0,
                       (float(np.linalg.norm(q[1] - q[0])) + float(np.linalg.norm(q[2] - q[3]))) / 2.0)
            (st['h_trovate'] if gi in ug else st['h_mancate']).append(h_px)
        for gi, pi, iou in coppie:
            st['iou_tp'].append(iou)
            pred = preds[pi]
            atteso, letto = gts[gi]['targa'], pred['ocr']
            lf = pred.get('ocr_frame')
            tr = st['tracce'][gts[gi]['track']]
            tr['targa'] = atteso
            tr['n_trovati'] += 1
            tr['iou'].append(iou)
            # Tabella e peso del fotogramma, per il consolidamento sull'intera traccia (sezione D)
            tab = leggi_tabella(pred['tabella']) if pred.get('tabella') else None
            if tab:
                tr['tabelle'].append(tab)
                tr['pesi'].append(peso_fotogramma(tab, pred['quad']))
            # Letture del singolo fotogramma, per il voto di maggioranza della sezione D
            if lf:
                tr['letture'].append(lf)
            if not atteso:
                continue
            if lf:
                d_frame = levenshtein(lf, atteso)
                if tr['lev_migliore'] is None or d_frame < tr['lev_migliore']:
                    tr['lev_migliore'] = d_frame
            st['ocr_tot'] += 1
            if not letto:
                continue
            st['ocr_letti'] += 1
            # Lettura del singolo fotogramma, da confrontare con quella consolidata sulla finestra
            if lf:
                st['lev_frame'].append(d_frame)
                if lf == atteso:
                    st['ocr_esatti_frame'] += 1
                # Stesso fotogramma senza vincolo LL-DDD-LL, per misurare il guadagno del solo vincolo
                if pred.get('ocr_raw') == atteso:
                    st['ocr_esatti_raw'] += 1
            if pred.get('n_finestra'):
                st['n_fin'].append(pred['n_finestra'])
            # Lettura consolidata dalla pipeline sulla finestra di fotogrammi
            d = levenshtein(letto, atteso)
            st['lev'].append(d)
            st['sim'].append(similarita(letto, atteso))
            conf = float(pred.get('conf_ocr') or 0.0)
            if d == 0:
                st['ocr_esatti'] += 1
                st['conf_ok'].append(conf)
            else:
                st['conf_ko'].append(conf)
    return st


def consolida(st):
    """
    Funzione che valuta la lettura di ogni traccia fisica (targa annotata), consolidando le
    tabelle di tutti i suoi fotogrammi con lo stesso metodo della pipeline.
    Per confronto conta anche:
    - le tracce corrette con il semplice voto di maggioranza sulle letture dei singoli fotogrammi;
    - le tracce con almeno un fotogramma letto correttamente da solo.
    Restituisce un dizionario con le metriche per traccia e il dettaglio di ognuna.
    """
    ris = {'tracce': 0, 'esatte': 0, 'lev': [], 'miglior_frame': 0, 'voto': 0, 'dettaglio': []}
    for tr in st['tracce'].values():
        atteso = tr['targa']
        if not atteso:
            continue
        ris['tracce'] += 1
        cons = consolida_tabelle(tr['tabelle'], pesi=tr['pesi']) if tr['tabelle'] else None
        letta = cons['testo'] if cons else None
        if tr['letture']:
            # Voto di maggioranza: la lettura del singolo fotogramma piu' frequente nella traccia
            conteggi = defaultdict(int)
            for x in tr['letture']:
                conteggi[x] += 1
            if max(conteggi.items(), key=lambda kv: kv[1])[0] == atteso:
                ris['voto'] += 1
        d = levenshtein(letta or '', atteso)
        ris['lev'].append(d)
        if d == 0:
            ris['esatte'] += 1
        if tr['lev_migliore'] == 0:
            ris['miglior_frame'] += 1
        ris['dettaglio'].append({
            'attesa': atteso, 'consolidata': letta, 'lev': d, 'n_frame': len(tr['tabelle']),
            'conf': round(cons['conf_media'], 3) if cons else 0.0,
            'n_gt': tr['n_gt'], 'n_trovati': tr['n_trovati'],
            'copertura': tr['n_trovati'] / tr['n_gt'] if tr['n_gt'] else 0.0,
            'iou_max': max(tr['iou_ogni']) if tr['iou_ogni'] else 0.0,
            'iou_medio': float(np.mean(tr['iou'])) if tr['iou'] else 0.0,
            'iou_mediano': float(np.median(tr['iou'])) if tr['iou'] else 0.0})
    return ris


def confronto_targhe_indipendente(gt, cartelle):
    """
    Funzione che confronta le targhe lette con quelle annotate, indipendentemente dalla
    localizzazione dei box: per ogni targa annotata in un video si cerca, fra le letture
    prodotte in quel video, quella con distanza di Levenshtein minima.
    """
    print(f"\n{'='*104}")
    print(" F. TARGHE LETTE vs TARGHE ANNOTATE  (confronto indipendente dalla localizzazione)")
    print(f"{'='*104}")
    tot_targhe = esatte = entro1 = entro2 = 0
    distanze, dettaglio = [], []
    letture_vicine = letture_tot = 0
    for cart in cartelle:
        tenute, _, _, _, _ = carica_predizioni(cart)
        if tenute is None:
            continue
        chiave = next((k for k in gt if k.lower() == cart.name.lower()), None)
        if chiave is None:
            continue
        attese = {b['targa'] for f in gt[chiave].values() for b in f if b['targa']}
        letture = [p['ocr'] for preds in tenute.values() for p in preds if p['ocr']]
        letture_tot += len(letture)
        for att in sorted(attese):
            tot_targhe += 1
            if not letture:
                distanze.append(len(att))
                dettaglio.append((cart.name, att, None, len(att), 0))
                continue
            migliore = min(letture, key=lambda L: levenshtein(L, att))
            d = levenshtein(migliore, att)
            quante = sum(1 for L in letture if levenshtein(L, att) == d)
            distanze.append(d)
            dettaglio.append((cart.name, att, migliore, d, quante))
            if d == 0:
                esatte += 1
            if d <= 1:
                entro1 += 1
            if d <= 2:
                entro2 += 1
        for L in letture:
            if attese and min(levenshtein(L, a) for a in attese) <= 2:
                letture_vicine += 1

    if not tot_targhe:
        print("  Nessuna targa annotata confrontabile.")
        return
    print(f"  Targhe annotate nei video elaborati                    : {tot_targhe}")
    print(f"  Lette ESATTAMENTE almeno una volta nel video           : {esatte} ({esatte/tot_targhe:.1%})")
    print(f"  Lette a <=1 carattere di distanza                      : {entro1} ({entro1/tot_targhe:.1%})")
    print(f"  Lette a <=2 caratteri di distanza                      : {entro2} ({entro2/tot_targhe:.1%})")
    if distanze:
        print(f"  Distanza di edit MINIMA, mediana sulle targhe          : {np.median(distanze):.1f} su 7")
    if letture_tot:
        print(f"  Letture prodotte vicine (<=2) a una targa annotata     : {letture_vicine}/{letture_tot} "
              f"({letture_vicine/letture_tot:.1%})")
    print(f"\n  {'video':14s} {'attesa':>9s} {'migliore letta':>15s} {'dist':>5s} {'n':>4s}")
    for v, att, letta, d, quante in sorted(dettaglio, key=lambda x: x[3])[:20]:
        print(f"  {v:14s} {att:>9s} {str(letta):>15s} {d:5d} {quante:4d}")
    if len(dettaglio) > 20:
        print(f"  ... e altre {len(dettaglio)-20} targhe (ordinate dalla piu' vicina)")


# ================================================================================================
# ANALISI PER CARATTERE: MATRICE DI CONFUSIONE E CONFIDENZA TEMPORALE
# ================================================================================================
POS_LETTERA = (0, 1, 5, 6)  # le altre tre posizioni (2, 3, 4) sono cifre
ETICHETTE_POS = ''.join('L' if p in POS_LETTERA else 'D' for p in range(7))


def decodifica_posizioni(posizioni, vincolata=True, formato=FORMATO_TARGA_IT):
    """
    Funzione che decodifica una tabella di probabilita' posizione per posizione (Alg. 3.6) e
    restituisce la lista dei caratteri scelti e la probabilita' di ciascuno. A differenza di
    decodifica(), le posizioni vuote restano nella lista, cosi' ogni indice corrisponde a una
    posizione della targa.
    """
    caratteri, probs = [], []
    for p, cand in enumerate(posizioni):
        if not cand:
            caratteri.append(VUOTO)
            probs.append(0.0)
            continue
        scelto = None
        if vincolata and p < len(formato):
            scelto = next(((c, pr) for c, pr in cand if _ammesso(c, formato[p])), None)
        if scelto is None:
            scelto = cand[0]
        caratteri.append(scelto[0])
        probs.append(float(scelto[1]))
    return caratteri, probs


def prob_carattere(posizioni, p, carattere):
    """Probabilita' che la tabella assegna a un carattere specifico in una data posizione."""
    if p >= len(posizioni):
        return 0.0
    for c, pr in posizioni[p]:
        if c == carattere:
            return float(pr)
    return 0.0


def _nuovo_contenitore():
    """Strutture vuote per accumulare la matrice di confusione e l'accuratezza per posizione,
    sia per la lettura consolidata ('conf', 'pos') sia per il singolo fotogramma ('conf_frame',
    'pos_frame')."""
    return {'conf': defaultdict(lambda: defaultdict(int)),
            'conf_frame': defaultdict(lambda: defaultdict(int)),
            'pos': np.zeros((7, 2), int), 'pos_frame': np.zeros((7, 2), int),
            'n_letture': 0, 'n_lunghezza_errata': 0}


def _accumula(cont, atteso, letto, quale):
    """
    Funzione che aggiorna la matrice di confusione e i contatori di accuratezza per posizione,
    confrontando carattere per carattere la targa attesa con quella letta.
    Le letture di lunghezza diversa da 7 non si possono allineare alle posizioni: vengono
    escluse e contate a parte.
    """
    if not letto or len(letto) != 7:
        if quale == 'conf':
            cont['n_lunghezza_errata'] += 1
        return
    if quale == 'conf':
        cont['n_letture'] += 1
    matrice = cont[quale]
    contatore = cont['pos' if quale == 'conf' else 'pos_frame']
    # Una cella della matrice per ogni coppia nella stessa
    # posizione, piu' il contatore corretti/totali per l'accuratezza per posizione.
    for p in range(7):
        matrice[atteso[p]][letto[p]] += 1
        contatore[p, 1] += 1
        if atteso[p] == letto[p]:
            contatore[p, 0] += 1


def raccogli_caratteri(gt, cartelle, soglia):
    """
    Funzione che raccoglie i dati per la matrice di confusione, l'accuratezza per posizione e la
    vista temporale del report PDF, a livello globale e per video.
    Per ogni traccia salva i fotogrammi in ordine temporale, con tabella, peso e probabilita'
    assegnata al carattere vero in ogni posizione, e calcola la lettura consolidata sull'intera
    traccia con lo stesso metodo della pipeline.
    """
    D = {'globale': _nuovo_contenitore(), 'video': {}}
    for cart in cartelle:
        tenute, _sc, _mt, _so, _nr = carica_predizioni(cart)
        if tenute is None:
            continue
        chiave = next((k for k in gt if k.lower() == cart.name.lower()), None)
        if chiave is None:
            continue
        V = D['video'][cart.name] = _nuovo_contenitore()
        V['tracce'] = defaultdict(lambda: {'atteso': None, 'frame': [], 'iou': []})
        V['tp'] = V['fp'] = V['fn'] = 0
        gtv = gt[chiave]
        for f in sorted(set(gtv) | set(tenute)):
            gts, preds = list(gtv.get(f, [])), list(tenute.get(f, []))
            coppie, ug, up = accoppia(gts, preds, soglia)
            V['tp'] += len(coppie)
            V['fn'] += len(gts) - len(ug)
            V['fp'] += len(preds) - len(up)
            for gi, pi, iou in coppie:
                atteso = gts[gi]['targa']
                if not atteso or len(atteso) != 7:
                    continue
                pred = preds[pi]
                for quale, valore in (('conf', pred.get('ocr')),
                                      ('conf_frame', pred.get('ocr_frame'))):
                    _accumula(V, atteso, valore, quale)
                    _accumula(D['globale'], atteso, valore, quale)
                tr = V['tracce'][gts[gi]['track']]
                tr['atteso'] = atteso
                tr['iou'].append(iou)
                tab = pred.get('tabella')
                pos = leggi_tabella(tab) if tab else None
                if pos:
                    car, pr = decodifica_posizioni(pos)
                    tr['frame'].append({
                        'f': f, 'scelti': car, 'prob_scelti': pr, 'tabella': pos,
                        'peso': peso_fotogramma(pos, pred['quad']),
                        'prob_vero': [prob_carattere(pos, p, atteso[p]) for p in range(7)]})
    # Lettura consolidata sull'intera traccia, per la griglia delle targhe nel PDF
    for nome, V in D['video'].items():
        V['targhe'] = []
        for tid, tr in V['tracce'].items():
            if not tr['atteso']:
                continue
            fr = sorted(tr['frame'], key=lambda d: d['f'])
            cons = consolida_traccia(fr)
            letto = cons['testo'] if cons else ''
            V['targhe'].append({
                'track': tid, 'atteso': tr['atteso'], 'letto': letto,
                'lev': levenshtein(letto, tr['atteso']), 'n_frame': len(fr),
                'conf': cons['conf_media'] if cons else 0.0,
                'iou': float(np.mean(tr['iou'])) if tr['iou'] else 0.0})
        V['targhe'].sort(key=lambda t: (t['lev'], -t['n_frame']))
    return D


def consolida_traccia(frame):
    """Funzione che consolida i fotogrammi di una traccia (lista di dizionari con 'tabella' e
    'peso') con il voto pesato della pipeline. Restituisce None se la traccia e' vuota."""
    if not frame:
        return None
    return consolida_tabelle([d['tabella'] for d in frame], pesi=[d['peso'] for d in frame])


def stampa_confusione(D):
    """
    Funzione che stampa la matrice di confusione dei caratteri, l'accuratezza per posizione e le
    sostituzioni di caratteri piu' frequenti.
    """
    G = D['globale']
    conf, conf_f = G['conf'], G['conf_frame']
    tot = sum(v for r in conf.values() for v in r.values())
    print(f"\n{'='*104}\n G. MATRICE DI CONFUSIONE DEI CARATTERI\n{'='*104}")
    if tot == 0:
        print("  Nessuna lettura di lunghezza 7 confrontabile.")
        return
    giusti = sum(conf[c][c] for c in conf)
    print(f"  Letture confrontate (lunghezza 7)        : {G['n_letture']}")
    if G['n_lunghezza_errata']:
        print(f"  Escluse perche' di lunghezza diversa da 7 : {G['n_lunghezza_errata']}")
    print(f"  Caratteri confrontati                    : {tot}")
    print(f"  Caratteri corretti                       : {giusti} ({giusti/tot:.1%})")
    tot_f = sum(v for r in conf_f.values() for v in r.values())
    if tot_f:
        gf = sum(conf_f[c][c] for c in conf_f)
        print(f"  Accuratezza per carattere, singolo fotogramma : {gf/tot_f:.1%}")
        print(f"  Accuratezza per carattere, consolidata        : {giusti/tot:.1%} "
              f"({giusti/tot - gf/tot_f:+.1%})")

    print(f"\n  Accuratezza per POSIZIONE (formato LL-DDD-LL):")
    print(f"    {'pos':>4s} {'tipo':>7s} {'corretti':>9s} {'totali':>8s} {'accuratezza':>12s}")
    for p in range(7):
        ok, n = G['pos'][p]
        print(f"    {p+1:4d} {('lettera' if p in POS_LETTERA else 'cifra'):>7s} "
              f"{ok:9d} {n:8d} {ok/max(1,n):12.1%}")

    coppie = sorted(((n, v, l) for v, r in conf.items() for l, n in r.items() if v != l),
                    reverse=True)
    n_err = tot - giusti
    print(f"\n  Le 20 sostituzioni piu' frequenti (vero -> letto):")
    print(f"    {'vero':>5s} {'letto':>6s} {'n':>6s} {'% errori':>9s}")
    for n, v, l in coppie[:20]:
        print(f"    {v:>5s} {l:>6s} {n:6d} {n/max(1,n_err):9.1%}")
    if coppie:
        top10 = sum(n for n, _, _ in coppie[:10])
        q = top10 / max(1, n_err)
        print(f"\n  Le prime 10 coppie coprono il {q:.1%} degli errori "
              f"({len(coppie)} coppie distinte).")
        if q > 0.5:
            print("  -> Errore CONCENTRATO: poche confusioni ricorrenti, quindi strutturale e in")
            print("     principio correggibile.")
        else:
            print("  -> Errore DIFFUSO: nessuna confusione domina, coerente con un limite di")
            print("     risoluzione piuttosto che con un bias del classificatore.")
    per_vero = sorted(((sum(r.values()) - r[v], v, sum(r.values())) for v, r in conf.items()),
                      reverse=True)
    print(f"\n  Caratteri veri piu' sbagliati:")
    print(f"    {'vero':>5s} {'errori':>7s} {'occorrenze':>11s} {'tasso':>8s}")
    for err, v, n in per_vero[:10]:
        if n:
            print(f"    {v:>5s} {err:7d} {n:11d} {err/n:8.1%}")


# ================================================================================================
# REPORT PDF
# ================================================================================================
A4 = (8.27, 11.69)
C_OK, C_KO, C_GREY = '#2a7f62', '#b4453c', '#8a8a8a'  # verde = corretto, rosso = errore, grigio = neutro
C_BLU = '#3a6ea5'  # colore della lettura consolidata nel grafico "Accuracy per position"


def _matrice_su_ax(ax, conf, simboli, titolo, mostra_y=True):
    """Disegna una matrice di confusione normalizzata per riga, sui soli simboli osservati."""
    simboli = [c for c in simboli if sum(conf[c].values()) > 0]
    if not simboli:
        ax.axis('off')
        ax.set_title(titolo, fontsize=9)
        return None
    M = np.array([[conf[v][l] for l in simboli] for v in simboli], float)
    Mn = M / np.maximum(1, M.sum(1, keepdims=True)) * 100  # percentuali, coerenti con la colorbar
    im = ax.imshow(Mn, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(simboli))); ax.set_xticklabels(simboli, fontsize=6.5)
    ax.set_yticks(range(len(simboli))); ax.set_yticklabels(simboli, fontsize=6.5)
    if mostra_y:
        ax.set_ylabel("true character", fontsize=8)
    ax.set_xlabel("predicted character", fontsize=8)
    ax.set_title(titolo, fontsize=9)
    for i in range(len(simboli)):
        for j in range(len(simboli)):
            ax.text(j, i, f"{Mn[i,j]:.0f}", ha='center', va='center', fontsize=5.2,
                    color='white' if Mn[i, j] > 55 else '#333')
    return im


def _griglia_targhe(ax, targhe, titolo):
    """
    Disegna una griglia con le targhe attese e quelle lette (consolidate sull'intera traccia),
    evidenziando in verde i caratteri corretti e in rosso quelli sbagliati, con numero di
    fotogrammi, IoU e confidenza media. Se non ci sono targhe annotate, mostra un avviso.
    """
    ax.set_title(titolo, fontsize=10, loc='left')
    ax.axis('off')
    if not targhe:
        ax.text(0, 0.5, "no annotated plate in this video", fontsize=9, color=C_GREY)
        return
    n = len(targhe)
    ax.set_xlim(-0.2, 13.4)
    ax.set_ylim(-0.4, n * 1.06)
    for i, t in enumerate(targhe):
        # Riga i-esima della griglia: #traccia, atteso, letto, esito, n_frame, IoU, conf
        y = (n - 1 - i) * 1.06
        atteso, letto = t['atteso'], (t['letto'] or '')
        ax.text(-0.1, y + 0.52, f"#{t['track']}", fontsize=6.5, color=C_GREY, va='center')
        for p in range(7):
            cg = atteso[p]
            cp = letto[p] if p < len(letto) else '-'
            uguale = (cg == cp)
            col = C_OK if uguale else C_KO
            x = 1.15 + p * 0.78  # una colonna per ognuna delle 7 posizioni della targa
            ax.add_patch(Rectangle((x - 0.33, y + 0.03), 0.66, 0.94,
                                   facecolor=col, alpha=0.10 if uguale else 0.22,
                                   edgecolor=col, linewidth=0.8))
            ax.text(x, y + 0.70, cg, ha='center', va='center', fontsize=9,
                    family='monospace', color='#333')
            ax.text(x, y + 0.28, cp, ha='center', va='center', fontsize=9.5,
                    family='monospace', fontweight='bold', color=col)
        esito = "correct" if t['lev'] == 0 else f"{t['lev']} wrong"
        ax.text(7.2, y + 0.5, esito, fontsize=7.5, va='center',
                color=C_OK if t['lev'] == 0 else C_KO)
        ax.text(9.0, y + 0.5, f"{t['n_frame']} fr", fontsize=7.5, va='center', color=C_GREY)
        ax.text(10.4, y + 0.5, f"IoU {t['iou']:.2f}", fontsize=7.5, va='center', color=C_GREY)
        ax.text(12.1, y + 0.5, f"conf {t['conf']:.2f}", fontsize=7.5, va='center', color=C_GREY)
    ax.text(13.35, n * 1.06 - 0.10, "expected (top) / predicted (bottom)", fontsize=7,
            color=C_GREY, va='bottom', ha='right')


def _heatmap_confidenza(ax, traccia, max_col=40):
    """Disegna la probabilita' del carattere corretto, posizione per posizione e fotogramma per
    fotogramma. Il bordo nero segna le posizioni in cui il fotogramma,
    da solo, avrebbe scelto un carattere sbagliato. Il titolo riporta la lettura consolidata
    sull'intera traccia."""
    fr = sorted(traccia['frame'], key=lambda d: d['f'])
    if len(fr) < 2:
        ax.axis('off')
        ax.text(0, 0.5, "track too short for the temporal view", fontsize=9, color=C_GREY)
        return None
    cons = consolida_traccia(fr)
    nota_sub = ""
    if len(fr) > max_col:
        # Sottocampionamento delle colonne per stare nel formato A4 del PDF
        idx = np.linspace(0, len(fr) - 1, max_col).astype(int)
        fr = [fr[i] for i in idx]
        nota_sub = f" (subsampled to {max_col} of {len(traccia['frame'])} frames)"
    atteso = traccia['atteso']
    P = np.array([[d['prob_vero'][p] for d in fr] for p in range(7)])
    im = ax.imshow(P, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(fr)))
    ax.set_xticklabels([str(d['f']) for d in fr], fontsize=5.5, rotation=90)
    ax.set_yticks(range(7))
    ax.set_yticklabels([f"{p+1} {atteso[p]}" for p in range(7)], fontsize=8)
    ax.set_xlabel("frame", fontsize=8)
    ax.set_ylabel("position / true char", fontsize=8)
    for p in range(7):
        for j, d in enumerate(fr):
            if d['scelti'][p] != atteso[p]:
                ax.add_patch(Rectangle((j - 0.5, p - 0.5), 1, 1, fill=False,
                                       edgecolor='black', linewidth=0.6))
    letto = cons['testo'] if cons else ''
    esito = "correct" if letto == atteso else f"read as {letto or '-'}"
    ax.set_title(f"Temporal confidence - plate {atteso} ({esito}){nota_sub}"
                 f"   -   black outline: this frame alone would be wrong",
                 fontsize=9, loc='left')
    return im


def _pagina_titolo(pdf, D, tot, cons_tot, base, soglia):
    """Prima pagina del PDF: le cifre principali di tutto il dataset."""
    fig = plt.figure(figsize=A4)
    fig.text(0.07, 0.95, "Plate Recognition - Evaluation Report", fontsize=19, fontweight='bold')
    fig.text(0.07, 0.925, f"Results folder: {base}", fontsize=8, color=C_GREY)
    p, r, f1 = prf(tot)
    G = D['globale']
    totc = sum(v for row in G['conf'].values() for v in row.values())
    giusti = sum(G['conf'][c][c] for c in G['conf'])
    # Le coppie ("", "") sono righe vuote: lasciano uno spazio fra un gruppo di cifre e il successivo
    righe = [
        ("Videos evaluated", f"{len(D['video'])}"),
        ("Annotated boxes (ground truth)", f"{tot['tp'] + tot['fn']}"),
        ("Covered by the detector", f"{tot['gt_coperti']} "
                                    f"({tot['gt_coperti']/max(1,tot['tp']+tot['fn']):.1%})"),
        ("", ""),
        (f"True positives (IoU >= {soglia})", f"{tot['tp']}"),
        ("False positives", f"{tot['fp']}"),
        ("False negatives", f"{tot['fn']}"),
        ("Precision", f"{p:.3f}"),
        ("Recall", f"{r:.3f}"),
        ("F1-score", f"{f1:.3f}"),
        ("Mean IoU on true positives",
         f"{(float(np.mean(tot['iou_tp'])) if tot['iou_tp'] else 0):.3f}"),
        ("", ""),
        ("Readings evaluated", f"{tot['ocr_tot']}"),
        ("Exact match, single frame",
         f"{tot['ocr_esatti_frame']} ({tot['ocr_esatti_frame']/max(1,tot['ocr_tot']):.1%})"),
        ("Exact match, consolidated window",
         f"{tot['ocr_esatti']} ({tot['ocr_esatti']/max(1,tot['ocr_tot']):.1%})"),
        ("Mean Levenshtein distance",
         f"{(float(np.mean(tot['lev'])) if tot['lev'] else float('nan')):.2f} / 7"),
        ("Character accuracy", f"{giusti/max(1,totc):.1%}"),
        ("", ""),
        ("Physical plates (tracks)", f"{cons_tot['tracce']}"),
        ("Correct after consolidation",
         f"{cons_tot['esatte']} ({cons_tot['esatte']/max(1,cons_tot['tracce']):.1%})"),
        ("Correct by majority vote", f"{cons_tot['voto']}"),
        ("Correct in at least one frame", f"{cons_tot['miglior_frame']}"),
    ]
    y = 0.87
    for k, v in righe:
        if k:
            fig.text(0.10, y, k, fontsize=10.5)
            fig.text(0.66, y, v, fontsize=10.5, fontweight='bold')
        y -= 0.026
    fig.text(0.07, y - 0.02,
             "The report contains one or more pages per video: a character-by-character comparison of\n"
             "every annotated plate and the temporal evolution of the confidence along the selected\n"
             "tracks. Aggregated confusion matrices follow at the end.",
             fontsize=9, color=C_GREY, va='top')
    pdf.savefig(fig); plt.close(fig)


def _scegli_tracce(V, quante=2):
    """Sceglie le tracce da mostrare nella vista temporale: una letta male e una letta bene, se
    esistono. In ogni gruppo si prende la piu' lunga, per mostrare meglio il consolidamento."""
    esito = {t['track']: t['lev'] for t in V['targhe']}
    utili = [(tid, tr) for tid, tr in V['tracce'].items()
             if tr['atteso'] and len(tr['frame']) >= 2]
    utili.sort(key=lambda kv: -len(kv[1]['frame']))
    sbagliate = [tr for tid, tr in utili if esito.get(tid, 1) > 0]
    corrette = [tr for tid, tr in utili if esito.get(tid, 1) == 0]
    scelte = []
    for gruppo in (sbagliate, corrette):
        if gruppo and len(scelte) < quante:
            scelte.append(gruppo[0])
    # se una delle due categorie e' vuota, si completa con le tracce piu' lunghe rimaste
    for tid, tr in utili:
        if len(scelte) >= quante:
            break
        if tr not in scelte:
            scelte.append(tr)
    return scelte[:quante]


def _pagina_video(pdf, nome, V, soglia, max_targhe=10):
    """Pagine di un video: griglia di confronto delle targhe (al massimo max_targhe per pagina,
    quindi anche piu' pagine) e, solo nella prima, la vista temporale delle tracce scelte."""
    targhe = V['targhe']
    blocchi = [targhe[i:i + max_targhe] for i in range(0, len(targhe), max_targhe)] or [[]]
    tracce = _scegli_tracce(V)

    for b, blocco in enumerate(blocchi):
        fig = plt.figure(figsize=A4)
        p, r, f1 = prf(V)
        titolo = nome if len(blocchi) == 1 else f"{nome}  (page {b+1} of {len(blocchi)})"
        fig.text(0.07, 0.962, titolo, fontsize=17, fontweight='bold')
        esatte = sum(1 for t in targhe if t['lev'] == 0)
        tc = sum(x for row in V['conf'].values() for x in row.values())
        gc = sum(V['conf'][c][c] for c in V['conf'])
        fig.text(0.07, 0.938,
                 f"TP {V['tp']}   FP {V['fp']}   FN {V['fn']}   "
                 f"precision {p:.3f}   recall {r:.3f}   F1 {f1:.3f}",
                 fontsize=9, color='#444')
        fig.text(0.07, 0.920,
                 f"plates: {esatte}/{len(targhe)} fully correct"
                 + (f"      character accuracy: {gc/tc:.1%}  ({gc}/{tc})" if tc else ""),
                 fontsize=9, color='#444')

        y = 0.895  # cursore verticale: parte dall'alto e scende a ogni riquadro disegnato
        n = max(1, len(blocco))
        h = 0.028 + 0.036 * n
        ax1 = fig.add_axes([0.07, y - h, 0.88, h])
        _griglia_targhe(ax1, blocco, "Plate comparison")
        y -= h + 0.045

        # Vista temporale delle tracce scelte, solo nella prima pagina e solo se c'e' spazio
        if b == 0 and tracce:
            quante = max(0, min(len(tracce), int((y - 0.075) / 0.26)))
            for k in range(quante):
                h2 = 0.175
                base_y = y - h2 - 0.055
                ax = fig.add_axes([0.11, base_y, 0.78, h2])
                im = _heatmap_confidenza(ax, tracce[k])
                if im is not None:
                    cax = fig.add_axes([0.905, base_y, 0.013, h2])
                    cb = fig.colorbar(im, cax=cax)
                    cb.set_label("P(correct char)", fontsize=7.5)
                    cb.ax.tick_params(labelsize=6.5)
                y = base_y - 0.055
        pdf.savefig(fig); plt.close(fig)


def _pagina_matrici(pdf, D):
    """Ultima pagina del PDF: matrici di confusione aggregate, accuratezza per posizione e
    sostituzioni piu' frequenti."""
    G = D['globale']
    if not sum(v for r in G['conf'].values() for v in r.values()):
        return
    fig = plt.figure(figsize=A4)
    fig.text(0.07, 0.962, "Character confusion matrices (whole dataset)",
             fontsize=17, fontweight='bold')
    fig.text(0.07, 0.940, "Row-normalised: each cell is the share of that true character. "
                          "Values in per cent.", fontsize=8.5, color=C_GREY)
    ax1 = fig.add_axes([0.10, 0.545, 0.80, 0.355])
    im = _matrice_su_ax(ax1, G['conf'], sorted(LETTERE_VALIDE),
                        "Letters (positions 1, 2, 6, 7)")
    if im is not None:
        fig.colorbar(im, cax=fig.add_axes([0.915, 0.545, 0.014, 0.355]))
    ax2 = fig.add_axes([0.10, 0.285, 0.36, 0.20])
    _matrice_su_ax(ax2, G['conf'], sorted(CIFRE_VALIDE), "Digits (positions 3, 4, 5)")

    # Accuratezza per posizione: singolo fotogramma (grigio) contro lettura consolidata (blu)
    ax3 = fig.add_axes([0.58, 0.285, 0.34, 0.20])
    x = np.arange(7)
    acc = np.array([G['pos'][p, 0] / max(1, G['pos'][p, 1]) for p in range(7)])
    accf = np.array([G['pos_frame'][p, 0] / max(1, G['pos_frame'][p, 1]) for p in range(7)])
    ax3.bar(x - 0.2, accf, 0.38, color=C_GREY, alpha=0.85, label="single frame")
    ax3.bar(x + 0.2, acc, 0.38, color=C_BLU, alpha=0.9, label="temporal consolidation")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{p+1}\n{ETICHETTE_POS[p]}" for p in range(7)], fontsize=7.5)
    ax3.set_ylim(0, max(0.1, max(acc.max(), accf.max()) * 1.28))
    ax3.set_ylabel("character accuracy", fontsize=8)
    ax3.set_title("Accuracy per position", fontsize=9)
    ax3.legend(fontsize=7, frameon=False)
    ax3.tick_params(labelsize=7)
    for sp in ('top', 'right'):
        ax3.spines[sp].set_visible(False)

    conf = G['conf']
    tot = sum(v for r in conf.values() for v in r.values())
    giusti = sum(conf[c][c] for c in conf)
    coppie = sorted(((n, v, l) for v, r in conf.items() for l, n in r.items() if v != l),
                    reverse=True)
    # Elenco delle sostituzioni a sinistra e diagnosi a destra
    y = 0.235
    fig.text(0.07, y, "Most frequent substitutions", fontsize=11, fontweight='bold')
    y -= 0.021
    fig.text(0.10, y, f"{'true':>5s} {'pred':>6s} {'count':>7s} {'% of errors':>12s}",
             fontsize=8.5, family='monospace', color=C_GREY)
    n_err = tot - giusti
    yy = y
    for n, v, l in coppie[:10]:
        yy -= 0.016
        fig.text(0.10, yy, f"{v:>5s} {l:>6s} {n:7d} {n/max(1,n_err):11.1%}",
                 fontsize=8.5, family='monospace')
    if coppie:
        top10 = sum(n for n, _, _ in coppie[:10])
        q = top10 / max(1, n_err)
        diagnosi = ("CONCENTRATED error: a few recurring\nconfusions, hence structural and in\n"
                    "principle correctable by post-processing\nor targeted retraining."
                    if q > 0.5 else
                    "DIFFUSE error: no single confusion\ndominates, which is consistent with a\n"
                    "resolution limit rather than with a bias\nof the classifier.")
        fig.text(0.52, y,
                 f"Character accuracy: {giusti/max(1,tot):.1%}\n"
                 f"Distinct substitution pairs: {len(coppie)}\n"
                 f"Share of errors in the top ten: {q:.1%}\n\n{diagnosi}",
                 fontsize=9, color='#444', va='top', linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


def genera_pdf(percorso, D, tot, cons_tot, base, soglia):
    """Genera il report PDF: pagina riassuntiva, pagine per video e pagina delle matrici."""
    if not HA_MATPLOTLIB:
        print("\n[!] matplotlib non disponibile: report PDF saltato (pip install matplotlib)")
        return
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(percorso) as pdf:
        _pagina_titolo(pdf, D, tot, cons_tot, base, soglia)
        for nome in sorted(D['video']):
            _pagina_video(pdf, nome, D['video'][nome], soglia)
        _pagina_matrici(pdf, D)
        info = pdf.infodict()
        info['Title'] = "Plate Recognition - Evaluation Report"
        info['Subject'] = str(base)
    print(f"\nReport PDF -> {percorso}")


def prf(d):
    """Precision, recall e F1 da un dizionario con tp, fp, fn."""
    tp, fp, fn = d['tp'], d['fp'], d['fn']
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ================================================================================================
# MAIN
# ================================================================================================
def main():
    """Funzione che esegue la valutazione con i parametri della sezione CONFIGURAZIONE, stampa le
    sezioni da A a G e genera il report PDF."""
    soglia = SOGLIA_IOU
    if not Path(ANNOTATIONS_XML).is_file():
        print(f"ERRORE: annotations.xml non trovato in {ANNOTATIONS_XML}")
        print("Correggi ANNOTATIONS_XML in testa al file.")
        return 1
    base = Path(BASE_OUTPUT_SEGMENTATION)
    if not base.exists():
        print(f"ERRORE: cartella degli output non trovata: {base}")
        print("Correggi BASE_OUTPUT_SEGMENTATION in testa al file.")
        return 1

    gt, problemi = parse_cvat(ANNOTATIONS_XML)
    n_box = sum(len(b) for v in gt.values() for b in v.values())
    targhe = {b['targa'] for v in gt.values() for f in v.values() for b in f if b['targa']}
    print(f"Ground truth : {len(gt)} video, {n_box} box, {len(targhe)} targhe distinte")
    print(f"Output       : {base}")
    if problemi:
        print(f"ATTENZIONE: {len(problemi)} anomalie di parsing (prime 3): {problemi[:3]}")

    # La cartella puo' essere quella di un solo video o quella che contiene tutti i video
    cartelle = ([base] if (base / "segmenter_boxes.json").is_file()
                else sorted(d for d in base.iterdir() if d.is_dir()))
    if not cartelle:
        print(f"Nessun risultato della pipeline in {base}")
        return 1

    tot = {'tp': 0, 'fp': 0, 'fn': 0, 'iou_tp': [], 'gt_coperti': 0, 'gt_persi': 0,
           'ocr_tot': 0, 'ocr_letti': 0, 'ocr_esatti': 0, 'ocr_esatti_raw': 0,
           'ocr_esatti_frame': 0, 'lev_frame': [], 'n_fin': [],
           'lev': [], 'sim': [], 'conf_ok': [], 'conf_ko': [], 'motivi': defaultdict(int),
           'h_trovate': [], 'h_mancate': [], 'n_scartate': 0, 'area_ratio': [],
           'multi': {s: {'tp': 0, 'fp': 0, 'fn': 0} for s in SOGLIE_EXTRA}}
    stati_ocr_tot = defaultdict(int)
    cons_tot = {'tracce': 0, 'esatte': 0, 'lev': [], 'miglior_frame': 0, 'voto': 0, 'dettaglio': []}
    mancanti, senza_tabelle, ricostruite_tot = [], 0, 0

    # ------------------------------------------------------------------------------------------
    # A. LOCALIZZAZIONE
    # ------------------------------------------------------------------------------------------
    print(f"\n{'='*104}\n A. LOCALIZZAZIONE (soglia IoU {soglia})\n{'='*104}")
    print(f"{'video':16s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'prec':>6s} {'rec':>6s} {'F1':>6s} "
          f"{'IoU_m':>6s} {'OCR ok':>9s} {'lev':>6s}")
    print("-" * 104)
    for cart in cartelle:
        tenute, scartate, mancanti_tab, stati, n_ric = carica_predizioni(cart)
        if tenute is None:
            continue
        senza_tabelle += mancanti_tab
        ricostruite_tot += n_ric
        for k, v in stati.items():
            stati_ocr_tot[k] += v
        chiave = next((k for k in gt if k.lower() == cart.name.lower()), None)
        if chiave is None:
            mancanti.append(cart.name)
            continue
        st = valuta_video(gt[chiave], tenute, scartate, soglia)
        cons = consolida(st)
        p, r, f1 = prf(st)
        iou_m = float(np.mean(st['iou_tp'])) if st['iou_tp'] else 0.0
        lev_m = float(np.mean(st['lev'])) if st['lev'] else float('nan')
        ocr = f"{st['ocr_esatti']}/{st['ocr_tot']}" if st['ocr_tot'] else "-"
        print(f"{cart.name:16s} {st['tp']:6d} {st['fp']:6d} {st['fn']:6d} {p:6.3f} {r:6.3f} "
              f"{f1:6.3f} {iou_m:6.3f} {ocr:>9s} {lev_m:6.2f}")

        # Somma le statistiche del video dentro 'tot' e 'cons_tot': i contatori si sommano e le
        # liste si concatenano, cosi' alla fine 'tot' contiene il dataset intero
        for k in ('tp', 'fp', 'fn', 'gt_coperti', 'gt_persi', 'ocr_tot', 'ocr_letti',
                  'ocr_esatti', 'ocr_esatti_raw', 'ocr_esatti_frame', 'n_scartate'):
            tot[k] += st[k]
        for k in ('iou_tp', 'lev', 'sim', 'conf_ok', 'conf_ko', 'h_trovate', 'h_mancate',
                  'lev_frame', 'n_fin', 'area_ratio'):
            tot[k].extend(st[k])
        for m, c in st['motivi'].items():
            tot['motivi'][m] += c
        for s in SOGLIE_EXTRA:
            for k in ('tp', 'fp', 'fn'):
                tot['multi'][s][k] += st['multi'][s][k]
        for k in ('tracce', 'esatte', 'miglior_frame', 'voto'):
            cons_tot[k] += cons[k]
        cons_tot['lev'].extend(cons['lev'])
        cons_tot['dettaglio'].extend(cons['dettaglio'])

    p, r, f1 = prf(tot)
    print("-" * 104)
    print(f"{'TOTALE':16s} {tot['tp']:6d} {tot['fp']:6d} {tot['fn']:6d} {p:6.3f} {r:6.3f} {f1:6.3f} "
          f"{(float(np.mean(tot['iou_tp'])) if tot['iou_tp'] else 0):6.3f} "
          f"{str(tot['ocr_esatti'])+'/'+str(tot['ocr_tot']):>9s} "
          f"{(float(np.mean(tot['lev'])) if tot['lev'] else float('nan')):6.2f}")

    print("\nRisultati con soglie IoU diverse:")
    print(f"  {'soglia':>8s} {'precision':>10s} {'recall':>8s} {'F1':>8s}")
    for s in SOGLIE_EXTRA:
        pp, rr, ff = prf(tot['multi'][s])
        print(f"  {s:8.2f} {pp:10.3f} {rr:8.3f} {ff:8.3f}")
    print("  Se il recall crolla passando da 0.30 a 0.70 il contorno e' impreciso;")
    print("  se e' gia' basso a 0.30 la targa non viene trovata.")

    # ------------------------------------------------------------------------------------------
    # B. COSTO DEI FILTRI
    # ------------------------------------------------------------------------------------------
    print(f"\n{'='*104}\n B. COSTO DEL FILTRO ANTI-FALSI-POSITIVI\n{'='*104}")
    gt_tot = tot['tp'] + tot['fn']
    if gt_tot:
        print(f"  Targhe di ground truth                             : {gt_tot}")
        print(f"  Coperte da una detection ASRED (tenuta o scartata) : {tot['gt_coperti']} "
              f"({tot['gt_coperti']/gt_tot:.1%})  <- tetto raggiungibile")
        print(f"  Conservate dopo i controlli                        : {tot['tp']} ({tot['tp']/gt_tot:.1%})")
        print(f"  PERSE DAI CONTROLLI                                : {tot['gt_persi']} "
              f"({tot['gt_persi']/max(1,tot['gt_coperti']):.1%} di quelle disponibili)")
        if tot['motivi']:
            print("  Motivi (sono i parametri da allentare):")
            for m, c in sorted(tot['motivi'].items(), key=lambda kv: -kv[1])[:8]:
                print(f"      {c:6d}  {m}")
        if tot['n_scartate'] == 0:
            print("  NOTA: rejected_log.json non contiene detection scartate con poligono.")
        else:
            print(f"  (detection scartate lette dal log: {tot['n_scartate']})")

        # Altezza delle targhe trovate e di quelle mancate
        if tot['h_mancate'] and tot['h_trovate']:
            ht = np.array(tot['h_trovate'])
            hm = np.array(tot['h_mancate'])
            print(f"\n  Altezza in pixel delle targhe annotate:")
            print(f"      {'':12s} {'n':>7s} {'mediana':>9s} {'25o':>7s} {'75o':>7s} {'max':>7s}")
            print(f"      {'TROVATE':12s} {len(ht):7d} {np.median(ht):9.1f} "
                  f"{np.percentile(ht,25):7.1f} {np.percentile(ht,75):7.1f} {ht.max():7.1f}")
            print(f"      {'MANCATE':12s} {len(hm):7d} {np.median(hm):9.1f} "
                  f"{np.percentile(hm,25):7.1f} {np.percentile(hm,75):7.1f} {hm.max():7.1f}")
            if np.median(hm) < np.median(ht) * 0.7:
                print("      -> le targhe mancate sono piu' piccole: la distanza dalla camera e' una causa")
            else:
                print("      -> le targhe mancate hanno dimensioni simili a quelle trovate: la causa non e' la distanza")

            grandi = hm[hm >= np.percentile(ht, 25)]
            if grandi.size:
                print(f"      Targhe mancate pur essendo grandi almeno quanto il 25o percentile delle")
                print(f"      trovate: {grandi.size} ({grandi.size/len(hm):.1%} delle mancate)")

    # ------------------------------------------------------------------------------------------
    # C. OCR PER RILEVAZIONE
    # ------------------------------------------------------------------------------------------
    print(f"\n{'='*104}\n C. OCR PER RILEVAZIONE: SINGOLO FOTOGRAMMA vs FINESTRA CONSOLIDATA\n{'='*104}")
    if tot['n_fin']:
        nf = np.array(tot['n_fin'])
        print(f"  Consolidamento della pipeline: finestra di {np.median(nf):.0f} fotogrammi in mediana "
              f"(min {nf.min()}, max {nf.max()})")
        if tot['ocr_tot']:
            a = tot['ocr_esatti_frame'] / tot['ocr_tot']
            b = tot['ocr_esatti'] / tot['ocr_tot']
            print(f"  {'':34s} {'esatte':>10s} {'Levenshtein':>13s}")
            print(f"  {'singolo fotogramma':34s} {tot['ocr_esatti_frame']:5d} ({a:5.1%}) "
                  f"{(float(np.mean(tot['lev_frame'])) if tot['lev_frame'] else float('nan')):13.2f}")
            print(f"  {'finestra consolidata':34s} {tot['ocr_esatti']:5d} ({b:5.1%}) "
                  f"{(float(np.mean(tot['lev'])) if tot['lev'] else float('nan')):13.2f}")
            g = tot['ocr_esatti'] - tot['ocr_esatti_frame']
            print(f"  {'GUADAGNO della finestra':34s} {g:+5d} ({b-a:+5.1%})")
            if g <= 0:
                print("     -> il consolidamento non e' stato di aiuto.")
        print()
    if ricostruite_tot:
        print(f"  {ricostruite_tot} letture RICOSTRUITE dai file deepplate/tables/: nel JSON non")
        print(f"  c'erano, ma le tabelle sono sul disco e contengono tutto il necessario.")
    if stati_ocr_tot and tot['ocr_letti'] == 0:
        print(f"  Stato dell'OCR registrato dalla pipeline: {dict(stati_ocr_tot)}")
    if tot['ocr_tot']:
        print(f"  Targhe localizzate con GT nota   : {tot['ocr_tot']}")
        print(f"  Con una lettura prodotta         : {tot['ocr_letti']} ({tot['ocr_letti']/tot['ocr_tot']:.1%})")
        print(f"  Lette ESATTAMENTE (finestra)     : {tot['ocr_esatti']} ({tot['ocr_esatti']/tot['ocr_tot']:.1%})")
        print(f"  Singolo fotogramma, senza vincolo: {tot['ocr_esatti_raw']}  "
              f"(guadagno del vincolo LL-DDD-LL: {tot['ocr_esatti_frame']-tot['ocr_esatti_raw']:+d})")
        if tot['lev']:
            lev = np.array(tot['lev'])
            print(f"  Levenshtein medio                : {lev.mean():.2f} caratteri su 7")
            print(f"  Similarita' media                : {float(np.mean(tot['sim'])):.1%}")
            print(f"  Distribuzione errori             : "
                  f"{ {int(k): int((lev==k).sum()) for k in range(9) if (lev==k).sum()} }")
            print(f"  Letture a <=1 carattere dal vero : {int((lev<=1).sum())} "
                  f"({int((lev<=1).sum())/len(lev):.1%})")
    else:
        print("  Nessuna lettura OCR disponibile (campo ocr_text vuoto in segmenter_boxes.json).")

    # ------------------------------------------------------------------------------------------
    # D. OCR PER TRACCIA
    # ------------------------------------------------------------------------------------------
    print(f"\n{'='*104}\n D. OCR CONSOLIDATO SULL'INTERA TRACCIA (voto pesato, Alg. 3.7)\n{'='*104}")
    if cons_tot['tracce']:
        n = cons_tot['tracce']
        print(f"  Targhe fisiche (tracce con GT nota)                : {n}")
        print(f"  Corrette dopo il consolidamento pesato             : {cons_tot['esatte']} ({cons_tot['esatte']/n:.1%})")
        print(f"  Corrette col voto di maggioranza sui fotogrammi    : {cons_tot['voto']} "
              f"({cons_tot['voto']/n:.1%})   guadagno: {cons_tot['esatte']-cons_tot['voto']:+d}")
        print(f"  Corrette in almeno UN singolo fotogramma           : {cons_tot['miglior_frame']} "
              f"({cons_tot['miglior_frame']/n:.1%})")
        if cons_tot['lev']:
            print(f"  Levenshtein medio sulla lettura consolidata        : {np.mean(cons_tot['lev']):.2f}")
        sbagliate = [d for d in cons_tot['dettaglio'] if d['lev'] > 0]
        if sbagliate:
            print(f"\n  Targhe consolidate ancora errate (prime 15):")
            print(f"      {'attesa':>9s} {'letta':>9s} {'lev':>4s} {'frame':>6s} {'conf':>6s}")
            for d in sorted(sbagliate, key=lambda x: x['lev'])[:15]:
                print(f"      {d['attesa']:>9s} {str(d['consolidata']):>9s} {d['lev']:4d} "
                      f"{d['n_frame']:6d} {d['conf']:6.2f}")
    else:
        print("  Nessuna traccia consolidabile: mancano i file deepplate/tables/*.txt.")
    if senza_tabelle:
        print(f"  (attenzione: {senza_tabelle} tabelle DeepPlate non trovate sul disco)")

    # ------------------------------------------------------------------------------------------
    # E. CONFIDENZA DI DEEPPLATE
    # ------------------------------------------------------------------------------------------
    print(f"\n{'='*104}\n E. LA CONFIDENZA DI DEEPPLATE E' USABILE COME FILTRO?\n{'='*104}")
    cg, cs = tot['conf_ok'], tot['conf_ko']
    if cg and cs:
        mg, ms = float(np.mean(cg)), float(np.mean(cs))
        print(f"  Confidenza media quando e' GIUSTA   : {mg:.3f}  (n={len(cg)})")
        print(f"  Confidenza media quando e' SBAGLIATA: {ms:.3f}  (n={len(cs)})")
        print(f"  Separazione                         : {mg-ms:+.3f}")
        if mg - ms < 0.05:
            print("  -> Le distribuzioni coincidono: filtrare sulla confidenza NON serve.")
        else:
            # Soglia di confidenza che massimizza l'F1 fra letture giuste tenute e sbagliate tenute
            best, bf1 = None, -1
            for s in np.arange(0.10, 0.95, 0.05):
                tp = sum(1 for c in cg if c >= s)
                fp = sum(1 for c in cs if c >= s)
                pp = tp / (tp + fp) if tp + fp else 0
                rr = tp / len(cg)
                f = 2 * pp * rr / (pp + rr) if pp + rr else 0
                if f > bf1:
                    bf1, best = f, (s, pp, rr)
            print(f"  -> Soglia migliore: conf >= {best[0]:.2f} "
                  f"(precision {best[1]:.1%}, recall {best[2]:.1%})")
    else:
        print("  Dati insufficienti per valutare la calibrazione.")

    # ------------------------------------------------------------------------------------------
    # F. CONFRONTO INDIPENDENTE, G. MATRICE DI CONFUSIONE E REPORT PDF
    # ------------------------------------------------------------------------------------------
    confronto_targhe_indipendente(gt, cartelle)

    DC = raccogli_caratteri(gt, cartelle, soglia)
    stampa_confusione(DC)
    destinazione = base.parent / "evaluation_report_nuovo.pdf"
    genera_pdf(destinazione, DC, tot, cons_tot, base, soglia)

    if mancanti:
        print(f"\nCartelle senza ground truth corrispondente: {', '.join(mancanti)}")
        print("  (il nome della cartella deve coincidere col video annotato in CVAT)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
