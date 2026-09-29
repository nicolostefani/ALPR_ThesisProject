"""
******************************************************************+
PIPELINE SOBEL 4K
******************************************************************+
Pipeline video -> ASRED -> ritagli -> segmentazione (Blackhat + Sobel X) -> DeepPlate -> segmenter_boxes.json.
Elabora in batch tutti i video 4K di una cartella: per ognuno chiama ASRED per detection/tracking,
ritaglia le bounding box, ne stima il contorno della targa e manda il ritaglio a DeepPlate per l'OCR.

Il JSON finale (segmenter_boxes.json) ha le stesse coordinate del frame originale, pronto per essere
confrontato con il ground truth in Evaluation.py.

"""
import os
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURAZIONE PERCORSI
# ==========================================
BASE_VIDEO_FOLDER = r"E:\DatasetNuovo\Video4k"
BASE_ASRED_FOLDER = r"E:\DatasetNuovo\pipeline_sobel_4k\Asred"
BASE_FRAMES_FOLDER = r"E:\DatasetNuovo\pipeline_sobel_4k\Frame_video"
BASE_OUTPUT_SEGMENTATION = r"E:\DatasetNuovo\pipeline_sobel_4k\output"

# Eseguibili esterni richiamati via subprocess (hash del video, tracking ASRED, OCR DeepPlate).
STRUMENTI_FOLDER = r"C:\Users\root\Desktop\programmi"
SCRIPT_HASH_PATH = os.path.join(STRUMENTI_FOLDER, "createHash.py")
ASRED_FOLDER = os.path.join(STRUMENTI_FOLDER, "asred")
ASRED_EXE_PATH = os.path.join(ASRED_FOLDER, "asred.exe")
DEEPPLATE_EXE_PATH = os.path.join(STRUMENTI_FOLDER, "deepPlate_thesis", "deepPlate_thesis.exe")

ESTENSIONI_VIDEO_VALIDE = ('.mp4',)


# ==========================================
# FUNZIONI ASRED E RITAGLI
# ==========================================
def filtra_json_targhe_semplice(file_input, file_output):
    """Filtra solo le detection/tracking con object_class == 'license_plate' e riscrive il JSON di ASRED."""
    try:
        with open(file_input, 'r', encoding='utf-8') as f:
            dati = json.load(f)

        nuovi_frames = []
        for frame in dati.get('frames', []):
            detection_filtrate = [d for d in frame.get('detection', []) if d.get('object_class') == 'license_plate']
            tracking_filtrati = [t for t in frame.get('tracking', []) if t.get('object_class') == 'license_plate']
            nuovi_frames.append({
                "idx_frame": frame.get("idx_frame"),
                "detection": detection_filtrate,
                "tracking": tracking_filtrati
            })

        with open(file_output, 'w', encoding='utf-8') as f:
            json.dump({"frames": nuovi_frames}, f, indent=4)
        logger.info(f"  [+] Filtraggio JSON completato.")
    except Exception as e:
        logger.error(f"  [-] Errore durante il filtraggio JSON: {e}")


def estrai_frame_e_ritagli(video_path, json_path, crop_dir, frames_dir):
    """Estrae i frame originali, salva i ritagli BB e ritorna un dizionario con gli offset (x,y) di ogni ritaglio."""
    logger.info("  [*] Avvio estrazione frame e crop delle Bounding Box...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dati = json.load(f)
    except FileNotFoundError:
        logger.error(f"  [-] File JSON non trovato: {json_path}. Salto l'estrazione BB.")
        return {}

    frames_data = {
        frame.get("idx_frame"): frame.get("detection", [])
        for frame in dati.get("frames", [])
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"  [-] Errore nell'apertura del video: {video_path}")
        return {}

    # Serve a ricordare dove si trovava ogni ritaglio nel frame originale, cosi' da poter
    # rimappare le coordinate della targa una volta che verra' segmentata sul crop.
    crop_offsets = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        nome_frame = f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(os.path.join(frames_dir, nome_frame), frame)

        if frame_idx in frames_data and frames_data[frame_idx]:
            for det_idx, det in enumerate(frames_data[frame_idx]):
                bb = det.get("bounding_box")
                if bb and len(bb) == 4:
                    x_min, y_min, x_max, y_max = map(lambda x: int(float(x)), bb)
                    h_img, w_img = frame.shape[:2]
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w_img, x_max), min(h_img, y_max)

                    if x_max > x_min and y_max > y_min:
                        crop = frame[y_min:y_max, x_min:x_max]
                        nome_crop = f"frame_{frame_idx:05d}_crop_{det_idx}.jpg"
                        cv2.imwrite(os.path.join(crop_dir, nome_crop), crop)

                        crop_offsets[nome_crop] = {
                            "frame_idx": str(frame_idx),
                            "offset_x": x_min,
                            "offset_y": y_min
                        }

        frame_idx += 1
    cap.release()
    logger.info(f"  [+] Estrazione e crop completati.")
    return crop_offsets


# ==========================================
# SEGMENTATORE BLACKHAT + SOBEL X E DEEPPLATE
# ==========================================
def _ksize_dispari(n):
    """cv2.GaussianBlur richiede kernel dispari: arrotonda n al piu' vicino dispari >= 1."""
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


class LicensePlateSegmenterSobel:
    """ 
    Individua la targa in un ritaglio usando la trasformata Blackhat seguita dal gradiente di
    Sobel sull'asse X: la prima esalta i caratteri scuri su sfondo chiaro, la seconda ne marca
    i bordi verticali cosi' da ottenere un blob compatto facilmente riconoscibile come contorno.
    
    Una volta trovato un contorno con proporzioni plausibili, il ritaglio viene passato a DeepPlate per l'OCR.
    """

    def __init__(self, dp_output_dir: str, bb_output_dir: str, exe_path: str):
        self.dp_output_dir = dp_output_dir
        self.bb_output_dir = bb_output_dir
        self.exe_path = exe_path

        # Valori calibrate su crop 4K
        self.kernel_w, self.kernel_h = 13, 5
        self.blur_ksize = _ksize_dispari(5)
        self.min_area_frac = 0.02   # rapporto sull'area del ritaglio, non dipende dalla risoluzione
        self.max_area_frac = 0.60
        self.min_long_side = 40     # px, lato lungo minimo della targa
        self.min_short_side = 15    # px, lato corto minimo della targa

    def preprocess_for_extraction(self, image: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        # Ritagli piu' piccoli del kernel di blur/chiusura mandano in errore GaussianBlur
        # (il bordo riflesso di default non ha abbastanza pixel su cui appoggiarsi).
        if h <= self.blur_ksize or w <= self.blur_ksize:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # Blackhat evidenzia le regioni scure su sfondo chiaro
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_w, self.kernel_h))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # Il gradiente di Sobel sull'asse X esalta i bordi verticali dei caratteri
        sobelx = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
        sobelx = np.absolute(sobelx)
        # Un ritaglio quasi uniforme puo' generare NaN/Inf durante la
        # normalizzazione: li azzeriamo prima del cast a uint8 per evitare un GaussianBlur
        # su un array degenere.
        sobelx = np.nan_to_num(sobelx, nan=0.0, posinf=0.0, neginf=0.0)
        (minVal, maxVal) = (float(np.min(sobelx)), float(np.max(sobelx)))
        if maxVal - minVal > 1e-6:
            sobelx = 255 * ((sobelx - minVal) / (maxVal - minVal))
        else:
            sobelx = np.zeros_like(sobelx)
        sobelx = sobelx.astype("uint8")

        sobelx = cv2.GaussianBlur(sobelx, (self.blur_ksize, self.blur_ksize), 0)
        # OTSU sceglie la soglia in automatico dall'istogramma dell'immagine
        _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # La chiusura morfologica compatta i singoli caratteri, spesso frammentati dalla
        # sogliatura, in un unico blob rettangolare riconoscibile come contorno.
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel, iterations=2)
        return thresh

    def order_points_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """Ordina i 4 vertici del box in senso orario (TL, TR, BR, BL): cv2.boxPoints non
        garantisce un ordine fisso, ma DeepPlate si aspetta i punti in sequenza."""
        rect = np.zeros((4, 2), dtype="float32")
        # Il punto in alto a sinistra ha la somma x+y minima, quello in basso a destra la massima.
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # La differenza x-y minima individua l'alto a destra,
        # quella massima il basso a sinistra.
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def is_valid_plate(self, contour: np.ndarray, image_area: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Scarta i contorni troppo piccoli/grandi o con proporzioni incompatibili con una
        targa italiana, sia essa di un'auto (rettangolare) o di una moto (quasi quadrata)."""
        area = cv2.contourArea(contour)
        if area < (image_area * self.min_area_frac) or area > (image_area * self.max_area_frac):
            return False, None

        # minAreaRect trova il rettangolo (anche ruotato) di area minima che racchiude il
        # contorno: serve per avere i 4 vertici della targa anche quando e' inclinata.
        rect = cv2.minAreaRect(contour)
        box = np.float32(cv2.boxPoints(rect))

        dim1, dim2 = rect[1][0], rect[1][1]
        if dim1 == 0 or dim2 == 0:
            return False, None

        long_side, short_side = max(dim1, dim2), min(dim1, dim2)
        if long_side < self.min_long_side or short_side < self.min_short_side:
            return False, None

        # Due fasce di rapporto lato lungo/lato corto accettate: una per le targhe quasi
        # quadrate delle moto, una per le targhe rettangolari standard delle auto.
        aspect_ratio = long_side / short_side
        is_moto_or_square = 0.8 <= aspect_ratio <= 1.8
        is_auto_standard = 3.0 <= aspect_ratio <= 6.0
        if not (is_moto_or_square or is_auto_standard):
            return False, None
        return True, box

    def draw_overlay(self, image: np.ndarray, pts: np.ndarray, status_text: str, status_color: tuple) -> np.ndarray:
        """Disegna sul ritaglio il poligono individuato e un pallino per ciascun angolo,
        utile in fase di debug per verificare a vista l'ordinamento dei punti e l'esito
        della chiamata a DeepPlate."""
        output_image = image.copy()
        int_pts = pts.astype(np.int32)
        cv2.polylines(output_image, [int_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Un colore diverso per angolo: rosso=TL, giallo=TR, blu=BR, magenta=BL.
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
        labels = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(int_pts):
            cv2.circle(output_image, (x, y), 5, colors[i], -1)
            cv2.putText(output_image, labels[i], (x - 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        h, w = output_image.shape[:2]
        cv2.putText(output_image, status_text, (w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        return output_image

    def process_folder(self, input_folder: str, crop_offsets: dict, output_json_path: str, extensions: List[str] = None) -> None:
        """Segmenta ogni ritaglio della cartella, manda le targhe trovate a DeepPlate e
        salva in output_json_path le bounding box (in coordinate del frame originale)
        da confrontare poi con il ground truth in Evaluation.py."""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']

        image_files = []
        for ext in extensions:
            image_files.extend(Path(input_folder).glob(f"*{ext}"))

        image_files = sorted(image_files)
        if not image_files: return

        total_processed = 0
        segmenter_boxes = {}

        for image_file in image_files:
          try:
            image_path_str = str(image_file)
            filename = image_file.name
            image = cv2.imread(image_path_str)
            if image is None: continue

            image_area = image.shape[0] * image.shape[1]
            thresh = self.preprocess_for_extraction(image)
            if thresh is None:
                out_img_path = os.path.join(self.bb_output_dir, f"{image_file.stem}_SCARTATA.jpg")
                cv2.imwrite(out_img_path, image)
                continue
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Teniamo solo i 10 contorni piu' estesi: la targa, se presente, e' quasi sempre
            # tra i blob piu' grandi individuati dal Sobel.
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            targa_trovata = False
            for contour in contours:
                is_valid, box = self.is_valid_plate(contour, image_area)
                if is_valid and box is not None:
                    ordered_pts = self.order_points_clockwise(box)
                    flat_coords = [str(int(coord)) for point in ordered_pts for coord in point]

                    comando = [
                        self.exe_path, "--input_path", image_path_str,
                        "--output_dir", self.dp_output_dir, "--country", "it", "--pos"
                    ] + flat_coords

                    status_text = ""
                    status_color = (0, 255, 0)
                    try:
                        subprocess.run(comando, check=True, capture_output=True, text=True)
                        status_text = "DeepPlate: OK"
                        total_processed += 1

                        if filename in crop_offsets:
                            f_idx = crop_offsets[filename]["frame_idx"]
                            off_x = crop_offsets[filename]["offset_x"]
                            off_y = crop_offsets[filename]["offset_y"]

                            # I punti trovati sono relativi al ritaglio (origine 0,0 in alto a
                            # sinistra del crop): sommando l'offset li riportiamo nel sistema di
                            # coordinate del frame originale, come si aspetta Evaluation.py.
                            punti_assoluti = (ordered_pts + np.array([off_x, off_y])).astype(np.int32)
                            x_seg, y_seg, w_seg, h_seg = cv2.boundingRect(punti_assoluti)
                            bb_finale = [int(x_seg), int(y_seg), int(x_seg + w_seg), int(y_seg + h_seg)]

                            if f_idx not in segmenter_boxes:
                                segmenter_boxes[f_idx] = []
                            segmenter_boxes[f_idx].append({
                                "crop_name": filename,
                                "bounding_box": bb_finale
                            })

                    except subprocess.CalledProcessError:
                        status_text = "DeepPlate: ERRORE"
                        status_color = (0, 0, 255)
                    except FileNotFoundError:
                        status_text = "EXE NON TROVATO"
                        status_color = (0, 0, 255)

                    annotated_image = self.draw_overlay(image, ordered_pts, status_text, status_color)
                    out_img_path = os.path.join(self.bb_output_dir, f"{image_file.stem}_annotated.jpg")
                    cv2.imwrite(out_img_path, annotated_image)
                    targa_trovata = True
                    break  # un solo contorno valido per ritaglio: appena lo troviamo ci fermiamo

            if not targa_trovata:
                out_img_path = os.path.join(self.bb_output_dir, f"{image_file.stem}_SCARTATA.jpg")
                cv2.imwrite(out_img_path, image)
          except Exception as e:
            logger.error(f"  [-] Errore imprevisto su {image_file.name}: {e}")
            continue

        logger.info(f"  -> Elaborate con successo {total_processed} immagini (inviate all'OCR).")

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(segmenter_boxes, f, indent=4)


# ==========================================
# PIPELINE
# ==========================================
if __name__ == "__main__":
    # Cartelle di output create una volta sola, prima di entrare nel loop sui video.
    os.makedirs(BASE_ASRED_FOLDER, exist_ok=True)
    os.makedirs(BASE_FRAMES_FOLDER, exist_ok=True)
    os.makedirs(BASE_OUTPUT_SEGMENTATION, exist_ok=True)

    if not os.path.exists(BASE_VIDEO_FOLDER):
        logger.error(f"La cartella di input non esiste: {BASE_VIDEO_FOLDER}")
        exit()

    elementi_video = [f for f in os.listdir(BASE_VIDEO_FOLDER) if f.lower().endswith(ESTENSIONI_VIDEO_VALIDE)]
    logger.info(f"Inizio elaborazione. Trovati {len(elementi_video)} video.\n")

    # Un video alla volta, dall'esecuzione di ASRED fino alla scrittura di segmenter_boxes.json.
    for file_video in elementi_video:
        file_video_path = os.path.join(BASE_VIDEO_FOLDER, file_video)
        nome_video_base = os.path.splitext(file_video)[0]

        logger.info(f"{'='*50}")
        logger.info(f"--- Elaborazione Video: {file_video} ---")
        logger.info(f"{'='*50}")

        # Ogni video ha la propria sottocartella
        output_asred_folder = os.path.join(BASE_ASRED_FOLDER, nome_video_base)
        output_frames_folder = os.path.join(BASE_FRAMES_FOLDER, nome_video_base)
        crop_frames_folder = os.path.join(output_asred_folder, f"{nome_video_base}_asred_frame")

        base_output_path = Path(BASE_OUTPUT_SEGMENTATION) / nome_video_base
        dp_output_dir = base_output_path / "deepplate"
        bb_output_dir = base_output_path / "boundingbox"

        os.makedirs(output_asred_folder, exist_ok=True)
        os.makedirs(output_frames_folder, exist_ok=True)
        os.makedirs(crop_frames_folder, exist_ok=True)
        dp_output_dir.mkdir(parents=True, exist_ok=True)
        bb_output_dir.mkdir(parents=True, exist_ok=True)

        # ASRED richiede l'hash del video come parametro, viene calcolato per mezzo di uno script esterno
        try:
            res_hash = subprocess.run(["python", SCRIPT_HASH_PATH, file_video_path], check=True, capture_output=True, text=True)
            hash_value = res_hash.stdout.strip()
        except subprocess.CalledProcessError:
            logger.error("  [-] Errore calcolo hash.")
            continue

        # Modalita' "all" = detection + tracking, eseguiti su GPU.
        comando_asred = [
            ASRED_EXE_PATH, "--input-path", file_video_path, "--output-dir", output_asred_folder,
            "--hash", hash_value, "--mode", "all", "--device", "cuda", "--trc-max-age", "120"
        ]
        try:
            logger.info("  [*] Esecuzione ASRED in corso...")
            subprocess.run(comando_asred, cwd=ASRED_FOLDER, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"  [-] Errore ASRED: {e}")
            continue

        json_input = os.path.join(output_asred_folder, "data.json")
        json_filtered = os.path.join(output_asred_folder, "data_filtered.json")

        # Se ASRED non ha prodotto nulla (es. nessuna detection nel video) si prosegue
        # comunque con crop_offsets vuoto, invece di far fallire tutta la pipeline.
        crop_offsets = {}
        if os.path.exists(json_input):
            filtra_json_targhe_semplice(json_input, json_filtered)
            crop_offsets = estrai_frame_e_ritagli(file_video_path, json_filtered, crop_frames_folder, output_frames_folder)

        logger.info(f"  [*] Esecuzione Segmentazione Blackhat+Sobel e DeepPlate...")
        segmenter = LicensePlateSegmenterSobel(
            dp_output_dir=str(dp_output_dir),
            bb_output_dir=str(bb_output_dir),
            exe_path=DEEPPLATE_EXE_PATH
        )

        json_segmenter_boxes = os.path.join(base_output_path, "segmenter_boxes.json")
        segmenter.process_folder(str(crop_frames_folder), crop_offsets, json_segmenter_boxes)

        print("\n")

    logger.info("Elaborazione completata per tutti i video!")
