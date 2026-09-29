# ==================================================================================================
# PROJECT: ARTHADRISHTI - SEMANTIC EO RETRIEVAL & MULTI-TEMPORAL CHANGE ANALYSIS
# PROBLEM STATEMENT: NTRO / SIH26227
# ARCHITECTURE: FLEXIBLE AIR-GAPPED DUAL-ENGINE OPERATIONAL SYSTEM
# ==================================================================================================

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.io import MemoryFile
from scipy.ndimage import uniform_filter, label

import streamlit as st

# Safe imports for Machine Learning dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ==================================================================================================
# 1. GLOBAL CONFIGURATION & DIRECTORY SETUP
# ==================================================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_archive"
TILES_DIR = BASE_DIR / "tiles_db"
MODELS_DIR = BASE_DIR / "offline_models"
AUDIT_LOG_FILE = BASE_DIR / "analyst_audit_log.csv"
INDEX_FILE = TILES_DIR / "faiss_catalog.index"
METADATA_FILE = TILES_DIR / "metadata.json"

TILE_SIZE = 256
EMBEDDING_DIM = 512
SAR_VARIANCE_WINDOW = 7
DEFAULT_SAR_DELTA_THRESH = 0.09
DEFAULT_SAR_VAR_THRESH = 0.020
DEFAULT_NDVI_THRESH = 0.10
DEFAULT_MNDWI_THRESH = 0.0
DEFAULT_SLOPE_THRESH = 20.0
MIN_CONNECTED_PIXELS = 20

# Standard taxonomy for Automated Semantic Auditing
DEFAULT_TAXONOMY = [
    "high-density urban residential buildings",
    "open water bodies and rivers",
    "agricultural crop fields",
    "dense forest canopy",
    "bare soil and cleared land",
    "industrial warehouse structures"
]

def initialize_workspace():
    """Ensure all required local directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

initialize_workspace()

def find_raster_file(filename: str) -> Path:
    """Locate raster files gracefully across potential directories."""
    if (DATA_DIR / filename).exists():
        return DATA_DIR / filename
    if (BASE_DIR / filename).exists():
        return BASE_DIR / filename
    return DATA_DIR / filename


# ==================================================================================================
# 2. AUDIT TRAIL ENGINE
# ==================================================================================================

def init_audit_log():
    """Initialize the CSV ledger if it does not exist."""
    if not AUDIT_LOG_FILE.exists():
        df = pd.DataFrame(columns=[
            "timestamp", "tile_id", "query_text", "analyst_decision", 
            "confidence_metric", "bounds_wgs84", "source_file", "notes"
        ])
        df.to_csv(AUDIT_LOG_FILE, index=False)

def log_decision(tile_id: str, query_text: str, decision: str, confidence: float, bounds: Any, source_file: str, notes: str = ""):
    """Commit an analyst decision to the immutable CSV ledger."""
    init_audit_log()
    bounds_str = json.dumps(bounds) if isinstance(bounds, (list, dict)) else str(bounds)
    record = pd.DataFrame([[
        datetime.utcnow().isoformat() + "Z", tile_id, query_text, decision, 
        f"{confidence:.4f}", bounds_str, source_file, notes
    ]], columns=[
        "timestamp", "tile_id", "query_text", "analyst_decision", 
        "confidence_metric", "bounds_wgs84", "source_file", "notes"
    ])
    record.to_csv(AUDIT_LOG_FILE, mode='a', header=False, index=False)

def load_audit_log() -> pd.DataFrame:
    """Retrieve the current audit log."""
    init_audit_log()
    try:
        return pd.read_csv(AUDIT_LOG_FILE)
    except Exception:
        return pd.DataFrame()


# ==================================================================================================
# 3. DETERMINISTIC RADAR & ENVIRONMENTAL CHANGE ENGINE
# ==================================================================================================

class MultiCriteriaChangeEngine:
    """Handles deterministic physics-based pixel suppression for SAR anomaly detection."""
    
    @staticmethod
    def db_to_linear(sar_db: np.ndarray) -> np.ndarray:
        clipped_db = np.clip(sar_db, -45.0, 15.0)
        return np.power(10.0, clipped_db / 10.0)

    @staticmethod
    def calculate_spatial_variance(linear_img: np.ndarray, window_size: int = 7) -> np.ndarray:
        mean = uniform_filter(linear_img, size=window_size)
        mean_sq = uniform_filter(linear_img ** 2, size=window_size)
        return np.maximum(mean_sq - (mean ** 2), 0.0)

    @staticmethod
    def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        denom = nir + red
        denom[denom == 0] = 1e-6
        return (nir - red) / denom

    @staticmethod
    def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
        denom = green + swir
        denom[denom == 0] = 1e-6
        return (green - swir) / denom

    @classmethod
    def apply_morphological_filter(cls, binary_mask: np.ndarray, min_connected_size: int = 20) -> np.ndarray:
        labeled_array, num_features = label(binary_mask)
        if num_features == 0:
            return binary_mask
        unique, counts = np.unique(labeled_array, return_counts=True)
        component_sizes = dict(zip(unique, counts))
        cleaned_mask = np.zeros_like(binary_mask, dtype=bool)
        for comp_id, size in component_sizes.items():
            if comp_id != 0 and size >= min_connected_size:
                cleaned_mask[labeled_array == comp_id] = True
        return cleaned_mask

    @classmethod
    def run_inference(cls, sar_baseline_db, sar_target_db, optical_base, optical_target, dem_slope, 
                      delta_thresh, var_thresh, ndvi_thresh, mndwi_thresh, slope_thresh, min_pixels):
        base_lin = cls.db_to_linear(sar_baseline_db)
        targ_lin = cls.db_to_linear(sar_target_db)
        
        delta_intensity = targ_lin - base_lin
        c1 = delta_intensity > delta_thresh
        
        targ_variance = cls.calculate_spatial_variance(targ_lin, window_size=SAR_VARIANCE_WINDOW)
        c2 = targ_variance > var_thresh
        
        raw_candidates = c1 & c2

        ndvi_base = cls.compute_ndvi(optical_base['nir'], optical_base['red'])
        ndvi_targ = cls.compute_ndvi(optical_target['nir'], optical_target['red'])
        phenology_mask = np.abs(ndvi_targ - ndvi_base) < ndvi_thresh

        mndwi_targ = cls.compute_mndwi(optical_target['green'], optical_target['swir'])
        water_mask = mndwi_targ < mndwi_thresh
        terrain_mask = dem_slope < slope_thresh

        survived_matrix = raw_candidates & phenology_mask & water_mask & terrain_mask
        final_structural_mask = cls.apply_morphological_filter(survived_matrix, min_pixels)

        raw_count = int(np.sum(raw_candidates))
        final_count = int(np.sum(final_structural_mask))

        return {
            "final_mask": final_structural_mask,
            "target_variance": targ_variance,
            "diagnostics": {
                "total_pixels_scanned": raw_candidates.size,
                "raw_candidate_pixels": raw_count,
                "suppressed_by_vegetation": int(np.sum(raw_candidates & (~phenology_mask))),
                "suppressed_by_water": int(np.sum(raw_candidates & phenology_mask & (~water_mask))),
                "suppressed_by_terrain": int(np.sum(raw_candidates & phenology_mask & water_mask & (~terrain_mask))),
                "final_candidate_pixels": final_count,
                "false_alarm_reduction_pct": (((raw_count - final_count) / raw_count * 100.0) if raw_count > 0 else 0.0)
            }
        }


# ==================================================================================================
# 4. FLEXIBLE SEMANTIC INDEXER (PHASE-1)
# ==================================================================================================

class FlexibleSearchEngine:
    """Manages AI weight initialization, vector embedding, and FAISS similarity search."""
    
    def __init__(self):
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.model_loaded = False
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.metadata = []
        self.faiss_index = None
        self._load_index()

    def load_clip_model(self, force_offline=False):
        if not OPEN_CLIP_AVAILABLE:
            raise Exception("open_clip module is not installed.")
            
        if force_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)

        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip_model = model.to(self.device).eval()
        self.clip_preprocess = preprocess
        self.model_loaded = True

    def _fallback_embedding(self, text_or_image):
        """Mathematical fallback for environments lacking PyTorch/Model Weights."""
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        if isinstance(text_or_image, str):
            seed = sum(ord(c) * (31 ** (i % 8)) for i, c in enumerate(text_or_image))
            rng = np.random.RandomState(seed % (2**31 - 1))
            vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        elif isinstance(text_or_image, Image.Image):
            small = text_or_image.convert('RGB').resize((16, 16))
            arr = np.array(small, dtype=np.float32).flatten()
            vec[:min(len(arr), EMBEDDING_DIM)] = arr[:EMBEDDING_DIM]
        
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        return vec

    def _load_index(self):
        if INDEX_FILE.exists() and METADATA_FILE.exists():
            try:
                if FAISS_AVAILABLE:
                    self.faiss_index = faiss.read_index(str(INDEX_FILE))
                with open(METADATA_FILE, "r") as f:
                    self.metadata = json.load(f)
                return
            except Exception: pass

        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM) if FAISS_AVAILABLE else None
        self.metadata = []

    def encode_text(self, text: str) -> np.ndarray:
        if self.model_loaded and self.clip_model:
            tokens = self.clip_tokenizer([text]).to(self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_text(tokens)
                feat /= feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy().astype(np.float32)
        return np.expand_dims(self._fallback_embedding(text), axis=0)

    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        if self.model_loaded and self.clip_model:
            tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_image(tensor)
                feat /= feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy()[0].astype(np.float32)
        return self._fallback_embedding(pil_img)

    def search(self, query_text: str, top_k: int = 5):
        if not self.metadata: return []
        text_vec = self.encode_text(query_text)

        if FAISS_AVAILABLE and self.faiss_index and self.faiss_index.ntotal > 0:
            k = min(top_k, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(text_vec, k)
            results = []
            for rank, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.metadata):
                    item = dict(self.metadata[idx])
                    item['relevance_score'] = float(distances[0][rank])
                    results.append(item)
            return results
        else:
            stored_embeddings = [
                self.encode_image(Image.open(m["filepath"])) if Path(m["filepath"]).exists() 
                else np.zeros(EMBEDDING_DIM, dtype=np.float32) 
                for m in self.metadata
            ]
            scores = np.dot(np.array(stored_embeddings), text_vec[0])
            sorted_indices = np.argsort(scores)[::-1][:top_k]
            return [ {**self.metadata[idx], 'relevance_score': float(scores[idx])} for idx in sorted_indices ]


# ==================================================================================================
# 5. RASTER INGESTION PIPELINE (PHASE-1 LOCAL DATABASE)
# ==================================================================================================

def ingest_raster(engine, raster_path):
    """Processes physical disk files into the local FAISS catalog."""
    with rasterio.open(raster_path) as src:
        for y in range(0, src.height - TILE_SIZE + 1, TILE_SIZE):
            for x in range(0, src.width - TILE_SIZE + 1, TILE_SIZE):
                win = Window(x, y, TILE_SIZE, TILE_SIZE)
                data = src.read(window=win)
                if np.all(data == 0): continue

                if data.shape[0] >= 3:
                    r = data[1].astype(np.float32)
                    g = data[0].astype(np.float32)
                    b = data[0].astype(np.float32)
                    rgb_float = np.stack([r, g, b], axis=-1)

                    p2, p98 = np.percentile(rgb_float, 2), np.percentile(rgb_float, 98)
                    if p98 > p2:
                        rgb_scaled = np.clip((rgb_float - p2) / (p98 - p2) * 255.0, 0, 255)
                    else:
                        rgb_scaled = np.clip(rgb_float * 255.0, 0, 255)
                    rgb = rgb_scaled.astype(np.uint8)

                    pil_chip = Image.fromarray(rgb)
                    t_id = len(engine.metadata)
                    save_path = TILES_DIR / f"tile_{t_id}.png"
                    pil_chip.save(save_path)

                    bounds = rasterio.windows.bounds(win, src.transform)
                    meta = {
                        "tile_id": t_id, 
                        "filepath": str(save_path), 
                        "source_file": Path(raster_path).name,
                        "bounds": [bounds[0], bounds[1], bounds[2], bounds[3]], 
                        "window": [x, y, TILE_SIZE, TILE_SIZE]
                    }
                    
                    emb = engine.encode_image(pil_chip)
                    if engine.faiss_index: engine.faiss_index.add(np.expand_dims(emb, axis=0))
                    engine.metadata.append(meta)

    if engine.faiss_index: faiss.write_index(engine.faiss_index, str(INDEX_FILE))
    with open(METADATA_FILE, "w") as f: json.dump(engine.metadata, f, indent=2)


# ==================================================================================================
# 6. DYNAMIC IN-MEMORY AUTO-AUDITOR (PHASE-2 PIPELINE)
# ==================================================================================================

class DynamicAutoAuditor:
    """Handles on-the-fly raster parsing, true-color generation, and automatic semantic tagging."""
    
    @staticmethod
    def process_memory_raster(file_bytes) -> Tuple[Image.Image, np.ndarray, Any]:
        """Reads uploaded bytes, normalizes to True Color RGB, and returns PIL image & metadata."""
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                crs = str(src.crs)
                count = src.count
                bounds = src.bounds
                
                # Dynamic band routing
                if count >= 3:
                    # Assuming standard optical: Extract B4(Red), B3(Green), B2(Blue) equivalent
                    raw_data = src.read([1, 2, 3]) 
                else:
                    # Replicate single band (SAR/DEM) to pseudocolor
                    band1 = src.read(1)
                    raw_data = np.stack([band1, band1, band1])

                # 2nd-98th Percentile Contrast Stretch
                p2, p98 = np.percentile(raw_data, (2, 98))
                if p98 > p2:
                    stretched = np.clip((raw_data - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                else:
                    stretched = np.clip(raw_data * 255.0, 0, 255).astype(np.uint8)

                # Convert (C, H, W) to (H, W, C) for Image generation
                rgb_array = np.transpose(stretched, (1, 2, 0))
                full_image = Image.fromarray(rgb_array)
                
                return full_image, rgb_array, {"crs": crs, "bands": count, "bounds": bounds}

    @staticmethod
    def extract_chips(rgb_array: np.ndarray, chip_size: int = 512, max_chips: int = 12) -> List[Dict]:
        """Slices the array into spatial chips for localized semantic analysis."""
        h, w, _ = rgb_array.shape
        chips = []
        chip_id = 0
        
        # Grid slicing (with early stopping to prevent memory overload in browser)
        for y in range(0, h - chip_size + 1, chip_size):
            for x in range(0, w - chip_size + 1, chip_size):
                if chip_id >= max_chips:
                    break
                chip_data = rgb_array[y:y+chip_size, x:x+chip_size, :]
                
                # Skip entirely black/empty tiles
                if np.mean(chip_data) < 5: 
                    continue
                    
                chips.append({
                    "id": f"dyn_chip_{chip_id}",
                    "image": Image.fromarray(chip_data),
                    "window": [x, y, chip_size, chip_size]
                })
                chip_id += 1
            if chip_id >= max_chips:
                break
                
        # If the image is smaller than chip size, return the whole image as one chip
        if not chips:
            chips.append({
                "id": "dyn_chip_0",
                "image": Image.fromarray(rgb_array),
                "window": [0, 0, w, h]
            })
            
        return chips

    @classmethod
    def run_automated_audit(cls, engine: FlexibleSearchEngine, chips: List[Dict], taxonomy: List[str]) -> List[Dict]:
        """Runs parallel zero-shot semantic matching for all chips against the taxonomy."""
        results = []
        
        # Pre-compute text embeddings for the entire taxonomy
        text_features = {label: engine.encode_text(label) for label in taxonomy}
        
        for chip in chips:
            img_feat = engine.encode_image(chip["image"])
            
            best_label = "Unclassified"
            best_score = 0.0
            
            # Compare image embedding against all taxonomy texts
            for label, txt_feat in text_features.items():
                score = np.dot(img_feat, txt_feat[0])
                if score > best_score:
                    best_score = score
                    best_label = label
                    
            # Threshold to prevent forced false positives
            if best_score > 0.22:
                results.append({
                    "chip_id": chip["id"],
                    "image": chip["image"],
                    "window": chip["window"],
                    "detected_class": best_label,
                    "confidence": float(best_score)
                })
                
        return results


# ==================================================================================================
# 7. STREAMLIT INTERFACE EXECUTION
# ==================================================================================================

st.set_page_config(page_title="ArthaDrishti - Semantic EO Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    h1, h2, h3, h4 { color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
    .stMetric { background-color: #1e293b; border: 1px solid #334155; padding: 12px; border-radius: 4px; }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; }
    .stButton>button { background-color: #0284c7; color: white; border: none; border-radius: 4px; font-weight: 600; }
    .stButton>button:hover { background-color: #0369a1; }
    .audit-card { background-color: #1e293b; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

if 'search_engine' not in st.session_state:
    st.session_state.search_engine = FlexibleSearchEngine()
if 'dynamic_results' not in st.session_state:
    st.session_state.dynamic_results = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = "N/A"

engine = st.session_state.search_engine

with st.sidebar:
    st.title("System Telemetry")
    st.markdown(f"**Execution Device:** {engine.device.upper()}")
    st.markdown(f"**Model Status:** {'Loaded (CLIP)' if engine.model_loaded else 'Native Mathematical Fallback'}")
    st.markdown(f"**Vector Engine:** {'FAISS' if FAISS_AVAILABLE else 'NumPy Space'}")
    st.markdown(f"**Indexed Tiles:** {len(engine.metadata)}")
    
    st.divider()
    st.subheader("AI Model Controls")
    force_offline = st.checkbox("Enforce Strict Air-Gap (Offline)", value=False)
    if st.button("Initialize Vision AI Weights"):
        with st.spinner("Loading PyTorch Models..."):
            try:
                engine.load_clip_model(force_offline=force_offline)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    st.divider()
    st.subheader("Data Management")
    if st.button("Index Archive"):
        with st.spinner("Targeting optical index..."):
            target_optical = find_raster_file("real_sentinel2_target.tif")
            if target_optical.exists():
                ingest_raster(engine, target_optical)
                st.success("Indexing complete.")
            else:
                st.error("Cannot find real_sentinel2_target.tif.")

    if st.button("Purge Database"):
        if TILES_DIR.exists(): shutil.rmtree(TILES_DIR)
        initialize_workspace()
        engine._load_index()
        st.warning("Database reset.")

tab_search, tab_change, tab_prov, tab_docs, tab_dynamic = st.tabs([
    "1. Semantic Retrieval", "2. Anomaly Verification", "3. Audit Trail", "4. Architecture", "5. Automated Auditor"
])

# -------------------------------------------------------------------------
# TAB 1: SEMANTIC RETRIEVAL
# -------------------------------------------------------------------------
with tab_search:
    st.subheader("Multimodal Natural Language Archive Interrogation")
    c1, c2 = st.columns([4, 1])
    with c1: 
        preset = st.selectbox("Target Signature Presets:", [
            "Custom Free-Text Query...", 
            "High-density urban residential blocks", 
            "Large industrial warehouse structures", 
            "Fluvial channels and river boundaries", 
            "Exposed bare soil and cleared land"
        ])
        query_input = st.text_input("Semantic Query Input:", value="dense urban structures") if preset == "Custom Free-Text Query..." else preset
    with c2: 
        top_k = st.number_input("Max Results:", min_value=1, max_value=24, value=4)

    if st.button("Execute Semantic Query"):
        st.session_state.last_results = engine.search(query_input, top_k=top_k)

    if 'last_results' in st.session_state and st.session_state.last_results:
        cols = st.columns(min(len(st.session_state.last_results), 4))
        for i, res in enumerate(st.session_state.last_results):
            with cols[i % 4]:
                if Path(res["filepath"]).exists(): 
                    st.image(Image.open(res["filepath"]), use_container_width=True)
                st.markdown(f"**ID:** {res['tile_id']} | **Score:** {res['relevance_score']:.4f}")
                b1, b2 = st.columns(2)
                if b1.button("Confirm", key=f"c_{res['tile_id']}"): 
                    log_decision(res['tile_id'], query_input, "CONFIRMED", res['relevance_score'], res['bounds'], res['source_file'])
                    st.success("Logged!")
                if b2.button("Reject", key=f"r_{res['tile_id']}"): 
                    log_decision(res['tile_id'], query_input, "REJECTED", res['relevance_score'], res['bounds'], res['source_file'])
                    st.error("Rejected.")
    elif 'last_results' in st.session_state:
        st.warning("No tiles indexed.")

# -------------------------------------------------------------------------
# TAB 2: ANOMALY VERIFICATION
# -------------------------------------------------------------------------
with tab_change:
    st.subheader("Multi-Source Environmental False-Alarm Suppression")
    with st.expander("Filter Calibration Parameters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        p_delta = f1.slider("SAR Delta I:", 0.01, 0.30, DEFAULT_SAR_DELTA_THRESH, 0.01)
        p_var = f2.slider("Variance (7x7):", 0.005, 0.100, DEFAULT_SAR_VAR_THRESH, 0.005)
        p_ndvi = f3.slider("NDVI Mask:", 0.02, 0.30, DEFAULT_NDVI_THRESH, 0.01)
        p_slope = f4.slider("Slope Cutoff:", 5.0, 45.0, DEFAULT_SLOPE_THRESH, 1.0)

    sar_b = find_raster_file("real_sentinel1_baseline.tif")
    sar_t = find_raster_file("real_sentinel1_target.tif")
    s2_t = find_raster_file("real_sentinel2_target.tif")
    dem = find_raster_file("real_srtm_slope.tif")
    
    if sar_b.exists() and sar_t.exists() and s2_t.exists() and dem.exists():
        with rasterio.open(sar_b) as sb, rasterio.open(sar_t) as st_t, rasterio.open(s2_t) as s2, rasterio.open(dem) as dm:
            w = min(512, sb.width, st_t.width, s2.width, dm.width)
            h = min(512, sb.height, st_t.height, s2.height, dm.height)
            win = Window(0, 0, w, h)
            
            sb_d = sb.read(1, window=win)
            st_d = st_t.read(1, window=win)
            s2_d = s2.read(window=win)
            dm_d = dm.read(1, window=win)
            
            res = MultiCriteriaChangeEngine.run_inference(
                sb_d, st_d, 
                {'green': s2_d[0]*0.95, 'red': s2_d[1]*0.95, 'nir': s2_d[2]*0.9, 'swir': s2_d[3]*0.95},
                {'green': s2_d[0], 'red': s2_d[1], 'nir': s2_d[2], 'swir': s2_d[3]}, 
                dm_d, p_delta, p_var, p_ndvi, 0.0, p_slope, 20
            )

        diag = res["diagnostics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Pixels Scanned", f"{diag['total_pixels_scanned']:,}")
        m2.metric("Raw SAR Flags", f"{diag['raw_candidate_pixels']:,}")
        m3.metric("Suppressed", f"{diag['suppressed_by_vegetation']+diag['suppressed_by_water']+diag['suppressed_by_terrain']:,}")
        m4.metric("Verified", f"{diag['final_candidate_pixels']:,}")
        m5.metric("Noise Reduction", f"{diag['false_alarm_reduction_pct']:.1f}%")

        v1, v2, v3 = st.columns(3)
        with v1: st.image(((np.clip(sb_d, -30, 5)+30)/35*255).astype(np.uint8), caption="Baseline SAR", use_container_width=True)
        with v2:
            norm_t = ((np.clip(st_d, -30, 5)+30)/35*255).astype(np.uint8)
            overlay = np.stack([norm_t]*3, axis=-1)
            overlay[res["final_mask"]] = [255, 30, 30]
            st.image(overlay, caption="Target + Validated Structures", use_container_width=True)
        with v3: st.image((np.clip(res["target_variance"]/0.05, 0, 1)*255).astype(np.uint8), caption="Spatial Variance", use_container_width=True)
    else: 
        st.info("Ensure the real GeoTIFF rasters are available.")

# -------------------------------------------------------------------------
# TAB 3 & 4: AUDIT & ARCHITECTURE
# -------------------------------------------------------------------------
with tab_prov:
    st.subheader("Analyst Decision Ledger")
    df = load_audit_log()
    if df.empty: 
        st.info("No decisions logged yet.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "audit_log.csv", "text/csv")

with tab_docs:
    st.subheader("Architectural Specifications")
    st.markdown("1. Radiometric Calibration:")
    st.latex(r"I_{\text{linear}} = 10^{\frac{I_{\text{dB}}}{10}}")
    st.markdown("2. Local Spatial Variance:")
    st.latex(r"\text{Var}(I) = \frac{1}{N}\sum_{i=1}^{N}(I_i - \mu)^2")
    st.markdown("3. Exclusion Matrix:")
    st.latex(r"M_{\text{final}} = (\Delta I > 0.09) \cap (\text{Var} > 0.020) \cap (\Delta\text{NDVI} < 0.10) \cap (\text{MNDWI} < 0.0)")

# -------------------------------------------------------------------------
# TAB 5: AUTOMATED AUDITOR (PHASE-2 UI INGESTION)
# -------------------------------------------------------------------------
with tab_dynamic:
    st.subheader("Live Raster Ingestion & Automated Feature Detection")
    st.markdown("Upload raw `.tif` data. The system automatically normalizes a True Color composite, slices the scene, and runs parallel AI inferences against multiple environmental taxonomies to suggest detections.")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Drop Satellite Data Here", type=["tif", "tiff"])
    
    if uploaded_file is not None:
        st.session_state.uploaded_filename = uploaded_file.name
        
        try:
            with st.spinner("Parsing Raster & Generating True Color Composite..."):
                full_image, rgb_array, meta = DynamicAutoAuditor.process_memory_raster(uploaded_file.getvalue())
                
            st.success(f"Processing Complete | CRS: {meta['crs']} | Bounds: {meta['bounds']}")
            
            # Display Full Image
            st.image(full_image, caption="Generated True Color Composite", use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Automated Semantic Sweep")
            st.markdown("Select target classifications for the AI to detect across this scene:")
            
            # Allow user to customize the taxonomy for the sweep
            selected_taxonomy = st.multiselect("Detection Taxonomy:", options=DEFAULT_TAXONOMY, default=DEFAULT_TAXONOMY[:3])
            
            if st.button("Run Global Detection Sweep"):
                if not engine.model_loaded:
                    st.error("Error: Please 'Initialize Vision AI Weights' from the sidebar first.")
                elif not selected_taxonomy:
                    st.warning("Please select at least one taxonomy class.")
                else:
                    with st.spinner("Extracting spatial chips and running parallel zero-shot matching..."):
                        # Slicing and Detection Execution
                        chips = DynamicAutoAuditor.extract_chips(rgb_array, chip_size=512)
                        detections = DynamicAutoAuditor.run_automated_audit(engine, chips, selected_taxonomy)
                        
                        st.session_state.dynamic_results = detections
            
            # Render Detection Ledger
            if st.session_state.dynamic_results is not None:
                results = st.session_state.dynamic_results
                
                if len(results) == 0:
                    st.info("No significant features from the taxonomy were detected in this scene.")
                else:
                    st.markdown(f"**Found {len(results)} potential structural/environmental matches.** Review and audit below:")
                    
                    # Create a grid layout for the detection cards
                    cols = st.columns(3)
                    for i, det in enumerate(results):
                        with cols[i % 3]:
                            st.markdown('<div class="audit-card">', unsafe_allow_html=True)
                            
                            st.image(det["image"], use_container_width=True)
                            st.markdown(f"**Classification:** {det['detected_class'].title()}")
                            st.markdown(f"**Confidence:** {det['confidence']*100:.1f}%")
                            
                            b1, b2 = st.columns(2)
                            if b1.button("✅ Confirm", key=f"d_c_{det['chip_id']}"):
                                log_decision(
                                    tile_id=det['chip_id'],
                                    query_text=det['detected_class'],
                                    decision="CONFIRMED",
                                    confidence=det['confidence'],
                                    bounds=det['window'],
                                    source_file=st.session_state.uploaded_filename
                                )
                                st.success("Logged.")
                                
                            if b2.button("❌ Reject", key=f"d_r_{det['chip_id']}"):
                                log_decision(
                                    tile_id=det['chip_id'],
                                    query_text=det['detected_class'],
                                    decision="REJECTED",
                                    confidence=det['confidence'],
                                    bounds=det['window'],
                                    source_file=st.session_state.uploaded_filename
                                )
                                st.error("Rejected.")
                                
                            st.markdown('</div>', unsafe_allow_html=True)
                            
        except Exception as e:
            st.error(f"Ingestion Error: {str(e)}")
            st.info("Ensure the uploaded file is a valid, uncorrupted GeoTIFF raster.")
