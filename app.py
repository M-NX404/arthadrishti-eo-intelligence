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
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from PIL import Image

import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
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

def initialize_workspace():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

initialize_workspace()

def find_raster_file(filename: str) -> Path:
    if (DATA_DIR / filename).exists():
        return DATA_DIR / filename
    if (BASE_DIR / filename).exists():
        return BASE_DIR / filename
    return DATA_DIR / filename


# ==================================================================================================
# 2. AUDIT TRAIL ENGINE
# ==================================================================================================

def init_audit_log():
    if not AUDIT_LOG_FILE.exists():
        df = pd.DataFrame(columns=[
            "timestamp", "tile_id", "query_text", "analyst_decision", 
            "confidence_metric", "bounds_wgs84", "source_file", "notes"
        ])
        df.to_csv(AUDIT_LOG_FILE, index=False)

def log_decision(tile_id, query_text, decision, confidence, bounds, source_file, notes=""):
    init_audit_log()
    record = pd.DataFrame([[
        datetime.utcnow().isoformat() + "Z", tile_id, query_text, decision, 
        f"{confidence:.4f}", json.dumps(bounds), source_file, notes
    ]], columns=[
        "timestamp", "tile_id", "query_text", "analyst_decision", 
        "confidence_metric", "bounds_wgs84", "source_file", "notes"
    ])
    record.to_csv(AUDIT_LOG_FILE, mode='a', header=False, index=False)

def load_audit_log() -> pd.DataFrame:
    init_audit_log()
    try:
        return pd.read_csv(AUDIT_LOG_FILE)
    except Exception:
        return pd.DataFrame()


# ==================================================================================================
# 3. DETERMINISTIC RADAR & ENVIRONMENTAL CHANGE ENGINE
# ==================================================================================================

class MultiCriteriaChangeEngine:
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
# 4. FLEXIBLE SEMANTIC INDEXER
# ==================================================================================================

class FlexibleSearchEngine:
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
# 5. RASTER INGESTION PIPELINE
# ==================================================================================================

def ingest_raster(engine, raster_path):
    with rasterio.open(raster_path) as src:
        for y in range(0, src.height - TILE_SIZE + 1, TILE_SIZE):
            for x in range(0, src.width - TILE_SIZE + 1, TILE_SIZE):
                win = Window(x, y, TILE_SIZE, TILE_SIZE)
                data = src.read(window=win)
                if np.all(data == 0): continue

                if data.shape[0] >= 3:
                    # GEE Optical Export: B3(Green), B4(Red), B8(NIR), B11(SWIR)
                    # Create Pseudo-True Color: R=Red(1), G=Green(0), B=Green(0)
                    r = data[1].astype(np.float32)
                    g = data[0].astype(np.float32)
                    b = data[0].astype(np.float32)
                    rgb_float = np.stack([r, g, b], axis=-1)

                    # Auto-Contrast to fix washed out or dark imagery
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
# 6. STREAMLIT INTERFACE
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
</style>
""", unsafe_allow_html=True)

if 'search_engine' not in st.session_state:
    st.session_state.search_engine = FlexibleSearchEngine()

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
            # STRICT FIX: Only index the optical Sentinel-2 file for visual AI search
            target_optical = find_raster_file("real_sentinel2_target.tif")
            
            if target_optical.exists():
                ingest_raster(engine, target_optical)
                st.success("Indexing complete.")
            else:
                st.error("Cannot find real_sentinel2_target.tif. Make sure it is in your folder.")

    if st.button("Purge Database"):
        if TILES_DIR.exists(): 
            shutil.rmtree(TILES_DIR)
        initialize_workspace()
        engine._load_index()
        st.warning("Database reset.")

tab_search, tab_change, tab_prov, tab_docs = st.tabs([
    "1. Semantic Retrieval", "2. Anomaly Verification", "3. Audit Trail", "4. Architecture"
])

with tab_search:
    st.subheader("Multimodal Natural Language Archive Interrogation")
    c1, c2 = st.columns([4, 1])
    with c1: 
        # NEW FEATURE: Tactical Preset Dropdown + Custom Input
        preset = st.selectbox("Target Signature Presets:", [
            "Custom Free-Text Query...", 
            "High-density urban residential blocks", 
            "Large industrial warehouse structures", 
            "Fluvial channels and river boundaries", 
            "Exposed bare soil and cleared land"
        ])
        if preset == "Custom Free-Text Query...":
            query_input = st.text_input("Semantic Query Input:", value="dense urban structures")
        else:
            query_input = preset
            st.info(f"Executing signature: **{query_input}**")

    with c2: 
        top_k = st.number_input("Max Results:", min_value=1, max_value=24, value=4)

    # Save results to session state so they persist when secondary buttons are clicked
    if st.button("Execute Semantic Query"):
        st.session_state.last_results = engine.search(query_input, top_k=top_k)

    if 'last_results' in st.session_state:
        results = st.session_state.last_results
        if not results:
            st.warning("No tiles indexed.")
        else:
            cols = st.columns(min(len(results), 4))
            for i, res in enumerate(results):
                with cols[i % 4]:
                    if Path(res["filepath"]).exists(): 
                        st.image(Image.open(res["filepath"]), width="stretch")
                    st.markdown(f"**ID:** {res['tile_id']} | **Score:** {res['relevance_score']:.4f}")
                    
                    b1, b2 = st.columns(2)
                    if b1.button("Confirm", key=f"c_{res['tile_id']}"): 
                        log_decision(res['tile_id'], query_input, "CONFIRMED", res['relevance_score'], res['bounds'], res['source_file'])
                        st.success("Logged!")
                    if b2.button("Reject", key=f"r_{res['tile_id']}"): 
                        log_decision(res['tile_id'], query_input, "REJECTED", res['relevance_score'], res['bounds'], res['source_file'])
                        st.error("Rejected.")
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
            x_off = min(200, sb.width - w) if sb.width > w else 0
            y_off = min(200, sb.height - h) if sb.height > h else 0
            win = Window(x_off, y_off, w, h)
            
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
        with v1: 
            st.image(((np.clip(sb_d, -30, 5)+30)/35*255).astype(np.uint8), caption="Baseline SAR", width="stretch")
        with v2:
            norm_t = ((np.clip(st_d, -30, 5)+30)/35*255).astype(np.uint8)
            overlay = np.stack([norm_t]*3, axis=-1)
            overlay[res["final_mask"]] = [255, 30, 30]
            st.image(overlay, caption="Target + Validated Structures", width="stretch")
        with v3: 
            st.image((np.clip(res["target_variance"]/0.05, 0, 1)*255).astype(np.uint8), caption="Spatial Variance", width="stretch")
    else: 
        st.info("Ensure the real GeoTIFF rasters are available.")

with tab_prov:
    st.subheader("Analyst Decision Ledger")
    df = load_audit_log()
    if df.empty: 
        st.info("No logs.")
    else:
        st.dataframe(df, width="stretch")
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "audit_log.csv", "text/csv")

with tab_docs:
    st.subheader("Architectural Specifications")
    st.markdown("1. Radiometric Calibration:")
    st.latex(r"I_{\text{linear}} = 10^{\frac{I_{\text{dB}}}{10}}")
    st.markdown("2. Local Spatial Variance:")
    st.latex(r"\text{Var}(I) = \frac{1}{N}\sum_{i=1}^{N}(I_i - \mu)^2")
    st.markdown("3. Exclusion Matrix:")
    st.latex(r"M_{\text{final}} = (\Delta I > 0.09) \cap (\text{Var} > 0.020) \cap (\Delta\text{NDVI} < 0.10) \cap (\text{MNDWI} < 0.0) \cap (\text{Slope} < 20^\circ)")