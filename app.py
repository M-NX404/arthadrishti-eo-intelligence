# ==================================================================================================
# PLATFORM: ARTHADRISHTI - NATIONAL LEVEL EARTH OBSERVATION INTELLIGENCE
# SIH 2026 PROBLEM STATEMENT: NTRO / 26227
# CAPABILITIES: Sensor-Agnostic Ingestion, Absolute-Threshold Zero-Shot AI, 
#               Multi-Temporal Anomaly Detection, Sovereign Analyst Provenance.
# ==================================================================================================

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile
from scipy.ndimage import uniform_filter, label

import streamlit as st

# ==================================================================================================
# 1. ACCELERATION & ML DEPENDENCIES (WITH FAILSAFES)
# ==================================================================================================
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
# 2. SYSTEM CONSTANTS & TAXONOMY
# ==================================================================================================
TILE_SIZE = 256
EMBEDDING_DIM = 512
SIMILARITY_THRESHOLD = 0.225 # Absolute threshold to prevent AI hallucinations

# Advanced Multi-Prompt Taxonomy for Zero-Shot Classification
TAXONOMY_KNOWLEDGE_BASE = {
    "Water Body / River / Lake": [
        "satellite aerial view of open water body, lake, river, or ocean",
        "deep blue or dark surface water reservoir from above"
    ],
    "Dense Urban / Built-up": [
        "satellite aerial view of urban buildings, city blocks, concrete infrastructure",
        "dense residential housing and road grid from space"
    ],
    "Agricultural Cropland": [
        "satellite aerial view of agricultural fields, cropland, farmland plots",
        "geometric cultivated rural farming fields"
    ],
    "Forest / Dense Vegetation": [
        "satellite aerial view of dense forest canopy, woods, and wild trees",
        "thick green natural woodland cover from above"
    ],
    "Barren Land / Cleared Soil": [
        "satellite aerial view of bare soil, cleared earth, dirt, and arid land",
        "dry exposed ground without vegetation"
    ],
    "Industrial / Commercial Facilities": [
        "satellite aerial view of large industrial warehouse structures and factories",
        "commercial logistics hub with flat roof storage facilities"
    ]
}

# ==================================================================================================
# 3. STREAMLIT UI CONFIGURATION
# ==================================================================================================
st.set_page_config(page_title="ArthaDrishti | Semantic EO", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    h1, h2, h3, h4 { color: #f8fafc; font-family: 'Segoe UI', Tahoma, sans-serif; font-weight: 600; }
    .stMetric { background-color: #1e293b; border: 1px solid #334155; padding: 12px; border-radius: 6px; border-left: 4px solid #0284c7; }
    div[data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    .stButton>button { background-color: #0284c7; color: white; border: none; border-radius: 4px; font-weight: 600; width: 100%; transition: all 0.2s; }
    .stButton>button:hover { background-color: #0369a1; border-color: #38bdf8; }
    .audit-card { background-color: #162032; padding: 16px; border-radius: 8px; margin-bottom: 16px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .status-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
    .badge-unclassified { background-color: #475569; color: white; }
    .badge-classified { background-color: #059669; color: white; }
</style>
""", unsafe_allow_html=True)

# ==================================================================================================
# 4. SENSOR-AGNOSTIC DATA PIPELINE
# ==================================================================================================
class EarthObservationPipeline:
    """Handles parsing of generic GeoTIFFs into normalized machine-learning ready arrays."""
    
    @staticmethod
    def synthesize_true_color(file_bytes: bytes, r_idx: int, g_idx: int, b_idx: int) -> Tuple[Image.Image, np.ndarray, dict]:
        """Dynamically routes bands and normalizes radiometric contrast for any optical sensor."""
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                b_count = src.count
                meta = {
                    "crs": str(src.crs) if src.crs else "Unprojected",
                    "bands": b_count,
                    "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
                    "width": src.width, "height": src.height
                }

                # Single-band handling (e.g., SAR or Grayscale)
                if b_count == 1:
                    raw = np.nan_to_num(src.read(1).astype(np.float32))
                    p2, p98 = np.percentile(raw, 2), np.percentile(raw, 98)
                    norm = np.clip((raw - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
                    norm_uint8 = (norm * 255.0).astype(np.uint8)
                    rgb_array = np.stack([norm_uint8, norm_uint8, norm_uint8], axis=-1)
                
                # Multi-band True Color Composite
                else:
                    # Failsafe bounds check for user band inputs
                    ri = min(max(1, r_idx), b_count)
                    gi = min(max(1, g_idx), b_count)
                    bi = min(max(1, b_idx), b_count)

                    r = np.nan_to_num(src.read(ri).astype(np.float32))
                    g = np.nan_to_num(src.read(gi).astype(np.float32))
                    b = np.nan_to_num(src.read(bi).astype(np.float32))

                    stack = np.stack([r, g, b], axis=-1)
                    p2, p98 = np.percentile(stack, (2, 98))
                    
                    if p98 > p2:
                        rgb_array = np.clip((stack - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        rgb_array = np.clip(stack * 255.0, 0, 255).astype(np.uint8)

                return Image.fromarray(rgb_array), rgb_array, meta

    @staticmethod
    def extract_analytical_chips(rgb_array: np.ndarray, max_chips: int = 24) -> List[Dict]:
        """Slices massive rasters into standardized TILE_SIZE analytical chips."""
        h, w, _ = rgb_array.shape
        chips = []
        c_id = 0

        # Dynamic step sizing to cover the image without generating thousands of chips
        y_step = max(TILE_SIZE, (h - TILE_SIZE) // 4) if h > TILE_SIZE else TILE_SIZE
        x_step = max(TILE_SIZE, (w - TILE_SIZE) // 4) if w > TILE_SIZE else TILE_SIZE

        for y in range(0, max(1, h - TILE_SIZE + 1), y_step):
            for x in range(0, max(1, w - TILE_SIZE + 1), x_step):
                if c_id >= max_chips: break
                
                chip_arr = rgb_array[y:y+TILE_SIZE, x:x+TILE_SIZE, :]
                
                # Pad edges if chip is too small
                if chip_arr.shape[0] < TILE_SIZE or chip_arr.shape[1] < TILE_SIZE:
                    padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                    padded[:chip_arr.shape[0], :chip_arr.shape[1], :] = chip_arr
                    chip_arr = padded

                # Ignore pure black/nodata areas
                if np.mean(chip_arr) < 5.0: continue

                chips.append({
                    "id": f"spatial_chip_{c_id}",
                    "image": Image.fromarray(chip_arr),
                    "window": [x, y, TILE_SIZE, TILE_SIZE]
                })
                c_id += 1

        if not chips: # Fallback for tiny images
            chips.append({"id": "spatial_chip_0", "image": Image.fromarray(rgb_array), "window": [0,0,w,h]})
        return chips


# ==================================================================================================
# 5. ABSOLUTE-THRESHOLD SEMANTIC AI ENGINE
# ==================================================================================================
class ZeroShotSemanticCore:
    """Manages AI embeddings and absolute cosine similarity search."""
    def __init__(self):
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.active = False
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.prompt_vectors = {}
        
        # In-Memory Database for SIH demonstration
        self.vector_index: List[Dict] = []

    def boot_neural_core(self):
        if not OPEN_CLIP_AVAILABLE: raise RuntimeError("OpenCLIP library missing.")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.active = True
        self._cache_taxonomy()

    def _fallback_embed(self, data: Any) -> np.ndarray:
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        if isinstance(data, str):
            seed = sum(ord(c) * (7 ** (i%5)) for i,c in enumerate(data))
            rng = np.random.RandomState(seed % (2**31 - 1))
            vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        elif isinstance(data, Image.Image):
            arr = np.array(data.convert("RGB").resize((16,16)), dtype=np.float32).flatten()
            vec[:min(len(arr), EMBEDDING_DIM)] = arr[:EMBEDDING_DIM]
        return vec / (np.linalg.norm(vec) + 1e-7)

    def embed_text(self, text: str) -> np.ndarray:
        if self.active:
            tokens = self.tokenizer([text]).to(self.device)
            with torch.no_grad():
                v = self.model.encode_text(tokens)
                v /= v.norm(dim=-1, keepdim=True)
                return v.cpu().numpy()[0].astype(np.float32)
        return self._fallback_embed(text)

    def embed_image(self, img: Image.Image) -> np.ndarray:
        if self.active:
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                v = self.model.encode_image(tensor)
                v /= v.norm(dim=-1, keepdim=True)
                return v.cpu().numpy()[0].astype(np.float32)
        return self._fallback_embed(img)

    def _cache_taxonomy(self):
        """Pre-calculates mean embeddings for the multi-prompt taxonomy."""
        for label, prompts in TAXONOMY_KNOWLEDGE_BASE.items():
            vecs = [self.embed_text(p) for p in prompts]
            mean_v = np.mean(vecs, axis=0)
            self.prompt_vectors[label] = mean_v / (np.linalg.norm(mean_v) + 1e-7)

    def absolute_classification(self, img: Image.Image, targets: List[str]) -> Tuple[str, float]:
        """
        CRITICAL FIX: Uses Absolute Cosine Similarity instead of Softmax.
        If the highest score is below SIMILARITY_THRESHOLD, it returns "Unclassified".
        """
        img_v = self.embed_image(img)
        best_score = -1.0
        best_label = "Unclassified / Background"

        for label in targets:
            txt_v = self.prompt_vectors.get(label, self.embed_text(label))
            score = float(np.dot(img_v, txt_v))
            if score > best_score:
                best_score = score
                best_label = label

        # The Anti-Hallucination Gate
        if best_score < SIMILARITY_THRESHOLD:
            return "Unclassified / Background", best_score
            
        return best_label, best_score

    def ingest_to_memory(self, chip: Dict, source: str):
        v = self.embed_image(chip["image"])
        self.vector_index.append({
            "id": chip["id"], "source": source, "image": chip["image"], 
            "window": chip["window"], "vector": v
        })

    def semantic_search(self, query: str, top_k: int = 4) -> List[Dict]:
        if not self.vector_index: return []
        q_v = self.embed_text(query)
        results = []
        for item in self.vector_index:
            score = float(np.dot(q_v, item["vector"]))
            results.append({**item, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ==================================================================================================
# 6. MULTI-TEMPORAL CHANGE ENGINE
# ==================================================================================================
class AnomalyDetectionEngine:
    @staticmethod
    def compute_change(b_arr: np.ndarray, t_arr: np.ndarray, threshold: float) -> Tuple[np.ndarray, dict]:
        """Computes structural deviations between two unaligned/raw arrays."""
        # Align extents
        min_h, min_w = min(b_arr.shape[0], t_arr.shape[0]), min(b_arr.shape[1], t_arr.shape[1])
        b_crop, t_crop = b_arr[:min_h, :min_w], t_arr[:min_h, :min_w]
        
        b_gray = np.mean(b_crop, axis=-1) / 255.0
        t_gray = np.mean(t_crop, axis=-1) / 255.0
        
        diff = np.abs(t_gray - b_gray)
        raw_mask = diff > threshold
        
        # Morphological Noise Suppression (removes speckle < 15 pixels)
        labeled, num_features = label(raw_mask)
        unique, counts = np.unique(labeled, return_counts=True)
        clean_mask = np.zeros_like(raw_mask, dtype=bool)
        for cid, size in zip(unique, counts):
            if cid != 0 and size >= 15:
                clean_mask[labeled == cid] = True
                
        metrics = {
            "total_pixels": raw_mask.size,
            "anomalies": int(np.sum(clean_mask)),
            "noise_suppressed": int(np.sum(raw_mask) - np.sum(clean_mask))
        }
        return clean_mask, metrics


# ==================================================================================================
# 7. STATE MANAGEMENT & SIDEBAR
# ==================================================================================================
if "ai_engine" not in st.session_state: st.session_state.ai_engine = ZeroShotSemanticCore()
if "audit_log" not in st.session_state: st.session_state.audit_log = []

ai: ZeroShotSemanticCore = st.session_state.ai_engine

with st.sidebar:
    st.title("ArthaDrishti Control")
    st.markdown(f"**Hardware:** `{ai.device.upper()}`")
    st.markdown(f"**AI Status:** `{'ONLINE' if ai.active else 'OFFLINE'}`")
    st.markdown(f"**Indexed Chips:** `{len(ai.vector_index)}`")
    
    st.divider()
    if not ai.active:
        if st.button("Initialize Neural Core", type="primary"):
            with st.spinner("Loading Vision Transformer..."):
                ai.boot_neural_core()
                st.rerun()
    else:
        st.success("System Ready for Operations.")
        
    st.divider()
    if st.button("Purge Database & Reset"):
        st.session_state.ai_engine = ZeroShotSemanticCore()
        st.session_state.audit_log = []
        for key in ["active_img", "active_arr", "active_name", "scan_results"]:
            if key in st.session_state: del st.session_state[key]
        st.rerun()


# ==================================================================================================
# 8. PRIMARY WORKFLOW INTERFACE
# ==================================================================================================
tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs([
    "1. Data Management & TCC", 
    "2. Semantic Scanner", 
    "3. Archive Retrieval", 
    "4. Anomaly Engine", 
    "5. Analyst Provenance"
])

# --------------------------------------------------------------------------------------------------
# TAB 1: DATA MANAGEMENT & TRUE COLOR SYNTHESIS
# --------------------------------------------------------------------------------------------------
with tab_1:
    st.subheader("Sensor-Agnostic Ingestion")
    st.info("⚠️ **CRITICAL FOR AI ACCURACY:** The Vision AI requires True Color (RGB). If you upload multi-spectral data (like Sentinel-2 or USGS Landsat), you MUST map the bands correctly below to generate a natural looking image. False Color (red vegetation) will cause the AI to fail.")
    
    upload = st.file_uploader("Upload GeoTIFF Raster", type=["tif", "tiff"])
    
    if upload:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### Band Alignment")
            r_idx = st.number_input("Red Channel (Band #)", min_value=1, max_value=20, value=1)
            g_idx = st.number_input("Green Channel (Band #)", min_value=1, max_value=20, value=2)
            b_idx = st.number_input("Blue Channel (Band #)", min_value=1, max_value=20, value=3)
            proc_btn = st.button("Synthesize Image", type="primary")
            
        if proc_btn or "active_img" in st.session_state:
            if proc_btn:
                with st.spinner("Parsing radiometry..."):
                    img, arr, meta = EarthObservationPipeline.synthesize_true_color(upload.getvalue(), r_idx, g_idx, b_idx)
                    st.session_state.active_img = img
                    st.session_state.active_arr = arr
                    st.session_state.active_name = upload.name
                    st.session_state.active_meta = meta
                    
            with c2:
                st.markdown("#### Telemetry")
                st.write(f"**Source:** `{st.session_state.active_name}` | **CRS:** `{st.session_state.active_meta['crs']}`")
                st.image(st.session_state.active_img, caption="True Color Composite (TCC)", use_container_width=True)

# --------------------------------------------------------------------------------------------------
# TAB 2: AUTOMATED SEMANTIC SCANNER (WITH ABSOLUTE THRESHOLD)
# --------------------------------------------------------------------------------------------------
with tab_2:
    st.subheader("Automated Multi-Phenomena Sweep")
    st.markdown("Slices the active raster into chips and uses absolute cosine similarity to prevent classification hallucinations.")
    
    if "active_arr" not in st.session_state:
        st.warning("Process a raster in Tab 1 first.")
    else:
        targets = st.multiselect("Select Target Phenomena:", list(TAXONOMY_KNOWLEDGE_BASE.keys()), default=["Water Body / River / Lake", "Dense Urban / Built-up"])
        
        if st.button("Run Semantic Sweep", type="primary"):
            if not ai.active: st.error("Initialize Neural Core first.")
            elif not targets: st.warning("Select targets.")
            else:
                with st.spinner("Extracting chips and classifying..."):
                    chips = EarthObservationPipeline.extract_analytical_chips(st.session_state.active_arr)
                    results = []
                    for c in chips:
                        ai.ingest_to_memory(c, st.session_state.active_name)
                        label, score = ai.absolute_classification(c["image"], targets)
                        results.append({"chip": c, "label": label, "score": score})
                    st.session_state.scan_results = results
                    st.success("Sweep Complete.")

        if "scan_results" in st.session_state:
            st.markdown("### Feature Detections & Analyst Review")
            cols = st.columns(4)
            for i, res in enumerate(st.session_state.scan_results):
                with cols[i % 4]:
                    st.markdown('<div class="audit-card">', unsafe_allow_html=True)
                    st.image(res["chip"]["image"], use_container_width=True)
                    
                    if "Unclassified" in res["label"]:
                        st.markdown(f'<span class="status-badge badge-unclassified">Unclassified Background</span>', unsafe_allow_html=True)
                        st.caption(f"Max Similarity: {res['score']:.2f} (Below Threshold)")
                    else:
                        st.markdown(f'<span class="status-badge badge-classified">DETECTED: {res["label"]}</span>', unsafe_allow_html=True)
                        st.markdown(f"**Confidence:** `{res['score']*100:.1f}%`")
                        
                        b1, b2 = st.columns(2)
                        if b1.button("✅", key=f"y_{res['chip']['id']}"):
                            st.session_state.audit_log.append({
                                "Time": datetime.utcnow().strftime("%H:%M:%S"), "Tile": res["chip"]["id"], 
                                "Class": res["label"], "Action": "CONFIRMED", "Source": st.session_state.active_name
                            })
                            st.toast("Confirmed.")
                        if b2.button("❌", key=f"n_{res['chip']['id']}"):
                            st.session_state.audit_log.append({
                                "Time": datetime.utcnow().strftime("%H:%M:%S"), "Tile": res["chip"]["id"], 
                                "Class": res["label"], "Action": "REJECTED", "Source": st.session_state.active_name
                            })
                            st.toast("Rejected.")
                    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------
# TAB 3: NATURAL LANGUAGE ARCHIVE RETRIEVAL
# --------------------------------------------------------------------------------------------------
with tab_3:
    st.subheader("Global Archive Interrogation")
    q = st.text_input("Enter physical characteristic to search memory bank:", "large dense urban housing blocks")
    n = st.number_input("Max Results", 1, 12, 4)
    
    if st.button("Search Archive", type="primary"):
        if not ai.active: st.error("Initialize Neural Core.")
        elif not ai.vector_index: st.warning("Archive empty. Sweep an image in Tab 2 to ingest chips.")
        else:
            with st.spinner("Computing cosine distances..."):
                hits = ai.semantic_search(q, n)
            cols = st.columns(4)
            for i, h in enumerate(hits):
                with cols[i % 4]:
                    st.image(h["image"], use_container_width=True)
                    st.markdown(f"**Score:** `{h['score']*100:.1f}%`")
                    st.caption(f"Src: {h['source']}")

# --------------------------------------------------------------------------------------------------
# TAB 4: DYNAMIC ANOMALY ENGINE
# --------------------------------------------------------------------------------------------------
with tab_4:
    st.subheader("Multi-Temporal Structural Change Matrix")
    c_b, c_t = st.columns(2)
    b_up = c_b.file_uploader("T0: Baseline Raster", type=["tif"])
    t_up = c_t.file_uploader("T1: Target Raster", type=["tif"])
    
    if b_up and t_up:
        thresh = st.slider("Absolute Difference Threshold", 0.05, 0.50, 0.15, 0.01)
        if st.button("Compute Morphological Change", type="primary"):
            with st.spinner("Processing differential arrays..."):
                _, b_arr, _ = EarthObservationPipeline.synthesize_true_color(b_up.getvalue(), 1, 2, 3)
                _, t_arr, _ = EarthObservationPipeline.synthesize_true_color(t_up.getvalue(), 1, 2, 3)
                
                mask, metrics = AnomalyDetectionEngine.compute_change(b_arr, t_arr, thresh)
                
                # Match target size for overlay
                overlay = t_arr[:mask.shape[0], :mask.shape[1]].copy()
                overlay[mask] = [255, 20, 20] # Bright red anomalies
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Area Scanned (px)", f"{metrics['total_pixels']:,}")
                m2.metric("Confirmed Anomalies", f"{metrics['anomalies']:,}")
                m3.metric("Speckle Suppressed", f"{metrics['noise_suppressed']:,}")
                
                v1, v2 = st.columns(2)
                v1.image(b_arr[:mask.shape[0], :mask.shape[1]], caption="Baseline (T0)", use_container_width=True)
                v2.image(overlay, caption="Target (T1) + Anomaly Matrix", use_container_width=True)

# --------------------------------------------------------------------------------------------------
# TAB 5: ANALYST PROVENANCE & AUDIT
# --------------------------------------------------------------------------------------------------
with tab_5:
    st.subheader("Immutable Analyst Chain of Custody")
    if not st.session_state.audit_log:
        st.info("No decisions logged. Validate features in Tab 2.")
    else:
        df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(df, use_container_width=True)
        st.download_button("Export SOV Audit Log", df.to_csv(index=False).encode('utf-8'), "audit.csv", "text/csv")
