# ==================================================================================================
# PROJECT: ARTHADRISHTI - SENSOR-AGNOSTIC EO INTELLIGENCE & SEMANTIC RETRIEVAL PLATFORM
# PROBLEM STATEMENT: NTRO / SIH26227
# ARCHITECTURE: 100% IN-MEMORY DYNAMIC INGESTION, ZERO-SHOT AI, DETERMINISTIC CHANGE DETECTION
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

# Safe imports for Machine Learning acceleration
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
# 1. CORE SYSTEM CONFIGURATION & UI STYLING
# ==================================================================================================

TILE_SIZE = 256
EMBEDDING_DIM = 512

# Remote Sensing Multi-Prompt Taxonomy for Zero-Shot Classification
EO_PROMPT_TAXONOMY = {
    "Water Body / River / Lake": [
        "satellite aerial view of open water body, lake, river, or ocean",
        "radar SAR low backscatter calm dark water surface",
        "deep blue or dark surface water reservoir"
    ],
    "Dense Urban / Built-up": [
        "satellite aerial view of urban buildings, city blocks, concrete infrastructure",
        "dense residential housing and road grid",
        "radar high backscatter bright metallic building reflection"
    ],
    "Agricultural Cropland": [
        "satellite aerial view of agricultural fields, cropland, farmland plots",
        "geometric cultivated rural farming fields",
        "vegetation crop rows and agricultural plantations"
    ],
    "Forest / Dense Vegetation": [
        "satellite aerial view of dense forest canopy, woods, and wild trees",
        "thick green natural woodland cover",
        "dense tropical jungle or hillside forest"
    ],
    "Barren Land / Cleared Soil": [
        "satellite aerial view of bare soil, cleared earth, dirt, and arid land",
        "dry exposed ground without vegetation",
        "sand, gravel, and unpaved excavation ground"
    ],
    "Industrial / Commercial Facilities": [
        "satellite aerial view of large industrial warehouse structures and factories",
        "commercial logistics hub with flat roof storage facilities",
        "large shipping and manufacturing installations"
    ]
}

st.set_page_config(
    page_title="ArthaDrishti | Semantic EO Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    h1, h2, h3, h4 { color: #f8fafc; font-family: 'Segoe UI', Tahoma, sans-serif; font-weight: 600; }
    .stMetric { background-color: #1e293b; border: 1px solid #334155; padding: 12px; border-radius: 6px; }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; }
    .stButton>button { background-color: #0284c7; color: white; border: none; border-radius: 4px; font-weight: 600; width: 100%; transition: all 0.2s; }
    .stButton>button:hover { background-color: #0369a1; border-color: #38bdf8; }
    .audit-card { background-color: #1e293b; padding: 14px; border-radius: 8px; margin-bottom: 16px; border: 1px solid #334155; }
    .tag-water { color: #38bdf8; font-weight: bold; }
    .tag-urban { color: #f97316; font-weight: bold; }
    .tag-agri { color: #4ade80; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==================================================================================================
# 2. SENSOR-AGNOSTIC IMAGE ENGINE (PURE NUMPY + PILLOW, NO OPENCV)
# ==================================================================================================

class SensorAgnosticProcessor:
    """Handles universal GeoTIFF ingestion, dynamic True-Color synthesis, and contrast normalization."""

    @staticmethod
    def process_raster_bytes(file_bytes: bytes, red_ch: int = 1, green_ch: int = 2, blue_ch: int = 3) -> Tuple[Image.Image, np.ndarray, dict]:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                band_count = src.count
                crs_str = str(src.crs) if src.crs else "Unprojected / Local"
                bounds = src.bounds
                width, height = src.width, src.height
                
                meta = {
                    "crs": crs_str,
                    "bands": band_count,
                    "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                    "dimensions": f"{width} x {height}",
                    "driver": src.driver
                }

                # Single-band: SAR (Sentinel-1), DEM (SRTM), or Panchromatic
                if band_count == 1:
                    raw = src.read(1).astype(np.float32)
                    # Replace NaN / Inf values
                    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # 2nd-98th percentile stretch for radar/topography
                    p2, p98 = np.percentile(raw, 2), np.percentile(raw, 98)
                    if p98 > p2:
                        norm = np.clip((raw - p2) / (p98 - p2), 0.0, 1.0)
                    else:
                        norm = np.clip(raw, 0.0, 1.0)
                    
                    norm_uint8 = (norm * 255.0).astype(np.uint8)
                    # 3-channel grayscale stack (compatible with standard vision models)
                    rgb_array = np.stack([norm_uint8, norm_uint8, norm_uint8], axis=-1)

                # Multi-band: Optical (Sentinel-2, Landsat/USGS, Planet)
                else:
                    # Guard band indices to prevent out-of-bounds reads
                    r_idx = min(max(1, red_ch), band_count)
                    g_idx = min(max(1, green_ch), band_count)
                    b_idx = min(max(1, blue_ch), band_count)

                    r_band = np.nan_to_num(src.read(r_idx).astype(np.float32))
                    g_band = np.nan_to_num(src.read(g_idx).astype(np.float32))
                    b_band = np.nan_to_num(src.read(b_idx).astype(np.float32))

                    stack = np.stack([r_band, g_band, b_band], axis=-1)

                    # Contrast stretch across the combined RGB array
                    p2, p98 = np.percentile(stack, 2), np.percentile(stack, 98)
                    if p98 > p2:
                        rgb_array = np.clip((stack - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        rgb_array = np.clip(stack * 255.0, 0, 255).astype(np.uint8)

                pil_image = Image.fromarray(rgb_array)
                return pil_image, rgb_array, meta

    @staticmethod
    def extract_spatial_chips(rgb_array: np.ndarray, chip_size: int = TILE_SIZE, max_chips: int = 16) -> List[Dict]:
        """Slices raster arrays into uniform chips for localized zero-shot classification."""
        h, w, _ = rgb_array.shape
        chips = []
        chip_counter = 0

        y_step = max(chip_size, (h - chip_size) // 4) if h > chip_size else chip_size
        x_step = max(chip_size, (w - chip_size) // 4) if w > chip_size else chip_size

        for y in range(0, max(1, h - chip_size + 1), y_step):
            for x in range(0, max(1, w - chip_size + 1), x_step):
                if chip_counter >= max_chips:
                    break
                
                sub_arr = rgb_array[y:y+chip_size, x:x+chip_size, :]
                # Pad if chip is smaller than requested tile size
                if sub_arr.shape[0] < chip_size or sub_arr.shape[1] < chip_size:
                    padded = np.zeros((chip_size, chip_size, 3), dtype=np.uint8)
                    padded[:sub_arr.shape[0], :sub_arr.shape[1], :] = sub_arr
                    sub_arr = padded

                # Skip completely black or zero-information chips
                if np.mean(sub_arr) < 3.0:
                    continue

                chips.append({
                    "chip_id": f"tile_{chip_counter + 1}",
                    "image": Image.fromarray(sub_arr),
                    "array": sub_arr,
                    "window": [x, y, chip_size, chip_size]
                })
                chip_counter += 1

            if chip_counter >= max_chips:
                break

        if not chips:
            chips.append({
                "chip_id": "tile_1",
                "image": Image.fromarray(rgb_array),
                "array": rgb_array,
                "window": [0, 0, w, h]
            })

        return chips


# ==================================================================================================
# 3. ADVANCED ZERO-SHOT SEMANTIC ENGINE
# ==================================================================================================

class EarthObservationSemanticEngine:
    """Manages Vision Transformer encoders, ensemble zero-shot prompts, and in-memory search."""

    def __init__(self):
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.model_loaded = False
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.in_memory_index: List[Dict] = []
        self.precomputed_prompt_embeddings: Dict[str, np.ndarray] = {}

    def load_model(self):
        if not OPEN_CLIP_AVAILABLE:
            raise RuntimeError("open_clip library is not available in environment.")
        
        # Load lightweight, accurate ViT-B-32 trained on LAION-2B
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.model_loaded = True
        self._precompute_taxonomy_vectors()

    def _fallback_vector(self, text_or_img: Any) -> np.ndarray:
        """Deterministic mathematical fallback vector if weights are not yet downloaded."""
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        if isinstance(text_or_img, str):
            seed = sum(ord(c) * (37 ** (i % 7)) for i, c in enumerate(text_or_img))
            rng = np.random.RandomState(seed % (2**31 - 1))
            vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        elif isinstance(text_or_img, Image.Image):
            resized = text_or_img.convert("RGB").resize((16, 16))
            arr = np.array(resized, dtype=np.float32).flatten()
            vec[:min(len(arr), EMBEDDING_DIM)] = arr[:EMBEDDING_DIM]
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-7)

    def encode_text(self, text: str) -> np.ndarray:
        if self.model_loaded and self.model:
            tokens = self.tokenizer([text]).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_text(tokens)
                feat /= feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy()[0].astype(np.float32)
        return self._fallback_vector(text)

    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        if self.model_loaded and self.model:
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(tensor)
                feat /= feat.norm(dim=-1, keepdim=True)
                return feat.cpu().numpy()[0].astype(np.float32)
        return self._fallback_vector(pil_img)

    def _precompute_taxonomy_vectors(self):
        """Precomputes prompt ensemble vectors to make scanning instant."""
        for label_name, prompt_list in EO_PROMPT_TAXONOMY.items():
            vectors = [self.encode_text(p) for p in prompt_list]
            # Average ensemble vector normalized to unit length
            mean_vec = np.mean(vectors, axis=0)
            mean_vec /= (np.linalg.norm(mean_vec) + 1e-7)
            self.precomputed_prompt_embeddings[label_name] = mean_vec

    def classify_chip(self, pil_img: Image.Image, candidate_classes: List[str]) -> Tuple[str, float, Dict[str, float]]:
        """Classifies a chip against candidate classes using normalized softmax probabilities."""
        img_vec = self.encode_image(pil_img)
        scores = {}

        for cls_name in candidate_classes:
            if cls_name in self.precomputed_prompt_embeddings:
                txt_vec = self.precomputed_prompt_embeddings[cls_name]
            else:
                txt_vec = self.encode_text(cls_name)
            
            # Cosine similarity
            cosine_sim = float(np.dot(img_vec, txt_vec))
            scores[cls_name] = cosine_sim

        # Softmax over cosine similarities with temperature scaling (T=0.07)
        labels = list(scores.keys())
        raw_sims = np.array([scores[l] for l in labels])
        exp_sims = np.exp((raw_sims - np.max(raw_sims)) / 0.07)
        probs = exp_sims / (np.sum(exp_sims) + 1e-7)
        prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}

        best_idx = int(np.argmax(probs))
        return labels[best_idx], prob_dict[labels[best_idx]], prob_dict

    def register_chip(self, chip_dict: Dict, source_filename: str):
        vec = self.encode_image(chip_dict["image"])
        self.in_memory_index.append({
            "chip_id": chip_dict["chip_id"],
            "source": source_filename,
            "image": chip_dict["image"],
            "window": chip_dict["window"],
            "embedding": vec
        })

    def search_index(self, query: str, top_k: int = 4) -> List[Dict]:
        if not self.in_memory_index:
            return []
        
        query_vec = self.encode_text(query)
        scored = []
        for item in self.in_memory_index:
            sim = float(np.dot(query_vec, item["embedding"]))
            scored.append({**item, "similarity": sim})
        
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]


# ==================================================================================================
# 4. AUDIT & PROVENANCE ENGINE (IN-MEMORY & DOWNLOADABLE)
# ==================================================================================================

if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []

def record_analyst_decision(tile_id: str, classification: str, confidence: float, decision: str, source: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tile_id": tile_id,
        "classification": classification,
        "confidence": f"{confidence * 100:.1f}%",
        "decision": decision,
        "source_file": source
    }
    st.session_state.audit_trail.append(entry)


# ==================================================================================================
# 5. STREAMLIT APPLICATION CONTROLS & SESSION STATE
# ==================================================================================================

if "semantic_engine" not in st.session_state:
    st.session_state.semantic_engine = EarthObservationSemanticEngine()

engine: EarthObservationSemanticEngine = st.session_state.semantic_engine

with st.sidebar:
    st.title("System Telemetry")
    st.markdown(f"**Execution Hardware:** `{engine.device.upper()}`")
    st.markdown(f"**Neural Engine:** `{'ViT-B-32 (Online)' if engine.model_loaded else 'Native Mathematical Fallback'}`")
    st.markdown(f"**Indexed In-Memory Tiles:** `{len(engine.in_memory_index)}`")
    
    st.divider()
    st.subheader("AI Acceleration Weights")
    if not engine.model_loaded:
        if st.button("Initialize Neural Weights", type="primary"):
            with st.spinner("Downloading/Loading OpenCLIP Weights..."):
                try:
                    engine.load_model()
                    st.success("Vision Transformer Weights Loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load weights: {e}")
    else:
        st.success("Model active and ready for inference.")

    st.divider()
    if st.button("Reset Session & Clear Cache"):
        st.session_state.semantic_engine = EarthObservationSemanticEngine()
        st.session_state.audit_trail = []
        if "active_raster" in st.session_state:
            del st.session_state["active_raster"]
        if "scan_results" in st.session_state:
            del st.session_state["scan_results"]
        st.rerun()


# ==================================================================================================
# 6. APPLICATION WORKSPACE TABS
# ==================================================================================================

tab_ingest, tab_scan, tab_search, tab_anomaly, tab_audit, tab_docs = st.tabs([
    "1. Dynamic Raster Ingest",
    "2. Automated Semantic Scanner",
    "3. Natural Language Search",
    "4. Dynamic Anomaly Engine",
    "5. Analyst Audit Ledger",
    "6. Architectural Specs"
])


# --------------------------------------------------------------------------------------------------
# TAB 1: DYNAMIC INGESTION & TRUE-COLOR COMPOSITE (TCC)
# --------------------------------------------------------------------------------------------------
with tab_ingest:
    st.subheader("Universal Raster Ingestion & Band Routing")
    st.markdown("Upload **any** GeoTIFF (Sentinel-1 SAR, Sentinel-2 Optical, USGS Landsat, SRTM DEM). The pipeline extracts metadata and applies 2nd-98th percentile radiometric contrast stretching.")

    uploaded_raster = st.file_uploader("Select GeoTIFF File (.tif / .tiff)", type=["tif", "tiff"], key="main_uploader")

    if uploaded_raster:
        col_ctrl, col_meta = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown("#### Band Synthesis Mapping")
            st.caption("For multi-band rasters, specify 1-indexed channels for RGB rendering:")
            r_idx = st.number_input("Red Band Channel:", min_value=1, max_value=32, value=1)
            g_idx = st.number_input("Green Band Channel:", min_value=1, max_value=32, value=2)
            b_idx = st.number_input("Blue Band Channel:", min_value=1, max_value=32, value=3)
            
            process_trigger = st.button("Process & Synthesize Imagery", type="primary")

        if process_trigger or ("active_raster" in st.session_state and st.session_state.get("active_filename") == uploaded_raster.name):
            with st.spinner("Decoding GeoTIFF & Normalizing Radiometry..."):
                pil_img, rgb_arr, meta = SensorAgnosticProcessor.process_raster_bytes(
                    uploaded_raster.getvalue(),
                    red_ch=r_idx,
                    green_ch=g_idx,
                    blue_ch=b_idx
                )
                st.session_state.active_raster = {
                    "image": pil_img,
                    "array": rgb_arr,
                    "meta": meta,
                    "filename": uploaded_raster.name
                }
                st.session_state.active_filename = uploaded_raster.name

            with col_meta:
                st.markdown("#### Raster Spatial Telemetry")
                m1, m2, m3 = st.columns(3)
                m1.metric("Raster Dimensions", meta["dimensions"])
                m2.metric("Band Count", meta["bands"])
                m3.metric("Spatial Reference", meta["crs"])
                st.caption(f"Bounding Envelope: {meta['bounds']}")

            st.divider()
            st.markdown("#### Synthesized Sensor View")
            st.image(st.session_state.active_raster["image"], caption=f"Normalized Visual Synthesis | Source: {uploaded_raster.name}", use_container_width=True)


# --------------------------------------------------------------------------------------------------
# TAB 2: AUTOMATED SEMANTIC SCANNER & ONE-CLICK AUDIT
# --------------------------------------------------------------------------------------------------
with tab_scan:
    st.subheader("Automated Multi-Phenomena Feature Extraction")
    st.markdown("Automatically extracts spatial chips and runs parallel zero-shot classification against environmental and structural classes.")

    if "active_raster" not in st.session_state:
        st.warning("Please upload and process a raster in Tab 1 before launching the automated scanner.")
    else:
        current_data = st.session_state.active_raster
        
        c_sel, c_act = st.columns([3, 1])
        with c_sel:
            available_classes = list(EO_PROMPT_TAXONOMY.keys())
            target_classes = st.multiselect("Select Target Phenomena to Detect:", available_classes, default=available_classes[:3])
        with c_act:
            st.markdown("<br>", unsafe_allow_html=True)
            scan_now = st.button("Run Semantic Sweep", type="primary")

        if scan_now:
            if not target_classes:
                st.error("Select at least one phenomenon category.")
            else:
                with st.spinner("Extracting spatial chips and evaluating against semantic vectors..."):
                    chips = SensorAgnosticProcessor.extract_spatial_chips(current_data["array"], chip_size=TILE_SIZE, max_chips=16)
                    
                    scan_results = []
                    for chip in chips:
                        # Index the chip into the searchable memory bank
                        engine.register_chip(chip, current_data["filename"])
                        
                        # Classify with softmax confidence
                        best_cls, confidence, all_probs = engine.classify_chip(chip["image"], target_classes)
                        
                        scan_results.append({
                            "chip_id": chip["chip_id"],
                            "image": chip["image"],
                            "classification": best_cls,
                            "confidence": confidence,
                            "probabilities": all_probs,
                            "source": current_data["filename"]
                        })

                    st.session_state.scan_results = scan_results
                    st.success(f"Successfully evaluated {len(chips)} spatial chips.")

        if "scan_results" in st.session_state:
            results = st.session_state.scan_results
            st.markdown(f"### Detected Features Across Archive ({len(results)} Chips)")

            cols = st.columns(4)
            for idx, res in enumerate(results):
                with cols[idx % 4]:
                    st.markdown('<div class="audit-card">', unsafe_allow_html=True)
                    st.image(res["image"], use_container_width=True)
                    
                    st.markdown(f"**Target:** `{res['classification']}`")
                    st.markdown(f"**Confidence:** `{res['confidence']*100:.1f}%`")
                    
                    btn_y, btn_n = st.columns(2)
                    if btn_y.button("Confirm", key=f"y_{res['chip_id']}_{idx}"):
                        record_analyst_decision(res["chip_id"], res["classification"], res["confidence"], "CONFIRMED", res["source"])
                        st.success("Confirmed")
                    if btn_n.button("Reject", key=f"n_{res['chip_id']}_{idx}"):
                        record_analyst_decision(res["chip_id"], res["classification"], res["confidence"], "REJECTED", res["source"])
                        st.error("Rejected")
                    
                    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------
# TAB 3: NATURAL LANGUAGE RETRIEVAL
# --------------------------------------------------------------------------------------------------
with tab_search:
    st.subheader("Multimodal Natural Language Archive Interrogation")
    st.markdown("Search across all indexed spatial chips in memory using free-text natural language queries.")

    q_col, k_col = st.columns([4, 1])
    search_query = q_col.text_input("Enter natural language query:", "calm water body or reservoir")
    top_k = k_col.number_input("Max Results:", min_value=1, max_value=16, value=4)

    if st.button("Execute Natural Language Query", type="primary"):
        if len(engine.in_memory_index) == 0:
            st.warning("Archive is currently empty. Upload and sweep a raster in Tab 2 to populate the search catalog.")
        else:
            with st.spinner("Computing cosine similarities across vector index..."):
                hits = engine.search_index(search_query, top_k=top_k)

            if not hits:
                st.info("No matching chips found.")
            else:
                st.markdown(f"**Retrieved {len(hits)} relevant targets:**")
                h_cols = st.columns(len(hits))
                for i, hit in enumerate(hits):
                    with h_cols[i]:
                        st.image(hit["image"], use_container_width=True)
                        st.markdown(f"**ID:** `{hit['chip_id']}`")
                        st.markdown(f"**Relevance:** `{hit['similarity']:.4f}`")
                        st.caption(f"Source: {hit['source']}")


# --------------------------------------------------------------------------------------------------
# TAB 4: DYNAMIC MULTI-TEMPORAL ANOMALY DETECTION
# --------------------------------------------------------------------------------------------------
with tab_anomaly:
    st.subheader("Dynamic Multi-Temporal Change & Anomaly Detection")
    st.markdown("Upload both a **Baseline** and a **Target** raster directly through the interface to perform pixel-level radiometric change detection and morphological suppression.")

    c1, c2 = st.columns(2)
    base_file = c1.file_uploader("Upload Baseline Raster (T0)", type=["tif", "tiff"], key="anom_base")
    targ_file = c2.file_uploader("Upload Target Raster (T1)", type=["tif", "tiff"], key="anom_targ")

    if base_file and targ_file:
        thresh_val = st.slider("Anomaly Deviation Threshold:", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
        
        if st.button("Execute Radiometric Change Engine", type="primary"):
            with st.spinner("Aligning grids and computing change matrix..."):
                _, b_arr, _ = SensorAgnosticProcessor.process_raster_bytes(base_file.getvalue())
                _, t_arr, _ = SensorAgnosticProcessor.process_raster_bytes(targ_file.getvalue())

                # Crop to common intersection to prevent array dimension mismatches
                min_h = min(b_arr.shape[0], t_arr.shape[0])
                min_w = min(b_arr.shape[1], t_arr.shape[1])
                b_cropped = b_arr[:min_h, :min_w]
                t_cropped = t_arr[:min_h, :min_w]

                # Convert to grayscale linear arrays
                b_gray = np.mean(b_cropped, axis=-1) / 255.0
                t_gray = np.mean(t_cropped, axis=-1) / 255.0

                # Differential Radiometry
                diff = np.abs(t_gray - b_gray)
                raw_mask = diff > thresh_val

                # Morphological filtering to suppress random speckle noise
                labeled_array, num_features = label(raw_mask)
                unique, counts = np.unique(labeled_array, return_counts=True)
                cleaned_mask = np.zeros_like(raw_mask, dtype=bool)
                for comp_id, size in zip(unique, counts):
                    if comp_id != 0 and size >= 15: # minimum 15 connected pixels
                        cleaned_mask[labeled_array == comp_id] = True

                # Overlay anomalies in high-visibility red
                overlay = t_cropped.copy()
                overlay[cleaned_mask] = [255, 30, 30]

                # Metrics
                total_pixels = raw_mask.size
                flagged = int(np.sum(cleaned_mask))
                reduction = ((np.sum(raw_mask) - flagged) / max(1, np.sum(raw_mask))) * 100.0

                m_a, m_b, m_c = st.columns(3)
                m_a.metric("Total Scanned Pixels", f"{total_pixels:,}")
                m_b.metric("Confirmed Anomalies", f"{flagged:,}")
                m_c.metric("Speckle Noise Suppressed", f"{reduction:.1f}%")

                v1, v2 = st.columns(2)
                v1.image(b_cropped, caption="Baseline Scene (T0)", use_container_width=True)
                v2.image(overlay, caption="Target Scene (T1) + Structural Anomaly Overlay", use_container_width=True)


# --------------------------------------------------------------------------------------------------
# TAB 5: ANALYST AUDIT LEDGER
# --------------------------------------------------------------------------------------------------
with tab_audit:
    st.subheader("Analyst Decision Ledger & Sovereign Audit Trail")
    st.markdown("All confirmation and rejection actions taken during the session are logged to maintain an immutable chain of custody.")

    if not st.session_state.audit_trail:
        st.info("No decisions logged yet in this session. Confirm or reject targets in Tab 2 to populate this ledger.")
    else:
        df_audit = pd.DataFrame(st.session_state.audit_trail)
        st.dataframe(df_audit, use_container_width=True)

        csv_data = df_audit.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Certified Audit Ledger (CSV)",
            data=csv_data,
            file_name=f"analyst_audit_log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# --------------------------------------------------------------------------------------------------
# TAB 6: ARCHITECTURAL SPECIFICATIONS & FORMULAS
# --------------------------------------------------------------------------------------------------
with tab_docs:
    st.subheader("Mathematical & Architectural Framework")
    st.markdown("### 1. Radiometric Calibration & Dynamic Contrast Stretch")
    st.latex(r"I_{\text{norm}}(x, y) = \text{clip}\left(\frac{I(x, y) - P_2}{P_{98} - P_2 + \epsilon}, 0, 1\right) \times 255")
    
    st.markdown("### 2. Multi-Prompt Zero-Shot Semantic Vectorization")
    st.latex(r"\mathbf{t}_{\text{ensemble}} = \frac{1}{M}\sum_{m=1}^{M} \frac{\mathcal{E}_{\text{text}}(p_m)}{\|\mathcal{E}_{\text{text}}(p_m)\|_2}, \quad \mathcal{S}(\mathbf{x}, \mathbf{t}) = \frac{\mathcal{E}_{\text{img}}(\mathbf{x}) \cdot \mathbf{t}_{\text{ensemble}}}{\|\mathcal{E}_{\text{img}}(\mathbf{x})\|_2}")
    
    st.markdown("### 3. Softmax Phenomenon Classification")
    st.latex(r"P(c_k \mid \mathbf{x}) = \frac{\exp\left(\mathcal{S}(\mathbf{x}, \mathbf{t}_k) / \tau\right)}{\sum_{j=1}^{K}\exp\left(\mathcal{S}(\mathbf{x}, \mathbf{t}_j) / \tau\right)}")
    
    st.markdown("### 4. Morphological Speckle Suppression")
    st.latex(r"M_{\text{final}} = \left\{ \mathbf{p} \in M_{\text{raw}} \mid |\text{ConnectedComponent}(\mathbf{p})| \ge 15 \right\}")
