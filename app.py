# ==================================================================================================
# PROJECT: ARTHADRISHTI - ADVANCED EO INTELLIGENCE PLATFORM (NATIONAL LEVEL ARCHITECTURE)
# CAPABILITIES: Dynamic Sensor Ingestion, Zero-Shot Semantic Retrieval, Automated Auditing, Change Detection
# ==================================================================================================

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile
from scipy.ndimage import uniform_filter, label

import streamlit as st

# Hardware Acceleration & ML Dependencies
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
# 1. CORE SYSTEM CONFIGURATION
# ==================================================================================================

TILE_SIZE = 256
EMBEDDING_DIM = 512
DEFAULT_TAXONOMY = [
    "deep open water body, river, or lake",
    "high-density urban residential buildings and concrete infrastructure",
    "agricultural crop fields and farms",
    "dense forest canopy and thick vegetation",
    "bare soil, cleared land, and dirt",
    "industrial warehouse structures and factories",
    "airport runway or large tarmac"
]

st.set_page_config(page_title="ArthaDrishti | Semantic EO", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    h1, h2, h3, h4 { color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
    .stMetric { background-color: #1e293b; border: 1px solid #334155; padding: 12px; border-radius: 4px; }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; }
    .stButton>button { background-color: #0284c7; color: white; border: none; border-radius: 4px; font-weight: 600; width: 100%; }
    .stButton>button:hover { background-color: #0369a1; }
    .audit-card { background-color: #1e293b; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)


# ==================================================================================================
# 2. SENSOR-AGNOSTIC IMAGE PROCESSING ENGINE
# ==================================================================================================

class SensorAgnosticProcessor:
    """Handles dynamic ingestion, normalization, and TCC generation for any GeoTIFF."""
    
    @staticmethod
    def process_upload(file_bytes, red_idx=0, green_idx=1, blue_idx=2) -> Tuple[Image.Image, np.ndarray, dict]:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                meta = {"crs": str(src.crs), "count": src.count, "bounds": src.bounds, "width": src.width, "height": src.height}
                
                # Handling 1-Band Data (SAR / DEM)
                if src.count == 1:
                    raw = src.read(1)
                    # 2-98% contrast stretch
                    p2, p98 = np.percentile(raw, (2, 98))
                    norm = np.clip((raw - p2) / (p98 - p2 + 1e-6), 0, 1)
                    
                    # Apply a pseudocolor map so the RGB-trained AI can "see" structural features
                    # cv2 colormaps expect uint8 0-255
                    norm_uint8 = (norm * 255).astype(np.uint8)
                    colorized = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_BONE) # Bone is excellent for SAR
                    rgb_array = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
                    
                # Handling Multi-Band Data (Optical: Sentinel, Landsat, USGS)
                else:
                    # Safely map user-selected bands (fallback to 0 if out of bounds)
                    r = src.read(red_idx + 1) if red_idx < src.count else src.read(1)
                    g = src.read(green_idx + 1) if green_idx < src.count else src.read(1)
                    b = src.read(blue_idx + 1) if blue_idx < src.count else src.read(1)
                    
                    raw_stack = np.stack([r, g, b], axis=-1).astype(np.float32)
                    
                    # Intelligent radiometric normalization across the RGB stack
                    p2, p98 = np.percentile(raw_stack, (2, 98))
                    rgb_array = np.clip((raw_stack - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

                pil_img = Image.fromarray(rgb_array)
                return pil_img, rgb_array, meta

    @staticmethod
    def extract_chips(rgb_array: np.ndarray, chip_size: int = TILE_SIZE) -> List[Dict]:
        h, w, _ = rgb_array.shape
        chips = []
        chip_id = 0
        
        for y in range(0, h - chip_size + 1, chip_size):
            for x in range(0, w - chip_size + 1, chip_size):
                chip_data = rgb_array[y:y+chip_size, x:x+chip_size, :]
                if np.mean(chip_data) < 5: continue # Skip empty/nodata chips
                
                chips.append({
                    "id": f"chip_{int(time.time())}_{chip_id}",
                    "image": Image.fromarray(chip_data),
                    "window": [x, y, chip_size, chip_size]
                })
                chip_id += 1
                
        # Fallback if image is smaller than tile size
        if not chips:
            chips.append({"id": f"chip_{int(time.time())}_0", "image": Image.fromarray(rgb_array), "window": [0, 0, w, h]})
        return chips


# ==================================================================================================
# 3. EMBEDDING & SEMANTIC RETRIEVAL ENGINE
# ==================================================================================================

class SemanticEngine:
    """Manages AI models and the dynamic in-memory vector database."""
    def __init__(self):
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.model_loaded = False
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.metadata = []
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM) if FAISS_AVAILABLE else None

    def initialize_model(self):
        if not OPEN_CLIP_AVAILABLE: raise Exception("open_clip not installed.")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip_model = model.to(self.device).eval()
        self.clip_preprocess = preprocess
        self.model_loaded = True

    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        if not self.model_loaded: return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.encode_image(tensor)
            feat /= feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()[0].astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        if not self.model_loaded: return np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
        tokens = self.clip_tokenizer([text]).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy().astype(np.float32)

    def add_to_index(self, chip: Dict, source_name: str):
        emb = self.encode_image(chip["image"])
        if self.faiss_index:
            self.faiss_index.add(np.expand_dims(emb, axis=0))
        chip_meta = {
            "id": chip["id"], 
            "source": source_name, 
            "window": chip["window"], 
            "image": chip["image"] # Storing PIL image directly in memory for UI rendering
        }
        self.metadata.append(chip_meta)

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        if not self.metadata or not self.faiss_index or self.faiss_index.ntotal == 0: return []
        text_vec = self.encode_text(query)
        
        k = min(top_k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(text_vec, k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item['score'] = float(distances[0][rank])
                results.append(item)
        return results


# ==================================================================================================
# 4. AUDIT LEDGER (SESSION-BASED TO PREVENT CLOUD CRASHES)
# ==================================================================================================

if 'audit_ledger' not in st.session_state:
    st.session_state.audit_ledger = pd.DataFrame(columns=["timestamp", "tile_id", "query_text", "decision", "confidence", "source_file"])

def log_decision(tile_id, query_text, decision, confidence, source_file):
    new_record = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tile_id": tile_id,
        "query_text": query_text,
        "decision": decision,
        "confidence": f"{confidence:.4f}",
        "source_file": source_file
    }])
    st.session_state.audit_ledger = pd.concat([st.session_state.audit_ledger, new_record], ignore_index=True)


# ==================================================================================================
# 5. UI INITIALIZATION & SIDEBAR
# ==================================================================================================

if 'engine' not in st.session_state:
    st.session_state.engine = SemanticEngine()
engine = st.session_state.engine

with st.sidebar:
    st.title("System Telemetry")
    st.markdown(f"**Device:** {engine.device.upper()}")
    st.markdown(f"**AI Core:** {'Active (ViT-B/32)' if engine.model_loaded else 'Standby'}")
    st.markdown(f"**Index Size:** {len(engine.metadata)} tiles stored in memory")
    
    st.divider()
    if not engine.model_loaded:
        if st.button("Initialize Neural Weights", type="primary"):
            with st.spinner("Booting Foundation Model..."):
                engine.initialize_model()
                st.rerun()
    else:
        st.success("Neural Weights Active")
        
    st.divider()
    if st.button("Purge Session Memory"):
        st.session_state.engine = SemanticEngine()
        st.session_state.audit_ledger = pd.DataFrame(columns=["timestamp", "tile_id", "query_text", "decision", "confidence", "source_file"])
        st.rerun()

tab_ingest, tab_search, tab_audit, tab_anomaly = st.tabs([
    "1. Dynamic Ingestion & Auditor", "2. Global Semantic Retrieval", "3. Audit Ledger", "4. Advanced Anomaly Det."
])

# ==================================================================================================
# TAB 1: DYNAMIC INGESTION & AUTOMATED AUDITOR
# ==================================================================================================
with tab_ingest:
    st.subheader("Dynamic Sensor Ingestion & Automated Feature Extraction")
    st.markdown("Upload any GeoTIFF. The engine will normalize the radiometry, allow TCC band mapping, and automatically sweep the scene for specific physical phenomena.")
    
    uploaded_file = st.file_uploader("Upload Raster (.tif/.tiff)", type=["tif", "tiff"])
    
    if uploaded_file:
        c_bands, c_action = st.columns([3, 1])
        with c_bands:
            st.markdown("**True Color Composite (TCC) Mapping:** *(For multi-band data)*")
            b_cols = st.columns(3)
            red_b = b_cols[0].number_input("Red Band Index", min_value=0, value=0)
            green_b = b_cols[1].number_input("Green Band Index", min_value=0, value=1)
            blue_b = b_cols[2].number_input("Blue Band Index", min_value=0, value=2)
            
        with c_action:
            st.markdown("<br>", unsafe_allow_html=True)
            process_btn = st.button("Parse & Render Image")
            
        if process_btn or 'current_image' in st.session_state:
            if process_btn:
                with st.spinner("Processing raster physics..."):
                    pil_img, rgb_array, meta = SensorAgnosticProcessor.process_upload(uploaded_file.getvalue(), red_b, green_b, blue_b)
                    st.session_state.current_image = pil_img
                    st.session_state.current_array = rgb_array
                    st.session_state.current_meta = meta
                    st.session_state.current_filename = uploaded_file.name
                    
            st.success(f"Rendered: {st.session_state.current_filename} | CRS: {st.session_state.current_meta['crs']} | Bands: {st.session_state.current_meta['count']}")
            st.image(st.session_state.current_image, caption="Normalized Composite View", use_container_width=True)
            
            st.divider()
            c_tax, c_sweep = st.columns([3, 1])
            with c_tax:
                selected_tax = st.multiselect("Select Phenomena for Automated Sweep:", DEFAULT_TAXONOMY, default=DEFAULT_TAXONOMY[:3])
            with c_sweep:
                st.markdown("<br>", unsafe_allow_html=True)
                sweep_btn = st.button("Execute Semantic Sweep", type="primary")
                
            if sweep_btn:
                if not engine.model_loaded:
                    st.error("Please Initialize Neural Weights in the sidebar first.")
                else:
                    with st.spinner("Slicing spatial chips and extracting semantic features..."):
                        chips = SensorAgnosticProcessor.extract_chips(st.session_state.current_array)
                        
                        # Add chips to the global retrieval index in the background
                        for c in chips: engine.add_to_index(c, st.session_state.current_filename)
                        
                        # Run the automated sweep
                        st.session_state.sweep_results = []
                        text_features = {label: engine.encode_text(label) for label in selected_tax}
                        
                        for chip in chips:
                            img_feat = engine.encode_image(chip["image"])
                            best_label, best_score = "Unclassified", 0.0
                            for label, txt_feat in text_features.items():
                                score = np.dot(img_feat, txt_feat[0])
                                if score > best_score:
                                    best_score, best_label = score, label
                                    
                            if best_score > 0.23: # Confidence threshold
                                st.session_state.sweep_results.append({
                                    "chip": chip, "label": best_label, "score": float(best_score)
                                })
                                
            # Render Sweep Results
            if 'sweep_results' in st.session_state:
                res = st.session_state.sweep_results
                if not res: st.info("No targeted phenomena detected above confidence threshold.")
                else:
                    st.markdown(f"### Detected Anomalies ({len(res)})")
                    grid = st.columns(4)
                    for i, det in enumerate(res):
                        with grid[i % 4]:
                            st.markdown('<div class="audit-card">', unsafe_allow_html=True)
                            st.image(det["chip"]["image"], use_container_width=True)
                            st.markdown(f"**Class:** {det['label'].title()}")
                            st.markdown(f"**Conf:** {det['score']*100:.1f}%")
                            
                            b1, b2 = st.columns(2)
                            if b1.button("✅", key=f"y_{det['chip']['id']}"):
                                log_decision(det['chip']['id'], det['label'], "CONFIRMED", det['score'], st.session_state.current_filename)
                                st.toast("Confirmed!")
                            if b2.button("❌", key=f"n_{det['chip']['id']}"):
                                log_decision(det['chip']['id'], det['label'], "REJECTED", det['score'], st.session_state.current_filename)
                                st.toast("Rejected.")
                            st.markdown('</div>', unsafe_allow_html=True)


# ==================================================================================================
# TAB 2: GLOBAL SEMANTIC RETRIEVAL
# ==================================================================================================
with tab_search:
    st.subheader("Global Archive Interrogation")
    st.markdown("Query the active vector database containing all previously uploaded and indexed tiles using natural language.")
    
    q_col, n_col = st.columns([4, 1])
    query = q_col.text_input("Enter physical characteristic:", "high density urban structures")
    top_n = n_col.number_input("Max Results:", 1, 20, 4)
    
    if st.button("Search Archive"):
        if not engine.model_loaded:
            st.error("Initialize Neural Weights first.")
        elif len(engine.metadata) == 0:
            st.warning("Archive is empty. Please upload and sweep an image in Tab 1 first.")
        else:
            results = engine.search(query, top_k=top_n)
            
            grid = st.columns(4)
            for i, r in enumerate(results):
                with grid[i % 4]:
                    st.image(r["image"], use_container_width=True)
                    st.markdown(f"**Source:** {r['source']}")
                    st.markdown(f"**Match:** {r['score']*100:.1f}%")


# ==================================================================================================
# TAB 3: AUDIT LEDGER
# ==================================================================================================
with tab_audit:
    st.subheader("Immutable Analyst Ledger")
    df = st.session_state.audit_ledger
    if df.empty:
        st.info("No analyst actions logged in this session.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button("Export Ledger (CSV)", df.to_csv(index=False).encode('utf-8'), "audit_log.csv", "text/csv")


# ==================================================================================================
# TAB 4: ADVANCED ANOMALY DETECTION (DYNAMIC)
# ==================================================================================================
with tab_anomaly:
    st.subheader("Dynamic Multi-Temporal Change Analytics")
    st.markdown("Upload two aligned rasters (Baseline and Target) to compute structural anomalies dynamically.")
    
    c_base, c_targ = st.columns(2)
    base_file = c_base.file_uploader("Upload Baseline Raster", type=["tif"])
    targ_file = c_targ.file_uploader("Upload Target Raster", type=["tif"])
    
    if base_file and targ_file:
        diff_thresh = st.slider("Anomaly Threshold", 0.0, 1.0, 0.15)
        if st.button("Compute Change Matrix"):
            with st.spinner("Calculating pixel-wise deviations..."):
                # Dynamically process both uploads to unified arrays
                _, b_arr, _ = SensorAgnosticProcessor.process_upload(base_file.getvalue())
                _, t_arr, _ = SensorAgnosticProcessor.process_upload(targ_file.getvalue())
                
                # Match shapes in case of slight cropping differences
                min_h = min(b_arr.shape[0], t_arr.shape[0])
                min_w = min(b_arr.shape[1], t_arr.shape[1])
                b_arr = b_arr[:min_h, :min_w]
                t_arr = t_arr[:min_h, :min_w]
                
                # Compute normalized absolute difference
                b_gray = np.mean(b_arr, axis=2) / 255.0
                t_gray = np.mean(t_arr, axis=2) / 255.0
                diff = np.abs(t_gray - b_gray)
                
                # Apply threshold
                mask = diff > diff_thresh
                
                # Render Overlay
                overlay = t_arr.copy()
                overlay[mask] = [255, 0, 50] # Highlight anomalies in bright red
                
                v1, v2 = st.columns(2)
                v1.image(Image.fromarray(b_arr), caption="Baseline", use_container_width=True)
                v2.image(Image.fromarray(overlay), caption="Target with Anomaly Overlay", use_container_width=True)
                
                st.metric("Detected Anomalous Pixels", f"{np.sum(mask):,}")
