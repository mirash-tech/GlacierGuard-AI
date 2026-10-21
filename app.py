

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from analysis_engine import analyse_changes
from spectral_engine import (load_sentinel_data, false_color, index_image,
                             analyse_spectral_data, REFERENCE_PIXEL_AREA_M2, overlap_bounds, crop_to_bounds)

ROOT = Path(__file__).resolve().parent


FINAL_RESULTS = ROOT / "research" / "results"
FINAL_METADATA = FINAL_RESULTS / "analysis_metadata.json"
FINAL_STATS = FINAL_RESULTS / "validated_preliminary_statistics.csv"
FINAL_CLUSTERS = FINAL_RESULTS / "cluster_counts.csv"
ORIGINAL_RESULTS_CSV = ROOT / "data" / "spectral_results.csv"
ORIGINAL_RESULTS_MAP = ROOT / "assets" / "spectral_results_reference.png"
DEMO_FOLDER = ROOT / "sample_images"
SAT_FOLDER = DEMO_FOLDER / "nepal_preliminary"
SPECTRAL_FILE = ROOT / "data" / "glacierguard_data.npz"

@st.cache_resource(show_spinner="Loading aligned Sentinel-2 arrays...")
def get_spectral_data():
    return load_sentinel_data(SPECTRAL_FILE)


st.set_page_config(
    page_title="GlacierGuard AI | Nepal",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",  
)


STYLESHEET = ROOT / "assets" / "styles.css"
if STYLESHEET.exists():
    st.markdown(f"<style>{STYLESHEET.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(
    """
<div class="gg-hero">
  <div class="gg-hero-orbit"></div>
  <div class="gg-hero-top">
    <div class="gg-brand"><span class="gg-logo">✦</span><span>GLACIERGUARD <span class="gg-brand-suffix">AI</span></span></div>
    <div class="gg-build">● &nbsp; SATELLITE MONITORING · V1.0</div>
  </div>
  <div class="gg-hero-grid">
    <div class="gg-hero-copy">
      <div class="gg-kicker">EARTH OBSERVATION &nbsp; / &nbsp; NEPAL</div>
      <h1><span class="gg-heading-white">See change.</span><br><em>Understand impact.</em></h1>
      <p>Satellite-powered landscape change intelligence for glacial environments — built for the CDU Sydney AI Challenge.</p>
      <div class="gg-hero-tags"><span>◈ &nbsp; Satellite imagery</span><span>◎ &nbsp; Change analysis</span><span>↗ &nbsp; Nepal case study</span></div>
    </div>
    <div class="gg-landscape" aria-hidden="true">
      <div class="gg-halo"></div><div class="gg-mountain gg-back"></div><div class="gg-mountain gg-mid"></div><div class="gg-mountain gg-front"></div>
      <span class="gg-coord">28° N &nbsp; / &nbsp; 85° E</span>
      <span class="gg-glacier-label">HIMALAYAN REGION <i></i></span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="gg-note"><div class="gg-note-icon">ⓘ</div><div>
    <strong>Langtang, Nepal · 2026 observations</strong>
    <span>Compare Sentinel-2 spectral indices, inspect landscape changes, and explore unsupervised satellite-image clustering.</span>
    </div><span class="gg-note-status">RESEARCH PROJECT</span></div>""",
    unsafe_allow_html=True,
)


def png_bytes(array: np.ndarray) -> bytes:
    
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_array(source) -> np.ndarray:
    
    with Image.open(source) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        if rgb.width * rgb.height > 16_000_000:
            raise ValueError(
                "Image exceeds the supported size (16 million pixels). "
                "Use a smaller, geospatially aligned export."
            )
        return np.asarray(rgb, dtype=np.uint8)


def render_overlay(after: np.ndarray, mask: np.ndarray) -> np.ndarray:
    
    overlay = after.copy()
    changed = np.asarray(mask) > 0
    if np.any(changed):
        tint = np.array([255, 86, 91], dtype=np.float32)
        overlay[changed] = np.clip(
            after[changed].astype(np.float32) * 0.35 + tint * 0.65,
            0, 255,
        ).astype(np.uint8)
    return overlay


with st.sidebar:
    st.markdown('<div class="gg-sidebar-eyebrow">MISSION CONTROL</div><div class="gg-sidebar-title">🛰️ Image workspace</div>', unsafe_allow_html=True)
    has_spectral_data = SPECTRAL_FILE.is_file() or all((SPECTRAL_FILE.parent / f"glacierguard_part_{i}.npz").is_file() for i in range(1, 4))
    source_options = ["Sentinel-2 spectral analysis"] if has_spectral_data else []
    source_options += ["Sample imagery (synthetic)", "Satellite image preview", "Upload your own images"]
    source_type = st.radio(
        "Choose image source", source_options,
        help="Use the aligned Sentinel-2 multispectral observations.",
    )
    uploaded_before = uploaded_after = None
    if source_type == "Upload your own images":
        uploaded_before = st.file_uploader(
            "Before image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="before"
        )
        uploaded_after = st.file_uploader(
            "After image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="after"
        )
        st.caption("Upload the same geographic extent, resolution, and alignment.")
        before_date = st.text_input("Before observation date", placeholder="YYYY-MM-DD")
        after_date = st.text_input("After observation date", placeholder="YYYY-MM-DD")
        site_name = st.text_input("Study site", value="Nepal (site not yet verified)")
        source_reference = st.text_input("Imagery source / product ID", placeholder="Sentinel-2 product ID or provider link")
        alignment_confirmed = st.checkbox(
            "I verified both images show the same aligned area",
            value=False,
            help="Equal image sizes are not proof of geographic co-registration.",
        )
    elif source_type == "Sentinel-2 spectral analysis":
        before_date, after_date = "2026-08-12", "2026-10-13"
        site_name = "Langtang region, Nepal (study area; exact flood site unverified)"
        source_reference = "User-supplied Sentinel-2 L2A B03/B04/B08/B11 exports; product IDs unverified"
        alignment_confirmed = True
        st.caption("Aligned Sentinel-2 bands · NDWI and NDSI measurements")
    elif source_type == "Satellite image preview":
        before_date, after_date = "2026-08-12", "2026-10-13"
        site_name = "Langtang, Nepal (user-provided study region)"
        source_reference = "Sentinel-2 L2A bands from user-provided archives (product IDs unverified)"
        alignment_confirmed = True  
        st.info("Prepared, geographically aligned false-colour observations. Spectral classifications are unvalidated and not ML.")
    else:
        before_date, after_date = "Synthetic · before", "Synthetic · after"
        site_name = "Synthetic demonstration site (not Nepal)"
        source_reference = "Locally generated synthetic image pair"
        alignment_confirmed = True

    st.divider()
    if source_type == "Sentinel-2 spectral analysis":
        water_rule = st.radio(
            "Water detection method",
            ["NDWI only", "NDWI with snow exclusion"],
            help="The NDWI-only setting reproduces the supplied spectral measurements. Snow exclusion is an additional sensitivity check."
        )
        exclude_snow = water_rule.startswith("NDWI with")
        st.caption("Water: NDWI > 0.3 · Snow: NDSI > 0.4")
    else:
        st.caption("RGB image comparison uses a pixel-difference baseline.")
    if source_type == "Sentinel-2 spectral analysis":
        threshold = None
    else:
        threshold = st.slider(
            "Visual change sensitivity (baseline only)",
            min_value=10, max_value=120, value=35, step=5,
            help="Higher values flag fewer pixel differences. This is NOT a risk threshold.",
        )
    st.divider()
    st.caption("GlacierGuard AI · v1.0 · Mirash & Biraj")

ready = True
is_synthetic = source_type == "Sample imagery (synthetic)"
is_spectral = source_type == "Sentinel-2 spectral analysis"
satellite_data = None
try:
    if is_spectral:
        satellite_data = get_spectral_data()
        before = false_color(satellite_data, 'before')
        after = false_color(satellite_data, 'after')
    elif is_synthetic:
        before = image_array(DEMO_FOLDER / "demo_before.png")
        after = image_array(DEMO_FOLDER / "demo_after.png")
    elif source_type == "Satellite image preview":
        before = image_array(SAT_FOLDER / "langtang_before_aligned_falsecolor.png")
        after = image_array(SAT_FOLDER / "langtang_after_aligned_falsecolor.png")
    elif uploaded_before is not None and uploaded_after is not None:
        before = image_array(uploaded_before)
        after = image_array(uploaded_after)
    else:
        ready = False
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not read the selected images: {exc}")
    ready = False

if not ready:
    st.markdown(
        """<section class="gg-empty-state" aria-label="Image upload instructions">
          <div class="gg-empty-symbol">▧</div>
          <div class="gg-empty-kicker">YOUR WORKSPACE IS READY</div>
          <h2>Begin with two observations.</h2>
          <p>Upload images of the same area from two dates. Use matching resolution
          and geographic alignment for a reliable comparison.</p>
          <div class="gg-empty-steps">
            <span><b>01</b> Upload before</span>
            <span><b>02</b> Upload after</span>
            <span><b>03</b> Explore changes</span>
          </div>
          <div class="gg-empty-tip">To explore without uploads, select <strong>Sample imagery (synthetic)</strong> in Mission Control.</div>
        </section>""",
        unsafe_allow_html=True,
    )
    st.stop()

if not is_synthetic and not alignment_confirmed:
    st.warning(
        "**Unverified alignment:** The analysis below is exploratory only. "
        "Same-sized images may represent different ground footprints or offsets."
    )

if before.shape != after.shape:
    st.error(
        "Before and after images have different dimensions. Please export both "
        "from the **same region with matching resolution and alignment**. "
        "The app deliberately will not stretch one image to fit the other."
    )
    st.stop()

try:
    result = (analyse_spectral_data(satellite_data, exclude_snow=exclude_snow) if is_spectral else
              analyse_changes(before, after, threshold=threshold))
    mask = np.asarray(result["change_mask"], dtype=np.uint8)
    difference = np.asarray(result["difference_map"], dtype=np.uint8)
    if mask.shape != before.shape[:2] or difference.shape != before.shape[:2]:
        raise ValueError("Model masks must match the input image height and width.")
    overlay = render_overlay(after, mask)
except (ValueError, KeyError, TypeError, RuntimeError, ImportError, cv2.error) as exc:
    st.error(f"The analysis could not run: {exc}")
    st.stop()

height, width = mask.shape
changed_pixels = int(np.count_nonzero(mask))
total_pixels = int(result['counts']['valid_overlap_pixels']) if is_spectral else int(height * width)
changed_pct = 100 * changed_pixels / total_pixels if total_pixels else 0
method = str(result.get("method", "Unspecified method"))
is_ml = bool(result.get("is_ml", False))

st.markdown(
    f"""<div class="gg-stat-row">
      <div class="gg-stat"><span>ACTIVE DATASET</span><strong>{'SENTINEL-2 · BANDS' if is_spectral else ('SYNTHETIC TEST SET' if is_synthetic else ('SENTINEL-2 · PREVIEW' if source_type == 'Satellite image preview' else 'USER IMAGERY'))}</strong></div>
      <div class="gg-stat"><span>ANALYSIS ENGINE</span><strong>{'SPECTRAL INDICES' if is_spectral else ('ML MODEL' if is_ml else 'VISUAL BASELINE')}</strong></div>
      <div class="gg-stat"><span>IMAGE PAIR</span><strong>{width:,} × {height:,} PX</strong></div>
      <div class="gg-stat"><span>{'SPECTRAL TRANSITIONS' if is_spectral else 'PIXELS FLAGGED'}</span><strong>{changed_pct:.2f}% <small>OF VALID PIXELS</small></strong></div>
    </div>""", unsafe_allow_html=True,
)
display_bounds = None
if is_spectral:
    display_bounds = overlap_bounds(satellite_data['valid_mask'])

st.markdown("<div class='gg-section-eyebrow'>EXPLORE THE DATA</div><div class='gg-section-heading'>Your observation workspace</div>", unsafe_allow_html=True)

tab_compare, tab_analyse, tab_ml, tab_case, tab_science, tab_project = st.tabs(
    ["🛰️ Imagery", "📊 Change analysis", "🧠 ML model",
     "🇳🇵 Nepal case study", "🔬 Methodology", "📘 About"]
)

with tab_compare:
    st.markdown('<div class="gg-panel-heading">01 / OBSERVATION COMPARISON</div>', unsafe_allow_html=True)
    st.subheader("Satellite image comparison")
    col_before, col_after = st.columns(2, gap="medium")
    with col_before:
        st.image(crop_to_bounds(before, display_bounds), caption=f"BEFORE · {before_date or 'Date not provided'}", use_container_width=True)
    with col_after:
        st.image(crop_to_bounds(after, display_bounds), caption=f"AFTER · {after_date or 'Date not provided'}", use_container_width=True)
    st.markdown("#### Interactive comparison")
    fade = st.slider(
        "Blend BEFORE → AFTER", 0, 100, 50, 5,
        help="Visual blend only. Does not align or classify imagery.",
    )
    blended = Image.blend(Image.fromarray(crop_to_bounds(before, display_bounds)),
                          Image.fromarray(crop_to_bounds(after, display_bounds)), fade / 100)
    st.image(blended, caption=f"Before/after blend · {fade}% after", use_container_width=True)
    if is_synthetic:
        st.warning(
            "Sample imagery is synthetic and is not evidence from Nepal."
        )
    else:
        st.caption(
            "Visual comparison is meaningful only when the images are co-registered "
            "and have comparable seasons, cloud cover, and sensor characteristics."
        )

with tab_analyse:
    st.markdown('<div class="gg-panel-heading">02 / CHANGE ANALYTICS</div>', unsafe_allow_html=True)
    st.subheader("Detected visual changes")
    m1, m2, m3 = st.columns(3)
    st.caption(f"Study site: {site_name} · Imagery: {source_reference or 'source not supplied'}")
    m1.metric("Candidate pixels" if is_spectral else "Flagged pixels", f"{changed_pixels:,}")
    m2.metric("Spectral transitions" if is_spectral else "Flagged image area", f"{changed_pct:.2f}%")
    m3.metric("Analysis method", "NDWI / NDSI" if is_spectral else ("ML model" if is_ml else "Pixel difference"))
    st.caption(
        "Percent is the fraction of valid overlapping pixels flagged by NDWI/NDSI candidate transitions. "
        "It is NOT lake expansion, glacier area, model accuracy or flood risk."
        if is_spectral else
        "Percent is the fraction of image pixels flagged by the detector. "
        "It is NOT the amount of lake expansion, geographic area, or predicted flood risk."
    )
    if is_spectral:
        c = result['counts']
        st.markdown("#### Spectral-index candidate counts")
        w1, w2, s1, s2 = st.columns(4)
        w1.metric("Water candidates · Aug", f"{c['water_candidate_before_pixels']:,} px")
        w2.metric("Water candidates · Oct", f"{c['water_candidate_after_pixels']:,} px")
        s1.metric("Snow candidates · Aug", f"{c['snow_candidate_before_pixels']:,} px")
        s2.metric("Snow candidates · Oct", f"{c['snow_candidate_after_pixels']:,} px")
        if exclude_snow:
            st.warning("With snow exclusion, there are zero candidate-water pixels in October. "
                       "This is a threshold artefact possibility, NOT evidence that a glacial lake vanished.")
        else:
            st.info("The NDWI-only setting reproduces the supplied measurements. The NDWI rule "
                    "also flags 380 overlapping snow-candidate pixels in August and 234 in October; "
                    "lake extent cannot be inferred directly.")
        with st.expander("Show candidate pixel areas from the supplied grid metadata"):
            a1, a2 = st.columns(2)
            estimated_area_factor = REFERENCE_PIXEL_AREA_M2 / 1_000_000
            a1.metric("Candidate water · before", f"{c['water_candidate_before_pixels'] * estimated_area_factor:.4f} km²")
            a2.metric("Candidate water · after", f"{c['water_candidate_after_pixels'] * estimated_area_factor:.4f} km²")
            st.caption("Calculated only from pixel counts × approximately 99.99185 m²/pixel "
                       "as supplied in the original grid metadata. These are THRESHOLD-CLASSIFIED "
                       "pixel areas—not field-validated lake area or flood extent.")

    a, b = st.columns(2, gap="medium")
    with a:
        st.image(crop_to_bounds(overlay, display_bounds), caption="After image · flagged visual changes highlighted", use_container_width=True)
    with b:
        st.image(crop_to_bounds(mask, display_bounds), caption="Binary change mask · white = flagged", use_container_width=True, clamp=True)

    if is_spectral:
        st.warning(
            "Spectral-change maps show NDWI/NDSI threshold candidates, not confirmed lake or glacier boundaries. "
            "Seasonal snow and cloud conditions affect interpretation."
        )
        with st.expander("View NDWI/NDSI maps and candidate changes", expanded=False):
            v = satellite_data['valid_mask']
            a, b = st.columns(2)
            with a:
                st.image(crop_to_bounds(index_image(satellite_data['ndwi_before'], v), display_bounds), caption="NDWI · 12 Aug 2026 · colour scale -1 to +1", use_container_width=True)
                st.image(crop_to_bounds(index_image(satellite_data['ndsi_before'], v), display_bounds), caption="NDSI · 12 Aug 2026 · colour scale -1 to +1", use_container_width=True)
            with b:
                st.image(crop_to_bounds(index_image(satellite_data['ndwi_after'], v), display_bounds), caption="NDWI · 13 Oct 2026 · colour scale -1 to +1", use_container_width=True)
                st.image(crop_to_bounds(index_image(satellite_data['ndsi_after'], v), display_bounds), caption="NDSI · 13 Oct 2026 · colour scale -1 to +1", use_container_width=True)
            st.caption("Index colour legend: purple/blue = lower values; green/yellow/red = higher values. Light grey = outside overlapping valid area.")
            x, y = st.columns(2)
            with x:
                st.image(crop_to_bounds(result['water_change_image'], display_bounds), caption="Water candidate changes: green gain, orange loss, grey unchanged", use_container_width=True)
            with y:
                st.image(crop_to_bounds(result['snow_change_image'], display_bounds), caption="Snow candidate changes: green gain, orange loss, grey unchanged", use_container_width=True)
            st.caption("Water: NDWI > 0.3; optional exclusion for pixels with NDSI > 0.4. Snow: NDSI > 0.4. These are unvalidated spectral candidates, not lake or glacier boundaries.")
    elif source_type == "Satellite image preview":
        st.warning(
            "These false-colour satellite images are for visual comparison. RGB differences can reflect "
            "cloud, shadow or seasonal variation, and should not be interpreted as flood risk."
        )
        with st.expander("View NDWI/NDSI change maps", expanded=False):
            st.caption(
                "Based on candidate NDWI water threshold 0.3 and NDSI snow threshold 0.4. "
                "Green = candidate gain, orange = candidate loss, grey = unchanged, "
                "dark = no overlapping data. These are threshold rules, not trained machine learning."
            )
            c1,c2=st.columns(2)
            with c1:
                st.image(str(SAT_FOLDER / "water_candidate_changes.png"),caption="Candidate water change — exploratory spectral index")
            with c2:
                st.image(str(SAT_FOLDER / "snow_candidate_changes.png"),caption="Candidate snow/ice change — exploratory spectral index")
            st.caption("12 Aug and 13 Oct 2026 · valid overlap only · cloud masking not independently verified")

    with st.expander("Inspect spectral-index change distribution" if is_spectral else "Inspect pixel-difference distribution"):
        histogram_values = difference[satellite_data['valid_mask']] if is_spectral else difference
        counts, edges = np.histogram(histogram_values, bins=np.arange(0, 257, 16))
        chart = pd.DataFrame({
            "Difference range": [f"{int(lo)}–{int(hi-1)}" for lo, hi in zip(edges[:-1], edges[1:])],
            "Pixel count": counts,
        }).set_index("Difference range")
        st.bar_chart(chart)
        st.caption("Diagnostic histogram of scaled absolute index differences, not confidence." if is_spectral else "Diagnostic histogram of grayscale differences, not model confidence.")

    lake_before = result.get("before_lake_mask")
    lake_after = result.get("after_lake_mask")
    if lake_before is not None and lake_after is not None:
        lake_before = np.asarray(lake_before)
        lake_after = np.asarray(lake_after)
        if lake_before.shape == mask.shape and lake_after.shape == mask.shape:
            before_lake_px = int(np.count_nonzero(lake_before))
            after_lake_px = int(np.count_nonzero(lake_after))
            st.markdown("#### Classified lake extent (model output)")
            p1, p2 = st.columns(2)
            p1.metric("Lake pixels · before", f"{before_lake_px:,}")
            p2.metric("Lake pixels · after", f"{after_lake_px:,}",
                      delta=f"{after_lake_px-before_lake_px:+,} pixels")
            st.caption(
                "These are model-labelled pixel counts, not square kilometres. "
                "Georeferencing and validation are required for physical area estimates."
            )

    if is_spectral:
        st.caption("Change mask combines candidate water and snow transitions. Orange overlay marks candidate spectral transitions, not flood damage or confirmed lake boundaries.")
    st.markdown("#### Export reproducible outputs")
    downloads = st.columns(3)
    downloads[0].download_button(
        "⬇ Download overlay PNG", data=png_bytes(overlay),
        file_name="glacierguard_overlay.png", mime="image/png", use_container_width=True,
    )
    downloads[1].download_button(
        "⬇ Download mask PNG", data=png_bytes(mask),
        file_name="glacierguard_change_mask.png", mime="image/png", use_container_width=True,
    )
    report = {
        "project": "GlacierGuard AI",
        "analysis_type": "observed image change; not flood-risk prediction",
        "analysis_engine_selected": method,
        "data_type": "aligned user-supplied Sentinel-2 band arrays; unverified product IDs" if is_spectral else ("synthetic demo" if is_synthetic else "user-provided imagery; provenance not verified"),
        "before_date": before_date,
        "after_date": after_date,
        "site_name": site_name,
        "imagery_source": source_reference,
        "alignment_user_confirmed": alignment_confirmed,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "height_px": height,
        "width_px": width,
        "threshold": threshold,
        "spectral_thresholds": {"ndwi_water": 0.3, "ndsi_snow": 0.4} if is_spectral else None,
        "method": method,
        "is_machine_learning": is_ml,
        "changed_pixels": changed_pixels,
        "flagged_image_area_percent": round(changed_pct, 4),
        "flagged_percentage_denominator": "valid overlapping pixels" if is_spectral else "all image pixels",
        "spectral_candidate_counts": result['counts'] if is_spectral else None,
        "water_candidate_rule": result.get('water_rule') if is_spectral else None,
        "approximate_pixel_area_m2_from_supplied_metadata": REFERENCE_PIXEL_AREA_M2 if is_spectral else None,
        "notes": result.get("explanation", ""),
        "limitations": (
            "Pixel flags alone do not identify glacier-lake boundaries, establish "
            "flood risk, or prove causal links to a historical disaster."
        ),
    }
    downloads[2].download_button(
        "⬇ Download results JSON", data=json.dumps(report, indent=2),
        file_name="glacierguard_results.json", mime="application/json", use_container_width=True,
    )

with tab_case:
    st.markdown('<div class="gg-panel-heading">04 / FIELD CONTEXT</div>', unsafe_allow_html=True)
    st.subheader("Nepal, 26 August 2026 — glacier-related flood")
    st.markdown(
        "**Why this case matters:** A major slope failure involving glacier ice and rock "
        "in the Langtang region initiated a destructive debris flow and flash flood affecting "
        "communities downstream in Rasuwa. The precise initiating mechanism was still being "
        "investigated in early official assessments."
    )
    st.warning(
        "**Different hazards:** This 2026 event should not automatically be presented "
        "as a glacial-lake outburst flood. Glacier/rock collapse and glacial-lake "
        "expansion are related research concerns, but they are not the same process."
    )
    a, b, c = st.columns(3)
    a.metric("Event date", "26 Aug 2026")
    b.metric("Nepal focus", "Rasuwa / Langtang")
    c.metric("This app's output", "Visual changes")
    st.markdown("#### Evidence and imagery")
    st.markdown(
        "- [USGS — 2026 Nepal Debris Avalanche and Flash Flood]"
        "(https://www.usgs.gov/programs/landslide-hazards/science/2026-nepal-debris-avalanche-and-flash-flood) "
        "— event description, mechanism uncertainty and satellite mapping.\n"
        "- [ESA — Nepal flash flood imaged by satellites]"
        "(https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Nepal_flash_flood_imaged_by_satellites) "
        "— before/after satellite views and context.\n"
        "- [Copernicus Emergency Management — activation EMSR927]"
        "(https://mapping.emergency.copernicus.eu/activations/EMSR927/) "
        "— mapped flood impacts and reference products."
    )
    st.markdown("#### Interpretation of the results")
    st.markdown(
        "A properly aligned image pair can help **locate and visualise observed surface changes**. "
        "A pixel-difference mask alone cannot determine *why* a change occurred, prove "
        "that a lake is growing, or reliably forecast another flood. Our case study is "
        "motivational until site-specific, legally reusable images and a validated model "
        "are added. The synthetic demo is **not evidence from Nepal**."
    )
    st.markdown("#### Suggested validation plan")
    st.markdown(
        "1. Pick a defined area of interest and document imagery dates, product IDs and licence.\n"
        "2. Co-register comparable images, mask clouds/shadows and document the resolution.\n"
        "3. Label an independent reference set of lake/non-lake pixels.\n"
        "4. Evaluate a lake segmentation model with precision, recall and IoU.\n"
        "5. Only then interpret *mapped lake changes* separately from wider flood impact."
    )

with tab_science:
    st.markdown('<div class="gg-panel-heading">05 / METHODOLOGY</div>', unsafe_allow_html=True)
    st.subheader("Method and limitations")
    st.write(f"**Current approach:** {method}")
    st.write(result.get("explanation", "No method description provided."))
    st.markdown(
        """
**Important scientific safeguards**

- **Change is not flood risk.** A changed satellite pixel may represent water,
  snow, cloud, shade, illumination, land cover, or imperfect image registration.
- **Same-place imagery is essential.** Align and co-register images before analysis.
- **Water mapping needs appropriate data.** Genuine glacial-lake mapping is better
  supported by multispectral Sentinel-2 imagery (including green and near-infrared
  bands), validated segmentation, and site-specific reference labels.
- **No emergency warning claims.** This student research prototype cannot forecast
  glacier collapse or glacial-lake outburst floods.
- **No area in km² without coordinates.** Geographic measurements require known
  pixel size/projection and meaningful validation.
- **Research integrity.** Record image source, capture dates, preprocessing steps,
  model version, metrics, and limitations in the report.
"""
    )

with tab_project:
    st.markdown('<div class="gg-panel-heading">06 / PROJECT OVERVIEW</div>', unsafe_allow_html=True)
    st.subheader("GlacierGuard AI")
    st.write("Satellite-based glacier-environment monitoring for the Langtang region of Nepal.")
    st.markdown("""
**Study objective**

Analyse multispectral Sentinel-2 observations from 12 August and 13 October 2026,
visualise spectral-index changes, and examine unsupervised land-surface clustering.

**Technical approach**

- Preprocessed Sentinel-2 B03, B04, B08 and B11 imagery on a shared grid.
- Calculated NDWI and NDSI for candidate water and snow observations.
- Trained a four-group MiniBatchKMeans model using six spectral features.
- Compared outputs across two dates and reported internal cluster separation.
- Published the analysis as an interactive Streamlit dashboard.

**Study scope**

Results describe spectral change within the observed area. They do not establish
glacial-lake boundaries, flood causation, or real-time early warning.

**Scientific data sources**

- [Copernicus Data Space](https://browser.dataspace.copernicus.eu/)
- [ICIMOD](https://www.icimod.org/)
- [NASA Earthdata](https://www.earthdata.nasa.gov/)

CDU IT Code Fair Sydney 2026 · AI Challenge
""")



with tab_ml:
    st.markdown('<div class="gg-panel-heading">03 / MACHINE LEARNING</div>', unsafe_allow_html=True)
    st.subheader("Six-feature spectral clustering")
    if FINAL_METADATA.is_file() and FINAL_CLUSTERS.is_file():
        final_info = json.loads(FINAL_METADATA.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("Training observations", "40,000")
        cols[1].metric("Spectral groups", str(final_info.get("ml_clusters", 4)))
        score = final_info.get("ml_internal_silhouette")
        cols[2].metric("Silhouette score", f"{score:.4f}" if isinstance(score, (float, int)) else "N/A")
        cols[3].metric("Image overlap", f"{final_info.get('overlap_reference_percent', 0):.2f}%")
        st.caption("MiniBatchKMeans · bands B03, B04, B08, B11, NDWI and NDSI · 20,000 training samples per date")
        cluster_pic = FINAL_RESULTS / "unsupervised_clusters.png"
        if cluster_pic.is_file():
            st.image(str(cluster_pic), caption="Unsupervised spectral groups · August and October 2026", use_container_width=True)
        st.markdown("#### Group distribution")
        st.dataframe(pd.read_csv(FINAL_CLUSTERS), hide_index=True, use_container_width=True)
        st.download_button("Download group counts", data=FINAL_CLUSTERS.read_bytes(),
                           file_name="cluster_counts.csv", mime="text/csv")
        with st.expander("Additional plots and index statistics"):
            for fname, cap in [
                ("false_colour_comparison.png", "False-colour Sentinel-2 imagery"),
                ("index_maps.png", "NDWI and NDSI maps"),
                ("candidate_change_map.png", "Candidate spectral transitions"),
            ]:
                path = FINAL_RESULTS / fname
                if path.is_file():
                    st.image(str(path), caption=cap, use_container_width=True)
            if FINAL_STATS.is_file():
                st.dataframe(pd.read_csv(FINAL_STATS), hide_index=True, use_container_width=True)
                st.download_button("Download spectral statistics", data=FINAL_STATS.read_bytes(),
                                   file_name="spectral_statistics.csv", mime="text/csv")
        st.info("The silhouette score measures cluster separation, not classification accuracy. "
                "The spectral groups are not independently labelled as lakes or glaciers, "
                "and this model does not predict floods.")
    else:
        st.warning("Model output files are unavailable.")


with tab_science:
    if ORIGINAL_RESULTS_CSV.is_file():
        with st.expander("Numerical consistency audit — reconciled NDWI/NDSI export"):
            st.success(
                "**Reconciled:** All eight exported CSV areas match the supplied NPZ when "
                "the NDWI-only (>0.3) water rule and NDSI-only (>0.4) snow "
                "rule are used with the pixel area from the reference-grid metadata. "
                "The optional snow-exclusion sensitivity test produces DIFFERENT water counts."
            )
            st.warning(
                "Agreement between code and CSV confirms computational consistency, NOT "
                "that the threshold classifications, georeferencing or disaster attribution "
                "have been independently validated."
            )
            st.dataframe(pd.read_csv(ORIGINAL_RESULTS_CSV), hide_index=True, use_container_width=True)
            if ORIGINAL_RESULTS_MAP.is_file():
                st.image(str(ORIGINAL_RESULTS_MAP),
                         caption="Original spectral summary figure",
                         use_container_width=True)
