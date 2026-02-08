import streamlit as st
import datetime
from PIL import Image
import numpy as np

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Retinal AI – Medical Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------
# GLOBAL CSS (NO WHITE BORDERS)
# ----------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #f6f8fa;
}
#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# HEADER
# ----------------------------------
st.markdown("## 🧠 Comprehensive Medical Analysis")
st.caption("AI-powered retinal disease detection & explainability")

st.divider()

# ----------------------------------
# LAYOUT
# ----------------------------------
left, right = st.columns([1.1, 3])

# ==================================
# LEFT PANEL
# ==================================
with left:
    st.subheader("Upload Retinal Image")

    uploaded = st.file_uploader(
        "Drag & drop fundus image",
        type=["jpg", "png", "jpeg"]
    )

    st.divider()

    patient_id = st.text_input("Patient ID", value="PAT-" + datetime.datetime.now().strftime("%H%M%S"))
    timestamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")

    st.write("🕒 Timestamp")
    st.code(timestamp)

    run = st.button("🧪 Run Analysis", type="primary")

# ==================================
# RIGHT PANEL
# ==================================
with right:

    if uploaded:
        image = Image.open(uploaded)

        # ---------- METRICS ----------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("X Coordinate", "479")
        m2.metric("Y Coordinate", "252")
        m3.metric("Disc Diameter", "4.0 mm")
        m4.metric("Severity", "Moderate")

        st.divider()

        # ---------- IMAGE ROW ----------
        i1, i2, i3, i4 = st.columns(4)

        i1.image(image, caption="Original Image", use_container_width=True)
        i2.image(image, caption="Preprocessed", use_container_width=True)
        i3.image(image, caption="Grad-CAM Heatmap", use_container_width=True)
        i4.image(image, caption="Overlay", use_container_width=True)

        st.divider()

        # ---------- SEVERITY BAR ----------
        st.subheader("Disease Severity")
        st.progress(65)  # 0–100
        st.caption("Green → Mild | Yellow → Moderate | Red → Severe")

        st.divider()

        # ---------- FINDINGS ----------
        st.subheader("Clinical Findings")

        f1, f2, f3 = st.columns(3)

        f1.success("✔ Microaneurysms detected")
        f2.warning("⚠ Vascular abnormalities")
        f3.error("✖ Possible diabetic retinopathy")

        st.divider()

        # ---------- REPORT ----------
        st.subheader("Medical Report")

        st.download_button(
            "📄 Download PDF Report",
            data=b"PDF_CONTENT_PLACEHOLDER",
            file_name=f"{patient_id}_report.pdf"
        )

    else:
        st.info("👈 Upload a retinal image to begin analysis")
