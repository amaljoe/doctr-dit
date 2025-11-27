import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

from translation import translate_lines, update_page_with_layout
from utils import synthesize_page, extract_source_target_pairs
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page

forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(det_archs, reco_archs):
    """Build a streamlit layout"""
    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("End to End Document Image Translation")
    # For newline
    st.write("\n")
    # Instructions
    st.markdown(
        "*Hint: click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    cols = st.columns((1, 1, 1, 1))
    cols[0].subheader("1. Input page")
    cols[1].subheader("2. OCR output")
    cols[2].subheader("3. Layout Analysis")
    cols[3].subheader("5. Page reconstruction")

    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader(
        "Upload files", type=["pdf", "png", "jpeg", "jpg"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox(
            "Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        page = doc[page_idx]
        cols[0].image(page)

    # Model selection
    st.sidebar.title("Model selection")
    st.sidebar.markdown("**Backend**: PyTorch")
    det_arch = st.sidebar.selectbox("Text detection model", det_archs)
    reco_arch = st.sidebar.selectbox("Text recognition model", reco_archs)
    layout_models = ["No Layout Analysis", "LayoutLM + Decoder (Ours)"]
    layout_model = st.sidebar.selectbox("Layout analysis model", layout_models, index=len(layout_models) - 1)
    langs = ["English", "Chinese", "German", "French", "Hindi", "Malayalam", "Telugu", "Tamil"]
    lang_dict = {
        "English": ("eng_Latn", "noto.ttf"),
        "Simplified Chinese": ("zho_Hans", "noto-chinese.ttf"),
        "German": ("deu_Latn", "noto.ttf"),
        "French": ("fra_Latn", "noto.ttf"),
        "Hindi": ("hin_Deva", "noto-deva.ttf"),
        "Malayalam": ("mal_Mlym", "noto-mal.ttf"),
        "Telugu": ("tel_Telu", "noto-telugu.ttf"),
        "Tamil": ("tam_Taml", "noto-tamil.ttf"),
    }

    target_language = st.sidebar.selectbox("Target language", langs, index=5)

    # For newline
    st.sidebar.write("\n")
    # Only straight pages or possible rotation
    # st.sidebar.title("Parameters")
    assume_straight_pages = True
    disable_page_orientation = False
    disable_crop_orientation = False
    straighten_pages = False
    export_straight_boxes = False
    bin_thresh = 0.3
    box_thresh = 0.1
    # bin_thresh = st.sidebar.slider(
    #     "Binarization threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
    # st.sidebar.write("\n")
    # box_thresh = st.sidebar.slider(
    #     "Box threshold", min_value=0.1, max_value=0.9, value=0.1, step=0.1)
    # st.sidebar.write("\n")

    if st.sidebar.button("Analyze page"):
        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner("Loading model..."):
                predictor = load_predictor(
                    det_arch=det_arch,
                    reco_arch=reco_arch,
                    assume_straight_pages=assume_straight_pages,
                    straighten_pages=straighten_pages,
                    export_as_straight_boxes=export_straight_boxes,
                    disable_page_orientation=disable_page_orientation,
                    disable_crop_orientation=disable_crop_orientation,
                    bin_thresh=bin_thresh,
                    box_thresh=box_thresh,
                    device=forward_device,
                )

            with st.spinner("Detecting text..."):
                # # Forward the image to the model
                # seg_map = forward_image(predictor, page, forward_device)
                # seg_map = np.squeeze(seg_map)
                # seg_map = cv2.resize(
                #     seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)

                # # Plot the raw heatmap
                # fig, ax = plt.subplots()
                # ax.imshow(seg_map)
                # ax.axis("off")
                # cols[1].pyplot(fig)

                # Plot OCR output
                out = predictor([page])
                fig = visualize_page(out.pages[0].export(
                ), out.pages[0].page, interactive=False, add_labels=False)
                cols[1].pyplot(fig)
                cols[2].pyplot(fig)
                
            with st.spinner("Understanding layout..."):
                # Page reconsitution under input page
                page_export = out.pages[0].export()
                page_export = update_page_with_layout(page, page_export)
            with st.spinner("Translating text..."):
                page_export = translate_lines(page_export, lang=lang_dict[target_language][0])
                # Display JSON
                pairs = extract_source_target_pairs(page_export)
                st.subheader("\n4. Translation input and output:")
                st.json(pairs, expanded=True)
                
                if assume_straight_pages or (not assume_straight_pages and straighten_pages):
                    img = synthesize_page(
                        page_export,
                        out.pages[0].page,
                        font_family=f"fonts/{lang_dict[target_language][1]}",
                    )
                    cols[3].image(img, clamp=True)

                


if __name__ == "__main__":
    main(DET_ARCHS, RECO_ARCHS)
