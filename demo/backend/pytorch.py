# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import copy
import numpy as np
import torch
from typing import Any, Optional

from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor

DET_ARCHS = [
    "fast_base",
    "fast_small",
    "fast_tiny",
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]
RECO_ARCHS = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "parseq",
    "viptr_tiny",
]


def load_predictor(
    det_arch: str,
    reco_arch: str,
    assume_straight_pages: bool,
    straighten_pages: bool,
    export_as_straight_boxes: bool,
    disable_page_orientation: bool,
    disable_crop_orientation: bool,
    bin_thresh: float,
    box_thresh: float,
    device: torch.device,
) -> OCRPredictor:
    """Load a predictor from doctr.models

    Args:
        det_arch: detection architecture
        reco_arch: recognition architecture
        assume_straight_pages: whether to assume straight pages or not
        straighten_pages: whether to straighten rotated pages or not
        export_as_straight_boxes: whether to export boxes as straight or not
        disable_page_orientation: whether to disable page orientation or not
        disable_crop_orientation: whether to disable crop orientation or not
        bin_thresh: binarization threshold for the segmentation map
        box_thresh: minimal objectness score to consider a box
        device: torch.device, the device to load the predictor on

    Returns:
        instance of OCRPredictor
    """
    predictor = ocr_predictor(
        det_arch,
        reco_arch,
        pretrained=True,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_orientation=not assume_straight_pages,
        disable_page_orientation=disable_page_orientation,
        disable_crop_orientation=disable_crop_orientation,
    ).to(device)
    predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
    predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
    return predictor


def forward_image(predictor: OCRPredictor, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Forward an image through the predictor

    Args:
        predictor: instance of OCRPredictor
        image: image to process
        device: torch.device, the device to process the image on

    Returns:
        segmentation map
    """
    with torch.no_grad():
        processed_batches = predictor.det_predictor.pre_processor([image])
        out = predictor.det_predictor.model(processed_batches[0].to(device), return_model_output=True)
        seg_map = out["out_map"].to("cpu").numpy()

    return seg_map


def translate_page_export(
    page_export: dict[str, Any],
    source_lang: str,
    target_lang: str,
    translator: Optional[Any] = None,
) -> dict[str, Any]:
    """Translate all words in a page export dictionary
    
    Args:
        page_export: exported Page object dictionary
        source_lang: source language code (e.g., 'en', 'fr', 'de')
        target_lang: target language code (e.g., 'en', 'fr', 'de')
        translator: optional pre-loaded translation pipeline
    
    Returns:
        translated page export dictionary
    """
    # Only translate if source and target languages are different
    if source_lang == target_lang or not source_lang or not target_lang:
        return page_export
    
    # Lazy import to avoid loading transformers if not needed
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("transformers library is required for translation. Install it with: pip install transformers")
    
    # MBART language code mapping (mbart uses different codes)
    mbart_lang_map = {
        "ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX",
        "et": "et_EE", "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN",
        "it": "it_IT", "ja": "ja_XX", "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT",
        "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP", "nl": "nl_XX", "ro": "ro_RO",
        "ru": "ru_RU", "si": "si_LK", "tr": "tr_TR", "vi": "vi_VN", "zh": "zh_CN",
        "af": "af_ZA", "az": "az_AZ", "bn": "bn_IN", "fa": "fa_IR", "he": "he_IL",
        "hr": "hr_HR", "id": "id_ID", "ka": "ka_GE", "km": "km_KH", "mk": "mk_MK",
        "ml": "ml_IN", "mn": "mn_MN", "mr": "mr_IN", "pl": "pl_PL", "ps": "ps_AF",
        "pt": "pt_XX", "sv": "sv_SE", "sw": "sw_KE", "ta": "ta_IN", "uk": "uk_UA",
        "ur": "ur_PK", "xh": "xh_ZA", "gl": "gl_ES", "sl": "sl_SI"
    }
    
    # Load translation pipeline if not provided
    if translator is None:
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        use_mbart = False
        try:
            translator = pipeline("translation", model=model_name, device=0 if torch.cuda.is_available() else -1)
        except Exception:
            # Fallback to a multilingual model if language pair not available
            use_mbart = True
            try:
                translator = pipeline(
                    "translation",
                    model="facebook/mbart-large-50-many-to-many-mmt",
                    device=0 if torch.cuda.is_available() else -1,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load translation model: {e}")
    else:
        # Check if translator is mbart
        use_mbart = hasattr(translator, "model") and hasattr(translator.model, "config") and "mbart" in translator.model.config.model_type.lower()
    
    # Deep copy to preserve all fields
    translated_export = copy.deepcopy(page_export)
    
    # Translate each word in each block
    for block in translated_export["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                original_text = word["value"]
                
                # Translate the text
                if original_text.strip():  # Only translate non-empty text
                    try:
                        # Use the translation pipeline
                        if use_mbart:
                            # Map to mbart language codes
                            src_lang_mbart = mbart_lang_map.get(source_lang, f"{source_lang}_XX")
                            tgt_lang_mbart = mbart_lang_map.get(target_lang, f"{target_lang}_XX")
                            translator.tokenizer.src_lang = src_lang_mbart
                            result = translator(original_text, src_lang=src_lang_mbart, tgt_lang=tgt_lang_mbart)
                        else:
                            result = translator(original_text)
                        
                        # Extract translated text
                        if isinstance(result, list) and len(result) > 0:
                            translated_text = result[0].get("translation_text", original_text)
                        elif isinstance(result, dict):
                            translated_text = result.get("translation_text", original_text)
                        else:
                            translated_text = str(result)
                        
                        word["value"] = translated_text
                    except Exception as e:
                        # If translation fails, keep original text
                        print(f"Translation failed for '{original_text}': {e}")
                        word["value"] = original_text
    
    return translated_export
