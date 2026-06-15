import os
import re
import threading
import uuid
import webbrowser
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for
from werkzeug.utils import secure_filename

from auth_store import authenticate_farmer, register_farmer

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.environ.get("FARMX_SECRET_KEY", "farmx-dev-secret-change-in-production")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# -------------------------------------------------------------
# 1. CORE ALEXNET CONFIGURATIONS & INFERENCE PIPELINE
# -------------------------------------------------------------
MODEL_PATH = str(BASE_DIR / "AlexNetModel.hdf5")
FALLBACK_MODELS = ["rec_add_attention_v1", "densenet169_v1", "mobilenetv2_v1"]
LOW_CONFIDENCE_THRESHOLD = 0.55
PREDICTION_TOP_K = 3
IMAGE_TARGET_SIZE = (224, 224)

model = None
_model_load_attempted = False
_fallback_predictor = None
_fallback_predictor_name = None
_fallback_predictor_failed = False

SUPPORTED_LANGS = ("en", "hi", "es", "kn", "te", "mr", "bn")
SPEECH_LANG_CODES = {
    "en": "en-US",
    "hi": "hi-IN",
    "es": "es-ES",
    "kn": "kn-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
}


def get_model():
    """Load AlexNet weights once (only when the model file exists)."""
    global model, _model_load_attempted
    if _model_load_attempted:
        return model

    _model_load_attempted = True
    if not os.path.exists(MODEL_PATH):
        print(f" [!] {MODEL_PATH} not found. Using built-in PlantVillage model.")
        return None

    try:
        from tensorflow.keras.models import load_model

        print(" ** Loading User Custom AlexNet Architecture weights... **")
        model = load_model(MODEL_PATH)
        print(" ** Neural Network Weights Applied Successfully **")
    except ImportError:
        print(" [!] TensorFlow is not installed. Run: pip install -r requirements.txt")
    except Exception as exc:
        print(f" [!] Could not load model: {exc}")

    return model


def get_fallback_predictor():
    """Load highest-accuracy available PlantVillage model when custom weights are missing."""
    global _fallback_predictor, _fallback_predictor_name, _fallback_predictor_failed
    if _fallback_predictor_failed:
        return None
    if _fallback_predictor is not None:
        return _fallback_predictor

    try:
        from plantdoc_predictor import Predictor
    except ImportError:
        print(" [!] Install dependencies: pip install -r requirements.txt")
        _fallback_predictor_failed = True
        return None

    for model_name in FALLBACK_MODELS:
        try:
            print(f" ** Loading built-in classifier ({model_name})... **")
            _fallback_predictor = Predictor(model_name=model_name, verbose=False)
            _fallback_predictor_name = model_name
            print(f" ** Built-in classifier ready: {model_name} **")
            return _fallback_predictor
        except Exception as exc:
            print(f" [!] Could not load {model_name}: {exc}")

    _fallback_predictor_failed = True
    return None


def parse_class_label(label: str):
    """Split a PlantVillage label into crop and condition names."""
    parts = label.split("___", 1)
    crop_name = parts[0].replace("_", " ").replace(",", ",") if parts else "Unknown Crop"
    condition_name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown Condition"
    return crop_name, condition_name

CLASSES_LIST = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DISEASE_TREATMENTS = {
    "Apple___Apple_scab": "Apply sulfur or captan fungicide early in season. Prune for airflow and remove fallen leaves.",
    "Apple___Black_rot": "Remove mummified fruit and infected branches. Spray copper fungicide during bloom.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar/juniper hosts if possible. Use myclobutanil or sulfur sprays.",
    "Apple___healthy": "No disease detected. Maintain balanced fertilizer and regular pruning.",
    "Blueberry___healthy": "No disease detected. Keep acidic soil (pH 4.5–5.5) and mulch roots.",
    "Cherry_(including_sour)___Powdery_mildew": "Improve airflow, avoid overhead watering. Spray sulfur or potassium bicarbonate.",
    "Cherry_(including_sour)___healthy": "No disease detected. Monitor for powdery mildew in humid weather.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, use resistant hybrids. Apply strobilurin fungicide if severe.",
    "Corn_(maize)___Common_rust_": "Plant rust-resistant varieties. Apply fungicide if rust covers more than 5% of leaves.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids, rotate with soybeans. Apply fungicide at tasseling if needed.",
    "Corn_(maize)___healthy": "No disease detected. Ensure adequate nitrogen and weed control.",
    "Grape___Black_rot": "Remove infected fruit and leaves. Apply mancozeb or captan from bud break through bloom.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood during dry weather. No cure—focus on prevention and vineyard hygiene.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Remove infected leaves. Apply copper or mancozeb fungicide.",
    "Grape___healthy": "No disease detected. Maintain canopy airflow and balanced irrigation.",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure available. Remove infected trees and control Asian citrus psyllid vectors.",
    "Peach___Bacterial_spot": "Use copper sprays in dormancy. Plant resistant varieties and avoid overhead irrigation.",
    "Peach___healthy": "No disease detected. Apply dormant copper spray as preventive care.",
    "Pepper,_bell___Bacterial_spot": "Use disease-free seed, rotate crops. Apply copper bactericide preventively.",
    "Pepper,_bell___healthy": "No disease detected. Avoid working wet plants to prevent spread.",
    "Potato___Early_blight": "Apply chlorothalonil or mancozeb. Rotate crops and remove infected foliage.",
    "Potato___Late_blight": "Destroy infected plants immediately. Apply copper or chlorothalonil; use certified seed.",
    "Potato___healthy": "No disease detected. Hill soil around stems and avoid over-irrigation.",
    "Raspberry___healthy": "No disease detected. Prune canes annually and maintain good drainage.",
    "Soybean___healthy": "No disease detected. Rotate with corn and test soil nutrients.",
    "Squash___Powdery_mildew": "Spray neem oil or sulfur. Plant resistant varieties and improve spacing.",
    "Strawberry___Leaf_scorch": "Remove infected leaves. Apply fungicide and ensure good drainage.",
    "Strawberry___healthy": "No disease detected. Mulch beds and rotate planting sites every 3 years.",
    "Tomato___Bacterial_spot": "Use copper spray, avoid overhead watering. Remove infected leaves and rotate crops.",
    "Tomato___Early_blight": "Remove lower infected leaves. Apply copper or chlorothalonil; mulch and stake plants.",
    "Tomato___Late_blight": "Remove and destroy infected plants. Apply copper fungicide; avoid wet foliage.",
    "Tomato___Leaf_Mold": "Improve greenhouse ventilation. Apply chlorothalonil or copper fungicide.",
    "Tomato___Septoria_leaf_spot": "Remove infected lower leaves. Apply copper or mancozeb fungicide weekly.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray neem oil or insecticidal soap. Increase humidity and inspect undersides of leaves.",
    "Tomato___Target_Spot": "Apply chlorothalonil or azoxystrobin. Rotate crops and remove plant debris.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with yellow sticky traps and neem oil. Remove infected plants.",
    "Tomato___Tomato_mosaic_virus": "No chemical cure. Remove infected plants and disinfect tools; use virus-free seed.",
    "Tomato___healthy": "No disease detected. Stake plants and water at the base, not on leaves.",
}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("farmer_email"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def _current_farmer():
    if not session.get("farmer_email"):
        return None
    return {"email": session["farmer_email"], "name": session.get("farmer_name", "Farmer")}


COMMUNITY_POSTS = [
    {
        "user": "Farmer_Rajesh",
        "crop": "Tomato",
        "text": "Are any folks nearby seeing early blight? Looking for organic remedy suggestions.",
        "replies": [
            "Try copper fungicide sprays early in the morning.",
            "Ensure proper crop spacing to lower humidity!",
        ],
    },
    {
        "user": "AgroExpert_Anjali",
        "crop": "Corn",
        "text": "Market alert: Rust is hitting local corn yields. Keep clear records for insurance claims.",
        "replies": ["Using resistant hybrids next season helps drastically."],
    },
]


def _prepare_leaf_image(img_path: str, output_path: str | None = None) -> str:
    """Enhance leaf photos for more reliable disease classification."""
    from PIL import Image, ImageEnhance, ImageOps

    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)

    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize(IMAGE_TARGET_SIZE, Image.Resampling.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Sharpness(img).enhance(1.08)

    save_to = output_path or img_path
    img.save(save_to, quality=95)
    return save_to


def _leaf_image_variants(img_path: str) -> list[str]:
    """Build augmented views used for test-time averaging."""
    from PIL import Image

    prepared = _prepare_leaf_image(img_path, str(UPLOAD_DIR / f"prepared_{uuid.uuid4().hex}.jpg"))
    variants = [prepared]
    base = Image.open(prepared).convert("RGB")
    variants.append(str(UPLOAD_DIR / f"flip_{uuid.uuid4().hex}.jpg"))
    base.transpose(Image.FLIP_LEFT_RIGHT).save(variants[-1], quality=95)
    return variants


def _keras_predict_with_tta(active_model, img_path: str, classes_list):
    """Average predictions across multiple leaf views for better accuracy."""
    import numpy as np
    from PIL import Image
    from tensorflow.keras.preprocessing import image as keras_image

    def _batch(path: str):
        img = keras_image.load_img(path, target_size=IMAGE_TARGET_SIZE)
        x = keras_image.img_to_array(img)
        return np.expand_dims(x, axis=0) / 255.0

    accum = None
    variants = _leaf_image_variants(img_path)
    for variant in variants:
        batch = _batch(variant)
        preds = active_model.predict(batch, verbose=0)
        accum = preds if accum is None else accum + preds
    avg = (accum / len(variants)).flatten()

    top_indices = np.argsort(avg)[::-1][:PREDICTION_TOP_K]
    top_predictions = []
    for idx in top_indices:
        label = classes_list[int(idx)]
        crop, condition = parse_class_label(label)
        top_predictions.append(
            {"crop": crop, "condition": condition, "confidence": float(avg[idx]), "label": label}
        )
    best = top_predictions[0]
    return best["crop"], best["condition"], best["confidence"], top_predictions


def _fallback_predict_with_tta(fallback, img_path: str):
    """Run built-in PlantVillage model on original and flipped leaf images."""
    from PIL import Image

    prepared = _prepare_leaf_image(img_path, str(UPLOAD_DIR / f"fb_{uuid.uuid4().hex}.jpg"))
    paths = [prepared]
    flipped = str(UPLOAD_DIR / f"fb_flip_{uuid.uuid4().hex}.jpg")
    Image.open(prepared).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT).save(flipped, quality=95)
    paths.append(flipped)

    label_scores: dict[str, float] = {}
    for path in paths:
        result = fallback.predict(path, top_k=PREDICTION_TOP_K)
        label_scores[result["label"]] = label_scores.get(result["label"], 0.0) + float(
            result.get("confidence", 0)
        )
        for item in result.get("top_k", []):
            label_scores[item["label"]] = label_scores.get(item["label"], 0.0) + float(
                item["confidence"]
            )

    for label in label_scores:
        label_scores[label] /= len(paths)

    ranked = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)[:PREDICTION_TOP_K]
    best_label, best_conf = ranked[0]
    crop_name, condition_name = parse_class_label(best_label)
    top_predictions = []
    for label, conf in ranked:
        crop, condition = parse_class_label(label)
        top_predictions.append(
            {"crop": crop, "condition": condition, "confidence": conf, "label": label}
        )
    return crop_name, condition_name, best_conf, top_predictions


def _treatment_for_label(label: str, condition_name: str) -> str:
    if label in DISEASE_TREATMENTS:
        return DISEASE_TREATMENTS[label]
    if "healthy" in label.lower():
        return "No disease detected. Continue regular crop care and monitoring."
    return (
        f"Consult a local agronomist for {condition_name}. "
        "Remove affected leaves, improve airflow, and avoid overhead watering."
    )


def _build_prediction_result(crop_name, condition_name, confidence, top_predictions, model_used):
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    best_label = top_predictions[0].get("label", "") if top_predictions else ""
    treatment = _treatment_for_label(best_label, condition_name)
    return {
        "crop": crop_name,
        "condition": condition_name,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "model_used": model_used,
        "low_confidence": low_confidence,
        "treatment": treatment,
        "label": best_label,
    }


def model_predict(img_path: str):
    """Returns a dict with crop, condition, confidence, top_predictions, and model metadata."""
    active_model = get_model()
    if active_model is not None:
        try:
            crop_name, condition_name, confidence, top_predictions = _keras_predict_with_tta(
                active_model, img_path, CLASSES_LIST
            )
            return _build_prediction_result(
                crop_name, condition_name, confidence, top_predictions, "AlexNet (TTA)"
            )
        except ImportError:
            pass
        except Exception as exc:
            print(f" [!] Custom model prediction failed: {exc}")

    fallback = get_fallback_predictor()
    if fallback is not None:
        try:
            crop_name, condition_name, confidence, top_predictions = _fallback_predict_with_tta(
                fallback, img_path
            )
            model_used = _fallback_predictor_name or "PlantVillage"
            return _build_prediction_result(
                crop_name, condition_name, confidence, top_predictions, f"{model_used} (TTA)"
            )
        except Exception as exc:
            print(f" [!] Built-in model prediction failed: {exc}")

    raise RuntimeError(
        "No disease model available. Run: pip install -r requirements.txt "
        "or place AlexNetModel.hdf5 in the project folder."
    )


CHATBOT_FALLBACKS = {
    "empty": {
        "en": "Please type a question about crops, diseases, prices, or farming care.",
        "hi": "कृपया फसल, रोग, भाव या खेती से जुड़ा सवाल लिखें।",
        "es": "Escriba una pregunta sobre cultivos, enfermedades o precios.",
        "kn": "ದಯವಿಟ್ಟು ಬೆಳೆ, ರೋಗ, ಬೆಲೆ ಅಥವಾ ಕೃಷಿ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಬರೆಯಿರಿ.",
        "te": "దయచేసి పంటలు, వ్యాధులు, ధరలు లేదా వ్యవసాయం గురించి ప్రశ్న అడగండి.",
        "mr": "कृपया पीक, रोग, भाव किंवा शेतीबद्दल प्रश्न विचारा.",
        "bn": "অনুগ্রহ করে ফসল, রোগ, দাম বা চাষাবাদ সম্পর্কে প্রশ্ন করুন।",
    },
    "default": {
        "en": (
            "I can help with crop diseases, fertilizers, irrigation, pests, organic remedies, "
            "and market prices. Try asking about a specific crop (e.g. 'tomato blight' or 'corn rust') "
            "or upload a leaf photo on the Home page for AI diagnosis."
        ),
        "hi": (
            "मैं फसल रोग, खाद, सिंचाई, कीट, जैविक उपचार और बाजार भाव में मदद कर सकता हूँ। "
            "किसी फसल के बारे में पूछें (जैसे 'टमाटर झुलसा') या होम पेज पर पत्ती की फोटो अपलोड करें।"
        ),
        "es": (
            "Puedo ayudar con enfermedades, fertilizantes, riego, plagas y precios. "
            "Pregunte por un cultivo específico o suba una foto de hoja en Inicio."
        ),
        "kn": (
            "ನಾನು ಬೆಳೆ ರೋಗ, ರಸಗೊಬ್ಬರ, ನೀರಾವರಿ, ಕೀಟಗಳು, ಸಾವಯವ ಪರಿಹಾರಗಳು ಮತ್ತು ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳಲ್ಲಿ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ. "
            "ನಿರ್ದಿಷ್ಟ ಬೆಳೆಯ ಬಗ್ಗೆ ಕೇಳಿ ಅಥವಾ ಮುಖಪುಟದಲ್ಲಿ ಎಲೆಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
        ),
        "te": (
            "నేను పంట వ్యాధులు, ఎరువులు, నీటిపారుదల, పురుగులు, సేంద్రీయ పరిష్కారాలు మరియు మార్కెట్ ధరలలో సహాయం చేస్తాను. "
            "నిర్దిష్ట పంట గురించి అడగండి లేదా హోమ్ పేజీలో ఆకు ఫోటో అప్‌లోడ్ చేయండి."
        ),
        "mr": (
            "मी पीक रोग, खते, सिंचन, कीट, सेंद्रिय उपाय आणि बाजार भाव यात मदत करू शकतो. "
            "विशिष्ट पिकाबद्दल विचारा किंवा होम पेजवर पानाचा फोटो अपलोड करा."
        ),
        "bn": (
            "আমি ফসলের রোগ, সার, সেচ, পোকা, জৈব প্রতিকার এবং বাজার দরে সাহায্য করতে পারি। "
            "নির্দিষ্ট ফসল সম্পর্কে জিজ্ঞাসা করুন বা হোম পেজে পাতার ছবি আপলোড করুন।"
        ),
    },
}


def _localized(entry: dict, lang: str) -> str:
    return entry.get(lang) or entry["en"]


CHATBOT_KNOWLEDGE = [
    {
        "keywords": ["hello", "hi", "hey", "namaste", "good morning", "good evening", "ನಮಸ್ಕಾರ", "నమస్కారం", "नमस्कार", "নমস্কার"],
        "en": "Hello! I am FarmX Advisor. Ask me about crop diseases, market prices, fertilizers, irrigation, or upload a leaf photo on the Home page for AI diagnosis.",
        "hi": "नमस्ते! मैं FarmX सलाहकार हूँ। फसल रोग, बाजार भाव, खाद, सिंचाई या पत्ती की तस्वीर अपलोड करके निदान के बारे में पूछें।",
        "es": "¡Hola! Soy el asesor FarmX. Pregúnteme sobre enfermedades, precios, fertilizantes, riego o suba una foto de hoja en Inicio.",
        "kn": "ನಮಸ್ಕಾರ! ನಾನು FarmX ಸಲಹೆಗಾರ. ಬೆಳೆ ರೋಗ, ಮಾರುಕಟ್ಟೆ ಬೆಲೆ, ರಸಗೊಬ್ಬರ, ನೀರಾವರಿ ಬಗ್ಗೆ ಕೇಳಿ ಅಥವಾ ಮುಖಪುಟದಲ್ಲಿ ಎಲೆಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "te": "నమస్కారం! నేను FarmX సలహాదారు. పంట వ్యాధులు, మార్కెట్ ధరలు, ఎరువులు, నీటిపారుదల గురించి అడగండి లేదా హోమ్ పేజీలో ఆకు ఫోటో అప్‌లోడ్ చేయండి.",
        "mr": "नमस्कार! मी FarmX सल्लागार आहे. पीक रोग, बाजार भाव, खते, सिंचन बद्दल विचारा किंवा होम पेजवर पानाचा फोटो अपलोड करा.",
        "bn": "নমস্কার! আমি FarmX উপদেষ্টা। ফসলের রোগ, বাজার দর, সার, সেচ সম্পর্কে জিজ্ঞাসা করুন বা হোম পেজে পাতার ছবি আপলোড করুন।",
    },
    {
        "keywords": ["tomato", "tamatar", "tomate", "early blight", "late blight", "septoria", "leaf mold", "ಟೊಮಾಟೊ", "టమాటా", "टोमॅटो", "টমেটো"],
        "en": "Tomato diseases: Early/Late blight shows brown spots and yellowing. Remove infected leaves, avoid overhead watering, use copper fungicide or neem spray, and rotate crops yearly. Upload a leaf photo for exact diagnosis.",
        "hi": "टमाटर रोग: अगेती/पछेती झुलसा में भूरे धब्बे और पीलापन होता है। संक्रमित पत्ते हटाएँ, ऊपर से पानी न दें, तांबा fungicide या नीम का छिड़काव करें। सटीक निदान के लिए पत्ती की फोटो अपलोड करें।",
        "es": "Enfermedades del tomate: tizón temprano/tardío causa manchas marrones. Elimine hojas infectadas, evite riego por aspersión y use fungicida de cobre o neem.",
        "kn": "ಟೊಮಾಟೊ ರೋಗ: ಆರಂಭಿಕ/ಅಂತಿಮ ಬ್ಲೈಟ್ ಕಂದು ಕಲೆಗಳು ಮತ್ತು ಹಳದಿ ಬಣ್ಣ ತೋರಿಸುತ್ತದೆ. ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಮೇಲಿನಿಂದ ನೀರು ಹಾಕಬೇಡಿ, ತಾಮ್ರ ಶಿಲೀಂಧ್ರನಾಶಕ ಅಥವಾ ವೇಪು ಬಳಸಿ.",
        "te": "టమాటా వ్యాధులు: ముందుగా/తర్వాత బ్లైట్‌లో గోధుమ రంగు మచ్చలు కనిపిస్తాయి. సోకిన ఆకులు తీసివేయండి, పైనుండి నీరు పోయకండి, రాగి శిలీంధ్రనాశకి లేదా వేపు ఉపయోగించండి.",
        "mr": "टोमॅटो रोग: लवकर/उशिरा ब्लाइटमध्ये तपकिरी डाग आणि पिवळसरपणा. संक्रमित पाने काढा, वरून पाणी देऊ नका, तांबे फंगिसाइड किंवा कडुलिंब फवारणी करा.",
        "bn": "টমেটো রোগ: আগে/পরে ব্লাইটে বাদামি দাগ ও হলুদ হয়। সংক্রমিত পাতা সরান, উপর থেকে পানি দেবেন না, তামা ছত্রাকনাশক বা নিম স্প্রে করুন।",
    },
    {
        "keywords": ["potato", "aloo", "papa", "potato blight", "early blight potato", "late blight potato", "ಆಲೂಗಡ್ಡೆ", "బంగాళాదుంప", "बटाटा", "আলু"],
        "en": "Potato blight spreads fast in humid weather. Late blight causes dark lesions; early blight shows concentric rings. Use certified seed, ensure drainage, apply mancozeb/chlorothalonil preventively, and destroy infected plants.",
        "hi": "आलू का झुलसा नम मौसम में तेज फैलता है। प्रमाणित बीज, अच्छा जल निकास, mancozeb छिड़काव और संक्रमित पौधे नष्ट करें।",
        "es": "El tizón de papa se expande con humedad. Use semilla certificada, buen drenaje y fungicidas preventivos como mancozeb.",
        "kn": "ಆಲೂಗಡ್ಡೆ ಬ್ಲೈಟ್ ತೇವಾಂಶದಲ್ಲಿ ವೇಗವಾಗಿ ಹರಡುತ್ತದೆ. ಪ್ರಮಾಣೀಕೃತ ಬೀಜ, ಉತ್ತಮ ಜಲನಿಕಾಸ, mancozeb ಸಿಂಪಣೆ ಮತ್ತು ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ನಾಶ ಮಾಡಿ.",
        "te": "బంగాళాదుంప బ్లైట్ తేమ వాతావరణంలో వేగంగా వ్యాపిస్తుంది. ధృవీకరించిన విత్తనం, మంచి డ్రైనేజ్, mancozeb స్ప్రే చేయండి.",
        "mr": "बटाट्याचा ब्लाइट ओल्या हवामानात वेगाने पसरतो. प्रमाणित बियाणे, चांगला निचरा, mancozeb फवारणी आणि संक्रमित झाडे नष्ट करा.",
        "bn": "আলুর ব্লাইট আর্দ্র আবহাওয়ায় দ্রুত ছড়ায়। প্রমাণিত বীজ, ভাল নিষ্কাশন, mancozeb স্প্রে করুন এবং সংক্রমিত গাছ ধ্বংস করুন।",
    },
    {
        "keywords": ["corn", "maize", "makka", "rust", "northern leaf blight", "cercospora", "ಮೆಕ್ಕೆಜೋಳ", "మొక్కజొన్న", "मका", "ভুট্টা"],
        "en": "Corn rust appears as orange-brown pustules on leaves. Use resistant hybrids, rotate with soybeans, remove crop residue, and apply fungicide if rust exceeds 5% leaf area before tasseling.",
        "hi": "मक्के में रतुआ नारंगी-भूरे धब्बे दिखाता है। प्रतिरोधी किस्म, फसल चक्र और अवशेष हटाना जरूरी है। 5% से अधिक संक्रमण पर fungicide लगाएँ।",
        "es": "La roya del maíz muestra pústulas anaranjadas. Use híbridos resistentes, rotación de cultivos y fungicida si la infección supera el 5%.",
        "kn": "ಮೆಕ್ಕೆಜೋಳ ಕಸ್ತೂರಿ ಎಲೆಗಳ ಮೇಲೆ ಕಿತ್ತಳೆ-ಕಂದು ಪುಟಗಳನ್ನು ತೋರಿಸುತ್ತದೆ. ಪ್ರತಿರೋಧಕ ಕ್ರಾಸ್, ಬೆಳೆ ಪರಿವರ್ತನೆ ಮತ್ತು ಅವಶೇಷ ತೆಗೆದುಹಾಕಿ.",
        "te": "మొక్కజొన్న తుప్పు ఆకులపై నారింజ-గోధుమ రంగు గుట్టలు కనిపిస్తాయి. నిరోధక హైబ్రిడ్‌లు, పంట మార్పిడి ఉపయోగించండి.",
        "mr": "मक्याचा गंज येणे पानांवर नारिंगी-तपकिरी डाग दाखवते. प्रतिरोधक वाण, पीक फेरपालट आणि अवशेष काढा.",
        "bn": "ভুট্টার মরিচা পাতায় কমলা-বাদামি ফোস্কা দেখায়। প্রতিরোধী হাইব্রিড, ফসল পরিবর্তন ও অবশিষ্টাংশ সরান।",
    },
    {
        "keywords": ["apple", "grape", "peach", "cherry", "strawberry", "blueberry", "orange", "citrus", "ಸೇಬು", "ద్రాక్ష", "सफरचंद", "আপেল"],
        "en": "Fruit crop diseases vary by species. Apple scab needs pruning and sulfur sprays; grape black rot needs canopy airflow; citrus greening has no cure—remove infected trees. Upload a clear leaf photo for species-specific diagnosis.",
        "hi": "फल फसलों के रोग अलग-अलग होते हैं। सेब scab में pruning और sulfur; अंगूर black rot में हवा का प्रवाह; citrus greening में संक्रमित पेड़ हटाएँ। सटीक निदान के लिए पत्ती की फोटो अपलोड करें।",
        "es": "Las enfermedades varían por cultivo. Poda y azufre para mancha de manzana; ventilación para uva; elimine árboles cítricos con greening.",
        "kn": "ಹಣ್ಣು ಬೆಳೆಗಳ ರೋಗಗಳು ಬೇರೆ ಬೇರೆ. ಸೇಬು ಸ್ಕ್ಯಾಬ್‌ಗೆ ಕತ್ತರಿಸುವಿಕೆ ಮತ್ತು ಗಂಧಕ; ದ್ರಾಕ್ಷಿ ಬ್ಲ್ಯಾಕ್ ರಾಟ್‌ಗೆ ಗಾಳಿ ಪ್ರವಾಹ; ಸಿಟ್ರಸ್ ಗ್ರೀನಿಂಗ್‌ಗೆ ಸೋಂಕಿತ ಮರಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
        "te": "పండ్ల పంట వ్యాధులు వేర్వేరుగా ఉంటాయి. ఆపిల్ స్కాబ్‌కు కత్తిరింపు మరియు సల్ఫర్; ద్రాక్ష బ్లాక్ రాట్‌కు గాలి ప్రవాహం; సిట్రస్ గ్రీనింగ్‌కు సోకిన చెట్లు తీసివేయండి.",
        "mr": "फळ पिकांचे रोग वेगळे असतात. सफरचंद स्कॅबसाठी छाटणी आणि गंधक; द्राक्ष ब्लॅक रॉटसाठी हवा प्रवाह; सिट्रस ग्रीनिंगसाठी संक्रमित झाडे काढा.",
        "bn": "ফলের ফসলের রোগ আলাদা। আপেল স্ক্যাবের জন্য ছাঁটাই ও সালফার; আঙুর ব্ল্যাক রটের জন্য বায়ু প্রবাহ; সাইট্রাস গ্রিনিংয়ে সংক্রমিত গাছ সরান।",
    },
    {
        "keywords": ["tomato price", "potato price", "corn price", "maize price", "market price", "selling price", "mandi", "ಬೆಲೆ", "ధర", "भाव", "দাম"],
        "en": "Current June averages (per quintal): Tomato ~$4000, Potato ~$1550, Corn ~$2300. Prices shift with season and supply—open the Market tab for trend charts and profit calculator.",
        "hi": "जून के औसत भाव (प्रति क्विंटल): टमाटर ~$4000, आलू ~$1550, मक्का ~$2300। रुझान और लाभ के लिए Market टैब देखें।",
        "es": "Promedios de junio (por quintal): Tomate ~$4000, Papa ~$1550, Maíz ~$2300. Vea la pestaña Mercado para tendencias.",
        "kn": "ಜೂನ್ ಸರಾಸರಿ ಬೆಲೆ (ಪ್ರತಿ ಕ್ವಿಂಟಾಲ್): ಟೊಮಾಟೊ ~$4000, ಆಲೂಗಡ್ಡೆ ~$1550, ಮೆಕ್ಕೆಜೋಳ ~$2300. ಟ್ರೆಂಡ್‌ಗಳಿಗಾಗಿ Market ಟ್ಯಾಬ್ ತೆರೆಯಿರಿ.",
        "te": "జూన్ సగటు ధరలు (క్వింటాల్‌కు): టమాటా ~$4000, బంగాళాదుంప ~$1550, మొక్కజొన్న ~$2300. ట్రెండ్‌ల కోసం Market ట్యాబ్ తెరవండి.",
        "mr": "जून सरासरी भाव (प्रति क्विंटल): टोमॅटो ~$4000, बटाटा ~$1550, मका ~$2300. ट्रेंडसाठी Market टॅब पहा.",
        "bn": "জুনের গড় দর (প্রতি কুইন্টাল): টমেটো ~$4000, আলু ~$1550, ভুট্টা ~$2300। ট্রেন্ডের জন্য Market ট্যাব দেখুন।",
    },
    {
        "keywords": ["price", "market", "sell", "profit", "revenue", "cost", "bazaar", "bajari", "paisa", "बाजार", "भाव", "dam", "rate", "ಮಾರುಕಟ್ಟೆ", "మార్కెట్", "बाजार", "বাজার"],
        "en": "Market prices depend on crop, season, and local demand. Tomato peaks in summer; potato is steadier year-round. Use the Market Analytics page for charts and the revenue calculator for your harvest.",
        "hi": "बाजार भाव फसल, मौसम और मांग पर निर्भर हैं। गर्मियों में टमाटर ऊँचा; आलू अधिक स्थिर। Market Analytics पेज पर चार्ट और लाभ कैलकुलेटर देखें।",
        "es": "Los precios dependen del cultivo y la temporada. Use la página de Análisis de Mercado para gráficos y calculadora de ingresos.",
        "kn": "ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳು ಬೆಳೆ, ಋತು ಮತ್ತು ಬೇಡಿಕೆಯ ಮೇಲೆ ಅವಲಂಬಿತ. ಟೊಮಾಟೊ ಬೇಸಿಗೆಯಲ್ಲಿ ಹೆಚ್ಚು; ಆಲೂಗಡ್ಡೆ ಸ್ಥಿರ. Market Analytics ಪುಟ ಬಳಸಿ.",
        "te": "మార్కెట్ ధరలు పంట, సీజన్ మరియు డిమాండ్‌పై ఆధారపడి ఉంటాయి. టమాటా వేసవిలో ఎక్కువ; బంగాళాదుంప స్థిరం. Market Analytics పేజీ చూడండి.",
        "mr": "बाजार भाव पीक, हंगाम आणि मागणीवर अवलंबून. उन्हाळ्यात टोमॅटो जास्त; बटाटा स्थिर. Market Analytics पेज वापरा.",
        "bn": "বাজার দর ফসল, মৌসুম ও চাহিদার উপর নির্ভর করে। গ্রীষ্মে টমেটো বেশি; আলু স্থির। Market Analytics পেজ দেখুন।",
    },
    {
        "keywords": ["fertilizer", "fertiliser", "npk", "urea", "compost", "manure", "khad", "खाद", "abono", "ರಸಗೊಬ್ಬರ", "ఎరువు", "खत", "সার"],
        "en": "Balanced NPK supports healthy growth. Nitrogen (urea) for leafy growth; phosphorus for roots; potassium for fruit quality. Apply compost before planting and split nitrogen doses—avoid over-fertilizing which invites disease.",
        "hi": "संतुलित NPK स्वस्थ वृद्धि देता है। नाइट्रोजन (यूरिया) पत्तियों के लिए; फास्फोरस जड़ों के लिए; पोटाश फल की गुणवत्ता के लिए। अधिक खाद से रोग बढ़ते हैं।",
        "es": "NPK equilibrado favorece el crecimiento. Nitrógeno para hojas, fósforo para raíces, potasio para frutos. Evite exceso de fertilizante.",
        "kn": "ಸಮತೋಲಿತ NPK ಆರೋಗ್ಯಕರ ಬೆಳವಣಿಗೆಗೆ ಸಹಾಯ. ನೈಟ್ರೋಜನ್ ಎಲೆಗಳಿಗೆ; ಫಾಸ್ಫರಸ್ ಬೇರುಗಳಿಗೆ; ಪೊಟಾಷ್ ಹಣ್ಣಿನ ಗುಣಮಟ್ಟಕ್ಕೆ. ಅತಿಯಾದ ರಸಗೊಬ್ಬರ ರೋಗಕ್ಕೆ ಕಾರಣ.",
        "te": "సమతుల్య NPK ఆరోగ్యకర వృద్ధికి సహాయం. నత్రజని ఆకులకు; భాస్వరం మూలాలకు; పొటాష్ పండ్ల నాణ్యతకు. ఎక్కువ ఎరువు వ్యాధులకు దారి.",
        "mr": "संतुलित NPK निरोगी वाढीस मदत करते. नायट्रोजन पानांसाठी; फॉस्फरस मुळांसाठी; पोटॅश फळांच्या गुणवत्तेसाठी. जास्त खत रोग वाढवते.",
        "bn": "সুষম NPK সুস্থ বৃদ্ধিতে সাহায্য করে। নাইট্রোজেন পাতার জন্য; ফসফরাস শিকড়ের জন্য; পটাশium ফলের মানের জন্য। বেশি সার রোগ বাড়ায়।",
    },
    {
        "keywords": ["water", "irrigate", "irrigation", "drip", "rain", "sinchai", "सिंचाई", "pani", "पानी", "riego", "ನೀರಾವರಿ", "నీటిపారుదల", "सिंचन", "সেচ"],
        "en": "Most crops need consistent moisture, not flooding. Drip irrigation saves water and keeps leaves dry—reducing fungal disease. Water early morning; avoid evening watering on tomato and potato.",
        "hi": "अधिकांश फसलों को नियमित नमी चाहिए, जलभराव नहीं। ड्रिप सिंचाई पानी बचाती है और पत्ते सूखे रखती है। सुबह पानी दें; शाम को टमाटर/आलू में पानी न दें।",
        "es": "Riegue de forma constante, sin encharcar. El riego por goteo ahorra agua y mantiene las hojas secas, reduciendo hongos.",
        "kn": "ಹೆಚ್ಚಿನ ಬೆಳೆಗಳಿಗೆ ಸ್ಥಿರ ತೇವಾಂಶ ಬೇಕು, ಜಲಾವೃತಿ ಅಲ್ಲ. ಡ್ರಿಪ್ ನೀರಾವರಿ ನೀರು ಉಳಿಸುತ್ತದೆ ಮತ್ತು ಎಲೆಗಳನ್ನು ಒಣವಾಗಿ ಇಡುತ್ತದೆ. ಬೆಳಿಗ್ಗೆ ನೀರು ಹಾಕಿ.",
        "te": "చాలా పంటలకు స్థిర తేమ అవసరం, వరద కాదు. డ్రిప్ నీటిపారుదల నీరు ఆదా చేస్తుంది మరియు ఆకులను ఎండగా ఉంచుతుంది. ఉదయం నీరు పోయండి.",
        "mr": "बहुतेक पिकांना सातत्यपूर्ण ओलावा हवा, पाण्याचा भुर्दांडा नको. ड्रिप सिंचन पाणी वाचवते आणि पाने कोरडी ठेवते. सकाळी पाणी द्या.",
        "bn": "বেশিরভাগ ফসলের ধারাবাহিক আর্দ্রতা দরকার, জলাবদ্ধতা নয়। ড্রিপ সেচ পানি বাঁচায় এবং পাতা শুকনো রাখে। সকালে পানি দিন।",
    },
    {
        "keywords": ["pest", "insect", "mite", "aphid", "worm", "keeda", "कीट", "plaga", "spider mite", "ಕೀಟ", "పురుగు", "कीड", "পোকা"],
        "en": "For pests: inspect leaves weekly, use neem oil or insecticidal soap for mild infestations, introduce beneficial insects where possible, and remove heavily infested plants to stop spread.",
        "hi": "कीट नियंत्रण: साप्ताहिक जाँच, हल्के संक्रमण पर नीम तेल या insecticidal soap, लाभकारी कीट जहाँ संभव, और भारी संक्रमित पौधे हटाएँ।",
        "es": "Para plagas: inspeccione semanalmente, use aceite de neem o jabón insecticida, y elimine plantas muy infestadas.",
        "kn": "ಕೀಟಗಳಿಗೆ: ವಾರಕ್ಕೊಮ್ಮೆ ಎಲೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ, ನೀಮ್ ಎಣ್ಣೆ ಅಥವಾ ಕೀಟನಾಶಕ ಸಾಬೂನ್ ಬಳಸಿ, ಹೆಚ್ಚು ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
        "te": "పురుగులకు: వారానికోసారి ఆకులు పరిశీలించండి, వేపు నూనె లేదా కీటకనాశక సబ్బు ఉపయోగించండి, ఎక్కువ సోకిన మొక్కలు తీసివేయండి.",
        "mr": "कीड नियंत्रण: साप्ताहिक तपासणी, कडुलिंब तेल किंवा कीटकनाशक साबण, जास्त संक्रमित झाडे काढा.",
        "bn": "পোকার জন্য: সাপ্তাহিক পাতা পরীক্ষা করুন, নিম তেল বা কীটনাশক সাবান ব্যবহার করুন, বেশি সংক্রমিত গাছ সরান।",
    },
    {
        "keywords": ["organic", "natural", "neem", "copper", "chemical free", "jaivik", "जैविक", "ಸಾವಯವ", "సేంద్రీయ", "सेंद्रिय", "জৈব"],
        "en": "Organic options: neem oil for pests/fungi, copper fungicide for blight, baking soda spray for powdery mildew, crop rotation, and compost for soil health. Always test on a few leaves first.",
        "hi": "जैविक उपाय: नीम तेल, तांबा fungicide, baking soda छिड़काव, फसल चक्र और compost। पहले कुछ पत्तों पर परीक्षण करें।",
        "es": "Opciones orgánicas: aceite de neem, fungicida de cobre, bicarbonato para oídio, rotación y compost.",
        "kn": "ಸಾವಯವ ಆಯ್ಕೆಗಳು: ನೀಮ್ ಎಣ್ಣೆ, ತಾಮ್ರ ಶಿಲೀಂಧ್ರನಾಶಕ, ಬೇಕಿಂಗ್ ಸೋಡಾ ಸ್ಪ್ರೇ, ಬೆಳೆ ಪರಿವರ್ತನೆ ಮತ್ತು ಕಂಪೋಸ್ಟ್. ಮೊದಲು ಕೆಲವು ಎಲೆಗಳ ಮೇಲೆ ಪರೀಕ್ಷಿಸಿ.",
        "te": "సేంద్రీయ ఎంపికలు: వేపు నూనె, రాగి శిలీంధ్రనాశకి, బేకింగ్ సోడా స్ప్రే, పంట మార్పిడి మరియు కంపోస్ట్. ముందు కొన్ని ఆకులపై పరీక్షించండి.",
        "mr": "सेंद्रिय उपाय: कडुलिंब तेल, तांबे फंगिसाइड, बेकिंग सोडा फवारणी, पीक फेरपालट आणि कंपोस्ट. आधी काही पानांवर चाचणी करा.",
        "bn": "জৈব উপায়: নিম তেল, তামা ছত্রাকনাশক, বেকিং সোডা স্প্রে, ফসল পরিবর্তন ও কম্পোস্ট। আগে কয়েকটি পাতায় পরীক্ষা করুন।",
    },
    {
        "keywords": ["healthy", "prevent", "prevention", "protect", "care", "grow", "yield", "harvest", "urvarak", "ರೋಗ", "వ్యాధి", "रोग", "রোগ"],
        "en": "Prevent disease with: disease-resistant seeds, proper spacing, weed control, balanced fertilizer, crop rotation, and timely harvesting. Healthy plants resist infection better than stressed ones.",
        "hi": "रोग रोकथाम: प्रतिरोधी बीज, उचित दूरी, खरपतवार नियंत्रण, संतुलित खाद, फसल चक्र और समय पर कटाई। स्वस्थ पौधे संक्रमण का बेहतर विरोध करते हैं।",
        "es": "Prevenga enfermedades con semillas resistentes, espaciado adecuado, control de malezas, rotación y cosecha oportuna.",
        "kn": "ರೋಗ ತಡೆಗಟ್ಟಲು: ರೋಗ-ಪ್ರತಿರೋಧಕ ಬೀಜ, ಸರಿಯಾದ ಅಂತರ, ಕಳೆ ನಿಯಂತ್ರಣ, ಸಮತೋಲಿತ ರಸಗೊಬ್ಬರ, ಬೆಳೆ ಪರಿವರ್ತನೆ ಮತ್ತು ಸಮಯೋಚಿತ ಕೊಯ್ಲು.",
        "te": "వ్యాధి నివారణ: వ్యాధి-నిరోధక విత్తనాలు, సరైన దూరం, పేరుకు నియంత్రణ, సమతుల్య ఎరువు, పంట మార్పిడి మరియు సమయానుకూల పంట.",
        "mr": "रोग प्रतिबंध: प्रतिरोधक बियाणे, योग्य अंतर, तण नियंत्रण, संतुलित खत, पीक फेरपालट आणि वेळेवर कापणी.",
        "bn": "রোগ প্রতিরোধ: রোগ-প্রতিরোধী বীজ, সঠিক দূরত্ব, আগাছা নিয়ন্ত্রণ, সুষম সার, ফসল পরিবর্তন ও সময়মতো ফসল কাটা।",
    },
    {
        "keywords": ["upload", "photo", "image", "picture", "diagnose", "diagnosis", "detect", "scan", "analyze", "test leaf", "nidaan", "निदान", "ನಿದಾನ", "నిర్ధారణ", "निदान", "নির্ণয়"],
        "en": "To diagnose a disease: go to Crop Diagnosis on the Home page, upload a clear close-up of an affected leaf (good lighting, single leaf), and click Analyze Crop Health. The AI checks 38 crop-disease classes with top-3 predictions.",
        "hi": "रोग निदान: होम पेज पर Crop Diagnosis खोलें, प्रभावित पत्ती की स्पष्ट फोटो अपलोड करें और Analyze Crop Health दबाएँ। AI 38 फसल-रोग वर्गों की जाँच करता है।",
        "es": "Para diagnosticar: vaya a Diagnóstico de Cultivos, suba una foto clara de la hoja afectada y pulse Analizar. La IA clasifica 38 enfermedades.",
        "kn": "ರೋಗ ನಿದಾನ: ಹೋಮ್ ಪುಟದಲ್ಲಿ Crop Diagnosis ತೆರೆಯಿರಿ, ಸ್ಪಷ್ಟ ಎಲೆಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು Analyze Crop Health ಕ್ಲಿಕ್ ಮಾಡಿ. AI 38 ರೋಗ ವರ್ಗಗಳನ್ನು ಪರೀಕ್ಷಿಸುತ್ತದೆ.",
        "te": "వ్యాధి నిర్ధారణ: హోమ్ పేజీలో Crop Diagnosis తెరవండి, స్పష్టమైన ఆకు ఫోటో అప్‌లోడ్ చేసి Analyze Crop Health క్లిక్ చేయండి. AI 38 వ్యాధి తరగతులను తనిఖీ చేస్తుంది.",
        "mr": "रोग निदान: होम पेजवर Crop Diagnosis उघडा, पानाचा स्पष्ट फोटो अपलोड करा आणि Analyze Crop Health क्लिक करा. AI 38 रोग वर्ग तपासते.",
        "bn": "রোগ নির্ণয়: হোম পেজে Crop Diagnosis খুলুন, পাতার স্পষ্ট ছবি আপলোড করুন এবং Analyze Crop Health ক্লিক করুন। AI ৩৮ রোগ শ্রেণি পরীক্ষা করে।",
    },
    {
        "keywords": ["soil", "ph", "land", "field", "mitti", "मिट्टी", "suelo", "ಮಣ್ಣು", "నేల", "माती", "মাটি"],
        "en": "Most vegetables prefer soil pH 6.0–7.0. Test soil every 2–3 years, add lime if too acidic, add sulfur/organic matter if too alkaline, and avoid planting the same crop in the same spot each year.",
        "hi": "अधिकांश सब्जियों के लिए pH 6.0–7.0 उपयुक्त है। 2–3 साल में मिट्टी जाँचें, अम्लीय हो तो चूना, क्षारीय हो तो sulfur/organic matter डालें।",
        "es": "La mayoría de hortalizas prefieren pH 6.0–7.0. Analice el suelo cada 2–3 años y rote cultivos.",
        "kn": "ಹೆಚ್ಚಿನ ತರಕಾರಿಗಳಿಗೆ ಮಣ್ಣು pH 6.0–7.0 ಸೂಕ್ತ. 2–3 ವರ್ಷಗಳಿಗೊಮ್ಮೆ ಮಣ್ಣು ಪರೀಕ್ಷಿಸಿ, ಅಮ್ಲೀಯವಾದರೆ ಸುಣ್ಣ, ಕ್ಷಾರೀಯವಾದರೆ ಗಂಧಕ/ಸಾವಯವ ಪದಾರ್ಥ ಸೇರಿಸಿ.",
        "te": "చాలా కూరగాయలకు నేల pH 6.0–7.0 అనుకూలం. 2–3 సంవత్సరాలకోసారి నేల పరీక్షించండి, ఆమ్లత ఉంటే చున్నం, క్షారత ఉంటే సల్ఫర్/సేంద్రీయ పదార్థం జోడించండి.",
        "mr": "बहुतेक भाज्यांसाठी माती pH 6.0–7.0 योग्य. 2–3 वर्षांनी माती तपासा, आम्लीय असल्यास चुना, क्षारीय असल्यास गंधक/सेंद्रिय पदार्थ घाला.",
        "bn": "বেশিরভাগ সবজির জন্য মাটির pH 6.0–7.0 উপযুক্ত। ২–৩ বছরে মাটি পরীক্ষা করুন, অম্লীয় হলে চুন, ক্ষারীয় হলে সালফার/জৈব পদার্থ দিন।",
    },
    {
        "keywords": ["weather", "climate", "season", "monsoon", "winter", "summer", "mosam", "मौसम", "ಹವಾಮಾನ", "వాతావరణం", "हवामान", "আবহাওয়া"],
        "en": "Season matters: plant warm-season crops (tomato, corn) after frost; cool-season crops (potato) in milder months. High humidity increases fungal risk—improve airflow and avoid wet foliage.",
        "hi": "मौसम महत्वपूर्ण: गर्म मौसम की फसलें (टमाटर, मक्का) frost के बाद; ठंडी फसलें (आलू) हल्के महीनों में। अधिक नमी में fungal रोग बढ़ते हैं।",
        "es": "La temporada importa: cultivos de calor después de heladas; cultivos frescos en meses suaves. La humedad alta aumenta hongos.",
        "kn": "ಋತು ಮುಖ್ಯ: ಬೆಚ್ಚಗಿನ ಬೆಳೆಗಳನ್ನು (ಟೊಮಾಟೊ, ಮೆಕ್ಕೆಜೋಳ) ಫ್ರಾಸ್ಟ್ ನಂತರ; ತಂಪಾದ ಬೆಳೆಗಳನ್ನು (ಆಲೂಗಡ್ಡೆ) ಮೃದು ತಿಂಗಳುಗಳಲ್ಲಿ ನೆಡಿ. ಹೆಚ್ಚಿನ ತೇವಾಂಶ ಶಿಲೀಂಧ್ರ ಅಪಾಯ ಹೆಚ್ಚಿಸುತ್ತದೆ.",
        "te": "సీజన్ ముఖ్యం: వేడి పంటలు (టమాటా, మొక్కజొన్న) ఫ్రాస్ట్ తర్వాత; చల్లని పంటలు (బంగాళాదుంప) మృదువైన నెలల్లో నాటండి. ఎక్కువ తేమ శిలీంధ్ర ప్రమాదం పెంచుతుంది.",
        "mr": "हंगाम महत्त्वाचा: उष्ण हंगामाची पिके (टोमॅटो, मका) गारठा नंतर; थंड हंगामाची पिके (बटाटा) हलक्या महिन्यांत लावा. जास्त आर्द्रता बुरशीचा धोका वाढवते.",
        "bn": "মৌসুম গুরুত্বপূর্ণ: গরম মৌসুমের ফসল (টমেটো, ভুট্টা) তুষারপাতের পর; ঠান্ডা ফসল (আলু) নরম মাসে লাগান। বেশি আর্দ্রতা ছত্রাকের ঝুঁকি বাড়ায়।",
    },
    {
        "keywords": ["thank", "thanks", "dhanyavad", "धन्यवाद", "gracias", "shukriya", "ಧನ್ಯವಾದ", "ధన్యవాదాలు", "धन्यवाद", "ধন্যবাদ"],
        "en": "You're welcome! Feel free to ask more farming questions anytime.",
        "hi": "आपका स्वागत है! कभी भी और farming सवाल पूछें।",
        "es": "¡De nada! Pregúnteme cuando quiera sobre agricultura.",
        "kn": "ಸ್ವಾಗತ! ಯಾವುದೇ ಸಮಯದಲ್ಲಿ ಹೆಚ್ಚು ಕೃಷಿ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ.",
        "te": "స్వాగతం! ఎప్పుడైనా మరిన్ని వ్యవసాయ ప్రశ్నలు అడగండి.",
        "mr": "स्वागत आहे! कधीही अधिक शेती प्रश्न विचारा.",
        "bn": "স্বাগতম! যেকোনো সময় আরও চাষাবাদের প্রশ্ন করুন।",
    },
]


def _keyword_matches(msg: str, keyword: str) -> bool:
    if len(keyword) <= 3:
        return re.search(rf"\b{re.escape(keyword)}\b", msg) is not None
    return keyword in msg


def get_chatbot_reply(user_msg: str, lang: str = "en") -> str:
    """Pick the best matching farming answer for the user's question."""
    if lang not in SUPPORTED_LANGS:
        lang = "en"

    msg = user_msg.lower().strip()
    if not msg:
        return CHATBOT_FALLBACKS["empty"].get(lang, CHATBOT_FALLBACKS["empty"]["en"])

    best_score = 0
    best_reply = None
    for entry in CHATBOT_KNOWLEDGE:
        score = sum(len(kw) for kw in entry["keywords"] if _keyword_matches(msg, kw))
        if score > best_score:
            best_score = score
            best_reply = _localized(entry, lang)

    if best_reply:
        return best_reply

    return CHATBOT_FALLBACKS["default"].get(lang, CHATBOT_FALLBACKS["default"]["en"])


# -------------------------------------------------------------
# 2. AUTH PAGES
# -------------------------------------------------------------
AUTH_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FarmX - Farmer Login</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #1e5631 0%, #39D2B4 100%);
            display: flex; align-items: center; justify-content: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .auth-card {
            width: 100%; max-width: 440px;
            background: #fff; border-radius: 16px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .auth-header {
            background: #34495E; color: #fff;
            padding: 28px 24px; text-align: center;
        }
        .auth-header h2 { margin: 0; font-weight: 700; }
        .auth-tabs { display: flex; border-bottom: 1px solid #eee; }
        .auth-tabs a {
            flex: 1; text-align: center; padding: 14px;
            text-decoration: none; color: #666; font-weight: 600;
        }
        .auth-tabs a.active { color: #1e5631; border-bottom: 3px solid #39D2B4; }
        .auth-body { padding: 28px 24px 32px; }
        .btn-farmx {
            background: #39D2B4; border: none; color: #fff;
            font-weight: 600; padding: 12px;
        }
        .btn-farmx:hover { background: #2bb89d; color: #fff; }
        .alert { font-size: 0.92rem; }
    </style>
</head>
<body>
<div class="auth-card">
    <div class="auth-header">
        <h2>🌾 FarmX Portal</h2>
        <p class="mb-0 mt-1 opacity-75">Smart farming for every farmer</p>
    </div>
    <div class="auth-tabs">
        <a href="{{ url_for('login') }}" class="{{ 'active' if mode == 'login' else '' }}">Log In</a>
        <a href="{{ url_for('signup') }}" class="{{ 'active' if mode == 'signup' else '' }}">Sign Up</a>
    </div>
    <div class="auth-body">
        {% if message %}
        <div class="alert alert-{{ message_type }}">{{ message }}</div>
        {% endif %}

        {% if mode == 'login' %}
        <form method="POST" action="{{ url_for('login') }}">
            <input type="hidden" name="next" value="{{ next_url }}">
            <div class="mb-3">
                <label class="form-label fw-semibold">Email ID</label>
                <input type="email" name="email" class="form-control form-control-lg" placeholder="farmer@example.com" required autofocus>
            </div>
            <div class="mb-4">
                <label class="form-label fw-semibold">Password</label>
                <input type="password" name="password" class="form-control form-control-lg" placeholder="Enter password" required>
            </div>
            <button type="submit" class="btn btn-farmx w-100 btn-lg">Log In</button>
        </form>
        <p class="text-center text-muted mt-3 mb-0 small">
            Not registered? <a href="{{ url_for('signup') }}">Create an account</a>
        </p>
        {% else %}
        <form method="POST" action="{{ url_for('signup') }}">
            <div class="mb-3">
                <label class="form-label fw-semibold">Full Name</label>
                <input type="text" name="name" class="form-control form-control-lg" placeholder="Your name" required autofocus>
            </div>
            <div class="mb-3">
                <label class="form-label fw-semibold">Email ID</label>
                <input type="email" name="email" class="form-control form-control-lg" placeholder="farmer@example.com" required>
            </div>
            <div class="mb-4">
                <label class="form-label fw-semibold">Password</label>
                <input type="password" name="password" class="form-control form-control-lg" placeholder="At least 6 characters" minlength="6" required>
            </div>
            <button type="submit" class="btn btn-farmx w-100 btn-lg">Sign Up</button>
        </form>
        <p class="text-center text-muted mt-3 mb-0 small">
            Already have an account? <a href="{{ url_for('login') }}">Log in</a>
        </p>
        {% endif %}
    </div>
</div>
</body>
</html>
"""


# -------------------------------------------------------------
# 3. FLASK CONTEXT REST API ENDPOINTS & ROUTES
# -------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FarmX - Smart Agriculture Platform</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: #f4f7f6; font-family: 'Segoe UI', sans-serif; padding-bottom: 60px; }
        .nav-farmx { background: #34495E; }
        .img-preview {
            width: 256px; height: 256px; position: relative;
            border: 5px solid #F8F8F8; box-shadow: 0px 2px 4px rgba(0,0,0,0.1); margin: 1em auto;
        }
        .img-preview>div { width: 100%; height: 100%; background-size: 256px 256px; background-repeat: no-repeat; background-position: center; }
        input[type="file"] { display: none; }
        .upload-label, #result {
            display: inline-block; padding: 12px 30px; background: #39D2B4; color: #fff;
            font-size: 1em; transition: all .4s; cursor: pointer; border-radius: 4px;
        }
        .upload-label:hover, #result:hover { background: #34495E; color: #39D2B4; }
        .loader {
            border: 8px solid #f3f3f3; border-top: 8px solid #3498db;
            border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        #chat-window { position: fixed; bottom: 20px; right: 20px; width: 380px; height: 480px; background: white; border-radius: 12px; box-shadow: 0 5px 25px rgba(0,0,0,0.25); display: flex; flex-direction: column; overflow: hidden; z-index: 1000; }
        #chat-box { flex-grow: 1; overflow-y: auto; height: 300px; background: #fafafa; padding: 15px; }
        .chat-mode-btn { font-size: 0.75rem; padding: 2px 10px; border: 1px solid rgba(255,255,255,0.4); background: transparent; color: #fff; border-radius: 12px; cursor: pointer; }
        .chat-mode-btn.active { background: #fff; color: #198754; font-weight: 600; }
        #btn-voice-mic { width: 42px; border-radius: 50%; }
        #btn-voice-mic.listening { background: #dc3545; border-color: #dc3545; color: #fff; animation: pulse-mic 1s infinite; }
        @keyframes pulse-mic { 0%,100% { box-shadow: 0 0 0 0 rgba(220,53,69,0.5); } 50% { box-shadow: 0 0 0 8px rgba(220,53,69,0); } }
        .chat-bubble { margin-bottom: 10px; padding: 8px 10px; border-radius: 8px; font-size: 0.9rem; }
        .chat-bubble.user { background: #e8f5e9; }
        .chat-bubble.bot { background: #fff; border: 1px solid #e0e0e0; }
        .chat-speak-btn { border: none; background: none; color: #198754; font-size: 0.85rem; cursor: pointer; padding: 0 4px; }
        #prediction-result .top-pred { font-size: 0.9rem; color: #555; }
        #prediction-result .low-conf { color: #dc3545; font-weight: 600; margin-top: 8px; }
    </style>
</head>
<body>

<nav class="navbar navbar-expand-lg navbar-dark nav-farmx p-3 shadow-sm">
    <div class="container">
        <a class="navbar-brand fw-bold text-success" href="/">🌾 FarmX Portal</a>
        <div class="navbar-nav ms-auto align-items-center">
            <a class="nav-link {% if section=='diagnosis' %}active fw-bold text-white{% endif %}" href="/">Crop Diagnosis</a>
            <a class="nav-link {% if section=='market' %}active fw-bold text-white{% endif %}" href="/market">Market Analytics</a>
            <a class="nav-link {% if section=='community' %}active fw-bold text-white{% endif %}" href="/community">Expert Forum</a>
            {% if farmer %}
            <span class="nav-link text-success small">👨‍🌾 {{ farmer.name }}</span>
            <a class="nav-link text-warning" href="/logout">Logout</a>
            {% endif %}
        </div>
    </div>
</nav>

<div class="container my-5">
    {% if section == 'diagnosis' %}
    <div class="row">
        <div class="col-md-6 mx-auto text-center">
            <div class="card p-5 border-0 shadow-sm bg-white rounded-4">
                <h3 class="fw-bold text-secondary mb-4">AI Crop Disease Diagnosis</h3>
                <p class="text-muted small mb-3">High-accuracy model with top-3 predictions and confidence scoring</p>
                <form id="upload-file" method="post" enctype="multipart/form-data">
                    <label for="imageUpload" class="upload-label">Choose Crop Leaf Image</label>
                    <input type="file" name="file" id="imageUpload">
                </form>
                <div class="image-section d-none text-center my-3">
                    <div class="img-preview"><div id="imagePreview"></div></div>
                    <button type="button" class="btn btn-dark px-4 py-2 mt-2" id="btn-predict">Analyze Crop Health</button>
                </div>
                <div class="loader mx-auto my-3 d-none"></div>
                <div id="result" class="mt-3 text-center" style="display:none;"></div>
            </div>
        </div>
    </div>

    {% elif section == 'market' %}
    <div class="row g-4">
        <div class="col-lg-7">
            <div class="card p-4 shadow-sm border-0 bg-white rounded-3">
                <h4 class="fw-bold mb-3 text-secondary">Market Price Trends (Per Quintal)</h4>
                <canvas id="marketChart" width="400" height="230"></canvas>
            </div>
        </div>
        <div class="col-lg-5">
            <div class="card p-4 shadow-sm border-0 bg-white rounded-3">
                <h4 class="fw-bold mb-3 text-success">Revenue Yield Estimator</h4>
                <div class="mb-3">
                    <label class="form-label">Select Target Crop</label>
                    <select id="calc-crop" class="form-select">
                        <option value="Tomato" data-price="4000">Tomato ($4000/Qtl)</option>
                        <option value="Potato" data-price="1550">Potato ($1550/Qtl)</option>
                        <option value="Corn" data-price="2300">Corn ($2300/Qtl)</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label class="form-label">Total Harvest Quantity (Quintals)</label>
                    <input type="number" id="calc-qty" class="form-control" value="15">
                </div>
                <div class="mb-3">
                    <label class="form-label">Logistics & Production Overheads ($)</label>
                    <input type="number" id="calc-cost" class="form-control" value="8000">
                </div>
                <button onclick="calculateRevenue()" class="btn btn-success w-100 fw-bold py-2">Evaluate Financial Projections</button>
                <hr>
                <div class="p-3 bg-light rounded-3">
                    <div class="mb-2">Gross Projected Income: <strong id="gross-res" class="float-end text-dark">$0</strong></div>
                    <div class="fw-bold fs-5 text-dark">Net Profit Yield: <strong id="net-res" class="float-end text-success">$0</strong></div>
                </div>
            </div>
        </div>
    </div>

    {% elif section == 'community' %}
    <div class="row g-4">
        <div class="col-md-4">
            <div class="card p-4 border-0 shadow-sm bg-white rounded-3">
                <h5 class="fw-bold text-success mb-3">Consult Farming Experts</h5>
                <form method="POST" action="/community">
                    <input type="text" name="username" class="form-control mb-2" placeholder="Your Name" required>
                    <input type="text" name="crop" class="form-control mb-2" placeholder="Affected Crop Specie" required>
                    <textarea name="text" class="form-control mb-3" rows="4" placeholder="Detail your problem..." required></textarea>
                    <button type="submit" class="btn btn-success w-100 fw-bold">Publish Ticket</button>
                </form>
            </div>
        </div>
        <div class="col-md-8">
            <h4 class="fw-bold mb-4 text-secondary">Expert Answer Hub Feed</h4>
            {% for post in posts %}
            <div class="card p-4 mb-3 shadow-sm border-0 bg-white rounded-3">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <strong class="text-primary fs-5">👨‍🌾 {{ post.user }}</strong>
                    <span class="badge bg-success-subtle text-success px-3 py-2 rounded-pill">{{ post.crop }}</span>
                </div>
                <p class="text-muted fs-6">{{ post.text }}</p>
                {% if post.replies %}
                <div class="mt-3 p-3 bg-light rounded border-start border-success border-3">
                    {% for reply in post.replies %}
                    <div class="py-1"><strong>🎓 Verified Expert Response:</strong> {{ reply }}</div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
</div>

<div id="chat-window">
    <div class="bg-success text-white p-3 fw-bold">
        <div class="d-flex justify-content-between align-items-center mb-2">
            <span>🤖 FarmX Advisor</span>
            <select id="chat-lang" class="bg-dark text-white border-0 rounded px-1 py-0" style="max-width:130px;font-size:0.8rem;">
                <option value="en">English</option>
                <option value="hi">Hindi (हिंदी)</option>
                <option value="es">Español</option>
                <option value="kn">Kannada (ಕನ್ನಡ)</option>
                <option value="te">Telugu (తెలుగు)</option>
                <option value="mr">Marathi (मराठी)</option>
                <option value="bn">Bengali (বাংলা)</option>
            </select>
        </div>
        <div class="d-flex gap-2 align-items-center">
            <button type="button" id="mode-text" class="chat-mode-btn active" title="Text chat">💬 Text</button>
            <button type="button" id="mode-voice" class="chat-mode-btn" title="Voice chat">🎙️ Voice</button>
            <small id="voice-status" class="ms-auto opacity-75" style="font-size:0.7rem;"></small>
        </div>
    </div>
    <div id="chat-box"></div>
    <div class="p-2 border-top d-flex bg-white align-items-center gap-1">
        <button type="button" id="btn-voice-mic" class="btn btn-outline-success d-none" title="Hold to speak">🎤</button>
        <input type="text" id="chat-msg" class="form-control" placeholder="Ask about markets, care...">
        <button id="send-chat" class="btn btn-success">Send</button>
    </div>
</div>

<script>
$(document).ready(function () {
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('.image-section').removeClass('d-none').hide().fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () { readURL(this); });

    function escapeHtml(text) {
        return $('<div>').text(text).html();
    }

    function formatPrediction(data) {
        var html = '<div><strong>Predicted Crop:</strong> ' + escapeHtml(data.crop) + '</div>';
        html += '<div><strong>Condition:</strong> ' + escapeHtml(data.condition) + '</div>';
        html += '<div><strong>Confidence:</strong> ' + data.confidence_pct + '%</div>';
        html += '<div class="text-muted small mt-1">Model: ' + escapeHtml(data.model_used) + '</div>';
        if (data.top_predictions && data.top_predictions.length > 1) {
            html += '<div class="mt-2"><strong>Top predictions:</strong><ol class="top-pred mb-0 ps-3">';
            data.top_predictions.forEach(function(p) {
                html += '<li>' + escapeHtml(p.crop) + ' — ' + escapeHtml(p.condition) +
                    ' (' + p.confidence_pct + '%)</li>';
            });
            html += '</ol></div>';
        }
        if (data.low_confidence) {
            html += '<div class="low-conf">⚠ Low confidence — try a clearer leaf photo with good lighting.</div>';
        }
        if (data.treatment) {
            html += '<div class="mt-3 p-3 bg-light rounded text-start"><strong>Recommended care:</strong><br>' +
                escapeHtml(data.treatment) + '</div>';
        }
        return html;
    }

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        $(this).hide(); $('.loader').removeClass('d-none').show();
        $('#result').hide();
        $.ajax({
            type: 'POST', url: '/predict', data: form_data, contentType: false, cache: false, processData: false,
            dataType: 'json',
            success: function (data) {
                $('.loader').hide(); $('#btn-predict').show();
                $('#result').html(formatPrediction(data)).fadeIn(600);
            },
            error: function(xhr) {
                $('.loader').hide(); $('#btn-predict').show();
                var msg = (xhr.responseJSON && xhr.responseJSON.error) ? xhr.responseJSON.error : 'Prediction failed.';
                $('#result').text(msg).fadeIn(600);
            }
        });
    });

    var chatMode = 'text';
    var speechSupported = ('webkitSpeechRecognition' in window) || ('SpeechRecognition' in window);
    var recognition = null;
    var isListening = false;
    var autoSpeak = false;

    var speechLangMap = {
        en: 'en-US', hi: 'hi-IN', es: 'es-ES',
        kn: 'kn-IN', te: 'te-IN', mr: 'mr-IN', bn: 'bn-IN'
    };

    function getSpeechLang() {
        return speechLangMap[$('#chat-lang').val()] || 'en-US';
    }

    function setChatMode(mode) {
        chatMode = mode;
        $('#mode-text').toggleClass('active', mode === 'text');
        $('#mode-voice').toggleClass('active', mode === 'voice');
        autoSpeak = (mode === 'voice');
        if (mode === 'voice' && speechSupported) {
            $('#btn-voice-mic').removeClass('d-none');
            $('#voice-status').text('Tap mic to speak');
        } else {
            $('#btn-voice-mic').addClass('d-none').removeClass('listening');
            $('#voice-status').text(mode === 'voice' && !speechSupported ? 'Voice not supported in this browser' : '');
            stopListening();
        }
    }

    function speakText(text) {
        if (!window.speechSynthesis || !text) return;
        window.speechSynthesis.cancel();
        var utter = new SpeechSynthesisUtterance(text);
        utter.lang = getSpeechLang();
        utter.rate = 0.95;
        window.speechSynthesis.speak(utter);
    }

    function appendUserBubble(text) {
        $('#chat-box').append('<div class="chat-bubble user"><strong>You:</strong> ' + escapeHtml(text) + '</div>');
        $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
    }

    function appendBotBubble(text) {
        var id = 'bot-' + Date.now();
        var html = '<div class="chat-bubble bot text-success" id="' + id + '">' +
            '<strong>Advisor:</strong> ' + escapeHtml(text) +
            ' <button type="button" class="chat-speak-btn" title="Read aloud">🔊</button></div>';
        $('#chat-box').append(html);
        $('#' + id + ' .chat-speak-btn').click(function() { speakText(text); });
        $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
        if (autoSpeak) speakText(text);
    }

    function sendChatMessage(msgText) {
        var selectedLang = $('#chat-lang').val();
        if (!msgText.trim()) return;
        appendUserBubble(msgText);
        $('#chat-msg').val('');
        $.ajax({
            type: 'POST', url: '/api/chatbot', contentType: 'application/json',
            data: JSON.stringify({ message: msgText, lang: selectedLang }),
            success: function(res) {
                appendBotBubble(res.reply);
            }
        });
    }

    function stopListening() {
        if (recognition && isListening) {
            try { recognition.stop(); } catch(e) {}
        }
        isListening = false;
        $('#btn-voice-mic').removeClass('listening');
    }

    function startListening() {
        if (!speechSupported || isListening) return;
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = getSpeechLang();
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.onstart = function() {
            isListening = true;
            $('#btn-voice-mic').addClass('listening');
            $('#voice-status').text('Listening...');
        };
        recognition.onresult = function(event) {
            var transcript = event.results[0][0].transcript;
            $('#chat-msg').val(transcript);
            sendChatMessage(transcript);
        };
        recognition.onerror = function() {
            $('#voice-status').text('Could not hear — try again');
        };
        recognition.onend = function() {
            isListening = false;
            $('#btn-voice-mic').removeClass('listening');
            if (chatMode === 'voice') $('#voice-status').text('Tap mic to speak');
        };
        recognition.start();
    }

    $('#mode-text').click(function() { setChatMode('text'); });
    $('#mode-voice').click(function() { setChatMode('voice'); });

    $('#send-chat').click(function() {
        sendChatMessage($('#chat-msg').val());
    });

    $('#chat-msg').keypress(function(e) {
        if (e.which === 13) { sendChatMessage($('#chat-msg').val()); return false; }
    });

    $('#btn-voice-mic').click(function() {
        if (isListening) stopListening();
        else startListening();
    });

    $('#chat-lang').change(function() {
        if (isListening) { stopListening(); startListening(); }
    });
});

if (document.getElementById('marketChart')) {
    fetch('/api/market-data').then(res => res.json()).then(data => {
        const ctx = document.getElementById('marketChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.months,
                datasets: [
                    { label: 'Tomato ($)', data: data.Tomato, borderColor: '#e74c3c', tension: 0.2, fill: false },
                    { label: 'Potato ($)', data: data.Potato, borderColor: '#f1c40f', tension: 0.2, fill: false },
                    { label: 'Corn ($)', data: data.Corn, borderColor: '#2ecc71', tension: 0.2, fill: false }
                ]
            }
        });
    });
}

function calculateRevenue() {
    const selectEl = document.getElementById('calc-crop');
    const price = parseFloat(selectEl.options[selectEl.selectedIndex].getAttribute('data-price'));
    const qty = parseFloat(document.getElementById('calc-qty').value) || 0;
    const overhead = parseFloat(document.getElementById('calc-cost').value) || 0;
    const gross = price * qty; const net = gross - overhead;
    document.getElementById('gross-res').innerText = '$' + gross.toLocaleString();
    document.getElementById('net-res').innerText = '$' + net.toLocaleString();
    document.getElementById('net-res').className = net < 0 ? "float-end text-danger" : "float-end text-success";
}
</script>
</body>
</html>
"""


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("farmer_email"):
        return redirect(url_for("index"))

    message = ""
    message_type = "danger"
    next_url = request.args.get("next") or request.form.get("next") or url_for("index")

    if request.method == "POST":
        ok, farmer, err = authenticate_farmer(
            request.form.get("email", ""), request.form.get("password", "")
        )
        if ok and farmer:
            session["farmer_email"] = farmer["email"]
            session["farmer_name"] = farmer["name"]
            return redirect(next_url)
        message = err

    return render_template_string(
        AUTH_TEMPLATE, mode="login", message=message, message_type=message_type, next_url=next_url
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("farmer_email"):
        return redirect(url_for("index"))

    message = ""
    message_type = "success"

    if request.method == "POST":
        ok, msg = register_farmer(
            request.form.get("name", ""),
            request.form.get("email", ""),
            request.form.get("password", ""),
        )
        message = msg
        message_type = "success" if ok else "danger"
        if ok:
            return render_template_string(
                AUTH_TEMPLATE,
                mode="login",
                message=msg,
                message_type="success",
                next_url=url_for("index"),
            )

    return render_template_string(
        AUTH_TEMPLATE, mode="signup", message=message, message_type=message_type, next_url=url_for("index")
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template_string(
        HTML_TEMPLATE, section="diagnosis", posts=COMMUNITY_POSTS, farmer=_current_farmer()
    )


@app.route("/market")
@login_required
def market():
    return render_template_string(
        HTML_TEMPLATE, section="market", posts=COMMUNITY_POSTS, farmer=_current_farmer()
    )


@app.route("/community", methods=["GET", "POST"])
@login_required
def community():
    if request.method == "POST":
        user = request.form.get("username", "Anonymous Farmer")
        crop = request.form.get("crop", "General")
        text = request.form.get("text", "")
        if text.strip():
            COMMUNITY_POSTS.insert(0, {"user": user, "crop": crop, "text": text, "replies": []})
    return render_template_string(
        HTML_TEMPLATE, section="community", posts=COMMUNITY_POSTS, farmer=_current_farmer()
    )


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _format_prediction_response(result: dict):
    pretty_condition = result["condition"].title().replace("_", " ")
    confidence_pct = round(result["confidence"] * 100, 1)
    top_predictions = []
    for item in result.get("top_predictions", []):
        top_predictions.append(
            {
                "crop": item["crop"],
                "condition": item["condition"].title().replace("_", " "),
                "confidence_pct": round(item["confidence"] * 100, 1),
            }
        )
    return {
        "crop": result["crop"],
        "condition": pretty_condition,
        "confidence_pct": confidence_pct,
        "model_used": result.get("model_used", "unknown"),
        "low_confidence": result.get("low_confidence", False),
        "top_predictions": top_predictions,
        "treatment": result.get("treatment", ""),
    }


@app.route("/predict", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file detected."}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "Invalid file upload."}), 400

    filename = secure_filename(f.filename) or f"leaf_{uuid.uuid4().hex}.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Use an image file."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(file_path)

    try:
        result = model_predict(file_path)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        return jsonify({"error": "Prediction failed. Check the server console for details."}), 500

    return jsonify(_format_prediction_response(result))


@app.route("/api/market-data")
@login_required
def get_market_data():
    return jsonify(
        {
            "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "Tomato": [2200, 2500, 2100, 3100, 3400, 4000],
            "Potato": [1200, 1300, 1250, 1400, 1600, 1550],
            "Corn": [1800, 1900, 2000, 2150, 2100, 2300],
        }
    )


@app.route("/api/chatbot", methods=["POST"])
@login_required
def chatbot():
    data = request.json or {}
    user_msg = str(data.get("message", ""))
    lang = data.get("lang", "en")
    reply = get_chatbot_reply(user_msg, lang)
    return jsonify({
        "reply": reply,
        "lang": lang if lang in SUPPORTED_LANGS else "en",
        "speech_lang": SPEECH_LANG_CODES.get(lang, "en-US"),
    })


@app.route("/api/speech-langs")
def speech_langs():
    return jsonify(SPEECH_LANG_CODES)


def _open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/login")


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.25, _open_browser).start()
    app.run(debug=True, host="127.0.0.1", port=5000)

