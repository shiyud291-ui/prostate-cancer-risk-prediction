# -*- coding: utf-8 -*-
"""Individual SHAP explanations for the full stacking prediction function."""

from pathlib import Path
from typing import Dict, List
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Use Times New Roman consistently in all Matplotlib/SHAP figures.
# On systems where Times New Roman is unavailable, Matplotlib falls back to
# a metrically similar serif font rather than switching to a sans-serif face.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

from predictor import RAW_FEATURES, impute_raw_for_explanation, predict_raw_dataframe

BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_PATH = BASE_DIR / "data" / "shap_background.csv"

# English labels shown in all SHAP graphics. Internal model feature names are kept
# only as dictionary keys because the fitted model was trained with those columns.
DISPLAY_NAMES: Dict[str, str] = {
    "年龄": "Age (years)",
    "总前列腺特异性抗原": "tPSA (ng/mL)",
    "游离前列腺特异性抗原": "fPSA (ng/mL)",
    "血浆D-二聚体测定": "Plasma D-Dimer (μg/L)",
    "氯": "Cl (mmol/L)",
    "活化部分凝血活酶时间": "APTT (s)",
    "碱性磷酸酶": "ALP (U/L)",
    "单核细胞比率": "MONO% (%)",
    "血红蛋白测定": "HGB (g/L)",
    "白蛋白": "ALB (g/L)",
    "总胆红素": "TBIL (μmol/L)",
    "血浆纤维蛋白原测定": "FIB (g/L)",
    "血小板计数": "PLT (10⁹/L)",
    "葡萄糖测定": "GLU (mmol/L)",
    "尿素": "Urea (mmol/L)",
    "肌酐": "Cr (μmol/L)",
    "尿酸": "UA (μmol/L)",
    "乳酸脱氢酶": "LDH (U/L)",
    "总胆固醇": "TC (mmol/L)",
    "内生肌酐清除率": "CCR (mL/min)",
    "有核红细胞比率": "NRBC% (%)",
    "超敏C反应蛋白": "hsCRP (mg/L)",
    "二氧化碳结合力": "CO2CP (mmol/L)",
    "同型半胱氨酸": "HCY (μmol/L)",
    "血清5`核苷酸酶测定": "5'-Nucleotidase (U/L)",
    "腺苷脱氨酶": "ADA (U/L)",
    "高荧光强度网织红细胞比": "HFR% (%)",
    "γ-谷氨酰基转移酶": "GGT (U/L)",
    "中性细胞计数": "NEUT# (10⁹/L)",
    "吸烟": "Smoking history",
    "糖尿病": "Diabetes history",
    "尿糖": "Urine glucose",
    "白细胞（酯酶）": "LEU",
    "颜色": "COL",
}

FEATURE_DISPLAY_NAMES: List[str] = [DISPLAY_NAMES[f] for f in RAW_FEATURES]

if not BACKGROUND_PATH.exists():
    raise FileNotFoundError(f"SHAP background file not found: {BACKGROUND_PATH}")

_BACKGROUND = pd.read_csv(BACKGROUND_PATH).to_numpy(dtype=float)
if _BACKGROUND.shape[1] != len(RAW_FEATURES):
    raise RuntimeError(
        f"SHAP background has {_BACKGROUND.shape[1]} columns; expected {len(RAW_FEATURES)}."
    )


def _predict_probability_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    frame = pd.DataFrame(array, columns=RAW_FEATURES)
    probability, _ = predict_raw_dataframe(frame)
    return probability


# Construct once at application start. Kernel SHAP is used because the deployed
# predictor is the full two-layer stacking function, not a single tree model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    EXPLAINER = shap.KernelExplainer(
        _predict_probability_array,
        _BACKGROUND,
        link="identity",
        feature_names=FEATURE_DISPLAY_NAMES,
    )


def explain_patient(patient_raw: pd.DataFrame, nsamples: int = 220) -> dict:
    """Explain one patient's full stacking PCa probability."""
    patient = impute_raw_for_explanation(patient_raw)
    values = patient.iloc[0].to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = EXPLAINER.shap_values(
            values.reshape(1, -1),
            nsamples=int(nsamples),
            silent=True,
        )

    shap_values = np.asarray(shap_values, dtype=float).reshape(-1)
    base_value = float(np.asarray(EXPLAINER.expected_value).reshape(-1)[0])
    probability = float(_predict_probability_array(values.reshape(1, -1))[0])

    return {
        "base_value": base_value,
        "shap_values": shap_values,
        "feature_values": values,
        "feature_names": FEATURE_DISPLAY_NAMES,
        "probability": probability,
    }


def _force_times_new_roman(fig) -> None:
    """Apply Times New Roman to every text artist created by SHAP/Matplotlib."""
    for text_artist in fig.findobj(match=lambda obj: hasattr(obj, "set_fontfamily")):
        try:
            text_artist.set_fontfamily(["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"])
        except Exception:
            pass


def make_waterfall_figure(explanation: dict):
    """Create an English SHAP waterfall plot for one prediction."""
    exp = shap.Explanation(
        values=np.asarray(explanation["shap_values"], dtype=float),
        base_values=float(explanation["base_value"]),
        data=np.asarray(explanation["feature_values"], dtype=float),
        feature_names=list(explanation["feature_names"]),
    )

    plt.close("all")
    shap.plots.waterfall(exp, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.8, 5.4)
    # Keep the title fully inside the Matplotlib canvas.  In the previous
    # version y > 1 placed part of the title outside the figure, and Shiny
    # clipped it when rendering the PNG.
    fig.suptitle(
        "SHAP Waterfall Plot — Individual Prediction",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.925])
    _force_times_new_roman(fig)
    return fig


def make_force_figure(explanation: dict):
    """Create a compact, readable static SHAP force plot."""
    shap_values = np.asarray(explanation["shap_values"], dtype=float)
    feature_values = np.asarray(explanation["feature_values"], dtype=float)
    names = list(explanation["feature_names"])
    short_names = {
        "tPSA (ng/mL)": "tPSA",
        "fPSA (ng/mL)": "fPSA",
        "Plasma D-Dimer (μg/L)": "D-Dimer",
        "Cl (mmol/L)": "Cl",
        "APTT (s)": "APTT",
        "ALP (U/L)": "ALP",
        "MONO% (%)": "MONO%",
        "HGB (g/L)": "HGB",
        "ALB (g/L)": "ALB",
        "TBIL (μmol/L)": "TBIL",
        "FIB (g/L)": "FIB",
        "PLT (10⁹/L)": "PLT",
        "GLU (mmol/L)": "GLU",
        "Urea (mmol/L)": "Urea",
        "Cr (μmol/L)": "Cr",
        "UA (μmol/L)": "UA",
        "LDH (U/L)": "LDH",
        "TC (mmol/L)": "TC",
        "CCR (mL/min)": "CCR",
        "NRBC% (%)": "NRBC%",
        "hsCRP (mg/L)": "hsCRP",
        "CO2CP (mmol/L)": "CO2CP",
        "HCY (μmol/L)": "HCY",
        "5'-Nucleotidase (U/L)": "5'-NT",
        "ADA (U/L)": "ADA",
        "HFR% (%)": "HFR%",
        "GGT (U/L)": "GGT",
        "NEUT# (10⁹/L)": "NEUT#",
        "Urine glucose": "Urine glucose",
        "LEU": "LEU",
        "COL": "COL",
        "Total PSA (tPSA)": "tPSA",
        "Free PSA (fPSA)": "fPSA",
        "Plasma D-dimer": "D-dimer",
        "Alkaline phosphatase (ALP)": "ALP",
        "Monocyte percentage (MONO%)": "MONO%",
        "Hemoglobin (HGB)": "HGB",
        "Albumin (ALB)": "ALB",
        "Total bilirubin (TBIL)": "TBIL",
        "Fibrinogen (FIB)": "FIB",
        "Platelet count (PLT)": "PLT",
        "Glucose (GLU)": "GLU",
        "Creatinine (Cr)": "Cr",
        "Uric acid (UA)": "UA",
        "Lactate dehydrogenase (LDH)": "LDH",
        "Total cholesterol (TC)": "TC",
        "Endogenous creatinine clearance (CCR)": "CCR",
        "CO2 combining power (CO2CP)": "CO2CP",
        "Homocysteine (HCY)": "HCY",
        "Serum 5'-nucleotidase (5'-NT)": "5'-NT",
        "Adenosine deaminase (ADA)": "ADA",
        "High-fluorescence reticulocyte fraction (HFR%)": "HFR%",
        "Gamma-glutamyl transferase (GGT)": "GGT",
        "Neutrophil count (NEUT#)": "NEUT#",
        "Smoking history": "Smoking",
        "Diabetes history": "Diabetes",
        "Urine glucose code": "Urine glucose",
        "Leukocyte esterase code": "Leukocyte esterase",
        "Urine color code": "Urine color",
    }

    # A static force plot becomes unreadable with all 34 labels. Keep the six
    # largest absolute contributions and aggregate the remaining contributions.
    order = np.argsort(np.abs(shap_values))[::-1]
    top = order[:6]
    rest = order[6:]

    reduced_values = list(shap_values[top])
    reduced_names = [short_names.get(names[i], names[i]) for i in top]
    reduced_feature_values = [f"{feature_values[i]:.4g}" for i in top]

    if len(rest):
        reduced_values.append(float(shap_values[rest].sum()))
        reduced_names.append("Other features")
        reduced_feature_values.append("combined")

    plt.close("all")
    shap.force_plot(
        float(explanation["base_value"]),
        np.asarray(reduced_values, dtype=float),
        reduced_feature_values,
        feature_names=reduced_names,
        matplotlib=True,
        show=False,
        text_rotation=0,
    )
    fig = plt.gcf()
    fig.set_size_inches(7.8, 4.2)
    # Reserve a dedicated title band inside the canvas so the force-plot
    # annotations cannot overlap or clip the heading in Shiny.
    fig.suptitle(
        "SHAP Force Plot — Individual Prediction",
        fontsize=13,
        fontweight="bold",
        y=0.955,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.885])
    _force_times_new_roman(fig)
    return fig
