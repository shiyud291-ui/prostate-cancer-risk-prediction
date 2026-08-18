# -*- coding: utf-8 -*-
"""Prostate cancer risk prediction web application - Version 2."""

import math
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from predictor import CUTOFF, MODEL_NAME, RAW_FEATURES, classify, predict_raw_dataframe
from shap_explainer import explain_patient, make_force_figure, make_waterfall_figure

# Internal keys match the fitted model. Every value below is English-only and is
# the only feature wording shown to web users.
LABELS: Dict[str, str] = {
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

FEATURE_IDS = {feature: f"f_{i:02d}" for i, feature in enumerate(RAW_FEATURES)}
BASELINE = ["年龄", "吸烟", "糖尿病"]
URINE = ["尿糖", "白细胞（酯酶）", "颜色"]
BLOOD = [f for f in RAW_FEATURES if f not in set(BASELINE + URINE)]


def numeric_input(feature: str):
    return ui.input_text(
        FEATURE_IDS[feature],
        LABELS[feature],
        value="",
        placeholder="Enter value; leave blank if unavailable",
        width="100%",
    )


def select_input(feature: str):
    if feature in ("吸烟", "糖尿病"):
        choices = {"": "Not available", "0": "No", "1": "Yes"}
    elif feature == "尿糖":
        choices = {"": "Not available", "0": "Code 0", "1": "Code 1", "2": "Code 2", "3": "Code 3"}
    elif feature == "白细胞（酯酶）":
        choices = {"": "Not available", "0": "Code 0", "1": "Code 1", "2": "Code 2"}
    elif feature == "颜色":
        choices = {
            "": "Not available", "0": "Code 0", "1": "Code 1", "2": "Code 2",
            "3": "Code 3", "4": "Code 4", "5": "Code 5",
        }
    else:
        raise ValueError("No selection definition is available for this feature.")

    return ui.input_select(
        FEATURE_IDS[feature],
        LABELS[feature],
        choices=choices,
        selected="",
        width="100%",
    )


def input_for(feature: str):
    if feature in ("吸烟", "糖尿病", "尿糖", "白细胞（酯酶）", "颜色"):
        return select_input(feature)
    return numeric_input(feature)


CSS = """
html, body, body *, input, select, textarea, button, .form-control, .form-select, .btn,
.card, .card *, .accordion, .accordion *, .navbar, .modal, .popover, .tooltip {
  font-family: "Times New Roman", Times, serif !important;
}
body { background: #f5f7fb; color: #1f2937; }
.app-header {
  background: linear-gradient(135deg, #163a5f 0%, #245b87 100%);
  color: white; border-radius: 0 0 20px 20px; padding: 30px 32px 26px;
  margin: 0 -12px 24px -12px; box-shadow: 0 8px 24px rgba(22,58,95,.16);
}
.app-header h1 { font-weight: 750; margin-bottom: 6px; font-size: 2rem; }
.app-header p { margin-bottom: 0; opacity: .94; }
.version-badge { display:inline-block; margin-top:10px; padding:4px 10px; border:1px solid rgba(255,255,255,.45); border-radius:999px; font-size:.82rem; font-weight:700; }
.card { border: 0; border-radius: 16px; box-shadow: 0 4px 18px rgba(15,23,42,.08); margin-bottom: 18px; }
.card-header { background: white; font-weight: 700; border-bottom: 1px solid #eef2f7; }
.section-note { color:#64748b; font-size:.91rem; margin-bottom:12px; }
.model-pill { display:inline-block; padding:5px 10px; background:#eaf2f8; color:#163a5f;
              border-radius:999px; font-size:.84rem; margin-bottom:8px; }
.result-empty { color:#64748b; text-align:center; padding:52px 16px; }
.probability-number { font-size:3.1rem; font-weight:800; line-height:1; margin:12px 0; }
.risk-high { color:#a63131; font-weight:700; font-size:1.25rem; }
.risk-low { color:#246b45; font-weight:700; font-size:1.25rem; }
.prob-track { height:14px; background:#e7edf4; border-radius:999px; overflow:hidden; margin:18px 0 8px; }
.prob-fill { height:100%; background:#245b87; border-radius:999px; }
.cutoff-line { color:#64748b; font-size:.9rem; }
.missing-note { margin-top:14px; padding:10px 12px; border-radius:10px; background:#fff7e8; color:#795700; }
.disclaimer { margin-top:16px; padding:12px 14px; border-left:4px solid #94a3b8; background:#f8fafc;
              color:#52606d; font-size:.88rem; }
.explain-note { padding:12px 14px; border-radius:10px; background:#eef5fb; color:#36566f; font-size:.9rem; }
.btn-primary { background:#245b87; border-color:#245b87; font-weight:700; padding:10px 16px; }
.btn-primary:hover { background:#163a5f; border-color:#163a5f; }
.btn-outline-primary { font-weight:700; padding:9px 16px; }
.accordion-button:not(.collapsed) { color:#163a5f; background:#eef5fb; }
.form-label { font-size:.90rem; font-weight:600; }
.shap-card-title { font-weight:700; margin-bottom:3px; }
.shap-subtitle { color:#64748b; font-size:.9rem; margin-bottom:12px; }
"""

app_ui = ui.page_fluid(
    ui.tags.style(CSS),
    ui.div(
        ui.h1("Prostate Cancer Risk Prediction"),
        ui.p("Stacking Top13 + Meta XGBoost using baseline characteristics, blood laboratory tests, and urinalysis"),
        ui.div("Version 7 · English + SHAP · Side-by-side plots", class_="version-badge"),
        class_="app-header",
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Patient / Participant Inputs"),
            ui.div(
                ui.span(MODEL_NAME, class_="model-pill"),
                ui.p(
                    "Enter raw clinical values exactly as measured. Do not standardize values before entry. "
                    "Blank inputs are handled by the fitted training-set imputation steps.",
                    class_="section-note",
                ),
                ui.accordion(
                    ui.accordion_panel(
                        "Baseline characteristics",
                        ui.layout_columns(*[input_for(f) for f in BASELINE], col_widths=(6, 6, 6)),
                    ),
                    ui.accordion_panel(
                        "blood indicators",
                        ui.layout_columns(*[input_for(f) for f in BLOOD], col_widths=(6, 6)),
                    ),
                    ui.accordion_panel(
                        "urine indicators",
                        ui.p(
                            "Urine glucose, LEU, and COL are displayed using the numeric codes used by the fitted model. The code-to-category dictionary can be added later without changing the model.",
                            class_="section-note",
                        ),
                        ui.layout_columns(*[input_for(f) for f in URINE], col_widths=(6, 6, 6)),
                    ),
                    open="Baseline characteristics",
                ),
                ui.div(style="height:14px"),
                ui.input_action_button("predict_btn", "Calculate PCa Risk", class_="btn-primary w-100"),
            ),
        ),
        ui.card(
            ui.card_header("Prediction Result"),
            ui.output_ui("result_panel"),
        ),
        col_widths=(7, 5),
    ),
    ui.card(
        ui.card_header("Individual Model Explanation"),
        ui.div(
            ui.p(
                "Generate an individual SHAP explanation for the full two-layer stacking prediction. "
                "The probability is calculated immediately; the explanation is generated separately because SHAP requires additional model evaluations.",
                class_="section-note",
            ),
            ui.input_action_button("explain_btn", "Generate SHAP Explanation", class_="btn-outline-primary"),
            ui.div(style="height:12px"),
            ui.output_ui("shap_status"),
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("SHAP Waterfall Plot"),
            ui.div(
                ui.p(
                    "Shows how the strongest feature contributions move the model output from the background probability to this participant's predicted probability.",
                    class_="shap-subtitle",
                ),
                ui.output_plot("waterfall_plot", height="500px"),
                class_="p-3",
            ),
        ),
        ui.card(
            ui.card_header("SHAP Force Plot"),
            ui.div(
                ui.p(
                    "Features pushing the predicted probability higher are shown in red; features pushing it lower are shown in blue. The remaining smaller contributions are aggregated as Other features for readability.",
                    class_="shap-subtitle",
                ),
                ui.output_plot("force_plot", height="500px"),
                class_="p-3",
            ),
        ),
        col_widths=(6, 6),
    ),
    ui.div(
        "Research-use clinical prediction aid. The output is a model-estimated probability and does not establish a diagnosis. "
        "SHAP values describe the model's prediction, not causal effects. Clinical decisions should incorporate symptoms, examination, imaging, pathology, and professional judgment.",
        class_="disclaimer",
    ),
)


def _parse_number(value) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "nan", "none"}:
        return np.nan
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("Only finite numeric values are allowed.")
    return number


def server(input: Inputs, output: Outputs, session: Session):
    result = reactive.value(None)
    patient_state = reactive.value(None)
    prediction_error = reactive.value(None)
    shap_state = reactive.value(None)
    shap_error = reactive.value(None)

    @reactive.effect
    @reactive.event(input.predict_btn)
    def _calculate_prediction():
        try:
            row = {}
            missing_count = 0
            for feature in RAW_FEATURES:
                raw_value = getattr(input, FEATURE_IDS[feature])()
                value = _parse_number(raw_value)
                if np.isnan(value):
                    missing_count += 1
                row[feature] = value

            patient = pd.DataFrame([row], columns=RAW_FEATURES)
            probability, _ = predict_raw_dataframe(patient)
            p = float(probability[0])

            patient_state.set(patient)
            result.set({"probability": p, "class": classify(p), "missing_count": missing_count})
            prediction_error.set(None)
            shap_state.set(None)
            shap_error.set(None)
        except Exception as exc:
            patient_state.set(None)
            result.set(None)
            shap_state.set(None)
            prediction_error.set(str(exc))

    @reactive.effect
    @reactive.event(input.explain_btn)
    def _calculate_shap():
        patient = patient_state.get()
        if patient is None:
            shap_state.set(None)
            shap_error.set("Calculate a prediction first, then generate the SHAP explanation.")
            return
        try:
            explanation = explain_patient(patient, nsamples=220)
            shap_state.set(explanation)
            shap_error.set(None)
        except Exception as exc:
            shap_state.set(None)
            shap_error.set(str(exc))

    @render.ui
    def result_panel():
        message = prediction_error.get()
        if message:
            return ui.div(
                ui.h5("Unable to calculate prediction"),
                ui.p(message),
                class_="alert alert-danger m-3",
            )

        data = result.get()
        if data is None:
            return ui.div(
                ui.h5("No prediction yet"),
                ui.p("Enter the available clinical variables and click Calculate PCa Risk."),
                ui.p(f"Fixed training-set cutoff: {CUTOFF:.6f}"),
                class_="result-empty",
            )

        p = float(data["probability"])
        percent = 100.0 * p
        high = int(data["class"]) == 1
        risk_text = "Higher predicted risk" if high else "Lower predicted risk"
        risk_class = "risk-high" if high else "risk-low"
        missing_count = int(data["missing_count"])

        missing_ui = None
        if missing_count > 0:
            missing_ui = ui.div(
                f"{missing_count} of {len(RAW_FEATURES)} model inputs were left blank and were handled by the fitted imputation steps.",
                class_="missing-note",
            )

        return ui.div(
            ui.div("Predicted probability of prostate cancer", class_="section-note"),
            ui.div(f"{percent:.1f}%", class_="probability-number"),
            ui.div(risk_text, class_=risk_class),
            ui.div(
                ui.div(class_="prob-fill", style=f"width:{max(0, min(100, percent)):.2f}%"),
                class_="prob-track",
            ),
            ui.div(f"P(PCa) = {p:.6f}  |  Fixed cutoff = {CUTOFF:.6f}", class_="cutoff-line"),
            missing_ui,
            ui.div(
                "The cutoff was fixed using training-set out-of-fold predictions and is not recalculated for this participant.",
                class_="disclaimer",
            ),
            class_="p-4",
        )

    @render.ui
    def shap_status():
        error_message = shap_error.get()
        if error_message:
            return ui.div(error_message, class_="alert alert-warning")

        explanation = shap_state.get()
        if explanation is None:
            if result.get() is None:
                return ui.div(
                    "No SHAP explanation yet. Calculate a prediction first.",
                    class_="explain-note",
                )
            return ui.div(
                "Prediction complete. Click Generate SHAP Explanation to create the waterfall and force plots.",
                class_="explain-note",
            )

        base_value = float(explanation["base_value"])
        probability = float(explanation["probability"])
        return ui.div(
            ui.strong("SHAP explanation generated successfully. "),
            f"Background probability = {base_value:.3f}; participant probability = {probability:.3f}. ",
            "Kernel SHAP explains the complete stacking prediction in probability units using a privacy-preserving synthetic background derived from the training-feature distribution.",
            class_="alert alert-success",
        )

    @render.plot
    def waterfall_plot():
        explanation = shap_state.get()
        if explanation is None:
            return None
        return make_waterfall_figure(explanation)

    @render.plot
    def force_plot():
        explanation = shap_state.get()
        if explanation is None:
            return None
        return make_force_figure(explanation)


app = App(app_ui, server)
