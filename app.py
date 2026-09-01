
import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "final_xgboost_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "feature_names.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "threshold.pkl"
)

EVALUATION_PATH = os.path.join(
    BASE_DIR,
    "evaluation_data.pkl"
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))

evaluation_data = joblib.load(EVALUATION_PATH)

y_test = np.asarray(evaluation_data["y_test"])
y_prob = np.asarray(evaluation_data["y_prob"])
y_pred = np.asarray(evaluation_data["y_pred"])


# ============================================================
# MODEL METRICS
# ============================================================

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

average_precision = average_precision_score(
    y_test,
    y_prob
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# MODEL COMPARISON
# Values obtained from final project evaluation
# ============================================================

comparison_results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Tuned XGBoost"
    ],

    "Accuracy": [
        0.956044,
        0.952381,
        0.953602,
        0.956044
    ],

    "Precision": [
        0.761905,
        0.693878,
        0.708333,
        0.671875
    ],

    "Recall": [
        0.551724,
        0.586207,
        0.586207,
        0.741379
    ],

    "F1 Score": [
        0.640000,
        0.635514,
        0.641509,
        0.704918
    ]
})


# ============================================================
# CUSTOM DARK THEME
# ============================================================

CSS = """

body {
    background: #061a2e !important;
    color: white !important;
}

.gradio-container {
    max-width: 1450px !important;
    background: #061a2e !important;
    color: white !important;
}

.dashboard-header {
    background: linear-gradient(
        135deg,
        #082944,
        #063a5c
    );

    border: 1px solid #11658d;
    border-radius: 22px;

    padding: 35px;
    margin-bottom: 25px;

    text-align: center;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.dashboard-subtitle {
    font-size: 19px;
    color: #38bdf8;
    font-weight: 600;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 18px;
}

.info-card {
    background: #0a2942;
    border: 1px solid #155e83;
    border-radius: 14px;
    padding: 20px;
}

.prediction-card {
    background: #0b2b43;
    border: 1px solid #1b6d94;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

footer {
    text-align: center;
}

"""


# ============================================================
# FLOOD PREDICTION FUNCTION
# ============================================================

def predict_flood(
    annual_rainfall,
    monsoon_rainfall,
    max_monthly_rainfall,
    rainfall_std
):

    values = np.array([[
        annual_rainfall,
        monsoon_rainfall,
        max_monthly_rainfall,
        rainfall_std
    ]])

    probability = float(
        model.predict_proba(values)[0][1]
    )

    prediction = int(
        probability >= threshold
    )

    percentage = probability * 100

    if prediction == 1:

        result = "🌊 FLOOD PREDICTED"

        risk = "🔴 High Flood Risk"

    else:

        result = "✅ NO FLOOD PREDICTED"

        risk = "🟢 Low Flood Risk"

    return (
        f"## {result}\n\n"
        f"### Flood Probability: **{percentage:.2f}%**\n\n"
        f"### {risk}\n\n"
        f"Decision threshold: **{threshold:.2f}**"
    )


# ============================================================
# FEATURE IMPORTANCE FIGURE
# ============================================================

def create_feature_importance():

    df = feature_importance_df.sort_values(
        "Importance",
        ascending=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Importance"],
            y=df["Feature"],
            orientation="h",
            text=[
                f"{x:.4f}"
                for x in df["Importance"]
            ],
            textposition="outside",
            marker_color="#6670f5"
        )
    )

    fig.update_layout(
        title="XGBoost Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_dark",
        height=480,
        margin=dict(
            l=20,
            r=30,
            t=70,
            b=50
        )
    )

    return fig


# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

def create_confusion_matrix():

    fig = go.Figure(
        data=go.Heatmap(

            z=cm,

            x=[
                "No Flood",
                "Flood"
            ],

            y=[
                "No Flood",
                "Flood"
            ],

            text=cm,

            texttemplate="%{text}",

            textfont={
                "size": 20
            },

            colorscale="Viridis",

            hovertemplate=
            "True: %{y}<br>"
            "Predicted: %{x}<br>"
            "Count: %{z}<extra></extra>"
        )
    )

    fig.update_layout(

        title="XGBoost Confusion Matrix",

        xaxis_title="Predicted Label",

        yaxis_title="True Label",

        template="plotly_dark",

        height=480
    )

    return fig


# ============================================================
# ROC CURVE
# ============================================================

def create_roc_curve():

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"XGBoost (AUC = {roc_auc:.3f})",
            line=dict(
                width=3,
                color="#6670f5"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(
                dash="dash",
                color="#ff6248"
            )
        )
    )

    fig.update_layout(
        title="XGBoost ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        height=480,
        xaxis=dict(
            range=[0, 1]
        ),
        yaxis=dict(
            range=[0, 1]
        )
    )

    return fig


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

def create_pr_curve():

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"XGBoost (AP = {average_precision:.3f})",
            line=dict(
                width=3,
                color="#6670f5"
            )
        )
    )

    fig.update_layout(
        title="XGBoost Precision–Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_dark",
        height=480,
        xaxis=dict(
            range=[0, 1]
        ),
        yaxis=dict(
            range=[0, 1]
        )
    )

    return fig


# ============================================================
# MODEL COMPARISON
# ============================================================

def create_model_comparison(metric):

    values = comparison_results[metric]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(

            x=comparison_results["Model"],

            y=values,

            text=[
                f"{value * 100:.2f}%"
                for value in values
            ],

            textposition="outside",

            marker_color="#6670f5",

            hovertemplate=
            "<b>%{x}</b><br>"
            + metric
            + ": %{y:.2%}<extra></extra>"
        )
    )

    fig.update_layout(

        title=f"Model {metric} Comparison",

        xaxis_title="Machine Learning Model",

        yaxis_title=metric,

        template="plotly_dark",

        height=500,

        yaxis=dict(
            range=[
                0,
                1
            ]
        )
    )

    return fig


# ============================================================
# DASHBOARD
# ============================================================

with gr.Blocks(
    title="Smart Flood Prediction System"
) as demo:


    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML("""
    <div class="dashboard-header">

        <div class="dashboard-title">
            🌊 Smart Flood Prediction System
        </div>

        <div class="dashboard-subtitle">
            AI/ML Powered • Tuned XGBoost Model
        </div>

        <p>
            Predict the likelihood of flood occurrence
            using historical rainfall statistics and
            a trained machine learning model.
        </p>

    </div>
    """)


    # ========================================================
    # TABS
    # ========================================================

    with gr.Tabs():


        # ====================================================
        # FLOOD PREDICTION
        # ====================================================

        with gr.Tab("🔮 Flood Prediction"):

            gr.Markdown(
                "# 🔮 Flood Prediction"
            )

            gr.Markdown(
                "Enter rainfall statistics to predict "
                "the likelihood of flood occurrence."
            )

            with gr.Row():

                with gr.Column():

                    annual_rainfall = gr.Number(
                        label="Previous Annual Rainfall",
                        value=1000
                    )

                    monsoon_rainfall = gr.Number(
                        label="Previous Monsoon Rainfall",
                        value=700
                    )

                with gr.Column():

                    max_monthly_rainfall = gr.Number(
                        label="Previous Maximum Monthly Rainfall",
                        value=300
                    )

                    rainfall_std = gr.Number(
                        label="Previous Rainfall Standard Deviation",
                        value=100
                    )


            predict_button = gr.Button(
                "🌊 PREDICT FLOOD",
                variant="primary"
            )


            prediction_output = gr.Markdown()


            predict_button.click(

                fn=predict_flood,

                inputs=[
                    annual_rainfall,
                    monsoon_rainfall,
                    max_monthly_rainfall,
                    rainfall_std
                ],

                outputs=prediction_output
            )


            gr.Markdown(
                f"""
                ### 📌 Model Information

                **Model:** Tuned XGBoost

                **Decision Threshold:** {threshold:.2f}

                **Features Used:** {len(feature_names)}

                **ROC-AUC:** {roc_auc:.4f}

                **Average Precision:** {average_precision:.4f}
                """
            )


        # ====================================================
        # MODEL ANALYSIS
        # ====================================================

        with gr.Tab("📊 Model Analysis"):

            gr.Markdown(
                "# 📊 Model Performance Analysis"
            )

            gr.Markdown(
                "Explore the final Tuned XGBoost model "
                "using interactive visualizations."
            )


            # ------------------------------------------------
            # TOP ROW
            # ------------------------------------------------

            with gr.Row():

                with gr.Column():

                    gr.Markdown(
                        "### 📊 Feature Importance"
                    )

                    feature_plot = gr.Plot(
                        value=create_feature_importance()
                    )


                with gr.Column():

                    gr.Markdown(
                        "### 📉 Confusion Matrix"
                    )

                    confusion_plot = gr.Plot(
                        value=create_confusion_matrix()
                    )


            gr.Markdown(
                f"""
                **Observation:**  
                **{feature_importance_df.iloc[0]["Feature"]}**
                is the most influential feature in the
                final XGBoost model.
                """
            )


            gr.Markdown("### Test-set results")

            gr.Markdown(
                f"""
                - **True Negatives:** {tn}
                - **False Positives:** {fp}
                - **False Negatives:** {fn}
                - **True Positives:** {tp}
                """
            )


            # ------------------------------------------------
            # ROC + PR
            # ------------------------------------------------

            with gr.Row():

                with gr.Column():

                    gr.Markdown(
                        "### 📈 ROC Curve"
                    )

                    roc_plot = gr.Plot(
                        value=create_roc_curve()
                    )

                    gr.Markdown(
                        f"**ROC-AUC:** `{roc_auc:.4f}`"
                    )

                    gr.Markdown(
                        "The ROC curve evaluates the model's "
                        "ability to distinguish between flood "
                        "and no-flood cases across different "
                        "classification thresholds."
                    )


                with gr.Column():

                    gr.Markdown(
                        "### 📈 Precision–Recall Curve"
                    )

                    pr_plot = gr.Plot(
                        value=create_pr_curve()
                    )

                    gr.Markdown(
                        f"**Average Precision:** `{average_precision:.4f}`"
                    )

                    gr.Markdown(
                        "The Precision–Recall curve shows the "
                        "trade-off between precision and recall "
                        "across thresholds."
                    )


            # ------------------------------------------------
            # MODEL COMPARISON
            # ------------------------------------------------

            gr.Markdown(
                "### 📊 Model Comparison"
            )

            metric_dropdown = gr.Dropdown(

                choices=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score"
                ],

                value="F1 Score",

                label="Select Metric"
            )


            comparison_plot = gr.Plot(
                value=create_model_comparison(
                    "F1 Score"
                )
            )


            metric_dropdown.change(

                fn=create_model_comparison,

                inputs=metric_dropdown,

                outputs=comparison_plot
            )


            # ------------------------------------------------
            # FINAL MODEL SELECTION
            # ------------------------------------------------

            gr.Markdown(
                f"""
                ## ⭐ Final Model Selection

                **Tuned XGBoost** achieved the highest
                recall and F1 score while maintaining
                **95.60% accuracy**.

                **Recall:** 74.14%

                **F1 Score:** 70.49%

                **ROC-AUC:** {roc_auc:.4f}

                **Average Precision:** {average_precision:.4f}
                """
            )


        # ====================================================
        # ABOUT
        # ====================================================

        with gr.Tab("ℹ️ About"):

            gr.Markdown(
                """
                # ℹ️ About the Project

                ## 🌊 Smart Flood Prediction System

                This project uses machine learning to predict
                the likelihood of flood occurrence from
                historical rainfall statistics.

                ### 🤖 Final Model

                **Tuned XGBoost Classifier**

                ### 📊 Input Features

                - Previous Annual Rainfall
                - Previous Monsoon Rainfall
                - Previous Maximum Monthly Rainfall
                - Previous Rainfall Standard Deviation

                ### 🎯 Decision Threshold

                **0.40**

                The lower decision threshold was selected
                to improve flood detection and recall.

                ### 📈 Key Performance

                - Accuracy: **95.60%**
                - Precision: **67.19%**
                - Recall: **74.14%**
                - F1 Score: **70.49%**
                - ROC-AUC: **0.9727**
                - Average Precision: **0.7592**

                ### 🛠️ Technologies

                Python • Pandas • NumPy • Scikit-learn
                • XGBoost • Plotly • Gradio
                """
            )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""
    <div style="
        text-align:center;
        padding:30px;
        margin-top:20px;
        color:#b9d7ea;
    ">

        🌊 Smart Flood Prediction System
        &nbsp; | &nbsp;
        📦 Tuned XGBoost Model
        &nbsp; | &nbsp;
        🎯 Threshold: 0.40
        &nbsp; | &nbsp;
        ⚡ Interactive Dashboard

    </div>
    """)


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(css=CSS)
