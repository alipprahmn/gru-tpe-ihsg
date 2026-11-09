import os

# ==========================================================
# REDUCE TENSORFLOW CONSOLE LOGS
# ==========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="IHSG Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# COLOR PALETTE & GLOBAL CSS
# ==========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN BACKGROUND
    ===================================================== */

    .stApp {
        background-color: #f3f7f4;
        color: #1f2937;
    }


    /* =====================================================
       MAIN CONTENT
    ===================================================== */

    .block-container {
        padding-top: 2.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    [data-testid="stSidebar"] {
        background-color: #064e3b;
        border-right: 1px solid #0f6b50;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.20);
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-weight: 500;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        color: #d1fae5 !important;
    }

    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff !important;
    }


    /* =====================================================
       TITLES
    ===================================================== */

    h1 {
        color: #14532d !important;
        font-weight: 700 !important;
    }

    h2 {
        color: #14532d !important;
        font-weight: 650 !important;
    }

    h3 {
        color: #14532d !important;
        font-weight: 600 !important;
    }


    /* =====================================================
       NORMAL TEXT
    ===================================================== */

    p,
    span,
    div {
        color: #4b5563;
    }


    /* =====================================================
       CARD
    ===================================================== */

    .custom-card {
        background-color: #ffffff;
        border: 1px solid #b7e4c7;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0px 5px 18px rgba(20, 83, 45, 0.08);
        margin-bottom: 20px;
    }


    /* =====================================================
       METRIC CARD
    ===================================================== */

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #b7e4c7;
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0px 5px 18px rgba(20, 83, 45, 0.07);
        min-height: 145px;
    }

    .metric-label {
        color: #4b5563 !important;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #166534 !important;
        font-size: 38px;
        font-weight: 700;
        line-height: 1.2;
    }


    /* =====================================================
       HERO SECTION
    ===================================================== */

    .hero-section {
        background-color: #ffffff;
        border: 1px solid #b7e4c7;
        border-left: 6px solid #166534;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 28px;
        box-shadow: 0px 5px 18px rgba(20, 83, 45, 0.08);
    }

    .hero-title {
        color: #14532d !important;
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-text {
        color: #4b5563 !important;
        font-size: 17px;
        line-height: 1.7;
    }


    /* =====================================================
       BUTTON
    ===================================================== */

    .stButton > button {
        background-color: #166534 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
    }

    .stButton > button:hover {
        background-color: #14532d !important;
        color: #ffffff !important;
        border: none !important;
    }

    .stButton > button p {
        color: #ffffff !important;
    }


    /* =====================================================
       DATAFRAME
    ===================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #b7e4c7;
        border-radius: 12px;
        overflow: hidden;
    }


    /* =====================================================
       FOOTER
    ===================================================== */

    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #b7e4c7;
        text-align: center;
        color: #4b5563 !important;
        font-size: 15px;
    }


    /* =====================================================
       DIVIDER
    ===================================================== */

    hr {
        border-color: #b7e4c7 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_main_data():

    return pd.read_csv(
        "final_data.csv"
    )


@st.cache_data
def load_train_data():

    return pd.read_csv(
        "train_data.csv"
    )


@st.cache_data
def load_model_comparison():

    return pd.read_csv(
        "final_model_comparison.csv"
    )


@st.cache_data
def load_optimization_results():

    return pd.read_csv(
        "optuna_gru_results.csv"
    )


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_optimized_model():

    return load_model(
        "optimized_gru.keras"
    )


# ==========================================================
# GET REALTIME DATA FROM YAHOO FINANCE
# ==========================================================

@st.cache_data(ttl=900)
def download_market_data():

    ihsg = yf.download(
        "^JKSE",
        period="2y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )


    djia = yf.download(
        "^DJI",
        period="2y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )


    # ------------------------------------------------------
    # HANDLE MULTIINDEX
    # ------------------------------------------------------

    if isinstance(
        ihsg.columns,
        pd.MultiIndex
    ):

        ihsg.columns = ihsg.columns.get_level_values(
            0
        )


    if isinstance(
        djia.columns,
        pd.MultiIndex
    ):

        djia.columns = djia.columns.get_level_values(
            0
        )


    return ihsg, djia


# ==========================================================
# UPDATE DATA WITH YAHOO FINANCE
# ==========================================================

def get_latest_yahoo_data(base_data):

    try:

        # --------------------------------------------------
        # COPY BASE DATA
        # --------------------------------------------------

        base_data = base_data.copy()


        base_data["Date"] = pd.to_datetime(
            base_data["Date"]
        )


        base_data = base_data[
            [
                "Date",
                "IHSG_Close",
                "DJIA_Lag1"
            ]
        ].copy()


        # --------------------------------------------------
        # DOWNLOAD LATEST DATA
        # --------------------------------------------------

        ihsg, djia = download_market_data()


        # --------------------------------------------------
        # VALIDATE DOWNLOAD
        # --------------------------------------------------

        if ihsg.empty:

            return (
                base_data,
                "IHSG data could not be retrieved from Yahoo Finance."
            )


        if djia.empty:

            return (
                base_data,
                "DJIA data could not be retrieved from Yahoo Finance."
            )


        # --------------------------------------------------
        # PREPARE IHSG
        # --------------------------------------------------

        ihsg = ihsg.reset_index()


        ihsg = ihsg[
            [
                "Date",
                "Close"
            ]
        ].copy()


        ihsg = ihsg.rename(
            columns={
                "Close": "IHSG_Close"
            }
        )


        ihsg["Date"] = pd.to_datetime(
            ihsg["Date"]
        )


        # --------------------------------------------------
        # PREPARE DJIA
        # --------------------------------------------------

        djia = djia.reset_index()


        djia = djia[
            [
                "Date",
                "Close"
            ]
        ].copy()


        djia = djia.rename(
            columns={
                "Close": "DJIA_Close"
            }
        )


        djia["Date"] = pd.to_datetime(
            djia["Date"]
        )


        # --------------------------------------------------
        # SORT
        # --------------------------------------------------

        ihsg = ihsg.sort_values(
            "Date"
        )


        djia = djia.sort_values(
            "Date"
        )


        # --------------------------------------------------
        # CREATE DJIA LAG 1
        #
        # IMPORTANT:
        # DJIA_Lag1 dibuat dari nilai DJIA
        # satu observasi sebelumnya.
        # --------------------------------------------------

        djia["DJIA_Lag1"] = djia[
            "DJIA_Close"
        ].shift(
            1
        )


        djia = djia[
            [
                "Date",
                "DJIA_Lag1"
            ]
        ]


        # --------------------------------------------------
        # MERGE IHSG + DJIA LAG 1
        # --------------------------------------------------

        realtime_data = pd.merge_asof(

            ihsg.sort_values(
                "Date"
            ),

            djia.sort_values(
                "Date"
            ),

            on="Date",

            direction="backward"

        )


        # --------------------------------------------------
        # REMOVE MISSING VALUES
        # --------------------------------------------------

        realtime_data = realtime_data.dropna(
            subset=[
                "IHSG_Close",
                "DJIA_Lag1"
            ]
        )


        # --------------------------------------------------
        # KEEP REQUIRED COLUMNS ONLY
        # --------------------------------------------------

        realtime_data = realtime_data[
            [
                "Date",
                "IHSG_Close",
                "DJIA_Lag1"
            ]
        ].copy()


        # --------------------------------------------------
        # COMBINE ORIGINAL + REALTIME DATA
        # --------------------------------------------------

        combined_data = pd.concat(
            [
                base_data,
                realtime_data
            ],
            ignore_index=True
        )


        # --------------------------------------------------
        # SORT AND REMOVE DUPLICATES
        #
        # Yahoo Finance data takes priority.
        # --------------------------------------------------

        combined_data = combined_data.sort_values(
            "Date"
        )


        combined_data = combined_data.drop_duplicates(
            subset="Date",
            keep="last"
        )


        combined_data = combined_data.reset_index(
            drop=True
        )


        return combined_data, None


    except Exception as e:

        return (
            base_data,
            f"Yahoo Finance update failed: {str(e)}"
        )


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        """
        # IHSG Prediction

        **Prediction System Using Optimized GRU**
        """
    )


    st.divider()


    st.markdown(
        """
        ### NAVIGATION
        """
    )


    page = st.radio(
        "Select a Page",
        [
            "Home",
            "Data",
            "Model Comparison",
            "Prediction",
            "Optimization Results"
        ]
    )


    st.divider()


    st.markdown(
        """
        **GRU + Optuna Hyperparameter Optimization**
        """
    )


# ==========================================================
# PAGE: HOME
# ==========================================================

if page == "Home":

    st.title(
        "IHSG Prediction Dashboard"
    )


    st.markdown(
        """
        <div class="hero-section">

        <div class="hero-title">
        IHSG Prediction Dashboard
        </div>

        <div class="hero-text">
        This application presents the results of IHSG prediction using
        LSTM, GRU, and Optimized GRU models. The hyperparameter optimization
        process is performed using Optuna with the Tree-Structured Parzen
        Estimator (TPE) algorithm.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader(
        "Research Information"
    )


    st.markdown(
        """
        <div class="custom-card">

        <p><b>Title:</b></p>

        <p>
        Optimasi Hyperparameter Model GRU Menggunakan Optuna
        untuk Prediksi Indeks Harga Saham Gabungan (IHSG)
        </p>

        <p><b>Models:</b></p>

        <p>
        LSTM, GRU, and Optimized GRU
        </p>

        <p><b>Optimization Method:</b></p>

        <p>
        Optuna with Tree-Structured Parzen Estimator (TPE)
        </p>

        <p><b>Input Features:</b></p>

        <p>
        IHSG Close and DJIA Lag 1
        </p>

        <p><b>Evaluation Metrics:</b></p>

        <p>
        RMSE, MAE, and MAPE
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# PAGE: DATA
# ==========================================================

elif page == "Data":

    st.title(
        "IHSG Dataset"
    )


    data = load_main_data()


    if data is not None:

        st.markdown(
            """
            <div class="hero-section">

            <div class="hero-title">
            Dataset Overview
            </div>

            <div class="hero-text">
            Historical data used in the IHSG prediction model.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(
            3
        )


        with col1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                Total Observations
                </div>

                <div class="metric-value">
                {len(data):,}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                """
                <div class="metric-card">

                <div class="metric-label">
                Model Features
                </div>

                <div class="metric-value">
                2
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            missing_values = data.isnull().sum().sum()


            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                Missing Values
                </div>

                <div class="metric-value">
                {missing_values}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.subheader(
            "Dataset Preview"
        )


        st.dataframe(
            data,
            width="stretch",
            height=500
        )


# ==========================================================
# PAGE: MODEL COMPARISON
# ==========================================================

elif page == "Model Comparison":

    st.title(
        "Model Performance Comparison"
    )


    comparison = load_model_comparison()


    if comparison is not None:

        st.markdown(
            """
            <div class="hero-section">

            <div class="hero-title">
            Model Comparison
            </div>

            <div class="hero-text">
            Comparison of LSTM, baseline GRU, and Optimized GRU
            based on prediction performance.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.subheader(
            "Performance Results"
        )


        st.dataframe(
            comparison,
            width="stretch"
        )


        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.bar(
            comparison["Model"],
            comparison["RMSE"]
        )

        ax.set_title(
            "RMSE Comparison"
        )

        ax.set_xlabel(
            "Model"
        )

        ax.set_ylabel(
            "RMSE"
        )

        st.pyplot(
            fig
        )


        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.bar(
            comparison["Model"],
            comparison["MAE"]
        )

        ax.set_title(
            "MAE Comparison"
        )

        ax.set_xlabel(
            "Model"
        )

        ax.set_ylabel(
            "MAE"
        )

        st.pyplot(
            fig
        )


        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.bar(
            comparison["Model"],
            comparison["MAPE (%)"]
        )

        ax.set_title(
            "MAPE Comparison"
        )

        ax.set_xlabel(
            "Model"
        )

        ax.set_ylabel(
            "MAPE (%)"
        )

        st.pyplot(
            fig
        )


# ==========================================================
# PAGE: PREDICTION
# ==========================================================

elif page == "Prediction":

    st.title(
        "IHSG Future Prediction"
    )


    st.markdown(
        """
        Generate the next IHSG prediction using the Optimized GRU model
        based on the latest 30 observations.
        """
    )


    try:

        # --------------------------------------------------
        # LOAD BASE DATA
        # --------------------------------------------------

        base_data = load_main_data()


        # --------------------------------------------------
        # FETCH LATEST MARKET DATA
        # --------------------------------------------------

        with st.spinner(
            "Fetching latest IHSG and DJIA data..."
        ):

            data, yahoo_error = get_latest_yahoo_data(
                base_data
            )


        # --------------------------------------------------
        # DATA STATUS
        # --------------------------------------------------

        if yahoo_error is None:

            latest_available_date = pd.to_datetime(
                data["Date"]
            ).max()


            st.success(
                f"Latest market data loaded successfully. "
                f"Latest available date: "
                f"{latest_available_date.strftime('%d %B %Y')}"
            )

        else:

            st.warning(
                yahoo_error
            )


        # --------------------------------------------------
        # LOAD TRAINING DATA AND MODEL
        # --------------------------------------------------

        train_data = load_train_data()

        model = load_optimized_model()


        # --------------------------------------------------
        # FIXED MODEL FEATURES
        # --------------------------------------------------

        feature_columns = [
            "IHSG_Close",
            "DJIA_Lag1"
        ]


        target_column = "IHSG_Close"


        # --------------------------------------------------
        # VALIDATE MODEL INPUT
        # --------------------------------------------------

        model_feature_count = model.input_shape[-1]


        if model_feature_count != 2:

            st.error(
                f"The model expects "
                f"{model_feature_count} features, "
                f"but this application uses 2 features."
            )

            st.stop()


        # --------------------------------------------------
        # PREPARE DATA
        # --------------------------------------------------

        data["Date"] = pd.to_datetime(
            data["Date"]
        )


        data = data.sort_values(
            "Date"
        )


        data = data.dropna(
            subset=feature_columns
        )


        data = data.reset_index(
            drop=True
        )


        # --------------------------------------------------
        # GET LATEST 30 OBSERVATIONS
        # --------------------------------------------------

        latest_data = data.tail(
            30
        ).copy()


        if len(latest_data) < 30:

            st.error(
                "At least 30 observations are required."
            )

            st.stop()


        # --------------------------------------------------
        # LATEST VALUES
        # --------------------------------------------------

        latest_ihsg = latest_data[
            "IHSG_Close"
        ].iloc[-1]


        latest_djia = latest_data[
            "DJIA_Lag1"
        ].iloc[-1]


        latest_date = latest_data[
            "Date"
        ].iloc[-1]


        # --------------------------------------------------
        # METRIC CARDS
        # --------------------------------------------------

        col1, col2, col3 = st.columns(
            3
        )


        with col1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                Latest IHSG
                </div>

                <div class="metric-value">
                {latest_ihsg:,.2f}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                DJIA Lag 1
                </div>

                <div class="metric-value">
                {latest_djia:,.2f}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                """
                <div class="metric-card">

                <div class="metric-label">
                Input Window
                </div>

                <div class="metric-value">
                30 Days
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.caption(
            f"Latest observation date: "
            f"{latest_date.strftime('%d %B %Y')}"
        )


        # --------------------------------------------------
        # LATEST 30 OBSERVATIONS
        # --------------------------------------------------

        st.subheader(
            "Latest 30 Observations"
        )


        latest_table = latest_data[
            [
                "Date",
                "IHSG_Close",
                "DJIA_Lag1"
            ]
        ].copy()


        latest_table["Date"] = latest_table[
            "Date"
        ].dt.strftime(
            "%Y-%m-%d"
        )


        latest_table = latest_table.rename(
            columns={
                "IHSG_Close": "IHSG Close",
                "DJIA_Lag1": "DJIA Lag 1"
            }
        )


        latest_table.index = np.arange(
            1,
            len(latest_table) + 1
        )


        st.dataframe(
            latest_table,
            width="stretch",
            height=500
        )


        # --------------------------------------------------
        # GENERATE PREDICTION
        # --------------------------------------------------

        st.subheader(
            "Generate Prediction"
        )


        if st.button(
            "Generate IHSG Prediction",
            width="content"
        ):


            # ----------------------------------------------
            # TRAINING FEATURES
            # ----------------------------------------------

            train_features = train_data[
                feature_columns
            ].copy()


            # ----------------------------------------------
            # LATEST FEATURES
            # ----------------------------------------------

            latest_features = latest_data[
                feature_columns
            ].copy()


            # ----------------------------------------------
            # NORMALIZATION
            # ----------------------------------------------

            feature_scaler = MinMaxScaler()

            feature_scaler.fit(
                train_features
            )


            normalized_latest = feature_scaler.transform(
                latest_features
            )


            # ----------------------------------------------
            # RESHAPE FOR GRU
            # ----------------------------------------------

            X_input = normalized_latest.reshape(
                1,
                30,
                2
            )


            # ----------------------------------------------
            # MODEL PREDICTION
            # ----------------------------------------------

            normalized_prediction = model.predict(
                X_input,
                verbose=0
            )


            # ----------------------------------------------
            # TARGET SCALER
            # ----------------------------------------------

            target_scaler = MinMaxScaler()

            target_scaler.fit(
                train_data[
                    [target_column]
                ]
            )


            # ----------------------------------------------
            # DENORMALIZATION
            # ----------------------------------------------

            prediction = target_scaler.inverse_transform(
                normalized_prediction.reshape(
                    -1,
                    1
                )
            )[0][0]


            # ----------------------------------------------
            # NEXT BUSINESS DAY
            # ----------------------------------------------

            prediction_date = (
                pd.to_datetime(
                    latest_date
                )
                + pd.offsets.BDay(
                    1
                )
            )


            # ----------------------------------------------
            # DISPLAY RESULT
            # ----------------------------------------------

            st.success(
                "IHSG prediction generated successfully."
            )


            st.markdown(
                f"""
                <div class="custom-card">

                <div class="metric-label">
                Predicted IHSG
                </div>

                <div class="metric-value">
                {prediction:,.2f}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.caption(
                f"Prediction target date: "
                f"{prediction_date.strftime('%d %B %Y')}"
            )


            # ----------------------------------------------
            # PREDICTION CHART
            # ----------------------------------------------

            st.subheader(
                "Latest IHSG Trend and Prediction"
            )


            fig, ax = plt.subplots(
                figsize=(12, 6)
            )


            ax.plot(
                latest_data["Date"],
                latest_data["IHSG_Close"],
                marker="o",
                label="Actual IHSG"
            )


            ax.scatter(
                prediction_date,
                prediction,
                s=120,
                marker="o",
                label="Predicted IHSG"
            )


            ax.set_title(
                "IHSG Prediction Based on Latest 30 Observations"
            )


            ax.set_xlabel(
                "Date"
            )


            ax.set_ylabel(
                "IHSG Value"
            )


            ax.legend()

            ax.grid(
                alpha=0.3
            )


            plt.xticks(
                rotation=45
            )


            plt.tight_layout()


            st.pyplot(
                fig
            )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ==========================================================
# PAGE: OPTIMIZATION RESULTS
# ==========================================================

elif page == "Optimization Results":

    st.title(
        "Hyperparameter Optimization Results"
    )


    st.markdown(
        """
        Results of GRU hyperparameter optimization using Optuna
        with the Tree-Structured Parzen Estimator (TPE).
        """
    )


    try:

        optimization_results = load_optimization_results()


        if "state" in optimization_results.columns:

            complete_trials = optimization_results[
                optimization_results["state"] == "COMPLETE"
            ].copy()

        else:

            complete_trials = optimization_results.copy()


        best_trial = complete_trials.loc[
            complete_trials["value"].idxmin()
        ]


        st.subheader(
            "Best Hyperparameters"
        )


        col1, col2, col3, col4 = st.columns(
            4
        )


        with col1:

            units = best_trial.get(
                "params_units",
                "N/A"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-label">Best GRU Units</div>
                <div class="metric-value">{units}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:

            batch_size = best_trial.get(
                "params_batch_size",
                "N/A"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-label">Best Batch Size</div>
                <div class="metric-value">{batch_size}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            dropout_rate = best_trial.get(
                "params_dropout_rate",
                "N/A"
            )


            if isinstance(
                dropout_rate,
                (float, np.floating)
            ):

                dropout_display = (
                    f"{dropout_rate:.2f}"
                )

            else:

                dropout_display = dropout_rate


            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-label">Best Dropout Rate</div>
                <div class="metric-value">{dropout_display}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col4:

            learning_rate = best_trial.get(
                "params_learning_rate",
                "N/A"
            )


            if isinstance(
                learning_rate,
                (float, np.floating)
            ):

                learning_rate_display = (
                    f"{learning_rate:.6f}"
                )

            else:

                learning_rate_display = learning_rate


            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-label">Best Learning Rate</div>
                <div class="metric-value">{learning_rate_display}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        st.subheader(
            "Best Optimization Result"
        )


        best_value = best_trial[
            "value"
        ]


        st.markdown(
            f"""
            <div class="custom-card">

            <div class="metric-label">
            Best Objective Value
            </div>

            <div class="metric-value">
            {best_value:.8f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.subheader(
            "Optimization History"
        )


        fig, ax = plt.subplots(
            figsize=(11, 5)
        )


        ax.plot(
            complete_trials["number"],
            complete_trials["value"],
            marker="o"
        )


        ax.set_title(
            "Optuna Optimization History"
        )


        ax.set_xlabel(
            "Trial Number"
        )


        ax.set_ylabel(
            "Objective Value"
        )


        ax.grid(
            alpha=0.3
        )


        st.pyplot(
            fig
        )


        st.subheader(
            "All Optimization Trials"
        )


        display_columns = [

            "number",
            "value",
            "params_batch_size",
            "params_dropout_rate",
            "params_learning_rate",
            "params_units",
            "state"

        ]


        available_columns = [

            column

            for column in display_columns

            if column in optimization_results.columns

        ]


        optimization_display = optimization_results[
            available_columns
        ].copy()


        optimization_display.columns = [

            column.replace(
                "params_",
                ""
            ).replace(
                "_",
                " "
            ).title()

            for column in optimization_display.columns

        ]


        st.dataframe(
            optimization_display,
            width="stretch",
            height=500
        )


    except Exception as e:

        st.error(
            f"Optimization Results error: {e}"
        )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">

    IHSG Prediction System |
    GRU Hyperparameter Optimization Using Optuna

    </div>
    """,
    unsafe_allow_html=True
)