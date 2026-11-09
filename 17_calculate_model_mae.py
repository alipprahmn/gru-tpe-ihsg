import pandas as pd

from sklearn.metrics import mean_absolute_error


# ==========================================
# 1. DEFINE MODEL PREDICTION FILES
# ==========================================

model_files = {

    "LSTM": "baseline_lstm_predictions.csv",

    "GRU": "baseline_gru_predictions.csv",

    "Optimized GRU": "optimized_gru_predictions.csv"

}


# ==========================================
# 2. CALCULATE MAE FOR EACH MODEL
# ==========================================

mae_results = []


for model_name, file_name in model_files.items():

    predictions = pd.read_csv(

        file_name

    )


    mae = mean_absolute_error(

        predictions["Actual_IHSG"],

        predictions["Predicted_IHSG"]

    )


    mae_results.append({

        "Model": model_name,

        "MAE": mae

    })


    print(

        f"{model_name} MAE: {mae}"

    )


# ==========================================
# 3. CREATE MAE DATAFRAME
# ==========================================

mae_df = pd.DataFrame(

    mae_results

)


print(

    "\nMODEL MAE RESULTS"

)

print(

    mae_df

)


# ==========================================
# 4. LOAD FINAL MODEL COMPARISON
# ==========================================

comparison = pd.read_csv(

    "final_model_comparison.csv"

)


# ==========================================
# 5. MERGE MAE RESULTS
# ==========================================

comparison = comparison.merge(

    mae_df,

    on="Model",

    how="left"

)


# ==========================================
# 6. REORDER COLUMNS
# ==========================================

comparison = comparison[

    [

        "Model",

        "RMSE",

        "MAE",

        "MAPE (%)"

    ]

]


# ==========================================
# 7. DISPLAY FINAL RESULTS
# ==========================================

print(

    "\nFINAL MODEL COMPARISON"

)

print(

    comparison

)


# ==========================================
# 8. SAVE UPDATED COMPARISON
# ==========================================

comparison.to_csv(

    "final_model_comparison.csv",

    index=False

)


print(

    "\nUPDATED FILE SAVED SUCCESSFULLY"

)

print(

    "- final_model_comparison.csv"

)