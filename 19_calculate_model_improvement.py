import pandas as pd


# ==========================================
# 1. LOAD MODEL COMPARISON RESULTS
# ==========================================

comparison = pd.read_csv(

    "final_model_comparison.csv"

)


print(

    "MODEL COMPARISON DATA LOADED SUCCESSFULLY"

)


print(

    "\nMODEL PERFORMANCE RESULTS"

)

print(

    comparison

)


# ==========================================
# 2. GET BASELINE AND OPTIMIZED RESULTS
# ==========================================

gru = comparison[

    comparison["Model"] == "GRU"

].iloc[0]


optimized_gru = comparison[

    comparison["Model"] == "Optimized GRU"

].iloc[0]


# ==========================================
# 3. CALCULATE RMSE IMPROVEMENT
# ==========================================

rmse_improvement = (

    (gru["RMSE"] - optimized_gru["RMSE"])

    /

    gru["RMSE"]

) * 100


# ==========================================
# 4. CALCULATE MAE IMPROVEMENT
# ==========================================

mae_improvement = (

    (gru["MAE"] - optimized_gru["MAE"])

    /

    gru["MAE"]

) * 100


# ==========================================
# 5. CALCULATE MAPE IMPROVEMENT
# ==========================================

mape_improvement = (

    (gru["MAPE (%)"] - optimized_gru["MAPE (%)"])

    /

    gru["MAPE (%)"]

) * 100


# ==========================================
# 6. CREATE IMPROVEMENT DATAFRAME
# ==========================================

improvement = pd.DataFrame({

    "Metric":

    [

        "RMSE",

        "MAE",

        "MAPE (%)"

    ],

    "Baseline GRU":

    [

        gru["RMSE"],

        gru["MAE"],

        gru["MAPE (%)"]

    ],

    "Optimized GRU":

    [

        optimized_gru["RMSE"],

        optimized_gru["MAE"],

        optimized_gru["MAPE (%)"]

    ],

    "Improvement (%)":

    [

        rmse_improvement,

        mae_improvement,

        mape_improvement

    ]

})


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print(

    "\nOPTIMIZATION IMPROVEMENT RESULTS"

)

print(

    improvement

)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

improvement.to_csv(

    "model_improvement_results.csv",

    index=False

)


print(

    "\nIMPROVEMENT RESULTS SAVED SUCCESSFULLY"

)

print(

    "- model_improvement_results.csv"

)