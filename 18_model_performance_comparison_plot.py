import pandas as pd

import matplotlib.pyplot as plt


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
# 2. CREATE RMSE COMPARISON PLOT
# ==========================================

plt.figure(

    figsize=(8, 6)

)


plt.bar(

    comparison["Model"],

    comparison["RMSE"]

)


plt.title(

    "RMSE Comparison of Prediction Models"

)


plt.xlabel(

    "Model"

)


plt.ylabel(

    "RMSE"

)


plt.tight_layout()


plt.savefig(

    "rmse_model_comparison.png",

    dpi=300

)


plt.show()


# ==========================================
# 3. CREATE MAE COMPARISON PLOT
# ==========================================

plt.figure(

    figsize=(8, 6)

)


plt.bar(

    comparison["Model"],

    comparison["MAE"]

)


plt.title(

    "MAE Comparison of Prediction Models"

)


plt.xlabel(

    "Model"

)


plt.ylabel(

    "MAE"

)


plt.tight_layout()


plt.savefig(

    "mae_model_comparison.png",

    dpi=300

)


plt.show()


# ==========================================
# 4. CREATE MAPE COMPARISON PLOT
# ==========================================

plt.figure(

    figsize=(8, 6)

)


plt.bar(

    comparison["Model"],

    comparison["MAPE (%)"]

)


plt.title(

    "MAPE Comparison of Prediction Models"

)


plt.xlabel(

    "Model"

)


plt.ylabel(

    "MAPE (%)"

)


plt.tight_layout()


plt.savefig(

    "mape_model_comparison.png",

    dpi=300

)


plt.show()


# ==========================================
# 5. DISPLAY SAVED FILES
# ==========================================

print(

    "\nMODEL COMPARISON PLOTS SAVED SUCCESSFULLY"

)

print(

    "- rmse_model_comparison.png"

)

print(

    "- mae_model_comparison.png"

)

print(

    "- mape_model_comparison.png"

)