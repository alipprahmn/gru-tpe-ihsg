import pandas as pd

import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD PREDICTION RESULTS
# ==========================================

results = pd.read_csv(

    "optimized_gru_predictions.csv"

)


print(

    "PREDICTION DATA LOADED SUCCESSFULLY"

)


print(

    "\nFIRST 5 DATA"

)

print(

    results.head()

)


# ==========================================
# 2. CREATE PREDICTION VISUALIZATION
# ==========================================

plt.figure(

    figsize=(14, 6)

)


plt.plot(

    results["Actual_IHSG"],

    label="Actual IHSG"

)


plt.plot(

    results["Predicted_IHSG"],

    label="Predicted IHSG"

)


plt.title(

    "Actual vs Predicted IHSG Using Optimized GRU"

)


plt.xlabel(

    "Test Data Index"

)


plt.ylabel(

    "IHSG Value"

)


plt.legend()


plt.grid(

    True

)


plt.tight_layout()


# ==========================================
# 3. SAVE VISUALIZATION
# ==========================================

plt.savefig(

    "optimized_gru_prediction_plot.png",

    dpi=300

)


print(

    "\nVISUALIZATION SUCCESSFULLY SAVED"

)

print(

    "- optimized_gru_prediction_plot.png"

)


# ==========================================
# 4. DISPLAY VISUALIZATION
# ==========================================

plt.show()