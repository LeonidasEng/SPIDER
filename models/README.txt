SPIDER Model Inference
###############################################################################

In this directory you will find all saved models that are used by the rule
layer to generate new SPIDER forecasts.

Currently, there is only one type: Random Forest with Logistic Regression
calibration designed to estimate a probabilty of large forecast error: 

P(|ΔKp| > 1)

---

3-day Forecast (0030 Issue)
Lead Days:
LD0: 24-hr 
LD1: 48-hr 
LD2: 72-hr

- rf_cal_0030_LD0.pkl
- rf_cal_0030_LD1.pkl
- rf_cal_0030_LD2.pkl

3-day Forecast (1230 Issue)
Lead Days:
LD0: 24-hr
LD1: 48-hr
LD2: 72-hr

- rf_cal_1230_LD0.pkl
- rf_cal_1230_LD1.pkl
- rf_cal_1230_LD2.pkl

To use these models, simply run: 

    python src/decision/rule_layer.py 

You will be prompted to enter a forecast date (YYYYMMDD format)
from available data and choose an issue type ('0030' or '1230').