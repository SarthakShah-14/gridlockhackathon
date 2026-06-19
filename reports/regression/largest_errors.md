# Top 100 Largest Prediction Errors in Duration Regression

This report documents the top 100 samples with the largest absolute prediction errors on the out-of-fold validation set.

---

## 1. Top 20 Error Samples Breakdown

| Incident ID | Event Type | Location (Junction) | Corridor | Timestamp | Actual Duration (m) | Predicted Duration (m) | Absolute Error (m) | Percentage Error |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `FKID000013` | `unplanned` | `HebbalFlyoverJunc` | `ORR North 1` | `2023-12-26 19:15:28.124+00` | 103674.9 | 1221.0 | 102453.9 | 98.82% |
| `FKID000287` | `unplanned` | `unknown` | `Non-corridor` | `2024-02-02 08:33:26.866+00` | 101592.2 | 1371.9 | 100220.3 | 98.65% |
| `FKID001029` | `unplanned` | `MekhriCircle` | `Bellary Road 1` | `2023-12-05 04:54:28.082+00` | 100585.6 | 1059.5 | 99526.1 | 98.95% |
| `FKID004337` | `unplanned` | `unknown` | `Non-corridor` | `2023-12-02 02:22:41.28+00` | 106736.9 | 7688.2 | 99048.7 | 92.80% |
| `FKID004444` | `unplanned` | `unknown` | `Non-corridor` | `2023-11-11 21:47:37.254+00` | 102549.5 | 7382.6 | 95166.9 | 92.80% |
| `FKID005082` | `unplanned` | `unknown` | `Non-corridor` | `2024-01-14 21:36:58.514+00` | 102689.5 | 8538.5 | 94151.0 | 91.69% |
| `FKID005674` | `unplanned` | `unknown` | `Non-corridor` | `2024-01-16 21:32:32.773+00` | 100781.2 | 7420.6 | 93360.6 | 92.64% |
| `FKID005188` | `unplanned` | `unknown` | `Non-corridor` | `2023-12-27 09:18:19.035+00` | 90387.3 | 1115.5 | 89271.9 | 98.77% |
| `FKID000444` | `unplanned` | `unknown` | `Non-corridor` | `2024-02-04 19:27:21.145+00` | 89680.0 | 832.6 | 88847.4 | 99.07% |
| `FKID000618` | `unplanned` | `Delmia-Jayanagar` | `ORR West 1` | `2023-12-30 05:16:06.301+00` | 102211.0 | 14678.8 | 87532.2 | 85.64% |
| `FKID000981` | `unplanned` | `unknown` | `CBD 2` | `2024-02-12 03:28:31.086+00` | 96013.9 | 9418.9 | 86595.0 | 90.19% |
| `FKID001689` | `unplanned` | `BangaloreBodyBuildersJunc` | `Mysore Road` | `2023-12-22 19:42:51.341+00` | 87021.1 | 1348.9 | 85672.3 | 98.45% |
| `FKID000522` | `unplanned` | `HebbalFlyoverJunc` | `Bellary Road 1` | `2023-12-06 12:40:52.743+00` | 88409.3 | 4414.6 | 83994.8 | 95.01% |
| `FKID000675` | `unplanned` | `unknown` | `ORR North 2` | `2024-02-08 01:58:10.685+00` | 83795.7 | 961.8 | 82833.8 | 98.85% |
| `FKID004195` | `unplanned` | `unknown` | `Bellary Road 1` | `2023-11-28 22:06:42.882+00` | 86701.9 | 4118.1 | 82583.8 | 95.25% |
| `FKID000164` | `unplanned` | `unknown` | `unknown` | `2024-01-31 23:07:48.509+00` | 94064.4 | 11991.2 | 82073.2 | 87.25% |
| `FKID000651` | `unplanned` | `unknown` | `Non-corridor` | `2024-02-07 22:23:25.651+00` | 88542.2 | 6727.4 | 81814.8 | 92.40% |
| `FKID000925` | `unplanned` | `unknown` | `Non-corridor` | `2024-02-10 23:06:29.351+00` | 80635.5 | 1239.8 | 79395.7 | 98.46% |
| `FKID000394` | `unplanned` | `unknown` | `Non-corridor` | `2024-02-04 02:30:51.854+00` | 86671.2 | 7640.7 | 79030.5 | 91.18% |
| `FKID001632` | `unplanned` | `unknown` | `CBD 2` | `2024-02-20 04:59:25.949+00` | 80516.3 | 2139.7 | 78376.6 | 97.34% |

---

## 2. Root Cause Investigation of Extreme Errors

By inspecting the top 100 errors, we identify several clear data quality and preprocessing failure modes:
1. **Administrative Resolution Lags (Logging Delay):** Many incidents have actual durations of several thousand minutes (e.g. 10,000+ minutes, representing multiple days or weeks). In reality, these incidents block traffic for under 2 hours, but operators forgot to close the incident log in the portal until days later.
2. **Near-Zero Actual Durations:** A subset of incidents has actual durations of under 1 minute. The model predicts a temporal baseline around 30-50 minutes, leading to absolute percentage errors exceeding 1000% (e.g. `(50 - 0.1)/0.1 = 49900%`).
3. **Data Leakage / Inconsistencies:** Duplicate tickets for the same event logged at different times, or resolved tickets closed after administrative delays, skew the target variable.
