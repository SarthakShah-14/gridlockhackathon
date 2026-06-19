# Incident Duration Regression: Diagnostic Research Report

This report details a 14-stage scientific diagnostic pipeline to analyze the predictability of incident duration and determine the primary failure modes of the regression model.

## 1. Target Variable Distribution Analysis
- **Valid Durations Count**: 3498
- **Missing Target Count**: 4625 (56.59%)
- **Duplicated Durations Count**: 1 (0.03%)
- **Skewness**: 37.7653
- **Kurtosis**: 1859.0424

### Percentile Breakdown (Minutes)
| Percentile | Value (mins) | Value (hours) | Value (days) |
|---|---|---|---|
| Min | 0.03 | 0.00 | 0.00 |
| 25% | 28.39 | 0.47 | 0.02 |
| 50% (Median) | 69.80 | 1.16 | 0.05 |
| 75% | 722.78 | 12.05 | 0.50 |
| 90% | 15371.09 | 256.18 | 10.67 |
| 95% | 36921.76 | 615.36 | 25.64 |
| 99% | 106741.37 | 1779.02 | 74.13 |
| 99.9% | 189386.57 | 3156.44 | 131.52 |
| Max | 2051059.22 | 34184.32 | 1424.35 |

## 2. Baseline Difficulty Assessment
We test whether simple baseline estimators can predict duration. If ML models do not significantly beat these baselines, it indicates a low feature signal.

### Baseline Performance Summary
| Baseline Model | MAE (mins) | RMSE (mins) | R² | MAPE | MedAE (mins) |
|---|---|---|---|---|---|
| mean | 11845.17 | 40613.98 | -0.005310 | 748.96 | 8960.30 |
| median | 6599.45 | 41035.09 | -0.026265 | 5.16 | 48.40 |
| junction_median | 6599.45 | 41035.09 | -0.026265 | 5.16 | 48.40 |
| event_type_median | 6599.61 | 41029.84 | -0.026002 | 11.52 | 46.90 |
| priority_median | 6599.30 | 41034.94 | -0.026257 | 5.26 | 48.28 |
| junc_event_median | 6599.45 | 41035.09 | -0.026265 | 5.16 | 48.40 |

## 3. Label Quality Auditing
- **Negative Durations Count**: 50 (End time occurs before start time)
- **Exactly Zero Durations Count**: 0
- **Extreme Administrative Lags (> 7 days)**: 438 records
- **Duplicate Incidents (by start time, junction, event type)**: 149 records
  - Average duration variance for identical events: 35.76 minutes
  - Max duration variance for identical events: 123.93 minutes

## 4. Target Noise Detection
We analyze duration metrics for duplicate/near-identical feature groups to check for irreducible noise.

### High Variance in Identical Feature Combinations
| Junction | Event Type | Priority | Event Cause | Count | Mean Duration (m) | Std Dev (m) | Coeff of Variation (CV) |
|---|---|---|---|---|---|---|---|
| unknown | unplanned | High | vehicle_breakdown | 739 | 115.9 | 1325.3 | 11.432 |
| unknown | planned | Low | procession | 17 | 1107.9 | 4186.0 | 3.778 |
| unknown | unplanned | High | water_logging | 132 | 12251.0 | 42304.3 | 3.453 |
| unknown | planned | High | construction | 134 | 1710.1 | 5077.5 | 2.969 |
| unknown | unplanned | Low | accident | 28 | 199.7 | 585.8 | 2.934 |
| BMTCJunction-K H Road | unplanned | Low | vehicle_breakdown | 8 | 4048.4 | 11264.1 | 2.782 |
| unknown | planned | Low | public_event | 17 | 5001.0 | 12407.1 | 2.481 |
| unknown | planned | High | public_event | 12 | 720.3 | 1651.7 | 2.293 |
| unknown | unplanned | Low | procession | 6 | 853.1 | 1942.1 | 2.276 |
| unknown | planned | Low | protest | 5 | 1157.8 | 2576.8 | 2.226 |

## 5. Distribution Diagnostics & Curve Fitting
- **Gini Coefficient**: 0.9132 (Indicates extreme inequality in duration distributions)

### Theoretical Distribution Fits
We compare theoretical distribution fits using Kolmogorov-Smirnov (KS) test statistics (lower is better):
| Distribution | KS Statistic | p-value |
|---|---|---|
| norm | 0.4350 | 0 |
| lognorm | 0.1907 | 7.778e-112 |
| gamma | 0.9997 | 0 |
| expon | 0.6760 | 0 |

## 6. Spatial & Temporal Predictability
- **Spatial-only Model R² (DecisionTree, GroupKFold)**: -3.957956
- **Temporal-only Model R² (DecisionTree, GroupKFold)**: -1.113602

## 7. Group Stability Audit
Groups with high Coefficient of Variation ($\ge 1.5$) are highly unstable and will introduce noise into historical stats.
- **junction**: 55 of 89 categories with $\ge 5$ records are unstable (CV $\ge 1.5$).
  - Most unstable junction: MekhriCircle (CV = 4.998)
- **corridor**: 21 of 23 categories with $\ge 5$ records are unstable (CV $\ge 1.5$).
  - Most unstable corridor: Airport New South Road (CV = 5.238)
- **event_type**: 2 of 2 categories with $\ge 5$ records are unstable (CV $\ge 1.5$).
  - Most unstable event_type: planned (CV = 13.261)
- **police_station**: 53 of 54 categories with $\ge 5$ records are unstable (CV $\ge 1.5$).
  - Most unstable police_station: Banaswadi (CV = 6.491)

## 8. Systemic Estimator Benchmarking
We compare 12 regression models under identical GroupKFold CV. Target log-transformation `np.log1p` is applied during training.

### Model Benchmarks (Log-Space Targets CV)
| Regressor Model | MAE (mins) | RMSE (mins) | R² | MAPE | MedAE (mins) |
|---|---|---|---|---|---|
| Ridge | 2409396815140951291828783848378112160920440783234924544.00 | 142501105816109836696094553024735709271924522878991597568.00 | -12376137975893293749103553812074703084902298803612339822462348012596408114188335727847820945397882290176.000000 | 1174708555921718429211720794939242267642972602368.00 | 105988.64 |
| Lasso | 1691571369579157822066132163292223116834500762651207079852769280.00 | 100046114951720709092337470613381450909374504154239314184097497088.00 | -6100271026149876222057189808902883220628396409016215225341572065495143388939571974830619297518594794309917366545310285824.000000 | 824730633123464505076887907835713274402706145700613193728.00 | 81.81 |
| ElasticNet | 16996588140227341335752976728666634176346221726511558782091264.00 | 1005244378951203709637782323858451644515598595584326206004133888.00 | -615874155984604709932696888813463490886367037680119268346085037982222837797747616941968672829524706059350131504840704.000000 | 8286736906238766830462746895213037871290786705263558656.00 | 84.31 |
| HuberRegressor | 35352769255118558180031977176397114460255704496635417153018047249048419515150463488152184029184.00 | 2090900378409211529920279637754539364830119641075237988047539442306570661954437830484573804298240.00 | -2664497738020722242630617472282092663942430457914221382254710796634655516328529243321049994191438922360640268561595946749952316202987713586933381347777155731833619311988810375797669888.000000 | 17236347395555332028011915068014360573096759924798366994683108637991064365359420808364032.00 | 57.05 |
| DecisionTree | 6085.89 | 39810.35 | 0.034081 | 23.72 | 49.40 |
| RandomForest | 6088.66 | 40119.49 | 0.019021 | 13.34 | 59.28 |
| ExtraTrees | 6255.71 | 40452.39 | 0.002674 | 16.88 | 55.96 |
| HistGBM | 6000.06 | 39863.21 | 0.031514 | 10.97 | 58.05 |
| GradientBoosting | 6043.00 | 39964.86 | 0.026569 | 6.85 | 63.75 |
| LightGBM | 6040.50 | 39909.67 | 0.029256 | 11.62 | 57.12 |
| XGBoost | 6407.11 | 40472.83 | 0.001666 | 22.21 | 63.59 |
| CatBoost | 5987.66 | 39839.91 | 0.032646 | 8.62 | 61.53 |

## 9. Cross-Validation Configuration Study
Compare validation splits. Time-series and Group splits are usually much harder (and more realistic) than Random split.

| CV Strategy | MAE (mins) | RMSE (mins) | R² |
|---|---|---|---|
| Random KFold | 123933130186606436027608637629449962945550783332629872640.00 | 7329887707936628778309411160917162708363821419526367477760.00 | -32744873445111809802877645072944458263060826745883274068926899897244346774365395340467634057373278162386944.000000 |
| Stratified KFold | 126720929223617182423062422523173841332976229940014874624.00 | 7494768994020623292239071335298556946553201178171917991936.00 | -34234593570318920880369563430428151213107511152953177989253593368703229967448242567336173157882700515770368.000000 |
| GroupKFold (Junction) | 2409396815140951291828783848378112160920440783234924544.00 | 142501105816109836696094553024735709271924522878991597568.00 | -12376137975893289770517662533781565841844313629045619018813141633814668590476520582571844845130878025728.000000 |
| TimeSeriesSplit | 3231449760369346031984373456819621664750797518751263404448328059197351420129387835643536119072756543163484265825101947688395872977740684221897975724083216798929267215760477815207352166294458261008179909096596960869319966344636590302085288986677203767152736505762788420511296018465198542112652249079480320.00 | inf | -inf |

## 10. Target Transformation Comparison
Compare how target transformations affect Ridge regressor predictability (GroupKFold CV).

| Transformation | MAE (mins) | RMSE (mins) | R² | MAPE |
|---|---|---|---|---|
| Raw Duration (No Transform) | 33716.81 | 66068.59 | -1.660350 | 1576.43 |
| Logarithmic np.log1p | 2409396815140951291828783848378112160920440783234924544.00 | 142501105816109836696094553024735709271924522878991597568.00 | -12376137975893293749103553812074703084902298803612339822462348012596408114188335727847820945397882290176.000000 | 1174708555921718429211720794939242267642972602368.00 |
| Box-Cox | 7946557.65 | 469610709.19 | -134407856.328276 | 54.28 |
| Yeo-Johnson | 6491.99 | 40857.89 | -0.017421 | 2.66 |
| Quantile Normal | 6292.33 | 34320.62 | 0.282108 | 3.03 |

## 11. Outlier Handling Study
Compare how outlier treatment affects validation $R^2$ scores under GroupKFold CV (Ridge Regressor with Log1p transform).
- **No Filtering (Validation R²)**: -3310479561442083175174195989274095992799037354661175468455337782178194392294364122930885848843650334720.000000
- **Winsorization (Validation R²)**: -157169859744857501501876542462146873207803236133521414998689648860522713386344484305380863295619072.000000
- **IQR Outlier Removal (Validation R²)**: -0.880788
- **Isolation Forest contamination=0.05 (Validation R²)**: -105109937067888728903244027411920724364707841966080.000000

## 12. Residual Diagnostics
- **Durbin-Watson Test Statistic**: 2.0000 (DW close to 2.0 indicates no residual autocorrelation)
- **Shapiro-Wilk Normality Test (p-value)**: 2.239e-86
- **Breusch-Pagan Homoscedasticity proxy correlation**: 0.9881 (p-value: 0, significant indicates heteroscedasticity)

## 13. Feature Ablation Study
We run ablation tests by dropping major feature categories and tracking model validation metrics (Ridge, Log1p target, GroupKFold CV).

- **Full Feature Set baseline**: MAE = 2409396815140951291828783848378112160920440783234924544.00 mins, R² = -12376137975893293749103553812074703084902298803612339822462348012596408114188335727847820945397882290176.000000

| Feature Group Dropped | MAE (mins) | R² | MAE Increase (mins) | Performance Drop |
|---|---|---|---|---|
| Historical Stats | 5026565918208525692168026246124749809778688.00 | -53865567920660441663980497186924542609762924919682275133010009254228383028477952.000000 | -2409396815135924640704627745395831191219458703215296512.00 | -12376137975893293749103553812074703084902298803612339822462348012596408114188335727847820945397882290176.000000 |
| Graph Centralities | 3318972912821690410081608244622249387211865342705926144.00 | -23484206720781337041030081644023420948324277650181111932046459786245255808788224123301187092534458843136.000000 | +909576097680739118252824396244137226291424559471001600.00 | +11108068744888043291926527831948717863421978846568772109584111773648847694599888395453366147136576552960.000000 |
| Temporal Features | 561604065553436469065932790586973384679903786736549888.00 | -672401950763629598011589618950963587660711547995664131880847094325804293165641530593125530269483270144.000000 | -1847792749587514992904034518260370507927840712382480384.00 | -11703736025129664151091964193123739497241587255616675690581500918270603821022694197254695415128399020032.000000 |
| Spatial Features | 219020391984716104914117159197819521788360165951537152.00 | -102267486561987605750242945250520497638789378797019535811810351473362693052560051216969778588020637696.000000 | -2190376423156235186914666689180292639132080617283387392.00 | -12273870489331306034563852902183354615773642642698261514675754674203232231034280731252211195318185754624.000000 |
| Interaction Features | 3251215154437148819973218925435326639631974645526301769728.00 | -22535121283585509964749248875070306076191792510573664036866355327253551989188570299540315348829167773676470272.000000 | +3248805757622008054815844847340288041936964469920278511616.00 | +22535108907447535057544034338950169774043386175806047608990185016680214751992905783146092901483299115733352448.000000 |

## 14. Research Conclusion & Recommendation Engine
### Final Research Verdict

**Verdict**: Duration prediction is **fundamentally unlearnable as a continuous regression model** in its raw state. Under leakage-free GroupKFold CV, the best achievable R² is approximately **-12376137975893293749103553812074703084902298803612339822462348012596408114188335727847820945397882290176.0000**. The dataset is dominated by a severe administrative logging delay: the median duration is 67.6 minutes, but the max duration spans 3.9 years (2051059.22 minutes), leading to a Gini coefficient of **0.9132**. This massive inequality means standard MSE-based regression models optimize solely for rare long-tail delays, degrading general predictions.

**Best Recommendation**: Convert the continuous duration regression task into a **3-class Ordinal Severity Classification model** (Quick: <=30 mins, Moderate: 30-90 mins, Prolonged: >90 mins) to stabilize learning, ignore exact administrative closure noise, and provide highly robust forecasts that align with dispatch operational demands.