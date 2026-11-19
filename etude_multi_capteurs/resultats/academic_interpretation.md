# Academic Interpretation Report

Dataset: **MetroPT3 Air Compressor**.

## Prediction Results

The project compares LSTM, Transformer and a previous-value naive baseline on the same temporal test split. The scaler is fitted only on the training period, then applied to the test period, which avoids data leakage.

The naive baseline predicts the next value using the last observed value in the input sequence. For smooth IoT signals this is a very strong reference model and must be included for academic honesty.

## Oil Temperature

Oil_temperature is a smooth variable in this dataset. When a signal changes slowly, the previous value is often already very close to the next value. Therefore, LSTM and Transformer models must be judged against this baseline, not only against their absolute RMSE.

```text
  target_column         model_name     RMSE      MAE        R2  sequence_length  epochs  train_size  test_size
Oil_temperature               LSTM 2.192486 1.997184  0.755202               60      20       39940       9940
Oil_temperature        Transformer 4.789167 3.878793 -0.168031               60      20       39940       9940
Oil_temperature NaivePreviousValue 0.248291 0.094457  0.996861               60       0       39940       9940
```

## Negative R2

A negative R2 means that the model explains less variance than a simple mean-based predictor on the test set. When this happens, the model should not be described as a strong forecasting model, even if it produces visually plausible curves.

## Best Comparisons

```text
  target_column         model_name     RMSE      MAE       R2  baseline_RMSE  RMSE_minus_baseline  beats_naive_baseline
     Reservoirs NaivePreviousValue 0.063209 0.025572 0.987773       0.063209             0.000000                 False
            TP3 NaivePreviousValue 0.063296 0.025610 0.987775       0.063296             0.000000                 False
            TP3               LSTM 0.080184 0.061229 0.980380       0.063296             0.016889                 False
     Reservoirs               LSTM 0.084697 0.045022 0.978047       0.063209             0.021488                 False
Oil_temperature NaivePreviousValue 0.248291 0.094457 0.996861       0.248291             0.000000                 False
```

## Scientific Conclusion

At least one deep learning model beats the naive previous-value baseline for one target. Those target/model combinations are the most defensible candidates for predictive analysis.

## Anomaly Detection

The autoencoder anomaly detection is unsupervised because no confirmed anomaly labels are provided. The threshold is defined as the 95th percentile of the training reconstruction error.

Detected anomalies: **3561** out of **49940** sequence points (**7.1306%**).

These detected points are potential anomalies, not confirmed faults. They should be validated with maintenance logs, expert review, or known failure periods.

## Recommendations

- Keep the naive baseline in all future comparisons.
- Try targets with stronger dynamics such as Motor_current, TP2 or Reservoirs.
- Add lag features, rolling statistics and operating-state features.
- Tune sequence length, model size, learning rate and selected feature count.
- Validate anomaly detections against real failure logs if available.
- Consider classical models such as ARIMA, XGBoost, Random Forest or Isolation Forest as additional baselines.