# Rails GUI plan (future)

## Screens

### 1) Station dashboard
- current year predictions:
  - last spring frost
  - first fall frost
  - freeze-free days
  - winter severity outlook
- show P10/P50/P90 + baseline

### 2) Frost history
- plot last 50 years:
  - actual DOY
  - predicted P50
  - interval bands

### 3) Model comparison
- baseline vs model MAE
- coverage calibration

### 4) Winter severity
- percentile rank this year
- component breakdown (HDD, extremes, snap days)

## UX guardrails (reputation-safe)

- never show a single “answer”
- always show uncertainty + baseline
- show historical performance prominently
