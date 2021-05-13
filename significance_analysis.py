import significance_tests as s
import pandas as pd
metric_platelet_df=pd.read_csv('metrics_data.csv')

metrics=['Max','Slope','T-lag']
metrics_sig=pd.DataFrame()
for m in metrics:
    loop_sig=s.run_kruskall_posthoc('Shear Rate (s^-1)',m,
                                    metric_platelet_df)
    metrics_sig=metrics_sig.append(loop_sig,ignore_index=True)
metrics_sig.to_csv('metrics_significance_test.csv')
metrics_sig
