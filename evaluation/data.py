# evaluation/data.py
import os
import pandas as pd

def extract_coverage_data(stats_file):
    coverage_data, timeline_data = {}, None
    if not os.path.exists(stats_file):
        return coverage_data, timeline_data
    try:
        stats_df = pd.read_csv(stats_file)
        if not stats_df.empty:
            stats_df.columns = stats_df.columns.str.lower()
            last_row = stats_df.iloc[-1]
            coverage_data = last_row.to_dict()
            timeline_cols = [col for col in stats_df.columns if '_t' in col and col.split('_t')[-1].isdigit()]
            if timeline_cols:
                timeline_data = stats_df[timeline_cols].iloc[[-1]].copy()
                timeline_data.reset_index(drop=True, inplace=True)
    except Exception as e:
        print(f"  Error reading statistics file {stats_file}: {e}")
    return coverage_data, timeline_data
