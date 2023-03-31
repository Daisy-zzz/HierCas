import pandas as pd
g_df = pd.read_csv('./processed/ml_weibo.csv')
g_df.rename(columns={'size':'e_idx'}, inplace = True)
cas_l = g_df.cas
src_l = g_df.src
dst_l = g_df.target
e_l = g_df.e_idx # max=997
ts_l = g_df.ts
print(max(max(src_l), max(dst_l)), min(min(src_l), min(dst_l)))