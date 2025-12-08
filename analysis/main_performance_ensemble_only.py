import os
import pickle
import sys
import numpy as np
import pandas as pd
import pdb
from performance_functions import (
    baseflow_index, bias, flow_duration_curve, get_quant,
    high_flows, low_flows, nse, alpha_nse, beta_nse,
    kge, stdev_rat, zero_freq, FHV, FLV, mass_balance
)

# === User input ===
experiment = sys.argv[1]  # the performnace csv file name
#input_path = f"results_data/{experiment}.pkl"
# "/home/eecs/erichson/yihan/diffusion_ssm/runs/run_2305_1319_seed200/test_results.pkl"

# run_2705_1007_seed200 encdec_lstm
# run_0206_2217_seed200  seq2seq_lstm
# run_0206_2104_seed200 seq2seq_ssm
# run_2005_1634_seed200 run_2305_1319_seed200 diffusion lstm
# run_0706_2355_seed200 new diffusion_lstm with generic setup
# run_0806_1017_seed200 new diffusion_lstm with generic setup, using flex optimizer and scheduler (per-batch)
# run_1006_2026_seed200 new diffusion_lstm with generic setup, using ssm optimizer&scheduler, and using future precip
# run_1306_0902_seed200 diffusion_lstm with generic setup, and concatenate the future prcp for test
# run_1206_1256_seed200 diffusion_unet with generic setup, film, and h32
# run_1106_0103_seed200 diffusion_unet with generic setup, no film, h128
# run_1206_1257_seed200 diffusion_unet with generic setup, film, and h64
# run_1706_2101_seed200 diffusion_unet with generic setup with attention for 7-day future prcp
# run_2206_0024_seed200 diffusion_unet with correct cross attention
# run_2406_2231_seed200 diffusion_unet with time emb and film every block, film_v2 prcp add in hidden
# run_2606_0810_seed200 diffusion_unet film v2 concat prcp
# run_2606_2337_seed200 lstm concat prcp
# run_2606_1823_seed200 diffusion_unet film v3  concat prcp
# run_2606_2329_seed200 diffusion_unet attention film v2 h32
# run_2606_2319_seed200 diffusion_unet attention film v2 (h64)
# run_2706_0823_seed200 diffusion_unet attention film v3
# run_2806_1832_seed200 lstm concat temb and prcp
# run_0207_0349_seed200 ssm v2 original scheduler
# run_0307_0701_seed200 ssm v2 flex scheduler lr1e-4 cossteps100 dmodel256 dstate128 static full
# run_0307_0659_seed200 ssm v2 flex scheduler lr1e-4 cossteps100 dmodel256 dstate128 static&temb full
# run_0307_0552_seed200 ssm v2 flex scheduler lr1e-4 cossteps100 dmodel256 dstate128 allcond0
# run_0307_1717_seed200 ssm v2 flex scheduler lr1e-4 cossteps50 warmup 5% dmodel256 dstate128 ssm encoder last step of hidden allcond full
# run_0407_0144_seed200 ssm v2 flex scheduler lr1e-4 cossteps60 warmup 1 dmodel256 dstate256 ssm encoder last step of hidden allcond full #-----
# run_0507_2233_seed200 ssm v2 flex scheduler lr1e-4 warmup 1 dmodel256 dstate256 allcond full
# run_0607_1826_seed200 lstm flex scheduler
# run_0607_1914_seed200 ssm v2 flex schueduler lr4e-4
# run_0707_0709_seed200 ssm v2 flex lr4e-4 allcond0 paramgroups
# run_0707_0649_seed200 ssm v2 flex lr4e-4 allcond0 uniform params
# run_0607_0843_seed200 ssm unet flex lr1e-4 fullcond unicormparams 
# run_0707_0839_seed200 ssm lstm flex lr4e-4 fullcond groupparams (modified lstm with static attributes bias added into encoder hidden)
# run_0607_2129_seed200 ssm flex lr1e-4 fullcond uniform params with ssm encoder
# run_0807_0728_seed200 ssm-ssm lr4e-4 cond0 groupparams
# run_0907_0437_seed200 lstm-lstm lr1e-4 droupout0.5 (with static cond bias at bn)
# run_0907_0519_seed5534 ssm-ssm lr2e-4 cond0 grupparams
# run_1007_1805_seed5534 ssm-ssm lr2e-4 cond0 embmlp silureadout dropout0.2 seed5534
# run_1107_2105_seed5534 encdec lstm deterministic baseline
# run_1107_2236_seed5534 diffusion lstm nowcast 3 prcp
# run_1107_0814_seed5534 ssm-ssm lr2e-4 cond0 embmlp silureadout dropout0.2 seed5534 powerlaw pooling for encoder (refer to as ssm v2)
# run_1007_1804_seed5534 ssm-ssm lr2e-4 cond0 embmlp linearreadout dropout0.2 seed5534
# run_1207_2225_seed5534 diffusion lstm nowcast 15 forcings all
# run_1207_2019_seed5534 ssm-ssm lr2e-4 cond0 embmlp silureadout dropout0.2 seed5534 powerlaw pooling for encoder daymet 
# run_1407_0600_seed5534 ssm-ssm v2 dmodel 512 dstate 512 bs 64
# run_1607_1724_seed5534 diffusion lstm daymet 
# run_1507_2209_seed5534 ssm-ssm v2 daymet bs64
# run_1507_2208_seed5534 ssm-ssm v2 daymet 540 8 bs64
# run_1607_1758_seed5534/test_results_cfg2.pkl diffusion lstm allforcings nowcast noise cfg2
# run_1607_1758_seed5534/test_results_cfg1.pkl diffusion lstm allforcings nowcast noise cfg2
# run_1607_1725_seed5534 diffusion lstm daymet noise cfg0
# run_1807_2237_seed5534 ssm-ssm v2 daymet crossattention pooling concatprcp 
# run_1907_2236_seed5534 ssm-ssm v2 xtallconcat crossattention lion seed5534
# run_1907_2259_seed3407 ssm-ssm v2 xtallconcat crossattention lion seed3407

# run_2007_0900_seed3407 diffusion lstm xtallconcat lion seed3407
# run_2007_1642_seed3407 diffusion lstm xtallconcat lion seed3407 daymet
# run_2007_0830_seed3407 ssm-ssm v2 xtallconcat pp lion seed3407 daymet
# run_2007_2105_seed3407 encdec lstm seed3407 allforcings
# run_2107_0651_seed3407 encdec lstm seed3407 daymet
# run_2007_2029_seed3407 ssm-ssm v2 xtallconcat pp lion seed3407 all best epoch
# run_2007_2030_seed3407 ssm-ssm v2 xtallconcat pp lion seed3407 daymet best epoch


# run_2207_1832_seed3407 diffusion lstm aligned daymet ****
# run_2207_1829_seed3407/test_results_epoch60.pkl ssm-ssm daymet aligned
# run_2307_1833_seed3407 diffusion lstm aligned all
# run_2307_1732_seed3407 diffusion lstm aligned daymet 1-7fc
# run_2307_1834_seed3407/test_results_epoch60.pkl ssm-ssm all aligned
# run_2407_2122_seed3407 encdec_lstm aligned daymet
# run_2307_0655_seed3407/test_results_epoch100.pkl ssm-ssm daymet aligned epoch100
# run_2507_2120_seed3407/test_results_epoch60.pkl encoder_only_ssm daymet aligned epoch60 ****
# run_1108_0525_seed3407/test_results_epoch60.pkl ssm-ssm all aligned unnormy epoch60
# run_1208_0529_seed3407/test_results_epoch60.pkl encoder_only_ssm all aligned unnormy
# run_1208_1401_seed3407/test_results_epoch60.pkl encoder_only_ssm all aligned
# run_1409_1932_seed3407/ensembles_epochbest.npz encoder_only_lstm daymet aligned ****
# run_2910_1858_seed3407/deterministic_epoch60.npz seq2seq_ssm daymet 

input_path = "/home/yihan/diffusion_ssm/runs/run_2910_1858_seed3407/deterministic_epoch60.npz" # todo test_results_epoch19, test_results
print(f"Loading {input_path}")
#with open(input_path, 'rb') as f:
#    ens_dict = pickle.load(f)
ens_dict = np.load(input_path, allow_pickle=True)

# === Output dirs ===
output_dir = f"analysis/stats/"
os.makedirs(output_dir, exist_ok=True)

# Optional time series output
'''
timeseries_output_dir = os.path.join(output_dir, f"{experiment}_time_series/")
os.makedirs(timeseries_output_dir, exist_ok=True)

# Save each basin's full time series
for basin, df in ens_dict.items():
    out_fp = os.path.join(timeseries_output_dir, f"{basin}_timeseries.csv")
    # ensure index is meaningful, you may need df.index.name = "date"
    df.to_csv(out_fp, index=True, index_label="date")
    print(f"Saved time series for {basin} {out_fp}")
'''

# === Initialize ===
leadtime_stats = {lead: [] for lead in range(1, 9)} #todo

# === Main loop ===
for basin, df in ens_dict.items():
    print(f"Processing basin {basin}...")
    pdb.set_trace()

    if 'q_obs_t+1' not in df.columns:
        print(f"Skipping {basin}: no q_obs_t+1 column found.")
        continue

    # ensure dates are in chronological order
    df = df.sort_index()

    for lead in range(1,9):
        pred_col = f"qsim_t+{lead}"
        if pred_col not in df.columns:
            print(f"  Warning: {pred_col} missing for basin {basin}")
            continue

        # shift the 1-day-ahead obs up (lead-1) days to align with this lead
        obs_shifted = df['q_obs_t+1'].iloc[(lead-1):]
        pred_shifted =  df[pred_col].shift(lead-1)
        
        # build paired DataFrame and drop NaNs
        df_single = pd.DataFrame({
            'qsim': pred_shifted.clip(lower=0),
            'qobs': obs_shifted
        }).dropna()
        
        if df_single.empty:
            print(f"Skipping {basin} lead {lead}: no valid pairs after shift")
            continue

        # === Compute metrics ===
        obs5, sim5         = get_quant(df_single, 0.05)
        obs95, sim95       = get_quant(df_single, 0.95)
        obs0, sim0         = zero_freq(df_single)
        obsH, simH         = high_flows(df_single)
        obsL, simL         = low_flows(df_single)
        e_fhv_01            = FHV(df_single, 0.1) # percentile value, i.e., 0.1%
        e_flv              = FLV(df_single, 0.3) # not percentile value, i.e., 0.3 means 30%
        e_nse              = nse(df_single)
        e_nse_alpha        = alpha_nse(df_single)
        e_nse_beta         = beta_nse(df_single)
        e_kge, r, alpha, beta = kge(df_single)
        massbias_total, massbias_pos, massbias_neg = mass_balance(df_single)
        e_bias             = bias(df_single)
        e_stdev_rat        = stdev_rat(df_single)
        obsFDC, simFDC     = flow_duration_curve(df_single)

        leadtime_stats[lead].append({
            'basin': basin,
            'lead_time': lead,
            'nse': e_nse,
            'alpha_nse': e_nse_alpha,
            'beta_nse': e_nse_beta,
            'kge': e_kge,
            'kge_r': r,
            'kge_alpha': alpha,
            'kge_beta': beta,
            'fhv_01': e_fhv_01,
            'flv': e_flv,
            'massbias_total': massbias_total,
            'massbias_pos': massbias_pos,
            'massbias_neg': massbias_neg,
            'bias': e_bias,
            'stdev': e_stdev_rat,
            'obs5': obs5,
            'sim5': sim5,
            'obs95': obs95,
            'sim95': sim95,
            'obs0': obs0,
            'sim0': sim0,
            'obsL': obsL,
            'simL': simL,
            'obsH': obsH,
            'simH': simH,
            'obsFDC': obsFDC,
            'simFDC': simFDC
        })

# === Write CSVs ===
for lead, stats in leadtime_stats.items():
    df_stats = pd.DataFrame(stats)

    if df_stats.empty:
        print(f"No valid stats for lead {lead}")
        continue

    mean_stats   = df_stats.mean(numeric_only=True)
    median_stats = df_stats.median(numeric_only=True)
    mean_stats['basin']   = 'mean'
    median_stats['basin'] = 'median'

    df_stats = pd.concat(
        [df_stats, 
         mean_stats.to_frame().T, 
         median_stats.to_frame().T],
        ignore_index=True
    )

    out_csv = os.path.join(output_dir, f"{experiment}_lead{lead}.csv")
    df_stats.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

print("All evaluation results saved.")
