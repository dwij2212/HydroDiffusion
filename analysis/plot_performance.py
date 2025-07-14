import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# 1)  collect only CSVs that contain “generic_diffusion” but *not* “flexschdlr”
# ------------------------------------------------------------------
stats_dir = "/home/yihan/diffusion_ssm/analysis/analysis/stats"

# per batch versus per epoch scheduler
# "generic_diffusion_ssm_flexscdlr_lr4e-4"
# "generic_diffusion_ssm_v2",
# "generic_diffusion_lstm_concatprcp",
# "generic_diffusion_lstm_perbatch",

# param grouping 
# "generic_diffusion_ssm_flexschlr_lr4e-4_allcond0_groupparams"
# "generic_diffusion_ssm_v2_flexscdlr_lr4e-4_allcond0_uniformgroups"

# conditioning of ssm
# "generic_diffusion_ssm_flexscdlr_lr4e-4"
# "generic_diffusion_ssm_v2_flexscdlr_lr4e-4_allcond0_uniformgroups"

# ssm encoder
# "generic_diffusion_ssm_unet_flexschlr_lr1e-4_fullcond_uniformparams",
# "generic_diffusion_ssm_lstm_flexschlr_lr4e-4_fullcond_groupparams"
# "generic_diffusion_ssm_flexschlr_lr1e-4_fullcond_uniformparams_ssmencoder_epoch58"
# "generic_diffusion_ssm_ssm_flexschlr_lr4e-4_cond0_paramgroups"
# "generic_diffusion_ssm_flexschlr_lr2e-4_cond0_embmlp_dropout0.2_seed5534"
# "generic_diffusion_ssm_lr2e-4_cond0_embmlp_linearreadout_dropout0.2_seed5534"  

LABEL = {
    #"generic_diffusion_ssm_unet_flexschlr_lr1e-4_fullcond_uniformparams": "ssm-unet",
    #"generic_diffusion_ssm_lstm_flexschlr_lr4e-4_fullcond_groupparams": "ssm-lstm",
    "generic_diffusion_ssm_flexschlr_lr1e-4_fullcond_uniformparams_ssmencoder_epoch58": "ssm-ssm",
    #"generic_diffusion_ssm_ssm_flexschlr_lr4e-4_cond0_paramgroups": "ssm-ssm-2",
    #"generic_diffusion_lstm_perbatch": "lstm-lstm",
    "generic_diffusion_lstm_flexschlr_dropout0.5": "lstm-lstm",
    "generic_diffusion_ssm_flexschlr_lr2e-4_cond0_embmlp_dropout0.2_seed5534": "ssm-ssm-2",
    "generic_diffusion_ssm_lr2e-4_cond0_embmlp_dropout0.2_seed5534_encpower": "ssm-ssm-2-powerlawpooling",
    "diffusion_ssm_lr2e-4_cond0_silureadout_dropout0.2_seed5534_daymet_encpower": "ssm-ssm-2-powerlawpooling-daymet",

    "encdec_lstm_dropout0.5": "det encdec_lstm"
    #"generic_diffusion_unet_attention_film_v2_h64": "lstm-unet",
    #"generic_diffusion_ssm_flexschlr_lr4e-4_allcond0_groupparams": "lstm-ssm",
}
wanted = set(LABEL)                      # the long names we keep

# ────────────────────────────────────────────────────────────────
# 2)  collect the files we care about
# ────────────────────────────────────────────────────────────────
file_list = []
for fp in glob.glob(os.path.join(stats_dir, "*.csv")):
    stem = Path(fp).stem.lower()          # no extension
    stem = re.sub(r"_lead\d+$", "", stem) # drop “…_leadXX”
    if stem in wanted:
        file_list.append(fp)

# ────────────────────────────────────────────────────────────────
# 3)  group files by model-stem and sort by lead time
# ────────────────────────────────────────────────────────────────
model_files  = {}
lead_re = re.compile(r'lead(\d+)')

for f in file_list:
    base  = os.path.basename(f)
    model = re.sub(r'_lead\d+\.csv$', "", base.lower())   # same stem as above
    lead  = int(lead_re.search(base).group(1))
    model_files.setdefault(model, []).append((lead, f))

for m in model_files:
    model_files[m].sort(key=lambda t: t[0])               # sort by lead

# ────────────────────────────────────────────────────────────────
# 4)  plotting — use LABEL[model] for the legend text
# ────────────────────────────────────────────────────────────────
metric_keys   = ['nse', 'kge', 'kge_r']
metric_titles = ['Median NSE', 'Median KGE', 'Median Correlation']

STYLE = {
    'ssm' :  dict(linestyle='-',  marker='s', markersize=6),
    'unet': dict(linestyle='--', marker='^', markersize=6),
    'lstm': dict(linestyle='-.', marker='o', markersize=6),
}
family = lambda name: 'ssm' if 'ssm' in name else ('unet' if 'unet' in name else 'lstm')

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

for ax, metric, title in zip(axes, metric_keys, metric_titles):
    all_leads = set()

    for m_stem, files in model_files.items():
        leads, vals = [], []
        for lead, f in files:
            df = pd.read_csv(f, sep=None, engine="python")
            median = df.loc[df['basin'].str.lower() == 'median'].iloc[0]
            leads.append(lead)
            vals.append(float(median[metric]))
            all_leads.add(lead)

        ax.plot(
            leads,
            vals,
            label=LABEL[m_stem],         # ← short label instead of full name
            **STYLE[family(m_stem)]
        )

    ax.set_title(title)
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel("Performance")
    ax.set_xticks(sorted(all_leads))
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("skill_with_leadtime.png", dpi=600)
plt.close()

