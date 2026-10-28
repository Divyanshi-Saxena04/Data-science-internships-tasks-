import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportion_confint

sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
PAL = {"Lunch": "#E4A84B", "Dinner": "#2F5D8A"}
R = {}  # results store
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ 1. DATA
df = pd.read_csv("tips.csv")
df["tip_pct"] = df["tip"] / df["total_bill"] * 100
df["generous"] = (df["tip_pct"] >= 15).astype(int)
df["day"] = pd.Categorical(df["day"], ["Thur", "Fri", "Sat", "Sun"], ordered=True)

R["n"] = len(df)
R["missing"] = int(df.isna().sum().sum())
R["dupes"] = int(df.duplicated().sum())
R["desc"] = df[["total_bill", "tip", "tip_pct", "size"]].describe().round(2).to_dict()
R["counts"] = {c: df[c].value_counts().to_dict() for c in ["sex", "smoker", "day", "time"]}
R["tip_pct_skew"] = float(stats.skew(df.tip_pct))
R["tip_pct_kurt"] = float(stats.kurtosis(df.tip_pct))

# Fig 1: distributions
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
sns.histplot(df.total_bill, kde=True, ax=ax[0], color="#2F5D8A"); ax[0].set_title("Total bill ($)")
sns.histplot(df.tip, kde=True, ax=ax[1], color="#2F5D8A"); ax[1].set_title("Tip ($)")
sns.histplot(df.tip_pct, kde=True, ax=ax[2], color="#C0504D"); ax[2].set_title("Tip percentage (%)")
for a in ax: a.set_ylabel("Count"); a.set_xlabel("")
plt.tight_layout(); plt.savefig("fig1_distributions.png", dpi=200); plt.close()

# ----------------------------------------------- 2. H1: Lunch vs Dinner t-test
lunch = df.loc[df.time == "Lunch", "tip_pct"]
dinner = df.loc[df.time == "Dinner", "tip_pct"]
sh_l, sh_d = stats.shapiro(lunch), stats.shapiro(dinner)
lev = stats.levene(lunch, dinner, center="median")
t = stats.ttest_ind(lunch, dinner, equal_var=False)
diff = lunch.mean() - dinner.mean()
se = np.sqrt(lunch.var(ddof=1) / len(lunch) + dinner.var(ddof=1) / len(dinner))
dfw = se**4 / ((lunch.var(ddof=1) / len(lunch))**2 / (len(lunch) - 1)
               + (dinner.var(ddof=1) / len(dinner))**2 / (len(dinner) - 1))
ci = (diff - stats.t.ppf(.975, dfw) * se, diff + stats.t.ppf(.975, dfw) * se)
sp = np.sqrt(((len(lunch)-1)*lunch.var(ddof=1) + (len(dinner)-1)*dinner.var(ddof=1)) / (len(lunch)+len(dinner)-2))
d = diff / sp
mw = stats.mannwhitneyu(lunch, dinner, alternative="two-sided")
boot = [rng.choice(lunch, len(lunch)).mean() - rng.choice(dinner, len(dinner)).mean() for _ in range(10000)]
power = TTestIndPower().power(effect_size=abs(d), nobs1=len(lunch), ratio=len(dinner)/len(lunch), alpha=.05)
mde = TTestIndPower().solve_power(nobs1=len(lunch), ratio=len(dinner)/len(lunch), alpha=.05, power=.8)
R["h1"] = dict(n_l=len(lunch), n_d=len(dinner), mean_l=lunch.mean(), mean_d=dinner.mean(),
               sd_l=lunch.std(), sd_d=dinner.std(), med_l=lunch.median(), med_d=dinner.median(),
               shapiro_l=[sh_l.statistic, sh_l.pvalue], shapiro_d=[sh_d.statistic, sh_d.pvalue],
               levene=[lev.statistic, lev.pvalue], t=t.statistic, p=t.pvalue, df=dfw, diff=diff,
               ci=list(ci), d=d, mw_u=mw.statistic, mw_p=mw.pvalue,
               boot_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               power=power, mde=mde)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sns.boxplot(data=df, x="time", y="tip_pct", order=["Lunch", "Dinner"], palette=PAL, ax=ax[0], width=.5)
sns.stripplot(data=df, x="time", y="tip_pct", order=["Lunch", "Dinner"], color="k", alpha=.25, size=3, ax=ax[0])
ax[0].set_title("Tip % by meal time"); ax[0].set_xlabel(""); ax[0].set_ylabel("Tip (%)")
for k, g in [("Lunch", lunch), ("Dinner", dinner)]:
    sns.kdeplot(g, ax=ax[1], label=f"{k} (mean {g.mean():.2f}%)", color=PAL[k], fill=True, alpha=.3)
ax[1].set_title("Density of tip %"); ax[1].set_xlabel("Tip (%)"); ax[1].legend()
ax[2].hist(boot, bins=40, color="#7A7A7A", alpha=.8)
ax[2].axvline(0, color="red", ls="--", label="No difference")
ax[2].axvline(diff, color="k", label=f"Observed = {diff:.2f}")
ax[2].set_title("Bootstrap: Lunch − Dinner (pp)"); ax[2].set_ylim(0, 1150); ax[2].legend(loc="upper right", fontsize=8); ax[2].set_xlabel("Difference (pp)")
plt.tight_layout(); plt.savefig("fig2_h1_time.png", dpi=200); plt.close()

# ------------------------------------------- 3. H2: tip% across days (ANOVA)
groups = [g.tip_pct.values for _, g in df.groupby("day", observed=True)]
anova_m = smf.ols("tip_pct ~ C(day)", df).fit()
anova_t = sm.stats.anova_lm(anova_m, typ=2)
welch_a = stats.f_oneway(*groups)
lev2 = stats.levene(*groups, center="median")
kw = stats.kruskal(*groups)
ss_b, ss_t = anova_t.loc["C(day)", "sum_sq"], anova_t["sum_sq"].sum()
eta2 = ss_b / ss_t
resid_sh = stats.shapiro(anova_m.resid)
tuk = pairwise_tukeyhsd(df.tip_pct, df.day.astype(str), alpha=.05)
R["h2"] = dict(by_day=df.groupby("day", observed=True).tip_pct.agg(["count", "mean", "std", "median"]).round(3).to_dict("index"),
               F=anova_t.loc["C(day)", "F"], p=anova_t.loc["C(day)", "PR(>F)"],
               df1=int(anova_t.loc["C(day)", "df"]), df2=int(anova_t.loc["Residual", "df"]),
               eta2=eta2, levene=[lev2.statistic, lev2.pvalue], resid_shapiro=[resid_sh.statistic, resid_sh.pvalue],
               kw=[kw.statistic, kw.pvalue], tukey_min_p=float(min(tuk.pvalues)))
R["h2"]["tukey"] = [dict(g1=str(r[0]), g2=str(r[1]), diff=float(r[2]), p=float(r[3]), lo=float(r[4]), hi=float(r[5]), rej=bool(r[6]))
                    for r in tuk._results_table.data[1:]]

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sns.boxplot(data=df, x="day", y="tip_pct", palette="Blues", ax=ax[0])
ax[0].set_title("Tip % by day"); ax[0].set_xlabel(""); ax[0].set_ylabel("Tip (%)")
m = df.groupby("day", observed=True).tip_pct.agg(["mean", "sem", "count"])
ax[1].errorbar(range(4), m["mean"], yerr=1.96 * m["sem"], fmt="o", capsize=6, color="#2F5D8A", ms=9)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(m.index); ax[1].axhline(df.tip_pct.mean(), ls="--", color="grey", label="Grand mean")
ax[1].set_title("Mean tip % with 95% CI"); ax[1].set_ylabel("Tip (%)"); ax[1].legend()
sm.qqplot(anova_m.resid, line="s", ax=ax[2], markersize=3); ax[2].set_title("ANOVA residual Q-Q plot")
plt.tight_layout(); plt.savefig("fig3_h2_day.png", dpi=200); plt.close()

# ------------------- 4. H3: chi-square — generous tipping (>=15%) vs. smoker
def chi(col):
    ct = pd.crosstab(df[col], df.generous)
    c, p, dof, exp = stats.chi2_contingency(ct, correction=False)
    cy = stats.chi2_contingency(ct, correction=True)
    v = np.sqrt(c / (ct.values.sum() * (min(ct.shape) - 1)))
    return dict(table=ct.values.tolist(), index=list(map(str, ct.index)), chi2=c, p=p, dof=dof, yates_chi2=cy[0], yates_p=cy[1],
                cramers_v=v, min_expected=float(exp.min()),
                prop=df.groupby(col, observed=True).generous.mean().to_dict())
R["h3_time"] = chi("time")
R["h3_smoker"] = chi("smoker")
R["h3_sex"] = chi("sex")
R["h3_day"] = chi("day")
ct = pd.crosstab(df.time, df.generous)
fisher = stats.fisher_exact(ct.values)
R["h3_time"]["fisher_p"] = fisher.pvalue; R["h3_time"]["fisher_or"] = fisher.statistic
props = df.groupby("time").generous.agg(["mean", "sum", "count"])
R["h3_time"]["ci"] = {k: list(proportion_confint(r["sum"], r["count"], method="wilson")) for k, r in props.iterrows()}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
pr = df.groupby("time").generous.mean().loc[["Lunch", "Dinner"]] * 100
cis = [R["h3_time"]["ci"][k] for k in ["Lunch", "Dinner"]]
ax[0].bar(pr.index, pr.values, color=[PAL[k] for k in pr.index], width=.5,
          yerr=[[pr.values[i] - cis[i][0]*100 for i in range(2)], [cis[i][1]*100 - pr.values[i] for i in range(2)]], capsize=6)
ax[0].set_ylim(0, 100); ax[0].set_ylabel("% of parties tipping ≥ 15%"); ax[0].set_title("Generous tippers by meal time (95% Wilson CI)")
for i, v in enumerate(pr.values): ax[0].text(i, 4, f"{v:.1f}%", ha="center", color="white", fontweight="bold")
pr2 = df.groupby("smoker").generous.mean() * 100
ax[1].bar(pr2.index, pr2.values, color=["#7A9E7E", "#B5651D"], width=.5)
ax[1].set_ylim(0, 100); ax[1].set_title("Generous tippers by smoking section"); ax[1].set_ylabel("% tipping ≥ 15%")
for i, v in enumerate(pr2.values): ax[1].text(i, 4, f"{v:.1f}%", ha="center", color="white", fontweight="bold")
plt.tight_layout(); plt.savefig("fig4_h3_chi2.png", dpi=200); plt.close()

# --------------------------- 5. H4: bill size vs tip (correlation/regression)
pear = stats.pearsonr(df.total_bill, df.tip)
spear = stats.spearmanr(df.total_bill, df.tip)
pear_pct = stats.pearsonr(df.total_bill, df.tip_pct)
spear_pct = stats.spearmanr(df.total_bill, df.tip_pct)
ols = smf.ols("tip ~ total_bill", df).fit()
ols_rb = smf.ols("tip ~ total_bill", df).fit(cov_type="HC3")
multi = smf.ols("tip ~ total_bill + size + C(time) + C(smoker) + C(sex)", df).fit(cov_type="HC3")
bp = sm.stats.het_breuschpagan(ols.resid, ols.model.exog)
r_ci = pear.confidence_interval(0.95)
R["h4"] = dict(pearson=[pear.statistic, pear.pvalue, r_ci.low, r_ci.high], spearman=[spear.statistic, spear.pvalue],
               pearson_pct=[pear_pct.statistic, pear_pct.pvalue], spearman_pct=[spear_pct.statistic, spear_pct.pvalue],
               b0=ols.params.iloc[0], b1=ols.params.iloc[1], r2=ols.rsquared, b1_ci=list(ols.conf_int().loc["total_bill"]),
               b1_ci_rb=list(ols_rb.conf_int().loc["total_bill"]), b1_p=ols.pvalues["total_bill"], bp=[bp[0], bp[1]], resid_n=len(ols.resid),
               multi=dict(r2=multi.rsquared, adj=multi.rsquared_adj,
                          coefs={k: dict(b=float(multi.params[k]), se=float(multi.bse[k]), p=float(multi.pvalues[k]),
                                         lo=float(multi.conf_int().loc[k, 0]), hi=float(multi.conf_int().loc[k, 1])) for k in multi.params.index}))

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sns.regplot(data=df, x="total_bill", y="tip", ax=ax[0], scatter_kws=dict(alpha=.5, s=18, color="#2F5D8A"), line_kws=dict(color="#C0504D"))
ax[0].set_title(f"Tip vs. bill (r = {pear.statistic:.2f})"); ax[0].set_xlabel("Total bill ($)"); ax[0].set_ylabel("Tip ($)")
sns.regplot(data=df, x="total_bill", y="tip_pct", ax=ax[1], scatter_kws=dict(alpha=.5, s=18, color="#2F5D8A"), line_kws=dict(color="#C0504D"), lowess=True)
ax[1].set_title("Tip % vs. bill (LOWESS)"); ax[1].set_xlabel("Total bill ($)"); ax[1].set_ylabel("Tip (%)"); ax[1].set_ylim(0, 45)
ax[2].scatter(ols.fittedvalues, ols.resid, s=14, alpha=.5, color="#2F5D8A"); ax[2].axhline(0, color="red", ls="--")
ax[2].set_title("Residuals vs. fitted"); ax[2].set_xlabel("Fitted tip ($)"); ax[2].set_ylabel("Residual")
plt.tight_layout(); plt.savefig("fig5_h4_regression.png", dpi=200); plt.close()

# --------------------------------------- 6. Multiple-testing summary (Holm)
from statsmodels.stats.multitest import multipletests
ps = {"H1 Welch t-test (time)": R["h1"]["p"], "H2 ANOVA (day)": R["h2"]["p"],
      "H3 chi-square (time)": R["h3_time"]["p"], "H4 Pearson (bill~tip)": R["h4"]["pearson"][1]}
rej, adj, _, _ = multipletests(list(ps.values()), method="holm")
R["holm"] = {k: dict(raw=float(v), adj=float(a), rej=bool(r)) for (k, v), a, r in zip(ps.items(), adj, rej)}

# Fig 6: correlation heatmap
fig, ax = plt.subplots(figsize=(5.2, 4.2))
sns.heatmap(df[["total_bill", "tip", "tip_pct", "size"]].corr(method="spearman"), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title("Spearman correlation matrix")
plt.tight_layout(); plt.savefig("fig6_corr.png", dpi=200); plt.close()


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o

json.dump(clean(R), open("results.json", "w"), indent=1)
print(json.dumps(clean(R), indent=1))

# ---------------- 7. Extras: TOST equivalence + outlier sensitivity -------------
from statsmodels.stats.weightstats import ttost_ind
tost = ttost_ind(lunch, dinner, -2, 2, usevar="unequal")
q1, q3 = df.tip_pct.quantile([.25, .75]); iqr = q3 - q1
mask = df.tip_pct.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
d2 = df[mask]
t2 = stats.ttest_ind(d2.loc[d2.time == "Lunch", "tip_pct"], d2.loc[d2.time == "Dinner", "tip_pct"], equal_var=False)
f2 = stats.f_oneway(*[g.tip_pct.values for _, g in d2.groupby("day", observed=True)])
r2 = stats.pearsonr(d2.total_bill, d2.tip_pct)
E2 = dict(tost_p=float(tost[0]), tost_lo_p=float(tost[1][1]), tost_hi_p=float(tost[2][1]), n_outliers=int((~mask).sum()),
          fence=[float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)], out_vals=df.loc[~mask, "tip_pct"].round(2).tolist(),
          t_p=float(t2.pvalue), t_stat=float(t2.statistic), f=float(f2.statistic), f_p=float(f2.pvalue),
          r_pct=float(r2.statistic), r_pct_p=float(r2.pvalue),
          diff2=float(d2.loc[d2.time=="Lunch","tip_pct"].mean()-d2.loc[d2.time=="Dinner","tip_pct"].mean()))
R["extras"] = E2
json.dump(clean(R), open("results.json", "w"), indent=1)
print(json.dumps(clean(E2), indent=1))
