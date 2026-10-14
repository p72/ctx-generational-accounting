"""検証スクリプト（GPT版の検証の作法を移植）

1. 年齢を隠す検証（held-out）
   論文に書いていない前提（世帯値の割り振り・教育の扱い・公共投資・賃金・介護）を、
   表2の一部の年齢だけを見て選び、隠した年齢でどれだけ当たるかを測る。
   前提選びが表2に過剰に合わせ込んでいないかの確認になる。
2. 表3・表4での検証（out-of-sample）
   モデルも前提選びも表2しか見ていないので、延期ケース（表3）・8%据え置き（表4）は手付かずの検証データ。
   基本ケースからの「変化幅」を年齢別に比べる。

出力: ../out_validation_specs.csv, ../out_validation_heldout.csv, ../out_validation_table34.csv, ../out_validation_summary.json
"""
import copy, itertools, json
import numpy as np, pandas as pd
import ga_model as m, paper

P = m.load_population()
T2 = paper.table2()
T3 = pd.Series(paper.T3_net, index=paper.AGES5)
T4 = pd.Series(paper.T4_net, index=paper.AGES5)
BASE_ITEMS, BASE_G = copy.deepcopy(m.ITEMS), m.G_COLLECTIVE
DEFAULT = dict(mode="head", edu="G", gmode="net", wage="earner", care="ss_other")
GRID = dict(mode=["head", "percapita"], edu=["G", "benefit", "none"], gmode=["net", "gross"],
            wage=["earner", "household"], care=["ss_other", "age"])
FOLDS = {"A（5,15,…,85歳を隠す）": [5, 15, 25, 35, 45, 55, 65, 75, 85],
         "B（0,10,…,90歳を隠す）": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]}

def set_spec(mode, edu, gmode, wage, care):
    m.ITEMS, m.G_COLLECTIVE = copy.deepcopy(BASE_ITEMS), BASE_G
    if edu == "benefit":
        m.ITEMS = m.ITEMS + [("現物：教育", m.IND_EDU, "edu", -1)]; m.G_COLLECTIVE = BASE_G - m.IND_EDU
    elif edu == "none":
        m.G_COLLECTIVE = BASE_G - m.IND_EDU
    m.WAGE_MODE, m.CARE_MODE = wage, care
    H = m.per_capita_profiles(P.loc[m.BASE].values, m.load_profiles(mode))
    return H

def reset():
    m.ITEMS, m.G_COLLECTIVE, m.WAGE_MODE, m.CARE_MODE = copy.deepcopy(BASE_ITEMS), BASE_G, "earner", "ss_other"

def rel_rmse(est, ref):
    return float(np.sqrt((((est - ref) / ref) ** 2).mean()) * 100)

def evaluate(spec):
    H = set_spec(**spec)
    t, r = m.table("base", P, H, g_mode=spec["gmode"])
    return t, r, H

# ---------------- 1. 全仕様を計算 ----------------
rows, tables = [], {}
for vals in itertools.product(*GRID.values()):
    spec = dict(zip(GRID.keys(), vals))
    t, r, _ = evaluate(spec)
    key = tuple(vals); tables[key] = t
    rows.append({**spec, "不均衡(%)": r["imbalance"] * 100, "潜在的債務(兆円)": r["implicit_debt"] / 1e12,
                 "純負担RMSE(千円)": float(np.sqrt(((t["純負担"] - T2.net) ** 2).mean())),
                 "負担・受益の相対RMSE(%)": rel_rmse(pd.concat([t["負担"], t["受益"]]), pd.concat([T2.burden, T2.benefit]))})
specs = pd.DataFrame(rows).sort_values("純負担RMSE(千円)")

# ---------------- 2. 年齢を隠す検証 ----------------
held = []
for fold, test_ages in FOLDS.items():
    train_ages = [a for a in paper.AGES5 if a not in test_ages]
    def rmse(key, ages): return float(np.sqrt(((tables[key]["純負担"].loc[ages] - T2.net.loc[ages]) ** 2).mean()))
    best = min(tables, key=lambda k: rmse(k, train_ages))
    oracle = min(tables, key=lambda k: rmse(k, test_ages))
    dflt = tuple(DEFAULT.values())
    held.append({"分割": fold, "学習で選んだ仕様": "/".join(best),
                 "学習RMSE(千円)": rmse(best, train_ages), "検証RMSE(千円)": rmse(best, test_ages),
                 "検証で最良の仕様": "/".join(oracle), "検証の最良RMSE(千円)": rmse(oracle, test_ages),
                 "既定仕様の検証RMSE(千円)": rmse(dflt, test_ages)})
held = pd.DataFrame(held)

# ---------------- 3. 表3・表4での検証 ----------------
def delay_check(spec, label):
    H = set_spec(**spec)
    out = {}
    res = {sc: m.table(sc, P, H, g_mode=spec["gmode"]) for sc in ["base", "delay", "keep8"]}
    d3 = res["delay"][0]["純負担"] - res["base"][0]["純負担"]
    d4 = res["keep8"][0]["純負担"] - res["base"][0]["純負担"]
    p3, p4 = T3 - T2.net, T4 - T2.net
    df = pd.DataFrame({"論文_延期の変化": p3, f"{label}_延期の変化": d3,
                       "論文_据置の変化": p4, f"{label}_据置の変化": d4}).round(1)
    rb = res["base"][1]
    summ = {
        "延期_年齢別変化RMSE(千円)": float(np.sqrt(((d3 - p3) ** 2).mean())),
        "延期_変化の相関": float(np.corrcoef(d3, p3)[0, 1]),
        "延期_最大軽減の年齢": int(d3.idxmin()), "論文_最大軽減の年齢": int(p3.idxmin()),
        "延期_将来世代の増加(千円)": (res["delay"][1]["future"] - rb["future"]) / 1e3, "論文_延期_将来世代の増加(千円)": 311.5,
        "延期_潜在的債務の増加(兆円)": (res["delay"][1]["implicit_debt"] - rb["implicit_debt"]) / 1e12, "論文_延期_債務増(兆円)": 10.7,
        "延期_不均衡の変化(pt)": (res["delay"][1]["imbalance"] - rb["imbalance"]) * 100, "論文_延期_不均衡の変化(pt)": 4.1,
        "据置_年齢別変化RMSE(千円)": float(np.sqrt(((d4 - p4) ** 2).mean())),
        "据置_変化の相関": float(np.corrcoef(d4, p4)[0, 1]),
        "据置_潜在的債務の増加(兆円)": (res["keep8"][1]["implicit_debt"] - rb["implicit_debt"]) / 1e12, "論文_据置_債務増(兆円)": 107.2,
        "据置_不均衡の変化(pt)": (res["keep8"][1]["imbalance"] - rb["imbalance"]) * 100, "論文_据置_不均衡の変化(pt)": 84.5,
    }
    return df, summ

df_def, s_def = delay_check(DEFAULT, "既定")
best_all = dict(zip(GRID.keys(), specs.iloc[0][list(GRID.keys())]))
df_best, s_best = delay_check(best_all, "表2最良")
reset()
t34 = pd.concat([df_def, df_best.drop(columns=["論文_延期の変化", "論文_据置の変化"])], axis=1)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("== 仕様（上位8 / 全%d）==" % len(specs)); print(specs.head(8).round(1).to_string(index=False))
print("※ 教育の扱い（edu）と公共投資（gmode）は G＝将来世代側にしか効かず、現在世代の純負担は同一になる。"
      "表2の年齢別の値ではこの2つを区別できない（同点は格子の先頭を採用）。区別は不均衡・将来世代の値でしかできない")
print("\n== 年齢を隠す検証 =="); print(held.round(0).to_string(index=False))
print("\n== 表3・表4（基本ケースからの変化、千円）=="); print(t34.T.to_string())
print("\n== 表3・表4 要約 ==")
print(pd.DataFrame({"既定仕様": s_def, "表2最良仕様": s_best}).round(2).to_string())

specs.to_csv("../out_validation_specs.csv", index=False, encoding="utf-8-sig")
held.to_csv("../out_validation_heldout.csv", index=False, encoding="utf-8-sig")
t34.to_csv("../out_validation_table34.csv", encoding="utf-8-sig")
json.dump({"default_spec": DEFAULT, "best_spec_on_table2": best_all,
           "table34_default": s_def, "table34_best": s_best,
           "heldout": held.to_dict(orient="records")},
          open("../out_validation_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
