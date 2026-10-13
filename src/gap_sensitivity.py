"""40〜60歳の純負担のズレの原因を探る感応度分析
賃金プロファイル・年金改革（マクロ経済スライド・支給開始年齢）・介護給付の配分の前提を変えて、論文表2との距離を測る。
出力: ../out_gap_sensitivity.csv
"""
import numpy as np, pandas as pd, ga_model as m, paper

P = m.load_population()
t2 = paper.table2()
MID = [40, 45, 50, 55, 60]

def run(wage="earner", slide=0.009, until=2043, elig=True, contrib=True, care="ss_other"):
    m.WAGE_MODE, m.SLIDE_RATE, m.SLIDE_UNTIL, m.ELIG_ON, m.CARE_MODE = wage, slide, until, elig, care
    H = m.per_capita_profiles(P.loc[m.BASE].values, m.load_profiles("head"))
    cm = m.contrib_multiplier
    if not contrib:
        m.contrib_multiplier = lambda y: 1.0
    t, r = m.table("base", P, H)
    rd, _ = m.table("delay", P, H)
    rdl = m.generational_accounts("delay", P=P, H=H)
    m.contrib_multiplier = cm
    gap = t["純負担"] - t2.net
    return {
        "賃金": wage, "スライド率": slide, "スライド終了": until, "開始年齢引上げ": elig, "保険料引上げ": contrib, "介護の配分": care,
        "40-60歳の平均ズレ(千円)": gap.loc[MID].mean(),
        "全年齢RMSE(千円)": np.sqrt((gap ** 2).mean()),
        "潜在的債務(兆円)": r["implicit_debt"] / 1e12,
        "不均衡(%)": r["imbalance"] * 100,
        "延期の債務増(兆円)": (rdl["implicit_debt"] - r["implicit_debt"]) / 1e12,
    }

cases = [
    dict(),                                            # 現行の採用ケース
    dict(slide=0.0),                                   # スライドなし
    dict(until=2025), dict(until=2060), dict(until=2400),
    dict(slide=0.012), dict(slide=0.015),
    dict(elig=False), dict(contrib=False),
    dict(wage="household"),
    dict(wage="household", slide=0.0),
    dict(wage="household", until=2400),
    dict(wage="household", slide=0.012, until=2400),
    dict(care="age"),
    dict(care="age", wage="household"),
]
df = pd.DataFrame([run(**c) for c in cases])
m.WAGE_MODE, m.SLIDE_RATE, m.SLIDE_UNTIL, m.ELIG_ON, m.CARE_MODE = "earner", 0.009, 2043, True, "ss_other"
pd.set_option("display.width", 250)
print("論文: 潜在的債務 1472.5兆円 / 不均衡 605.1% / 延期の債務増 10.7兆円")
print(df.round({c: 1 for c in df.columns if c != "スライド率"}).to_string(index=False))
df.to_csv("../out_gap_sensitivity.csv", index=False, encoding="utf-8-sig")
