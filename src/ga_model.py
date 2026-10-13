"""
佐藤康仁（2019）「消費税率引き上げ延期に伴う財政負担に関する世代会計分析」
（東北学院大学経済学論集 第191号）表2〜4 の再現モデル。

推計基準年 2010年度 / 成長率 g=1.5% / 利子率 r=5.0% / 寿命上限 D=94歳
データ:
  - 平成22年度国民経済計算確報（2000年基準）一般政府・家計の部門別勘定（ESRI）
  - 平成26年全国消費実態調査（e-Stat）世帯主の年齢階級別 収支・資産
  - 平成22年度国勢調査 各歳人口（e-Stat）、社人研 平成29年推計（出生中位・死亡中位）
  - 国民医療費 年齢階級別 一人当たり（2014年度、e-Stat）
"""
import re
import numpy as np, pandas as pd

RAW = "/home/claude/ga/raw/"
BASE = 2010
G, R = 0.015, 0.05
D = 94                    # 寿命上限
AGES = np.arange(D + 1)
T_END = 2400              # 予算制約の打ち切り年（r>g なので十分収束）

# ------------------------------------------------------------------
# 1. 人口：2010国勢調査 → 2011-14 線形補間 → 2015-2115 社人研 → 以降は定常成長
# ------------------------------------------------------------------
def load_population():
    c = pd.read_csv(RAW + "census2010.csv", dtype={"cat02_code": str})
    c = c[c.cat02_name.str.match(r"^[０-９0-9]+歳$|^１００歳以上$|^100歳以上$")]
    z2h = str.maketrans("０１２３４５６７８９", "0123456789")
    c["age"] = c.cat02_name.str.translate(z2h).str.extract(r"(\d+)").astype(int)
    p2010 = c.groupby("age")["value"].sum().reindex(range(106), fill_value=0).astype(float)
    ip = pd.read_csv(RAW + "ipss_pop.csv", index_col=0)
    ip.columns = ip.columns.astype(int)
    pop = {BASE: p2010.values}
    for s in range(2011, 2015):
        w = (s - BASE) / 5
        pop[s] = (1 - w) * p2010.values + w * ip.loc[2015].values
    for s in ip.index:
        pop[s] = ip.loc[s].values
    # 2116年以降：2110→2115 の年齢別変化率で外挿
    gr = (ip.loc[2115].values / np.maximum(ip.loc[2110].values, 1)) ** (1 / 5)
    for s in range(2116, T_END + 1):
        pop[s] = pop[s - 1] * gr
    P = pd.DataFrame(pop).T.sort_index()          # 行=年, 列=年齢0..105
    return P.loc[:, :D]                            # 94歳まで（D=94）

# ------------------------------------------------------------------
# 2. 配分基準データ（全消2014）を各歳プロファイルへ
# ------------------------------------------------------------------
FLOW_CLASS = [  # 世帯主年齢階級 → 各歳
    ("25歳未満", 0, 24), ("25～29歳", 25, 29), ("30～34歳", 30, 34), ("35～39歳", 35, 39),
    ("40～44歳", 40, 44), ("45～49歳", 45, 49), ("50～54歳", 50, 54), ("55～59歳", 55, 59),
    ("60～64歳", 60, 64), ("65～69歳", 65, 69), ("70～74歳", 70, 74), ("75歳以上", 75, 120)]
STOCK_CLASS = [("30歳未満", 0, 29), ("30～39歳", 30, 39), ("40～49歳", 40, 49), ("50～59歳", 50, 59),
               ("60～69歳", 60, 69), ("70歳以上", 70, 120)]

def _expand(series, classes):
    out = np.zeros(D + 1)
    for name, lo, hi in classes:
        out[lo:min(hi, D) + 1] = series[name]
    return out

DETAIL_CLASS = [("30歳未満", 0, 29), ("30～34", 30, 34), ("35～39", 35, 39), ("40～44", 40, 44),
                ("45～49", 45, 49), ("50～54", 50, 54), ("55～59", 55, 59), ("60～64", 60, 64),
                ("65～69", 65, 69), ("70～74", 70, 74), ("75～79", 75, 79), ("80～84", 80, 84),
                ("85歳以上", 85, 120)]
OTHER_WAGE_SHARE = 0.3   # 「その他の世帯」(自営業等)の勤め先収入＝勤労者世帯の3割と仮定

def load_profiles(mode="head"):
    """mode="head": 世帯の値を世帯主年齢＝本人年齢として20歳以上に帰属（子どもは0）
       mode="percapita": 世帯の値を成人数で割り、子どもには親世代の一人当たり消費を帰属"""
    """全消2014 総世帯：収入は勤労者・無職世帯のみ表章 → 世帯数でウェイト付けして全世帯値を合成"""
    d = pd.read_csv(RAW + "nsfie_detail.csv", dtype=str)
    d["value"] = d.value.astype(float)
    d["cls"] = d.cat05_name.str.replace("（５歳階級）", "", regex=False)
    def tab(tab_code, item=None):
        x = d[d.tab_code == tab_code]
        if item: x = x[x.cat04_code == item]
        return x.pivot_table(index="cls", columns="cat02_name", values="value").fillna(0)
    n = tab("07-2019")                                  # 世帯数
    mem = tab("08-2019"); u18 = tab("09-2019")
    wW, wN, wO = (n["勤労者世帯"] / n["全世帯"], n["無職世帯"] / n["全世帯"], n["その他の世帯"] / n["全世帯"])
    def all_hh(item, other="avg"):
        t = tab("01-2019", item)
        W, N = t.get("勤労者世帯", 0), t.get("無職世帯", 0)
        O = W * OTHER_WAGE_SHARE if other == "wage" else (wW * W + wN * N) / (wW + wN)
        return wW * W + wN * N + wO * O
    def col(series): return _expand(series, DETAIL_CLASS)
    members = col(mem["全世帯"]); adults = np.maximum(members - col(u18["全世帯"]), 1.0)
    adult = (AGES >= 20).astype(float)                 # 所得・資産系は20歳以上に帰属
    div = adults if mode == "percapita" else np.ones(D + 1)
    per_adult = lambda item, other="avg": col(all_hh(item, other)) / div * adult
    tab_cons = tab("01-2019", "2101")["全世帯"]

    # 消費：世帯の一人当たり消費。子ども(0-19歳)は「親世代（年齢+30）」の世帯の一人当たり消費
    cons_pc_head = col(tab_cons) / members
    cons = np.array([cons_pc_head[min(a + 30, D)] if a < 20 else cons_pc_head[a] for a in AGES])
    if mode == "head":
        cons = col(tab_cons) * adult

    s = pd.read_csv(RAW + "nsfie_stock.csv", dtype=str)
    s["value"] = s.value.astype(float)
    st = s.pivot_table(index="cat01_name", columns="cat02_code", values="value")
    sm = _expand(st["00030"], STOCK_CLASS)
    s_adults = np.maximum(sm, 1.0) if mode == "percapita" else np.ones(D + 1)
    housing = _expand(st["00230"], STOCK_CLASS) / s_adults * adult
    saving = _expand(st["00130"], STOCK_CLASS) / s_adults * adult
    debt = _expand(st["00200"], STOCK_CLASS) / s_adults * adult

    # 国民医療費（2014年度・一人当たり）
    m = pd.read_csv(RAW + "medexp2014.csv")
    med = np.zeros(D + 1)
    for _, r in m.iterrows():
        nm = r.cat03_name
        if nm == "総数": continue
        lo = int(nm.split("～")[0].replace("歳以上", "").replace("歳", ""))
        hi = lo + 4 if "～" in nm else D
        med[lo:min(hi, D) + 1] = r.value

    # 学校教育費（一人当たり・相対値の仮定：幼稚園0.3, 小1.0, 中1.15, 高1.1, 大学等0.25）
    edu = np.zeros(D + 1)
    edu[3:6] = 0.3; edu[6:12] = 1.0; edu[12:15] = 1.15; edu[15:18] = 1.1; edu[18:22] = 0.25

    # 勤め先収入〔就業率で修正〕：勤労者世帯の有業者1人当たり勤め先収入 × 2010国勢調査の年齢別雇用者率
    earn = tab("11-2019")
    w_per_earner = tab("01-2019", "310101")["勤労者世帯"] / earn["勤労者世帯"]
    w_per_earner["85歳以上"] = w_per_earner["80～84"]          # 85歳以上の勤労者世帯は標本僅少
    emp = pd.read_csv(RAW + "census2010_emp.csv", index_col=0)
    emp_rate = np.zeros(D + 1)
    for nm, r in emp.iterrows():
        mm = re.match(r"^(\d+)～(\d+)歳$|^(\d+)歳以上$", nm)
        if not mm: continue
        lo = int(mm.group(1) or mm.group(3)); hi = int(mm.group(2)) if mm.group(2) else D
        emp_rate[lo:min(hi, D) + 1] = r["（再掲）雇用者（役員を含む），総数"] / r["総数（労働力状態）"]
    wage = col(w_per_earner) * emp_rate

    pension = per_adult("310103021")
    ss_other = np.maximum(per_adult("31010302") - pension, 0)
    rent = col(tab("01-2019", "21010201")["全世帯"]) / div * adult
    return dict(
        wage=wage, prop_inc=per_adult("31010301"), gift=per_adult("310201"),
        pension=pension, ss_other=ss_other, cons=cons,
        rent_debt=rent + debt * 0.03 / 12, housing=housing, saving=saving,
        med=med, edu=edu, equal=np.ones(D + 1))

# ------------------------------------------------------------------
# 3. SNA（平成22年度確報, 10億円）→ 項目別総額と配分基準（表1）
#    符号: 負担(+) / 受益(-)
# ------------------------------------------------------------------
VAT = 12675.2                    # 付加価値型税（消費税, 税率5%）
PROPERTY_TAX = 8900.0            # 固定資産税（SNAに内訳なし → 地方財政統計ベースの概数で仮定）
TPI = 39865.8                    # 生産・輸入品に課される税
ITEMS = [  # (名称, 総額, 配分基準, 符号)
    ("固定資産税",           PROPERTY_TAX,               "housing",   +1),
    ("消費税（VAT）",        VAT,                        "cons",      +1),
    ("その他生産輸入品税",   TPI - VAT - PROPERTY_TAX,   "cons",      +1),
    ("補助金",               3092.2,                     "cons",      -1),
    ("財産所得受取",         6854.2,                     "rent_debt", +1),
    ("財産所得支払",         10008.6,                    "prop_inc",  -1),
    ("所得税等（労働分）",   24619.2,                    "wage",      +1),
    ("所得税等（資本分）",   37471.5 - 24619.2,          "saving",    +1),
    ("社会負担",             57503.5,                    "wage",      +1),
    ("その他経常移転受取",   58743.8 - 57494.9,          "equal",     +1),
    ("社会給付：年金",       48888.5,                    "pension",   -1),
    ("社会給付：医療",       2610.9,                     "med",       -1),
    ("社会給付：その他",     67218.7 - 48888.5 - 2610.9, "ss_other",  -1),
    ("その他経常移転支払",   65145.5 - 57494.9,          "cons",      -1),
    ("現物：医療",           37142.1 - 7251.2,           "med",       -1),
    ("現物：介護等",         7251.2,                     "ss_other",  -1),
    ("資本税（相続贈与）",   1250.4,                     "gift",      +1),
]
# 非移転支出（G）：集合消費＋個別的非市場財＋公的純投資（固定資本形成−固定資本減耗＋在庫＋土地）＋その他資本移転（純）
# 個別的非市場財（≒教育, 18.0兆円）は移転支出でなく政府消費として G に含める（論文脚注7）
EDU_INDIVIDUAL = 18025.5
G_COLLECTIVE = 40603.3 + EDU_INDIVIDUAL
G_INVEST_GROSS = 15217.5 - 21.1 + 1572.0
DEPRECIATION = 14353.4
CAPTRANS_NET = 4516.7 - (7054.8 - 1250.4)
NET_DEBT = 543_000.0             # 2010年度末 一般政府純債務（論文の約543兆円）

def per_capita_profiles(P2010, prof):
    """各項目を (3) 式で各歳へ配分 → 一人当たり（円）"""
    out = {}
    for name, total, base, sign in ITEMS:
        a = prof[base]
        out[name] = sign * total * 1e9 * a / (a * P2010).sum()   # h_ij = H_i α_ij / Σ α P
    return pd.DataFrame(out, index=AGES)

# ------------------------------------------------------------------
# 4. 制度変更（既に決まっているもの）
# ------------------------------------------------------------------
def vat_multiplier(year, scenario):
    """基準年の消費税(5%)に掛ける倍率（年度・年途中改定は按分）"""
    if year < 2014: return 1.0
    if scenario == "keep8": return 1.6
    if scenario == "base":   # 2017年4月に10%
        return 1.6 if year < 2017 else 2.0
    if scenario == "delay":  # 2019年10月に10%（2019年度は半年）
        return 1.6 if year < 2019 else (1.8 if year == 2019 else 2.0)
    raise ValueError(scenario)

PENSION_SHARE_OF_CONTRIB = (30990.1 + 1029.8 + 2897.3 + 426.9) / 57503.5   # 社会負担のうち年金分

def contrib_multiplier(year):
    """平成16年改正：厚生年金保険料率 16.058%(2010)→18.3%(2017) を年0.354%ずつ"""
    rate = min(16.058 + 0.354 * (year - BASE), 18.3)
    return 1 + PENSION_SHARE_OF_CONTRIB * (rate / 16.058 - 1)

def slide_multiplier(year, slide=0.009, until=2043):
    """マクロ経済スライド：2015年度〜until年度、一人当たり年金の伸びを年slide分抑制"""
    k = min(max(year - 2014, 0), until - 2014)
    return (1 - slide) ** k

def eligibility(age, year):
    """報酬比例部分の支給開始年齢引上げ（男子1953年度生〜、女子は5年遅れ）→ 60-64歳の受給割合"""
    if age < 60 or age >= 65: return 1.0
    c = year - age
    def start(c):
        return 60 if c <= 1953 else min(65, 60 + (c - 1952) // 2)
    em = 1.0 if age >= start(c) else 0.0
    ew = 1.0 if age >= start(c - 5) else 0.0
    e = 0.5 * (em + ew)
    e14 = 0.5 * ((1.0 if age >= start(2014 - age) else 0.0) + (1.0 if age >= start(2014 - age - 5) else 0.0))
    base_floor = 0.3                                  # 共済・繰上げ・障害・遺族など
    return (base_floor + (1 - base_floor) * e) / (base_floor + (1 - base_floor) * e14)

# ------------------------------------------------------------------
# 5. 世代会計
# ------------------------------------------------------------------
def build_NT(H, scenario, reduced_rate=0.0, reforms=True, years=None, part="net"):
    sign = {n: s for n, _, _, s in ITEMS}
    cols = [c for c in H.columns if part == "net" or (part == "burden") == (sign[c] > 0)]
    NT = np.zeros((len(years), D + 1))
    for i, y in enumerate(years):
        gf = (1 + G) ** (y - BASE)
        row = np.zeros(D + 1)
        for c in cols:
            v = H[c].values.copy()
            if c == "消費税（VAT）":
                v = v * (1 + (vat_multiplier(y, scenario) - 1) * (1 - reduced_rate))
            if reforms and c == "社会負担":
                v = v * contrib_multiplier(y)
            if reforms and c == "社会給付：年金":
                v = v * slide_multiplier(y) * np.array([eligibility(a, y) for a in AGES])
            row += v
        NT[i] = row * gf
    return NT

def generational_accounts(scenario="base", reduced_rate=0.0, P=None, H=None,
                          reforms=True, g_mode="net", part="net"):
    years = np.arange(BASE, T_END + 1)
    gfac = (1 + G) ** (years - BASE)
    disc = (1 + R) ** -(years - BASE)
    NT = build_NT(H, scenario, reduced_rate, reforms, years, part)
    Pm = P.reindex(years).values

    # 現在世代（2010年に a 歳）の一人当たり生涯純負担（2010年価値）
    gen = np.zeros(D + 1)
    for a in AGES:
        n = min(D - a + 1, len(years))
        idx = np.arange(n)
        gen[a] = (NT[idx, a + idx] * Pm[idx, a + idx] / Pm[0, a] * disc[idx]).sum()
    gen = pd.Series(gen, index=AGES)
    if part != "net":
        return dict(gen=gen)

    PV_living = (gen * Pm[0, :]).sum()
    g_base = G_COLLECTIVE + CAPTRANS_NET + (G_INVEST_GROSS - (DEPRECIATION if g_mode == "net" else 0))
    Gs = g_base * 1e9 * gfac * (Pm.sum(axis=1) / Pm[0].sum())
    PV_G = (Gs * disc).sum()
    # 将来世代：現行制度のままの生涯純負担（b年生まれ・出生時点価値）
    fut_cur = 0.0; fut_weight = 0.0
    for i in range(1, len(years)):
        n = min(D + 1, len(years) - i)
        idx = np.arange(n)
        acc = (NT[i + idx, idx] * Pm[i + idx, idx] / Pm[i, 0] * disc[i + idx] / disc[i]).sum()
        fut_cur += acc * Pm[i, 0] * disc[i]
        fut_weight += gfac[i] * Pm[i, 0] * disc[i]
    implicit_debt = PV_G + NET_DEBT * 1e9 - PV_living     # 政府の潜在的債務残高
    extra = (implicit_debt - fut_cur) / fut_weight          # 将来世代1人あたり追加負担
    return dict(gen=gen, implicit_debt=implicit_debt, extra=extra, future=gen[0] + extra,
                imbalance=extra / gen[0], PV_G=PV_G, PV_living=PV_living, fut_cur=fut_cur)

def table(scenario, P, H, **kw):
    """論文表2形式（2010年時点の年齢, 負担, 受益, 純負担：千円）"""
    b = generational_accounts(scenario, P=P, H=H, part="burden", **kw)["gen"]
    bn = -generational_accounts(scenario, P=P, H=H, part="benefit", **kw)["gen"]
    r = generational_accounts(scenario, P=P, H=H, **kw)
    t = pd.DataFrame({"負担": b, "受益": bn, "純負担": r["gen"]}).loc[::5] / 1000
    return t.round(1), r

if __name__ == "__main__":
    P = load_population(); prof = load_profiles()
    H = per_capita_profiles(P.loc[BASE].values, prof)
    pd.set_option("display.width", 200)
    for sc in ["base", "delay", "keep8"]:
        t, r = table(sc, P, H)
        print(f"\n== {sc}: 潜在的債務 {r['implicit_debt']/1e12:,.1f}兆円  将来世代 {r['future']/1e3:,.0f}千円  "
              f"不均衡 {r['imbalance']*100:.1f}%  (PV_G {r['PV_G']/1e12:,.0f} / PV現在世代 {r['PV_living']/1e12:,.0f} / 将来現行 {r['fut_cur']/1e12:,.0f}兆円)")
        print(t.to_string())
