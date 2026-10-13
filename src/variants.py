import numpy as np, pandas as pd, ga_model as m, paper, copy
P=m.load_population()
base_items=copy.deepcopy(m.ITEMS); G0=m.G_COLLECTIVE
def run(mode="head", edu="benefit", gmode="gross", reforms=True):
    m.ITEMS=copy.deepcopy(base_items); m.G_COLLECTIVE=G0
    if edu!="benefit":
        m.ITEMS=[it for it in m.ITEMS if it[0]!="現物：教育"]
        if edu=="G": m.G_COLLECTIVE=G0+18025.5
    prof=m.load_profiles(mode); H=m.per_capita_profiles(P.loc[2010].values,prof)
    out={}
    for sc in ["base","delay","keep8"]:
        t,r=m.table(sc,P,H,reforms=reforms,g_mode=gmode); out[sc]=(t,r)
    t,r=out["base"]
    t2=paper.table2()
    rmse=np.sqrt(((t["純負担"].values-t2.net.values)**2).mean())
    return dict(mode=mode,edu=edu,G=gmode,reforms=reforms,
        debt=r["implicit_debt"]/1e12, d0=t["純負担"].iloc[0], future=r["future"]/1e3, imb=r["imbalance"]*100,
        dDelay=(out["delay"][1]["implicit_debt"]-r["implicit_debt"])/1e12,
        dKeep8=(out["keep8"][1]["implicit_debt"]-r["implicit_debt"])/1e12,
        dFut=(out["delay"][1]["future"]-r["future"])/1e3, rmse_net=rmse,
        ben0=t["受益"].iloc[0], bur0=t["負担"].iloc[0])
rows=[]
for mode in ["head","percapita"]:
    for edu in ["benefit","G","none"]:
        for gm in ["gross","net"]:
            rows.append(run(mode,edu,gm))
rows.append(run("head","G","gross",reforms=False))
df=pd.DataFrame(rows); pd.set_option("display.width",250)
print("論文: debt 1472.5, 0歳 10230, future 72136, imb 605.1, dDelay 10.7, dKeep8 107.2, dFut 312, ben0 12480, bur0 22710")
print(df.round(1).to_string(index=False))
df.to_csv("../out_variants.csv",index=False,encoding="utf-8-sig")
