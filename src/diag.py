import numpy as np, pandas as pd, ga_model as m
P=m.load_population(); prof=m.load_profiles(); H=m.per_capita_profiles(P.loc[m.BASE].values, prof)
pd.set_option("display.width",250)
print((H.iloc[[0,5,10,15,20,30,40,50,60,70,80,90]]/1000).round(0).to_string())
# PV by item for 0歳 and 40歳 in base scenario
def pv_item(col, a):
    Hz=H*0; Hz[col]=H[col]; 
    return m.generational_accounts("base",P=P,H=Hz)["gen"][a]
for a in [0,20,40,65]:
    print(a, {c: round(pv_item(c,a)/1e4) for c in H.columns})
