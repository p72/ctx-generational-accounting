"""raw/ の全データについて出典・取得方法・バイト数・SHA-256 を raw/manifest.json に記録する。
  python3 make_manifest.py          # 作成・更新
  python3 make_manifest.py --check  # 記録と現在のファイルが一致するか検証（不一致なら終了コード1）
"""
import hashlib, json, os, sys

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
ESRI = "https://www.esri.cao.go.jp/jp/sna/data/data_list/kakuhou/files/h22/tables/"
IPSS = "https://www.ipss.go.jp/pp-zenkoku/j/zenkoku2017/db_zenkoku2017/s_tables/"
KAIGO = "https://www.mhlw.go.jp/toukei/saikin/hw/kaigo/kyufu/10/xls/"
ESTAT = "e-Stat API getStatsData（src/es.py 経由）statsDataId="

SOURCES = {
    "22i4.xls":  (ESRI + "22i4_jp.xls",  "平成22年度国民経済計算確報（2005年基準・93SNA）制度部門別所得支出勘定 一般政府"),
    "22i5.xls":  (ESRI + "22i5_jp.xls",  "同 家計（個人企業を含む）"),
    "22c3.xls":  (ESRI + "22c3_jp.xls",  "同 制度部門別資本調達勘定 一般政府"),
    "22s6.xls":  (ESRI + "22s6_jp.xls",  "同 付表6 一般政府の部門別勘定（参照用）"),
    "22s7.xls":  (ESRI + "22s7_jp.xls",  "同 付表7 一般政府の機能別支出"),
    "22s9.xls":  (ESRI + "22s9_jp.xls",  "同 付表9 一般政府から家計への移転の明細表（社会保障関係）"),
    "22s10.xls": (ESRI + "22s10_jp.xls", "同 付表10 社会保障負担の明細表"),
    "ipss_1-9.xls":  (IPSS + "xls/1-9.xls", "社人研 平成29年推計 表1-9 男女年齢各歳別人口（出生中位・死亡中位）2015〜2065"),
    "ipss_s1-9.xls": (IPSS + "s1-9.xls",    "同 参考表1-9（2066〜2115）"),
    "ipss_pop.csv":  ("src/parse_ipss.py により上記2ファイルから作成", "各歳人口（年×年齢）"),
    "kaigo10_toukei.xls": (KAIGO + "toukei.xls", "介護給付費実態調査 平成22年度 統計表"),
    "kaigo10_hyo.xls":    (KAIGO + "hyo.xls",    "介護給付費実態調査 平成22年度 結果の概要の表"),
    "kaigo2010_recipients.csv": ("src/parse_kaigo.py により kaigo10_toukei.xls 統計表3-1 から作成",
                                 "年齢階級・要介護度別受給者数（平成22年11月審査分）"),
    "census2010.csv":     (ESTAT + "0003410439 cdTab=020 cdCat01=000 cdTime=2010000000", "平成22年国勢調査 年齢各歳別人口"),
    "census2010_emp.csv": (ESTAT + "0003033025 cdCat01=000 cdCat03=000 cdCat04=000,002,007 cdArea=00000",
                           "平成22年国勢調査 年齢5歳階級別 15歳以上人口・就業者・雇用者"),
    "nsfie_detail.csv": (ESTAT + "0003424369 cdCat01=0 cdCat03=0（収支8項目＋世帯数・人員・有業人員）",
                         "平成26年全国消費実態調査 世帯主年齢階級・世帯区分別 収支（総世帯）"),
    "nsfie_stock.csv":  (ESTAT + "0003132154 cdCat03=07005 cdCat04=02", "平成26年全国消費実態調査 世帯主年齢階級別 資産（総世帯）"),
    "nsfie_flow.csv":   (ESTAT + "0003126809 cdCat02=07001", "平成26年全国消費実態調査 フロー編第43表（現行モデルでは未使用）"),
    "medexp2014.csv":   (ESTAT + "0003356094 cdTab=180 cdCat01=100 cdCat02=100 cdTime=2014100000",
                         "国民医療費 年齢階級別 人口一人当たり（2014年度）"),
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def build():
    files = []
    for name in sorted(os.listdir(RAW)):
        if name == "manifest.json":
            continue
        p = os.path.join(RAW, name)
        src, desc = SOURCES.get(name, ("（出典未登録）", ""))
        files.append(dict(file=name, source=src, description=desc, bytes=os.path.getsize(p), sha256=sha256(p)))
    return files

if __name__ == "__main__":
    path = os.path.join(RAW, "manifest.json")
    if "--check" in sys.argv:
        rec = {f["file"]: f for f in json.load(open(path, encoding="utf-8"))["files"]}
        now = {f["file"]: f for f in build()}
        bad = [n for n in rec if n not in now or now[n]["sha256"] != rec[n]["sha256"]]
        new = [n for n in now if n not in rec]
        print(f"記録 {len(rec)} 件 / 不一致 {len(bad)} 件 / 未記録 {len(new)} 件")
        for n in bad: print("  不一致:", n)
        for n in new: print("  未記録:", n)
        sys.exit(1 if bad or new else 0)
    files = build()
    json.dump({"note": "raw/ のデータの出典と SHA-256。--check で照合できる", "files": files},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    unreg = [f["file"] for f in files if f["source"].startswith("（出典未登録")]
    print(f"{len(files)} ファイルを記録" + (f"（出典未登録: {unreg}）" if unreg else ""))
