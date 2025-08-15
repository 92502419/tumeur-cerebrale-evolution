# -*- coding: utf-8 -*-
import argparse
from pprint import pprint

# On réutilise les builders de main.py
from main import build_items_nested, build_items_flat, build_items_classic

def parse_args():
    ap = argparse.ArgumentParser(description="Sanity-check: lister les 3 premiers patients détectés")
    ap.add_argument("--data_root", type=str, default="data", help="Dossier racine des données")
    ap.add_argument("--scan_flat", action="store_true", help="Fichiers à plat")
    ap.add_argument("--scan_nested", action="store_true", help="Arborescence imbriquée <PID>/<DATE>/<file>")
    ap.add_argument("--modality", type=str, default="FLAIR", help="Modalité: FLAIR, T1, T2, T1CE, ...")
    ap.add_argument("--use_last", action="store_true", help="Prendre les 3 acquisitions les plus récentes")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.scan_nested:
        items = build_items_nested(args.data_root, modality=args.modality, use_last=args.use_last)
    elif args.scan_flat:
        items = build_items_flat(args.data_root, modality=args.modality, use_last=args.use_last)
    else:
        items = build_items_classic(args.data_root)
    print(f"Total patients détectés: {len(items)}")
    print("---- Aperçu (3 premiers) ----")
    for it in items[:3]:
        preview = {k: it.get(k) for k in ["t1","t2","t3","m1","m2","m3"]}
        pprint(preview)

if __name__ == "__main__":
    main()
