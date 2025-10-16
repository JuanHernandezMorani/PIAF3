"""
Ejecuta ablations: RGB-only vs RGB+PBR vs RGB+PBR+FiLM (+/- aux). Exporta tablas.
"""
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=str)
    ap.add_argument("--imgsz", default=512, type=int)
    ap.add_argument("--epochs", default=50, type=int)
    args = ap.parse_args()
    print("[ablation] TODO: implementar corridas y export de resultados CSV.")

if __name__ == "__main__":
    main()
