"""
clean_dataset.py
Script de línea de comandos para limpiar el dataset de EyeSky.

Detecta y elimina:
- Archivos con formato no permitido.
- Archivos corruptos o ilegibles.
- Fotografías borrosas (varianza del Laplaciano menor al umbral).
- Fotografías duplicadas dentro del dataset.

Por seguridad, sin argumentos solo hace una simulación (no borra nada).

Uso:
    python clean_dataset.py
    python clean_dataset.py --threshold 60
    python clean_dataset.py --apply
    python clean_dataset.py --apply --yes
    python clean_dataset.py --apply --yes --no-dedupe
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.dataset_quality import MIN_SHARPNESS, clean_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revisa y limpia las fotografías de dataset/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python clean_dataset.py                       # simulación\n"
            "  python clean_dataset.py --threshold 60        # simulación con otro umbral\n"
            "  python clean_dataset.py --apply               # pide confirmación y borra\n"
            "  python clean_dataset.py --apply --yes         # borra sin preguntar\n"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Borra los archivos problemáticos. Sin esta opción solo simula.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="No pedir confirmación al usar --apply.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=MIN_SHARPNESS,
        help=f"Umbral mínimo de nitidez (por defecto: {MIN_SHARPNESS}).",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="No eliminar fotografías duplicadas.",
    )
    return parser.parse_args()


def print_report(result: dict, title: str) -> None:
    print("=" * 64)
    print(title)
    print("=" * 64)
    print(f"Modo:              {'SIMULACIÓN (no se borra nada)' if result['dry_run'] else 'LIMPIEZA REAL'}")
    print(f"Umbral de nitidez: {result['threshold']}")
    print(f"Archivos revisados: {result['scanned']}")
    print(f"Archivos conservados: {result['kept']}")
    print(f"Archivos problemáticos: {result['removed_count']}")

    if result["removed"]:
        print("\nDetalle:")
        for item in result["removed"]:
            sharpness = item.get("sharpness")
            extra = f" | nitidez: {sharpness}" if sharpness is not None else ""
            print(f"  - [{item['reason']}] {item['path']}{extra}")

    if result["errors"]:
        print("\nErrores al borrar:")
        for error in result["errors"]:
            print(f"  - {error}")

    print(f"\n{result['message']}")


def confirm_delete(count: int) -> bool:
    answer = input(
        f"\n¿Eliminar {count} archivo(s) del dataset? Esta acción no se puede deshacer. [s/N]: "
    )
    return answer.strip().lower() in {"s", "si", "sí", "y", "yes"}


def main() -> int:
    args = parse_args()

    if args.threshold < 0:
        print("Error: el umbral de nitidez no puede ser negativo.")
        return 2

    remove_duplicates = not args.no_dedupe
    common_args = {
        "threshold": args.threshold,
        "remove_duplicates": remove_duplicates,
    }

    if not args.apply:
        result = clean_dataset(dry_run=True, **common_args)
        print_report(result, "LIMPIEZA DE DATASET (SIMULACIÓN)")
        print("\nNo se borró ningún archivo.")
        print("Para eliminarlos, ejecuta: python clean_dataset.py --apply")
        return 0

    if not args.yes:
        preview = clean_dataset(dry_run=True, **common_args)
        print_report(preview, "VISTA PREVIA ANTES DE BORRAR")

        if preview["removed_count"] == 0:
            print("\nNo hay archivos problemáticos. No es necesario borrar nada.")
            return 0

        if not confirm_delete(preview["removed_count"]):
            print("\nOperación cancelada. No se borró ningún archivo.")
            return 1

    result = clean_dataset(dry_run=False, **common_args)
    print_report(result, "LIMPIEZA DE DATASET (APLICADA)")
    print("\nSiguiente paso recomendado:")
    print("  python register_faces.py")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
