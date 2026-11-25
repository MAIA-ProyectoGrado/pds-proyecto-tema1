"""Reproduce el acuerdo nominal entre los dos anotadores del test de SCIF.

Desde la raíz del repositorio:
    python3 scripts/calcular_kappa.py
    python3 scripts/calcular_kappa.py --hoja
    python3 scripts/calcular_kappa.py --hoja revision.xlsx --referencia test.csv

El modo predeterminado lee la matriz publicada. El modo --hoja reconstruye la
matriz desde la hoja de David y la columna human_label de Juan David. La salida
JSON incluye los conteos, el acuerdo global y el kappa de cada clase frente al resto.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import posixpath
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.labels import CANONICAL_LABELS  # noqa: E402

MATRIX = ROOT / "docs/anotacion/matriz_acuerdo_humano.csv"
SHEET = ROOT / "dataset/anotacion/hoja_anotacion_test.xlsx"
TEST = ROOT / "dataset/test_enriquecido_top3_chunks.csv"
FIELDS = {"contexto_de_cita": "citation_context", "seccion": "rhetorical_section",
          "titulo_citado": "cited_paper_title", "abstract_citado": "cited_paper_abstract"}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def excel_rows(archive, title):
    book = ET.fromstring(archive.read('xl/workbook.xml'))
    sheet = next((s for s in book.findall('m:sheets/m:sheet', NS) if s.get('name') == title), None)
    if sheet is None:
        raise ValueError(f'Falta la pestaña {title}.')
    rel_id = sheet.get('{%s}id' % NS['r'])
    rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
    target = next(rel.get('Target') for rel in rels if rel.get('Id') == rel_id)
    name = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
    strings = []
    if 'xl/sharedStrings.xml' in archive.namelist():
        strings = [''.join(t.text or '' for t in s.findall('.//m:t', NS))
                   for s in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
    rows = []
    for row in ET.fromstring(archive.read(name)).findall('m:sheetData/m:row', NS):
        cells = {}
        for cell in row.findall('m:c', NS):
            ref = cell.get('r')
            col = re.match(r'[A-Z]+', ref).group()
            val = cell.findtext('m:v', '', NS)
            if title == 'Anotacion' and cell.find('m:f', NS) is not None:
                raise ValueError(f'La celda {ref} contiene una fórmula. Introduce la anotación como texto.')
            if cell.get('t') == 's':
                val = strings[int(val)]
            elif cell.get('t') == 'inlineStr':
                val = ''.join(t.text or '' for t in cell.findall('.//m:t', NS))
            cells[col] = val
        if any(cells.values()):
            rows.append((int(row.get('r')), cells))
    return rows


def read_annotations(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        rows = excel_rows(archive, 'Anotacion')
    if not rows:
        raise ValueError('La hoja de anotación está vacía.')
    header = rows[0][1]
    required = {'id', 'etiqueta', *FIELDS}
    if not required.issubset(header.values()) or len(set(header.values())) != len(header):
        raise ValueError('Se necesitan las columnas originales de la hoja humana.')
    sample = {}
    for row_number, cells in rows[1:]:
        record = {field: cells.get(col, '').strip() for col, field in header.items()}
        key = record['id']
        if not key or key in sample:
            raise ValueError(f'ID vacío o duplicado en la fila {row_number}.')
        if record['etiqueta'] not in CANONICAL_LABELS:
            raise ValueError(f'Etiqueta ausente o inválida en la fila {row_number}.')
        sample[key] = record
    return sample


def matrix_from_reviews(sheet: Path, reference: Path) -> Counter:
    sample = read_annotations(sheet)
    matrix = Counter()
    with reference.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'pair_id', 'human_label', *FIELDS.values()}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Faltan campos del test o la columna de referencia human_label.')
        for row_number, row in enumerate(reader, 2):
            if any(row[field] is None for field in required):
                raise ValueError(f'Fila incompleta en el CSV del test: {row_number}.')
            text_key = row['pair_id'] + '\x1f' + row['citation_context']
            key = hashlib.sha1(text_key.encode()).hexdigest()[:16]
            record = sample.pop(key, None)
            if record is None:
                raise ValueError(f'ID del test duplicado o ausente en la hoja: {key}.')
            if any(record[field] != row[col].strip() for field, col in FIELDS.items()):
                raise ValueError(f'El contenido de la cita {key} difiere entre los dos archivos.')
            if row['human_label'] not in CANONICAL_LABELS:
                raise ValueError(f'Referencia humana inválida para la cita {key}.')
            matrix[row['human_label'], record['etiqueta']] += 1
    if sample:
        raise ValueError(f'Hay {len(sample)} citas de la hoja que no aparecen en el test.')
    return matrix


def read_matrix(path: Path) -> Counter:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.reader(stream)
        columns = next(rows, [])[1:]
        if sorted(columns) != sorted(CANONICAL_LABELS):
            raise ValueError('Las columnas de la matriz deben contener las nueve funciones una vez.')
        matrix = Counter()
        seen = set()
        for row in rows:
            if len(row) != len(columns) + 1 or row[0] not in columns or row[0] in seen:
                raise ValueError('Fila incompleta, desconocida o duplicada en la matriz.')
            seen.add(row[0])
            for col, cell in zip(columns, row[1:]):
                count = int(cell)
                if count < 0:
                    raise ValueError('La matriz requiere conteos enteros no negativos.')
                matrix[row[0], col] = count
        if seen != set(columns):
            raise ValueError('Faltan clases en las filas de la matriz.')
    return matrix


def kappa(n: int, matches: int, marginal_product: int) -> float | None:
    denominator = n * n - marginal_product
    if denominator == 0:
        return None
    return (n * matches - marginal_product) / denominator


def agreement(matrix: Counter) -> dict:
    n = sum(matrix.values())
    if not n:
        raise ValueError('El cálculo requiere al menos una cita.')
    first, second = Counter(), Counter()
    for (jd, david), count in matrix.items():
        first[jd] += count
        second[david] += count
    matches = sum(matrix[c, c] for c in CANONICAL_LABELS)
    expected_pairs = sum(first[c] * second[c] for c in CANONICAL_LABELS)
    by_class = {}
    for c in CANONICAL_LABELS:
        both = matrix[c, c]
        neither = n - first[c] - second[c] + both
        expected = first[c] * second[c] + (n - first[c]) * (n - second[c])
        by_class[c] = dict(juan_david=first[c], david=second[c], ambos=both,
                           kappa=kappa(n, both + neither, expected))
    return dict(n_pares=n, coincidencias=matches, desacuerdos=n - matches,
                acuerdo_observado=matches / n, acuerdo_esperado=expected_pairs / n**2,
                kappa_cohen_nominal=kappa(n, matches, expected_pairs), por_clase=by_class)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--matriz', type=Path, default=MATRIX, help='Matriz CSV de conteos, con Juan David en filas.')
    group.add_argument('--hoja', type=Path, nargs='?', const=SHEET, metavar='XLSX',
                       help='Reconstruye el acuerdo desde el test completo y la hoja humana de David.')
    parser.add_argument('--referencia', type=Path, metavar='CSV', help='CSV del test para usar con --hoja.')
    args = parser.parse_args()
    if args.referencia and args.hoja is None:
        parser.error('--referencia requiere --hoja.')
    try:
        if args.hoja is None:
            matrix = read_matrix(args.matriz)
        else:
            matrix = matrix_from_reviews(args.hoja, args.referencia or TEST)
        result = agreement(matrix)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, ET.ParseError) as error:
        parser.exit(1, f'No se pudo calcular el acuerdo: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

