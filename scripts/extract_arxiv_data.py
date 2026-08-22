"""
Script de extracción de datos para el proyecto de recomendación local de citas
y clasificación de funciones de cita.

Descarga artículos científicos en inglés desde arXiv (área de computación por
defecto), extrae el texto de cada PDF y localiza los marcadores de cita
([1], [2,3], (Smith et al., 2021), etc.) junto con una ventana de texto
alrededor de cada marcador (el "contexto de cita"). El resultado se guarda en
dos archivos CSV dentro de la carpeta de salida:

  - papers_metadata.csv     -> un registro por artículo descargado
  - citation_contexts.csv   -> un registro por cada marcador de cita encontrado

Por defecto el script descarga TODOS los artículos que coincidan con la
consulta de búsqueda (sin límite de cantidad). Los resultados se guardan de
forma incremental y el script puede reanudarse: si se interrumpe y se vuelve
a ejecutar, no vuelve a descargar ni a procesar artículos ya guardados.

Uso:
    python extract_arxiv_data.py --query "cat:cs.CL" --out_dir data/raw
    python extract_arxiv_data.py --query "cat:cs.CL" --max_results 50 --out_dir data/raw

Requisitos (ver requirements.txt):
    arxiv, pdfplumber, pandas, tqdm
"""

import argparse
import csv
import re
import time
from pathlib import Path

import arxiv
import pdfplumber
from tqdm import tqdm


CITATION_PATTERN = re.compile(
    r"\[(\d+(?:\s*,\s*\d+)*)\]"
    r"|\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?,?\s+\d{4}[a-z]?)\)"
)
CONTEXT_WINDOW_CHARS = 400

METADATA_FIELDS = ["paper_id", "title", "abstract", "categories", "pdf_path"]
CONTEXT_FIELDS = ["paper_id", "citation_marker", "citation_context", "char_position"]


def load_processed_ids(csv_path: Path) -> set:
    """Lee un CSV de metadatos ya generado y devuelve el conjunto de paper_id
    que ya fueron procesados, para poder reanudar una ejecución interrumpida."""
    if not csv_path.exists():
        return set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["paper_id"] for row in reader}


def open_csv_writer(csv_path: Path, fieldnames):
    """Abre un CSV en modo append, escribiendo el encabezado solo si el
    archivo es nuevo. Devuelve el archivo abierto y el writer."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
    return f, writer


def extract_text(pdf_path: Path) -> str:
    """Extrae el texto plano de un PDF, página por página."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_citation_contexts(paper_id: str, text: str):
    """Localiza marcadores de cita en el texto y guarda el contexto alrededor."""
    contexts = []
    for match in CITATION_PATTERN.finditer(text):
        start = max(0, match.start() - CONTEXT_WINDOW_CHARS)
        end = min(len(text), match.end() + CONTEXT_WINDOW_CHARS)
        context = text[start:end].replace("\n", " ").strip()

        contexts.append(
            {
                "paper_id": paper_id,
                "citation_marker": match.group(0),
                "citation_context": context,
                "char_position": match.start(),
            }
        )
    return contexts


def process_paper(result, pdf_dir: Path):
    """Descarga el PDF de un artículo, extrae su texto y sus contextos de
    cita. Devuelve (metadata_row, context_rows) o (None, []) si falla."""
    paper_id = result.get_short_id()
    pdf_path = pdf_dir / f"{paper_id.replace('/', '_')}.pdf"

    if not pdf_path.exists():
        try:
            result.download_pdf(dirpath=str(pdf_dir), filename=pdf_path.name)
        except Exception as exc:
            print(f"No se pudo descargar {paper_id}: {exc}")
            return None, []

    metadata_row = {
        "paper_id": paper_id,
        "title": result.title,
        "abstract": result.summary,
        "categories": ",".join(result.categories),
        "pdf_path": str(pdf_path),
    }

    try:
        text = extract_text(pdf_path)
    except Exception as exc:
        print(f"No se pudo leer {pdf_path}: {exc}")
        return metadata_row, []

    context_rows = extract_citation_contexts(paper_id, text)
    return metadata_row, context_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="cat:cs.CL",
        help="Consulta de búsqueda en arXiv (ej: 'cat:cs.CL', 'cat:cs.AI', 'cat:cs.CL OR cat:cs.AI')",
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=None,
        help="Número máximo de artículos a descargar. Si se omite, descarga TODOS los "
        "artículos que coincidan con la consulta.",
    )
    parser.add_argument("--out_dir", default="data/raw", help="Carpeta de salida para los datos")
    parser.add_argument(
        "--page_size",
        type=int,
        default=100,
        help="Cantidad de resultados por página al consultar la API de arXiv",
    )
    parser.add_argument(
        "--delay_seconds",
        type=float,
        default=3.0,
        help="Segundos de espera entre páginas de resultados (respeta el límite de la API)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pdf_dir = out_dir / "pdfs"
    metadata_path = out_dir / "papers_metadata.csv"
    contexts_path = out_dir / "citation_contexts.csv"

    search = arxiv.Search(
        query=args.query,
        max_results=args.max_results,  # None -> sin límite
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(
        page_size=args.page_size,
        delay_seconds=args.delay_seconds,
        num_retries=5,
    )

    processed_ids = load_processed_ids(metadata_path)
    if processed_ids:
        print(f"Se encontraron {len(processed_ids)} artículos ya procesados; se omitirán.")

    metadata_file, metadata_writer = open_csv_writer(metadata_path, METADATA_FIELDS)
    contexts_file, contexts_writer = open_csv_writer(contexts_path, CONTEXT_FIELDS)

    total_papers = 0
    total_contexts = 0

    try:
        for result in tqdm(client.results(search), desc="Procesando artículos", unit="paper"):
            paper_id = result.get_short_id()
            if paper_id in processed_ids:
                continue

            metadata_row, context_rows = process_paper(result, pdf_dir)
            if metadata_row is None:
                continue

            metadata_writer.writerow(metadata_row)
            metadata_file.flush()
            total_papers += 1

            for row in context_rows:
                contexts_writer.writerow(row)
            contexts_file.flush()
            total_contexts += len(context_rows)

            processed_ids.add(paper_id)
    finally:
        metadata_file.close()
        contexts_file.close()

    print(f"Artículos nuevos descargados en esta ejecución: {total_papers}")
    print(f"Contextos de cita nuevos extraídos en esta ejecución: {total_contexts}")
    print(f"Metadatos acumulados en: {metadata_path}")
    print(f"Contextos acumulados en: {contexts_path}")


if __name__ == "__main__":
    main()
