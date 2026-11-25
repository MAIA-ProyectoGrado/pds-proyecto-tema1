import hashlib
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import calcular_kappa as calc  # noqa: E402


def test_matriz_publicada_reproduce_el_cierre():
    r = calc.agreement(calc.read_matrix(calc.MATRIX))
    assert (r['n_pares'], r['coincidencias'], r['desacuerdos']) == (2241, 1544, 697)
    assert r['kappa_cohen_nominal'] == pytest.approx(0.6083494998108164)
    assert len(r['por_clase']) == 9
    assert r['por_clase']['Basis']['kappa'] == pytest.approx(0.0608, abs=0.00005)
    assert r['por_clase']['Evidence']['kappa'] == pytest.approx(0.8647, abs=0.00005)


def test_acuerdo_sin_variacion():
    r = calc.agreement(Counter({('Gap', 'Gap'): 3}))
    assert r['acuerdo_observado'] == 1
    assert r['kappa_cohen_nominal'] is None
    assert r['por_clase']['Gap']['kappa'] is None
    with pytest.raises(ValueError, match='al menos una cita'):
        calc.agreement(Counter())


def test_cruce_por_id_usa_solo_la_referencia_humana(tmp_path, monkeypatch):
    csv = tmp_path / 'test.csv'
    csv.write_text('pair_id,citation_context,rhetorical_section,cited_paper_title,cited_paper_abstract,label,human_label\n'
                   'X,a,,,,Background,Gap\nY,b,,,,Background,Evidence\n', encoding='utf-8')
    sample = {}
    for pair, context, label in [('Y', 'b', 'Evidence'), ('X', 'a', 'Basis')]:
        key = hashlib.sha1((pair + '\x1f' + context).encode()).hexdigest()[:16]
        sample[key] = dict(contexto_de_cita=context, seccion='', titulo_citado='', abstract_citado='', etiqueta=label)
    monkeypatch.setattr(calc, 'read_annotations', lambda _: sample.copy())
    sheet = tmp_path / 'hoja.xlsx'
    assert calc.matrix_from_reviews(sheet, csv) == {('Gap', 'Basis'): 1, ('Evidence', 'Evidence'): 1}
    sample.popitem()
    with pytest.raises(ValueError, match='ausente en la hoja'):
        calc.matrix_from_reviews(sheet, csv)


def test_matriz_con_clase_duplicada(tmp_path):
    lines = calc.MATRIX.read_text(encoding='utf-8-sig').splitlines()
    path = tmp_path / 'matriz.csv'
    path.write_text('\n'.join([*lines[:-1], lines[1]]), encoding='utf-8')
    with pytest.raises(ValueError, match='duplicada'):
        calc.read_matrix(path)
