"""Pruebas de `scripts/evaluate_llm.py` con servicios simulados y un test sintético (sin red ni costo)."""
from __future__ import annotations

import json
from collections import Counter
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import scripts.evaluate_llm as E  # noqa: E402
from src.api.prompts_llm import ORDEN  # noqa: E402


def _test_sintetico(por_clase=3):
    filas = []
    for clase in ORDEN:
        for k in range(por_clase):
            filas.append({"id": f"{clase[:4]}{k}", "citation_context": f"Context {clase} {k}.",
                          "rhetorical_section": "Methods", "cited_paper_title": None, "cited_paper_abstract": None,
                          "human_label": clase, "label": clase, "david_label": clase,
                          "pred_scibert": clase, "pred_tfidf": ORDEN[0]})
    df = pd.DataFrame(filas)
    return df.set_index("id", drop=False)


def _lista(clase):
    return "[" + ", ".join("0.9" if c == clase else "0.0" for c in ORDEN) + "]"


class _Servicio:
    """Responde la clase correcta (por el texto de la cita), con tokens fijos."""

    def __init__(self, fallos=0, codigo="ThrottlingException", texto=None):
        self.fallos, self.codigo, self.texto, self.llamadas = fallos, codigo, texto, 0

    def llamar(self, cfg, sistema, usuario):
        self.llamadas += 1
        if self.llamadas <= self.fallos:
            exc = Exception("transitorio")
            exc.response = {"Error": {"Code": self.codigo}}
            raise exc
        clase = next(c for c in ORDEN if f"Context {c} " in usuario)
        return {"texto": self.texto if self.texto is not None else _lista(clase), "tokens_entrada": 1000,
                "tokens_cache": 0, "tokens_salida": 50, "tokens_razonamiento": 0, "latencia_servicio_ms": 100,
                "motivo_fin": "end_turn", "cortada": False, "parametros": {}}


# ── muestra, mejor enfoque y tareas ─────────────────────────────────────────

def test_muestra_estratificada_y_reproducible(monkeypatch):
    monkeypatch.setattr(E, "POR_CLASE", 2)
    monkeypatch.setattr(E, "N_ESTABILIDAD", 4)
    test = _test_sintetico()
    a, b = E.construir_muestra(test), E.construir_muestra(test)
    assert a == b and len(a["muestra"]) == 18 and len(a["estabilidad"]) == 4
    assert Counter_clases(test, a["muestra"]) == {c: 2 for c in ORDEN}
    assert set(a["estabilidad"]) <= set(a["muestra"])


def Counter_clases(test, ids):
    from collections import Counter
    return dict(Counter(test.at[i, "human_label"] for i in ids))


def _regs(pred_por_enfoque, etapa="1"):
    regs = {}
    for e, pred in pred_por_enfoque.items():
        for i, et in pred.items():
            r = {"id": i, "enfoque": e, "etapa": etapa, "repeticion": 1, "estado": "valido", "etiqueta": et}
            regs[E.clave(r)] = r
    return regs


def test_mejor_enfoque_y_desempate():
    ids = ["a", "b"]
    ref = {"a": "Gap", "b": "Basis"}
    bien, mal = {"a": "Gap", "b": "Basis"}, {"a": "Basis", "b": "Gap"}
    assert E.mejor_enfoque(_regs({"E1": mal, "E2": mal, "E3": bien, "E4": bien}), ref, ids) == "E3"
    assert E.mejor_enfoque(_regs({"E1": bien, "E2": bien, "E3": bien, "E4": bien}), ref, ids) == "E1"


def test_mejor_enfoque_exige_la_muestra_completa():
    with pytest.raises(SystemExit):
        E.mejor_enfoque(_regs({"E1": {"a": "Gap"}}), {"a": "Gap", "b": "Gap"}, ["a", "b"])


def test_tareas_por_etapa():
    test = _test_sintetico()
    muestra = {"muestra": list(test["id"][:5]), "estabilidad": list(test["id"][:2])}
    mejores = {m: "E2" for m in E.SIN_ASTRA}
    assert len(E.tareas("1", "gemma-3-4b", test, muestra, mejores)) == 20
    assert {t["enfoque"] for t in E.tareas("1b", "gpt-6.2-astra", test, muestra, mejores)} == {"E2"}
    assert len(E.tareas("2", "gemma-3-4b", test, muestra, mejores)) == len(test) - 5
    assert len(E.tareas("3", "gpt-6.2-astra", test, muestra, mejores)) == 2 * E.REPETICIONES
    with pytest.raises(SystemExit):
        E.tareas("1", "gpt-6.2-astra", test, muestra, mejores)
    with pytest.raises(SystemExit):
        E.tareas("2", "gemma-3-4b", test, muestra, {})


# ── llamadas, topes y reanudación ───────────────────────────────────────────

def _lista_tareas(test, n):
    return [{"id": i, "enfoque": "E1", "etapa": "1", "repeticion": 1} for i in list(test["id"])[:n]]


def test_reintenta_transitorios_y_registra(tmp_path):
    test = _test_sintetico()
    serv = _Servicio(fallos=2)
    res = E.correr("gemma-3-4b", _lista_tareas(test, 1), test.to_dict("index"), serv, tmp_path / "r.jsonl",
                   0.0, hilos=1, dormir=lambda s: None)
    assert res["estados"] == {"valido": 1} and serv.llamadas == 3
    r = next(iter(E.leer_registros(tmp_path / "r.jsonl").values()))
    assert r["intentos"] == 3 and r["respuesta"].startswith("[") and r["etiqueta"] == ORDEN[0]


def test_error_permanente_no_se_reintenta(tmp_path):
    test = _test_sintetico()
    serv = _Servicio(fallos=99, codigo="ValidationException")
    res = E.correr("gemma-3-4b", _lista_tareas(test, 1), test.to_dict("index"), serv, tmp_path / "r.jsonl",
                   0.0, hilos=1, dormir=lambda s: None)
    assert res["estados"] == {"error": 1} and serv.llamadas == 1


def test_tope_detiene_la_corrida(tmp_path, monkeypatch):
    test = _test_sintetico()
    costo_llamada = E.costo({"tokens_entrada": 1000, "tokens_salida": 50}, E.MODELOS["gpt-6.2-sol"]["precio"])
    monkeypatch.setitem(E.TOPE_MODELO, "gpt-6.2-sol", costo_llamada * 3.5)
    res = E.correr("gpt-6.2-sol", _lista_tareas(test, 10), test.to_dict("index"), _Servicio(), tmp_path / "r.jsonl",
                   0.0, hilos=1)
    assert res["detenido"] and 1 <= sum(res["estados"].values()) <= 3
    assert res["gasto_modelo_usd"] <= E.TOPE_MODELO["gpt-6.2-sol"]


def test_tope_de_servicio_cuenta_lo_gastado_por_otros(tmp_path):
    test = _test_sintetico()
    res = E.correr("gemma-3-4b", _lista_tareas(test, 3), test.to_dict("index"), _Servicio(), tmp_path / "r.jsonl",
                   E.TOPE_SERVICIO["bedrock"], hilos=1)
    assert res["detenido"] and res["estados"] == {}


def test_reanudar_da_lo_mismo_que_una_corrida_continua(tmp_path):
    test = _test_sintetico()
    filas, lista = test.to_dict("index"), _lista_tareas(test, 8)
    E.correr("gemma-3-4b", lista, filas, _Servicio(), tmp_path / "a.jsonl", 0.0, hilos=2, limite=3)
    serv = _Servicio()
    E.correr("gemma-3-4b", lista, filas, serv, tmp_path / "a.jsonl", 0.0, hilos=2)
    E.correr("gemma-3-4b", lista, filas, _Servicio(), tmp_path / "b.jsonl", 0.0, hilos=2)
    a, b = E.leer_registros(tmp_path / "a.jsonl"), E.leer_registros(tmp_path / "b.jsonl")
    assert serv.llamadas == 5
    assert {k: v["etiqueta"] for k, v in a.items()} == {k: v["etiqueta"] for k, v in b.items()}
    assert round(E.gasto(a, "gemma-3-4b"), 9) == round(E.gasto(b, "gemma-3-4b"), 9)


def test_respuesta_cortada_e_invalida(tmp_path):
    test = _test_sintetico()
    serv = _Servicio(texto="[0.1, 0.2")
    E.correr("gemma-3-4b", _lista_tareas(test, 1), test.to_dict("index"), serv, tmp_path / "r.jsonl", 0.0, hilos=1)
    r = next(iter(E.leer_registros(tmp_path / "r.jsonl").values()))
    assert r["estado"] == "sin_lista" and r["etiqueta"] is None


def test_costo_cobra_la_cache_aparte():
    r = {"tokens_entrada": 1_000_000, "tokens_cache": 900_000, "tokens_salida": 1_000_000}
    assert E.costo(r, (2.0, 10.0, 0.1)) == pytest.approx(0.2 + 0.09 + 10.0)


# ── adaptador de OpenAI ─────────────────────────────────────────────────────

class _BadRequestError(Exception):
    pass


_BadRequestError.__name__ = "BadRequestError"


class _OpenAI:
    def __init__(self, rechaza_temperatura):
        self.rechaza, self.kwargs = rechaza_temperatura, []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.kwargs.append(kw)
        if self.rechaza and "temperature" in kw:
            raise _BadRequestError("Unsupported parameter: 'temperature' is not supported with this model.")
        uso = SimpleNamespace(input_tokens=600, output_tokens=49,
                              input_tokens_details=SimpleNamespace(cached_tokens=0),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=0))
        return SimpleNamespace(output_text=_lista("Gap"), usage=uso, status="completed")


def test_openai_quita_la_temperatura_si_la_rechaza():
    cliente = _OpenAI(rechaza_temperatura=True)
    serv = E.OpenAIResponses(cliente)
    cfg = E.MODELOS["gpt-6-luna"]
    r1 = serv.llamar(cfg, "s", "u")
    serv.llamar(cfg, "s", "u")
    assert "temperature" not in r1["parametros"] and cfg["id"] in serv.sin_temperatura
    assert [("temperature" in k) for k in cliente.kwargs] == [True, False, False]
    assert cliente.kwargs[-1]["reasoning"] == {"effort": "low"}


def test_openai_sin_temperatura_configurada():
    cliente = _OpenAI(rechaza_temperatura=False)
    E.OpenAIResponses(cliente).llamar(E.MODELOS["gpt-6.2-sol"], "s", "u")
    assert "temperature" not in cliente.kwargs[0]


# ── análisis ────────────────────────────────────────────────────────────────

def test_analisis_sintetico():
    test = _test_sintetico()
    ids = list(test["id"])
    muestra = {"muestra": ids[:9], "estabilidad": ids[:2], "semilla": 42}
    registros = {}
    for m in ("gemma-3-4b", "gpt-6.2-sol", "gpt-6.2-astra"):
        regs = {}
        etapas = [("1b", "E1", ids[:9])] if m == "gpt-6.2-astra" else (
            [("1", e, ids[:9]) for e in ("E1", "E2", "E3", "E4")] + [("2", "E1", ids[9:])])
        for etapa, enfoque, lote in etapas:
            for i in lote:
                bueno = enfoque == "E1"
                r = {"id": i, "enfoque": enfoque, "etapa": etapa, "repeticion": 1, "estado": "valido" if bueno else "sin_lista",
                     "etiqueta": test.at[i, "human_label"] if bueno else None, "empate": False,
                     "tokens_entrada": 1000, "tokens_salida": 50, "latencia_cliente_ms": 100.0}
                regs[E.clave(r)] = r
        for k in range(1, 4):
            for i in ids[:2]:
                r = {"id": i, "enfoque": "E1", "etapa": "3", "repeticion": k, "estado": "valido",
                     "etiqueta": test.at[i, "human_label"] if (k < 3 or i == ids[0]) else ORDEN[-1],
                     "empate": False, "tokens_entrada": 1000, "tokens_salida": 50, "latencia_cliente_ms": 100.0}
                regs[E.clave(r)] = r
        registros[m] = regs
    res = E.analizar(test, muestra, registros)
    assert res["mejor_enfoque"] == {"gemma-3-4b": "E1", "gpt-6.2-sol": "E1"}
    assert res["comparacion"]["gemma-3-4b"]["test_completo"]["juan_david"]["f1_macro"] == 1.0
    assert res["comparacion"]["gpt-6.2-astra"]["citas"] == 9
    assert res["comparacion"]["tfidf"]["test_completo"]["juan_david"]["exactitud"] == pytest.approx(1 / 9, abs=1e-4)
    assert res["enfoques"]["gemma-3-4b"]["E2"]["tasa_error_formato"] == 1.0
    assert res["enfoques"]["gemma-3-4b"]["E2"]["juan_david"]["f1_macro"] == 0.0
    assert res["eficiencia"]["gemma-3-4b"]["estabilidad_misma_prediccion"] == 0.5
    assert res["eficiencia"]["gemma-3-4b"]["latencia_media_ms"] == 100.0
    assert res["enfoques"]["gemma-3-4b"]["E2"]["llamadas"] == 9
    assert res["enfoques"]["gemma-3-4b"]["E2"]["errores_formato"] == {"sin_lista": 9}
    # TF-IDF predice siempre la primera clase; los demás modelos aciertan todo.
    primera = res["ambiguedad"][0]
    assert primera["clase_predicha"] == ORDEN[0] and primera["maximo_modelos"] == 1.0
    assert primera["por_modelo"]["tfidf"] == 1.0
    assert primera["promedio_modelos"] == pytest.approx(1 / len(primera["por_modelo"]), abs=1e-4)
    json.dumps({k: v for k, v in res.items() if not k.startswith("_")})


def test_ambiguedad_promedia_entre_modelos_y_omite_invalidas():
    test = _test_sintetico(por_clase=2)
    a, b = ORDEN[0], ORDEN[1]
    ids_a, ids_b = list(test.index[test["human_label"] == a]), list(test.index[test["human_label"] == b])
    finales = {"m1": {ids_a[0]: b, ids_a[1]: a, ids_b[0]: b, ids_b[1]: E.INVALIDO},
               "m2": {ids_a[0]: a, ids_a[1]: a, ids_b[0]: a, ids_b[1]: b}}
    filas = {(f["clase_real"], f["clase_predicha"]): f for f in E._ambiguedad(test, finales)}
    assert filas[a, b]["por_modelo"] == {"m1": 0.5, "m2": 0.0}
    assert filas[a, b]["promedio_modelos"] == 0.25
    # En m1 la segunda cita de b es inválida: queda una sola cita real de b, bien clasificada.
    assert filas[b, a]["por_modelo"] == {"m1": 0.0, "m2": 0.5}
    assert (a, a) not in filas
    assert all(r != ORDEN[2] for r, _ in filas)


def test_etiqueta_ajustada_recupera_el_formato():
    base = {"id": "x", "enfoque": "E1", "etapa": "1", "repeticion": 1, "etiqueta": None}
    sin_corchetes = {**base, "estado": "sin_lista", "respuesta": "0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0"}
    negrita = {**base, "estado": "no_numerico", "respuesta": "[0.0, 0.0, **1.0**, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"}
    cortada = {**base, "estado": "cortada", "respuesta": "[0."}
    valida = {**base, "estado": "valido", "etiqueta": "Gap", "respuesta": "[0.0, 1.0, 0, 0, 0, 0, 0, 0, 0]"}
    assert E.etiqueta_ajustada(sin_corchetes) == "Gap"
    assert E.etiqueta_ajustada(negrita) == "Basis"
    assert E.etiqueta_ajustada(cortada) == E.INVALIDO
    assert E.etiqueta_ajustada(valida) == "Gap"
    regs = {E.clave(r): r for r in (sin_corchetes,)}
    assert E.predicciones(regs, "1", "E1") == {"x": E.INVALIDO}
    assert E.predicciones(regs, "1", "E1", ajustada=True) == {"x": "Gap"}


def test_openai_envia_clave_de_cache_por_mensaje_de_sistema():
    cliente = _OpenAI(rechaza_temperatura=False)
    serv = E.OpenAIResponses(cliente)
    for sistema in ("a", "a", "b"):
        serv.llamar(E.MODELOS["gpt-6.2-sol"], sistema, "u")
    claves = [k["prompt_cache_key"] for k in cliente.kwargs]
    assert claves[0] == claves[1] != claves[2]


def test_submuestra_de_razonamiento_estratificada_y_reproducible():
    test = _test_sintetico()
    muestra = {"muestra": list(test["id"]), "estabilidad": [], "semilla": 42}
    sub = E.submuestra_razonamiento(test, muestra)
    assert sub == E.submuestra_razonamiento(test, muestra)
    por_clase = Counter(test.set_index("id").loc[sub, "human_label"])
    assert all(n == min(E.POR_CLASE_R, (test["human_label"] == c).sum()) for c, n in por_clase.items())


def test_etapa_r_solo_para_variantes():
    test = _test_sintetico()
    muestra = {"muestra": list(test["id"]), "estabilidad": [], "semilla": 42}
    lista = E.tareas("R", "gpt-6.2-sol-high", test, muestra, {"gpt-6.2-sol": "E3"})
    assert {t["enfoque"] for t in lista} == {"E3"} and {t["etapa"] for t in lista} == {"R"}
    assert len(lista) == len(E.submuestra_razonamiento(test, muestra))
    completa = E.tareas("R", "gpt-6.2-sol-xhigh", test, muestra, {"gpt-6.2-sol": "E3"})
    assert [t["id"] for t in completa] == muestra["muestra"]
    assert E.MODELOS["gpt-6.2-sol-xhigh"]["razonamiento"] == "xhigh"
    assert "gpt-6.2-sol-high" not in E.SIN_ASTRA
    with pytest.raises(SystemExit):
        E.tareas("R", "gpt-6.2-sol", test, muestra, {"gpt-6.2-sol": "E3"})
