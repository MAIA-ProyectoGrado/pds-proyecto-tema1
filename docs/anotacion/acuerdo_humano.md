# Acuerdo entre anotadores del test

Dos integrantes anotaron por separado las 2.241 citas del test, sin ver la etiqueta
automática ni la del otro, siguiendo la [guía de anotación](GUIA_ANOTACION.md):
Juan David (`human_label` en el CSV del test) y David
(`dataset/anotacion/hoja_anotacion_test.xlsx`, con DVC).

| Medida | Valor |
|---|--:|
| Citas con la misma función | 1.544 de 2.241 (68,9 %) |
| Acuerdo esperado por azar | 20,6 % |
| **Kappa de Cohen (9 funciones)** | **0,608** |

## Por función

Kappa de cada función como decisión binaria (esa función frente al resto).

| Función | Juan David | David | Ambos | Kappa |
|---|--:|--:|--:|--:|
| Evidence | 259 | 259 | 228 | 0,865 |
| Application | 316 | 285 | 228 | 0,721 |
| Gap | 210 | 143 | 128 | 0,703 |
| Comparison | 279 | 244 | 187 | 0,678 |
| Modification_Improvement | 81 | 53 | 46 | 0,677 |
| Identification_of_the_Originator | 45 | 29 | 24 | 0,643 |
| Further_Reading | 88 | 50 | 45 | 0,642 |
| Background | 662 | 1.147 | 644 | 0,539 |
| Basis | 301 | 31 | 14 | 0,061 |

El desacuerdo principal es Basis: en 279 citas Juan David eligió Basis y David,
Background. La matriz completa está en [`matriz_acuerdo_humano.csv`](matriz_acuerdo_humano.csv)
(Juan David en filas, David en columnas).

## Efecto en la evaluación de los modelos

Mismas predicciones del test ([`predicciones_test.csv`](../modelos_v4/predicciones_test.csv)),
cambiando solo la referencia humana:

| Modelo | F1 macro frente a Juan David | F1 macro frente a David |
|---|--:|--:|
| SciBERT B | 0,532 | 0,408 |
| TF-IDF + regresión logística | 0,469 | 0,370 |

SciBERT B supera a TF-IDF con las dos referencias; la diferencia depende de quién anota.
Juan David se mantiene como referencia principal de la evaluación.

## Reproducir el kappa

Desde la raíz del repositorio, ejecuta el [script de cálculo](../../scripts/calcular_kappa.py)
con Python 3. El primer comando usa la matriz de conteos publicada y solo requiere
la biblioteca estándar. Imprime un JSON con el número de citas, el acuerdo y los nueve
kappas por función.

```sh
python3 scripts/calcular_kappa.py
```

Con la hoja humana cerrada en `dataset/anotacion/hoja_anotacion_test.xlsx` y
el archivo `dataset/test_enriquecido_top3_chunks.csv` disponibles, el segundo
comando reconstruye los conteos por ID y comprueba el texto de cada cita.
Para usar otras ubicaciones, pasa las rutas con `--hoja archivo.xlsx --referencia test.csv`.

```sh
python3 scripts/calcular_kappa.py --hoja
```

Con los insumos de esta evaluación, ambos comandos producen 2.241 pares, 1.544
coincidencias y un kappa global de 0,608349.
