# Guía de anotación de funciones de cita

Guía para la validación humana del test. Cada anotador trabaja por su
cuenta, sin ver la etiqueta automática, la de otro anotador ni la predicción de ningún
modelo. Con dos anotaciones independientes del mismo test se calcula el acuerdo
interanotador (kappa de Cohen).

## Qué se anota

Para cada cita se lee:

- **Contexto de cita:** la oración del artículo citante donde aparece la referencia.
- **Sección:** la parte del artículo citante donde está la cita.
- **Artículo citado:** su título y su abstract, cuando están disponibles.

Se registra una etiqueta y se puede añadir un comentario.

| Columna | Qué poner |
|---|---|
| `etiqueta` | Una sola de las 9 funciones (lista desplegable) |
| `comentario` | Opcional: la segunda función posible o por qué hubo duda |

La pregunta que se responde es: **¿para qué usa el autor esta referencia en esta
oración?** No se juzga la calidad del artículo citado ni si la afirmación es cierta.

## Las 9 funciones

Definiciones y criterios del enunciado del proyecto.

| Función | Definición | Señales típicas |
|---|---|---|
| **Background** | Da contexto general, resume el estado de un tema o traza su historia | Hechos generales del área, estado del arte, «has been widely used», «recent work» |
| **Gap** | Señala lo que falta o no se ha hecho, y así justifica el trabajo actual | «However», «few studies», «remains a challenge», «lack of» |
| **Basis** | Es el punto de partida intelectual o metodológico sobre el que se construye el trabajo | «Based on», «builds on», «following the framework of» |
| **Comparison** | Contrasta el trabajo actual con el citado, o compara trabajos citados entre sí | «Compared to», «outperforms», «in contrast to», «similar to» |
| **Application** | Usa un método, herramienta, dato o ecuación del trabajo citado **sin modificarlo** | «We use», «we employ», «using the toolkit of» |
| **Modification_Improvement** | Adapta, extiende o mejora un método del trabajo citado | «We extend», «a modified version of», «we adapt» |
| **Evidence** | Respalda una afirmación, hipótesis, decisión de diseño o resultado | «Consistent with», «as shown in», «supported by» |
| **Identification_of_the_Originator** | Reconoce la fuente original de una idea, método o teoría | «First proposed by», «introduced by», «coined by» |
| **Further_Reading** | Remite al lector a otras fuentes para más detalle | «See», «we refer the reader to», «for details, see» |

## Ejemplos

Tomados del conjunto de entrenamiento; ninguno está en el test.

| Función | Ejemplo |
|---|---|
| Background | RSs are widely used in areas such as e-commerce, social networking, entertainment, online learning, and native application platforms [2], [8]. |
| Gap | However, there are still few studies about the application of deep learning approaches in piRNA prediction [14, 12]. |
| Basis | The implemented deep learning model for real-time segmentation of iris and pupil is based on the MedFormer architecture described in 19. |
| Comparison | In contrast, models such as Llama 3.1 [11], Gemma 2 [51], and Mistral [3] show lower computation accuracy, particularly on the Avatar dataset. |
| Application | We use single-variable slice sampling (Neal, 2003) to sample from the posterior distribution of hyperparameters. |
| Modification_Improvement | We use a modified directed message-passing neural network (D-MPNN) 35,36 to predict retention order indices. |
| Evidence | The results linking neutrophil ROS production with NETosis and subsequent bacterial killing are consistent with several prior reports (18, 35, 36). |
| Identification_of_the_Originator | A novel role of endocytosis in turning on activation of the ERK/MAPK cascade by cell surface receptors was first reported for the EGF receptor [15]. |
| Further_Reading | Since a detailed survey of the available ML algorithms is out of the scope of this paper, we refer interested readers to [113, 157, 116, 29]. |

## Reglas para los casos dudosos

1. **Application o Modification_Improvement.** Si el método se usa tal cual, es
   Application. Si el texto dice que se adaptó, extendió o modificó («modified»,
   «adapted», «extended»), es Modification_Improvement, aunque también diga «we use».
2. **Application o Basis.** Si lo citado es una herramienta para calcular o
   experimentar, es Application. Si es la idea o arquitectura sobre la que se
   construye la propuesta, es Basis.
3. **Background o Identification_of_the_Originator.** Si la oración atribuye el
   origen de una idea o método («first», «introduced», «proposed by»), es
   Identification_of_the_Originator. Si solo describe el estado del tema, es Background.
4. **Background o Further_Reading.** Si la oración envía explícitamente al lector a la
   fuente («see», «refer to», «for details»), es Further_Reading.
5. **Background o Gap.** Si la oración destaca una limitación o algo que no se ha
   hecho, es Gap.
6. **Comparison o Evidence.** Si el propósito es mostrar en qué se parecen o difieren
   los resultados o métodos, es Comparison. Si el trabajo citado se usa para respaldar
   una afirmación, es Evidence. Si las dos lecturas son posibles, se elige la que mejor
   explica la oración. La otra se puede mencionar en el comentario.
7. **Varias funciones a la vez.** Se elige la función principal de la cita en la
   oración y se anota la otra en el comentario.
8. **Oraciones sin una referencia explícita.** Algunas citas del test no muestran un
   marcador como [4] o (Autor, año). Se clasifican según la función que cumpliría la
   afirmación si estuviera citada.
9. **Información incompleta.** Si no hay título ni abstract del artículo citado, se
   decide solo con el contexto y la sección.

## Procedimiento

- Anotar en el orden de la hoja: está mezclada al azar y balanceada por clase, de modo
  que cualquier tramo desde el inicio sirve como muestra.
- No consultar la etiqueta automática, la revisión de otro anotador ni el tablero.
- No cambiar una anotación después de ver otras etiquetas.
- Al terminar una sesión, guardar la hoja; el acuerdo se calcula con las filas que
  tengan `etiqueta`.
