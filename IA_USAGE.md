# IA_USAGE — Uso de IA generativa en el proyecto

Consigna §7: modelo de IA usado, prompt exacto, parte del código o decisión que salió de la IA, modificaciones hechas por el equipo y análisis crítico (aciertos, errores y limitaciones).

- **Modelo:** Claude (Anthropic), en la app de Claude. Sesión 1 en un chat normal; sesión 2 en una sesión de trabajo con archivos y terminal, identificador `claude-opus-5-5`.
- **Integrantes:** Samuel Uribe, Emilio Marquez y Juan Pablo Lopez trabajaron con la misma cuenta de Claude y en las mismas sesiones. Por eso hay un solo registro, que recoge los prompts del equipo.
- **Prompts:** se copian tal como se escribieron, con sus errores de tipeo. Cuando el mensaje fue solo una salida de terminal, una captura o un archivo, se indica entre corchetes.
- **Semana 8 (EDA y Visualizador 1):** no quedó registro de un chat con IA en el historial; si algún integrante la usó, agrega aquí su sesión con el mismo formato.

---

## Sesión 1 · Semana 9 — Detección (2 de octubre de 2026)

**Contexto entregado:** enunciado `.docx`, notebook del EDA de la semana 8 y, después, el notebook del Taller 3 con la FundidoraPC.

| # | Prompt exacto |
|---|---|
| 1 | "mira, este es el proyecto que nos mandaron, y ese fue el primer avance que entregamos, hoy tenemos que presentar otro avance, y el profesor nos dios esa pista para la reconstruccion de la imagen, analsa todo y me dices si tiene sentido, me dices si necesitas las imagenes que salieron del colab," |
| 2 | "creo que eso que dijo era para la entrega de hoy," |
| 3 | "si" |
| 4 | "no necesito que lo pruebes, solo dame las celdas, te tiras 15 min probando datos que no son los del trbajo" |
| 5 | "te mando el codigo de la fundidraPC?" |
| 6 | "aqui esta la fundidora" |
| 7 | "dame las celdas" |
| 8 | "dame las celdas para copiar y pegar, o junta ya los dos ipynb en uno solo, el que te mande y este que me diste" |
| 9 | "si esta en compartido conmigo, como cambio la ruta" |
| 10 | "cual es nuestra arquitectura propuesta, o que bloque tenemos, el encoder porque tenemos la fundidoraPC" |
| 11 | "esa foto era de otro grupo, solo te la ponia como ejemplo para que me dijeras nosotros como estabamos" |
| 12 | "dame un guion de como presentar, sobre todo la neuva parte" |
| 13 | "pero no hicimos transfer" |
| 14 | "dame el guion de nuevo" |
| 15 | "calculamos el gamma?" |
| 16 | "usamos CBAM?" |
| 17 | "por que usamos CBAM? el profesor le pregunto a otro grupo" |
| 18 | "dame el guion de nuevo" |
| 19 | "dame el guion ya con estos resultados, rapido, un guion sencillo de que se hizo," [adjunto: notebook ejecutado] |
| 20 | "como vamos a hacer la reconstruccion tridimensional" |
| 21 | "usamos dos veces dice, dos veces interseccion?" |
| 22 | "que estamos usamos en la perdida compuesta y calibracion lambda" |
| 23 | "que cstiga cada uno" |
| 24 | "que hemos logrado hasta el momento" |

**Qué salió de la IA**
- Interpretación de la pista del profesor (salida de 9×9 con ≤ 10 canales) como cabeza de detección de 8 canales por celda, justificada con la tabla de colisiones del EDA.
- Celdas de la semana 9: caché de cortes a 256 px, codificación del target, CBAM, cuello 16×16 → 9×9, cabezas de detección y clasificación, IoU y NMS propios, pérdida compuesta con λ calibrados, overfit, entrenamiento con AMP, mAP y latencia.
- Adaptación de la FundidoraPC del Taller 3 a CT de 1 canal y unión del notebook del EDA con las celdas nuevas.
- Guiones de presentación y respuestas para preguntas del profesor (CBAM, γ, pérdida, IoU).

**Modificaciones del equipo**
- Aportó el código real de la FundidoraPC en reemplazo del *placeholder* de la IA.
- Ejecutó el notebook en Colab con los datos reales y reportó los resultados: IoU 0.724, mAP@0.5 0.820, mAP@[0.5:0.95] 0.437 y overfit con IoU 0.93.
- Corrigió a la IA dos veces: la arquitectura de la imagen era de otro grupo, y el modelo se entrenó sin *transfer learning*.

**Análisis crítico**
- **Aciertos.** El 9×9 quedó justificado con datos propios (0 % de colisiones) y no por imitación de YOLO. La prueba de overfit validó la codificación antes de entrenar.
- **Errores.**
  - Gastó unos 15 minutos probando con datos sintéticos que no aportaban; el equipo tuvo que pedir que se detuviera.
  - Asumió que se usaba *transfer learning* y lo escribió en el primer guion.
  - Tomó como propia una arquitectura de otro grupo mostrada solo como ejemplo.
- **Limitaciones.** La IA no ejecutó el código con los datos reales; todos los números vienen de las corridas del equipo.

---

## Sesión 2 · Semanas 10 y 11 — Revisión, segmentación, dashboard, despliegue, informe y GitHub (13 al 17 de octubre de 2026)

### 2.1 Revisión y correcciones de la semana 10 (13–15 de octubre)
Esta parte de la sesión se compactó automáticamente. Los prompts 1 a 4 se conservan completos; del resto quedaron los fragmentos textuales que se listan.

| # | Prompt exacto |
|---|---|
| 1 | "mira, este es el proyecto y este es el avance hasta la tercera semana, solo me falta el ultimo avance de la ultima semana, estudia muy bien el docx, para saber que piden y que no nos falte nada, y ayudame a revisar lo que tenemos, y cuando tengamos listo eso, vamos con la ultima parte" [adjuntos: enunciado y notebook S8–S9] |
| 2 | [solo adjunto: notebook S8–S10] |
| 3 | "corrije todo, y dame el notebook listo, y tambien, cuando eso este listo, ya vamos con lo 11" |
| 4 | "no pruebes el colab, solo damelo y yo lo corro aca" |
| — | Fragmentos de mensajes posteriores durante la ejecución: "esta celda es para que" · "primero terminemos con esta verificacion hasta la sem 10" · "si me interesa" (sobre el slider de HU) · "te voy a ir mandando las imagenes… para el informe, donde solo van imagenes e interpretaciones" · "por que hay que volver a correr el eda" · "la seccion 10 no se guardo" · "que es la ablacion" · "o sea que ya quedo todo hasta la semana 10?" |
| — | Fragmentos sobre la semana 11 y Docker: "me gusta todo, pero hay posibilidad de montarlo en docker?" · "listo, entonces vamos asi con docker, crea el dcokerfile, y el compose, damelo todo en un zip, para montarlo en mi computador y tambien de una vez hacer lo del github" · "dame el paso a paso para montar todo… ya los colab estan corridos, eso ya no lo tengo que volver a tocar" · "como puedo dejar docker en un solo lugar" · "el trabajo pide que lo probemos en gpu y cpu, o como" · "eso culmina el trabajo?" · "que es lo del caso precalculado?" |

### 2.2 Dashboard, ablación, informe y pitch (16 de octubre)
| # | Prompt exacto |
|---|---|
| 5 | "tienes que darme una guia de  como usar la app, hay forma de simplificarla y que se vea mas bonita, y otra cosa, en el visualizador 1, volumen crudo (MIP), no eta girando" |
| 6 | "con respecto a la app, me gusto mucho, quedo mas entendible y chevere de ver" |
| 7 | "que queda faltando a parte de lo de github, y los .md de ia" |
| 8 | "quitemos lo de la fila 2D, no hagamos la ablacion de 2 y 2.5 ya que demora mucho, dame la celda para poner donde va, solo la voy a poner y guardar" |
| 9 | [celda de ablación editada] "asi quedo" |
| 10 | "ese que tengo en local, es la ultima version que descargue de lo que corri en colab?" |
| 11 | [solo adjunto: notebook S8–S10 actualizado] |
| 12 | "listo, empieza con el paper, ya tienes todas las imagenes no? solo faltan las de la app, ajustate a las limitacioens y requerimientos ue pide el profe, las inerpretaciones de las imagenes hazlas cortas y solo resaltando lo mas importante, el documento no puede quedar tan largo," |
| 13 | "autores, ya quite a andres, el no esta solo somos 3,quita de todo l de transfer learning, nunca s ehizo porqu no teniamos lo pesos, y la ablacion de 2D vs 2.5D" |
| 14 | [adjunto: `GUIA_GITHUB.md`] "si corrige esto" |
| 15 | "aqui estan las imagenes, faltan las del celular" [capturas del dashboard] |
| 16 | "listo corrijamoslo, y como es que saco la url de cloudare para verlo desde el celular" |
| 17 | "Aquí están las fotos desde el celular" |
| 18 | "Solo queda pendiente entojces el github y los .md de IA?" |
| 19 | "todo esoty ya lo hice, solo falta expoat, y el pithc y video, y quiuta lla mencion de transfer learning en el model card" |
| 20 | "haz el diseno mas sobrio, me gustan los colores pero que todas sean dde fondo blanco, en e ptch final, que no parezcan tan IA, y por ejemplo quita lo de separar conminutos en contacto sigue abierto y esta declarado enelmodel card, ya lo que se hizo se hizo, no hay siguiente entreg por lo que nada queda abierto o en opcio de mejora" |
| 21 | "si" (quitar también "trabajo futuro" del informe) |
| 22 | "cual de estas opciones uso para descargrla?" |
| 23 | [adjunto: enunciado] "la consigna que dice sobre como debe ser el informe tecnico, si cumplimos con las especificaciones?" |
| 24 | "listo agrega eso del transfer" |
| 25 | [cita del enunciado §4.1] "mir qu si es opcional, no es marcao como falta solo que no lo hicimos" |
| 26 | [cita del enunciado §5, extensión opcional] "exta extension opcional, es muy dificil de hacer?" |
| 27 | "si, dame el script, por puntos extras que de pronto nos suben en la sustentacion" |
| 28 | [resultados de `hd95_assd.py` y CSV] |
| 29 | "el profesor nos dijo que el puede subir el volumen .mha, o subir los cortes de ese volumen y se debe tener la capacidad de hacer la reconstruccion, asi lo tenemos o hay que cambiarlo? tambien, que las imagenes que nos va a subir puede cambiar en profundidad y dimensionalidad" |
| 30 | "cuantas paginas tiene el informe?" |
| 31 | "porque el profe puso limite de maximo 4 paginas" |
| 32 | "una pregunta, si el sube el mha. el procesamiento en vivo, lo hace con la cpu del computador?" |
| 33 | "el profe pide un video donde se muestre que funciona?" |
| 34 | "lo que suba el profe, puede tener unos cuantos cortes de mas o unos cortes de menos, tambien estamos blindados contra eso?" |

### 2.3 GitHub y video de avance (17 de octubre)
| # | Prompt exacto |
|---|---|
| 35 | "ya estamos los tres, cada uno con un computador, como hacemos lo de github, ya que cada uno tiene que subir algo" |
| 36 | "listo, me gusta el plan, entonces vamos primero con samuel, dame el paso a paso, mi usuario es PresiUribe" |
| 37 | [errores de `Compress-Archive` y `tar` en PowerShell] |
| 38 | "n se puede hacer un push directamente?" · "sin tener que hacer el zip" |
| 39 | [salida de `git push`: "Repository not found"] |
| 40 | [salida de `git status` de la rama temporal] |
| 41 | "antes de hacer el commit, si yo le doy push a esto, va a aparecer que yo subi todo? que subirian emilio y lopez despues" |
| 42 | "entiendo, o sea yo estoy subiendo todo a una rama temporal, y de ahi emilio y lopez lo suben al main, y en el registro que le mandamos al profe solo sale main, y os archivos que subio cada uno" |
| 43 | "bueno, ya hago este commit entonces" |
| 44 | "listo, ya esta arriba, ahora que sigue" |
| 45 | "ellos como tal no tienen ningun archiv, por lo que ese primer pull es para bajarlo de  la rama de archivos temporales?" |
| 46 | "dame un guion, sobre el avance del colab de la semana 11, es un videito de 5 min, lo que sea corta" |
| 47 | [salida de `git push` de Emilio: autenticación fallida] |
| 48 | [salida del push exitoso de Emilio] |
| 49 | "ya quedo?" [captura del PR #1 aprobado y fusionado] |
| 50 | "necesito que me des todos los prompts que he utilizado relaciondo con este poyecto lo he trabajado  en diferentes chats y de acuerdo pasame un reporte de IA USAGE" |

### Qué salió de la IA en la sesión 2
- **Revisión contra el enunciado.** Detectó cinco problemas en el notebook de la semana 10:
  - se usaba solo la mitad de los datos (`Path(a, b)` no carga dos zips);
  - la ablación no corría (`NameError`);
  - el Dice por fragmento estaba bajo la meta, con 60 falsos positivos;
  - el error de distancia de 0.00 mm no demostraba nada, porque los fragmentos estaban en contacto;
  - los conminutos fusionados aparecían como "no detectados".
- **Notebook S8–S10 corregido.** Carga de ambos zips, salidas en Drive con reanudación, MIP rotacional, post-proceso calibrado en validación, análisis de conminutos en contacto, ablación reanudable y exportación para el dashboard.
- **Semana 11.**
  - `pengwin_core.py`, `app.py` y el notebook S11: latencia, visualizadores 2 y 3, peores casos y generación del model card.
  - Dashboard Gradio rediseñado, con el MIP como GIF animado y la carga de un volumen o de sus cortes (DICOM, `.mha`, PNG, `.zip`).
  - `Dockerfile`, `docker-compose` con túnel de Cloudflare, pruebas y CI.
  - Script de HD95/ASSD.
- **Documentos.** `MODEL_CARD.md`, `GUIA_GITHUB.md`, informe técnico (mini-paper de 4 páginas), pitch de 15 diapositivas y guiones de video.
- **Procedimientos.** Pasos de GitHub para tres integrantes con rama temporal y PR por persona, y pasos para correr Docker en Windows.

### Modificaciones del equipo
- Ejecutó todo en Colab y en su PC; los resultados que aparecen en el informe vienen de esas corridas.
- **Decisiones del equipo:**
  - reducir la ablación a 4 épocas y 24 casos;
  - no correr la ablación 2D vs 2.5D por tiempo;
  - no usar *transfer learning*, por falta de pesos;
  - excluir a Andres del equipo;
  - pedir un estilo de pitch más sobrio, con fondo blanco y sin "trabajo futuro";
  - limitar el informe a 4 páginas.
- Corrigió versiones del notebook (no reportar la configuración 2D) y tomó las capturas del dashboard en PC y celular.

### Análisis crítico de la sesión 2
- **Aciertos.**
  - La revisión encontró fallas que las métricas no mostraban: datos a la mitad, ablación sin resultados y distancias triviales.
  - Calibrar el post-proceso en validación subió el Dice por fragmento de 0.562 a 0.627 y bajó los falsos positivos de 60 a 23.
  - El dashboard quedó probado en PC y celular y acepta CT de cualquier profundidad y tamaño.
- **Errores de la IA que el equipo tuvo que corregir o padecer.**
  - La primera versión de la sección 26 llenó la RAM de Colab.
  - El umbral de volumen mínimo se anclaba a un valor atípico de 282 mm³; se cambió al percentil 5.
  - El MIP salió invertido y con la camilla visible.
  - Una columna `gt` chocaba con un método de pandas.
  - Gradio abría un enlace `gradio.live` no pedido.
  - En el dashboard, el MIP no giraba porque Gradio no reproduce animaciones de Plotly.
  - El dashboard se caía si el modelo no detectaba hueso.
  - La primera versión del informe superaba el límite de 4 páginas.
  - El primer pitch tenía un estilo "de IA" y frases de trabajo futuro que el equipo no quería.
  - Comandos de PowerShell que fallaron por permisos de Windows.
  - Estimaciones de tiempo optimistas: ablación de 16 min por configuración, no 8–10.
- **Limitaciones.**
  - La IA no tuvo acceso al dataset ni a la GPU; probó el código solo con CT sintéticos.
  - Las interpretaciones de figuras y las hipótesis de los peores casos fueron revisadas por el equipo.
  - Parte del registro de la sesión se compactó, así que algunos prompts de los días 13 a 15 quedan como fragmentos.
- **Lo que no resolvió.** El Dice por fragmento (0.627) sigue bajo la meta de 0.85: el 91 % de los conminutos toca al hueso principal y 20 de 39 se fusionaron en test. Está declarado en el informe y en el model card.
