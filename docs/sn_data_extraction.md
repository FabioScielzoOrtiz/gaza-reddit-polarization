
## Extracción de datos

Los datos primarios serán extraidos de redes sociales

### **Twitter (X)**

#### **API Oficial**

- Tiene una API oficial para la extracción de datoos, pero desde mediados del 2023 cambió su política, eliminando el plan de research, que permetía usarla de forma gratuita para fines educativos y de investigación.

- Actualmente hay 3 planes: 

 | Plan      | Coste      | Límite de extracción de posts | Observaciones                                        |
| --------- | ---------- | ----------------------------- | ---------------------------------------------------- |
| **Free**  | 0 $        | 100 posts / mes               | Solo sirve para pruebas muy pequeñas.                |
| **Basic** | 200 $/mes  | 15 000 posts / mes            | Suficiente para un dataset inicial de investigación. |
| **Pro**   | 5000 $/mes | 1 000 000 posts / mes         | Enfocado a empresas y grandes volúmenes.             |


El plan Basic (200 $/mes) es el único viable hoy para investigación seria sobre Twitter/X.

Te da acceso a:

 - 15 000 tweets/mes,

 - Búsquedas recientes (últimos 7 días),

 - Metadatos (autor, idioma, engagement, etc.),

 - 2 entornos de desarrollo (puedes tener proyectos separados).

#### **`snscrape`**

Principal alternativa de codigo abierto a la API oficial antes del cambio de política de la misma.

Tras el cambio de política de la API oficial, `snscrape` está inhabilitado por Twitter, de modo que ya no funciona.


#### **Alternativa manual**

Obtener datos manualmente de un conjunto de actores de interes (politicos, periodistas, influencers...).

Estos datos complementarían los extraidos a través de otras fuentes.



### Reddit

Actualmente es la alternativa más robusta y utilizada en investigación académica como reemplazo de X/Twitter.

API: Totalmente abierta, gratuita y excelentemente documentada. La librería PRAW (Python Reddit API Wrapper) facilita enormemente la extracción de datos.

Datos Mixtos (Ejemplos):

Texto: submission.title (título del post), submission.selftext (cuerpo del post) y comment.body (cuerpo de los comentarios).

Cuantitativo (Quant): score (puntuación neta del post/comentario), upvote_ratio (ratio de votos positivos), num_comments (nº de comentarios en un post), author.karma (karma del autor).

Categórico (Cat): subreddit (la comunidad específica, p.ej., r/es, r/spain, r/politics), link_flair_text (etiqueta del post asignada por el usuario o moderadores, p.ej., "Política", "Debate").

Ventajas:

Estructura Natural: Los subreddits actúan como "clusters" temáticos naturales, permitiendo acotar la extracción de forma muy precisa (p.ej., comparar el discurso sobre Gaza en r/es vs r/europe).

Riqueza Textual: El relativo anonimato fomenta discusiones más honestas, detalladas y, a veces, crudas sobre temas sensibles, lo cual es excelente para el análisis textual.

Desafíos: El perfil demográfico de Reddit presenta sesgos conocidos (históricamente más joven, masculino y con afinidad tecnológica) que deben ser declarados como una limitación del estudio.

### Mastodon

La principal alternativa descentralizada y de código abierto. No es una sola entidad, sino una red de servidores (instancias) que se comunican entre sí.

API: Totalmente abierta, gratuita y bien documentada. Sigue estándares muy similares a la antigua API de Twitter v1, lo que facilita la migración de scripts y metodologías.

Datos Mixtos (Ejemplos):

Texto: Contenido del "toot" (el post).

Cuantitativo (Quant): boosts (equivalente a retweets), favorites (likes), replies_count (nº respuestas).

Categórico (Cat): instancia_origen (el servidor del usuario, p.ej., mstdn.es), visibilidad (público, no listado, solo seguidores), hashtags (usados intensivamente para la descubribilidad).

Ventajas:

Novedad Metodológica: El estudio de la propagación de información a través de instancias (federación) es un campo de investigación nuevo y publicable.

Ética de Datos: Su naturaleza abierta y centrada en el usuario es preferida éticamente.

Desafíos: Fragmentación. No existe un firehose central. La recolección debe hacerse instancia por instancia, seguir a usuarios específicos o monitorear hashtags a través de la red, lo que la hace más compleja.

### Bluesky

Una alternativa emergente, creada por el cofundador de Twitter (Jack Dorsey), basada en un protocolo abierto y descentralizado (AT Protocol).

API: El "Protocolo AT" (Authenticated Transfer Protocol) es la base de la plataforma y está diseñado para ser abierto y permitir la interoperabilidad y extracción de datos.

Datos Mixtos (Ejemplos):

Texto: Contenido del post (text).

Cuantitativo (Quant): likeCount (likes), repostCount (reposts).

Categórico (Cat): Custom Feeds. Una característica única es que los usuarios pueden crear y suscribirse a feeds algorítmicos personalizados. Analizar qué feeds se crean sobre un tema (p.ej., "GazaES") es una variable categórica nueva y potente.

Ventajas: Alto factor de novedad (pocos estudios publicados). La característica de los feeds personalizados ofrece un ángulo de análisis único sobre cómo los usuarios curan su propia información.

Desafíos: Base de usuarios mucho menor que X o Reddit. Se debe realizar una validación preliminar para asegurar que existe una masa crítica de conversación en español sobre el tema de estudio.

### Telegram

Plataforma de mensajería que se ha convertido en una herramienta clave para la comunicación unidireccional (canales) y comunitaria (grupos) de actores políticos y mediáticos.

API: Dispone de dos APIs muy potentes:

Bot API: Para crear bots (no útil para leer historiales ajenos).

Core API (MTProto): Permite actuar como un usuario. Mediante librerías como Telethon o Pyrogram (Python), se puede leer el historial completo de canales y grupos públicos.

Datos Mixtos (Ejemplos):

Texto: Contenido del mensaje.

Cuantitativo (Quant): views (nº de vistas del mensaje), recuento de reacciones (p.ej., nº de 👍, 🔥, o 👎).

Categórico (Cat): channel_id (identificador del canal de origen), message_type (texto, link, foto, vídeo), reaction_type (el set de emojis usado en las reacciones).

Ventajas: Permite capturar el discurso "oficial" o de élite. Es la fuente primaria para saber qué están comunicando los partidos políticos, medios de comunicación y grupos activistas clave a sus seguidores más fieles.

Desafíos: No es una "plaza pública" que se pueda buscar globalmente (p.ej., "buscar 'Gaza' en todo Telegram"). El investigador debe identificar y listar manualmente los canales y grupos públicos de interés antes de iniciar la extracción.