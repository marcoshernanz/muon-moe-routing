# Cambios en la clase tfgtfmthesisuam

## v3.2 (05/10/2026)

Modernización para TeX Live y Overleaf actuales, sin cambiar el aspecto del documento:
el manual se ha comparado página a página con la versión anterior.

### Cambio de sintaxis: subfiguras y subtablas

El antiguo comando `\subfigure[etiqueta]{pie}{contenido}` (y `\subtable{etiqueta}{pie}{contenido}`)
dependía del paquete `subfigure`, obsoleto desde 2005, que ya no funciona con TeX Live actual.
Se sustituye por los entornos estándar del paquete `subcaption`:

```latex
\begin{figure}{FIG:EJEMPLO}{Pie de la figura completa}
  \begin{subfigure}[b]{0.45\linewidth}
    \centering
    \image{3cm}{}{logoEPS}
    \caption{Pie de la subfigura}\label{SBFIG:UNA}
  \end{subfigure}
  \begin{subfigure}[b]{0.45\linewidth}
    ...
  \end{subfigure}
\end{figure}
```

Los documentos que usen la sintaxis antigua deben adaptarse. Las referencias siguen saliendo
como "figura 4.2(b)".

### Errores corregidos

- **El manual no compilaba** con TeX Live 2023 o posterior (`\cftsubtabfillnum` indefinido) por
  la combinación de `subfigure` y `tocloft[subfigure]`.
- **Referencias a capítulos equivocadas**: una `\label` dentro del título de un capítulo o
  apartado se volvía a definir en cada cabecera de página y en el índice, así que `\ref` acababa
  apuntando a otro sitio (el manual decía "apartado 4.8" donde debía decir "capítulo 4").
- **Entradas basura "(a) lof", "(b) lof"** en las listas de figuras y tablas por cada subfigura.
- **"Lista de cuadros" repetida** en español, y **ausente** en inglés.
- **Metadatos del PDF duplicados y con acentos rotos**: se escribían a mano con `\pdfinfo` además
  de con hyperref. El título salía corrupto, los marcadores con caracteres extraños
  ("IntroducciÃ³n") y el productor era "Eloy Anguiano Rey" en todas las memorias.
- **Enlaces internos a la página equivocada**: la numeración se reinicia (romanos/arábigos) y
  había dos destinos "página 1".
- **El entorno `algorithm` fallaba** con algorithm2e 5.x (`\SetVline` ya no existe).
- La opción `makeidx` se pasaba a la clase `book`, que no la conoce (aviso "Unused global option").

### Modernización

- Codificación `utf8` en lugar de `utf8x` (paquete `ucs`, abandonado).
- Eliminados paquetes obsoletos o sin uso: `times` (se fijan las mismas familias), `subfigure`,
  `etex`, `ifpdf`, `filecontents` (incluido en LaTeX), `grffile`.
- Con XeLaTeX o LuaLaTeX, error claro ("pdfTeX is required") en vez de un fallo críptico.
- Requiere LaTeX 2020-10 o posterior (Overleaf y cualquier TeX Live desde 2021).

### Ficheros

- Los logos y fotos de portada están solo en `img/` (antes, duplicados en la raíz). La clase los
  busca en `\logosdir`, luego en `img/` y por último en la carpeta del documento.
- Eliminados los ficheros generados de `gnuplottex/`.
- Nuevo esqueleto de memoria: `memoria_tfg.tex`.
- Paquete `tfgmarcas.sty`: marcas invisibles para la comprobación automática del formato en TFGTool.

## v3.1 (04/10/2026)

- Marcas invisibles de TFGTool (portada, inicio y fin del cuerpo, páginas en blanco).
