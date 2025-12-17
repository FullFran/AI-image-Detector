# Entrenamiento con Imágenes Reales

Para que el detector funcione con precisión en el mundo real, necesita aprender de ejemplos reales.

## Instrucciones

1.  **Imágenes Reales**: Coloca fotografías reales (fotos de cámara, sin editar o editadas mínimamente) en la carpeta `real/`.

    - Ejemplos: Paisajes, retratos, objetos cotidianos.
    - Formatos soportados: PNG, JPG, JPEG.
    - Recomendado: Al menos 20 imágenes variadas.

2.  **Imágenes IA (Fake)**: Coloca imágenes generadas por IA (Midjourney, DALL-E, Stable Diffusion, Flux) en la carpeta `fake/`.
    - Ejemplos: Imágenes generadas con prompts variados.
    - Recomendado: Al menos 20 imágenes.

## Nota Importante

El modelo se re-entrenará automáticamente **cada vez que reinicies el servicio (API)**. Si estas carpetas están vacías, el sistema utilizará datos sintéticos (ruido matemático) que son solo para demostración y no funcionan bien con fotos reales.
