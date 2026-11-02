[README.md](https://github.com/user-attachments/files/34296119/README.md)
# Finetuning de Stable Diffusion v1.4 con LoRA: ilustraciones de libros antiguos

Entrega final del módulo **Modelos de Generación de Imagen**.

## Objetivo

Finetunear con la técnica **LoRA** el modelo pre-entrenado [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4) usando el dataset [`gigant/oldbookillustrations`](https://huggingface.co/datasets/gigant/oldbookillustrations), para que genere imágenes con el estilo de las ilustraciones de libros antiguos.

## Modelo finetuneado en Hugging Face

**https://huggingface.co/Roxana80/sd-v1-4-lora-oldbookillustrations**

El repositorio contiene:
- `pytorch_lora_weights.safetensors`: los pesos de la LoRA (unos 5 MB).
- El modelo completo con la LoRA ya fusionada en la UNet (float16), cargable directamente con `from_pretrained`.

## Contenido de este repositorio

| Archivo | Descripción |
|---|---|
| `finetuning_lora_oldbookillustrations.ipynb` | Notebook con todo el proceso: carga de datos, finetuning con LoRA, subida a Hugging Face y carga del modelo desde Hugging Face |
| `link_huggingface.txt` | Link al repositorio de Hugging Face con el modelo finetuneado |
| `comparacion_antes_despues.png` | Mismo prompt y misma semilla, antes y después del finetuning |
| `comparacion_castillo.png` | Comparación con un prompt nuevo, usando el modelo cargado desde Hugging Face |
| `grafica_loss.png` | Evolución de la loss media por época |

## Adaptaciones respecto al código visto en clase

1. **Columnas del dataset.** Solo se cargan `1600px` (imagen) e `info_alt` (descripción), con `load_dataset(..., columns=[...])`. El dataset incluye también la columna `rawscan`, con escaneos de hasta unos 11.000 px de ancho. Si no se excluye, cada acceso a una fila decodifica también esa imagen, lo que hace el entrenamiento mucho más lento y consume mucha RAM.
2. **Imágenes no cuadradas.** En lugar de `Resize((512, 512))`, que deforma la imagen, se usa `Resize(512)` seguido de `CenterCrop(512)`. Primero se escala el lado corto a 512 manteniendo la proporción y después se recorta el centro. Por ejemplo, una imagen de 1158×1600 pasa a 512×707 y queda en 512×512 sin deformación.
3. **Conversión a RGB.** Todas las imágenes se convierten con `.convert("RGB")`, porque el modelo espera 3 canales y algunas imágenes pueden venir en escala de grises (1 canal).
4. **Descripciones largas.** Algunas descripciones superan los 3.000 caracteres. Se tokenizan con `truncation=True` para ajustarlas al límite de 77 tokens de CLIP.
5. **Subconjunto aleatorio.** Se barajó el dataset (`shuffle(seed=42)`) y se seleccionaron 600 imágenes, para mezclar estilos de muchos libros distintos y mantener un tiempo de entrenamiento razonable en Colab.
6. **Guardado por época.** Los pesos de la LoRA se guardan en Google Drive al final de cada época, para no perder el entrenamiento si Colab se desconecta.

## Configuración del entrenamiento

| Parámetro | Valor |
|---|---|
| GPU | NVIDIA Tesla T4 (Google Colab) |
| Imágenes de entrenamiento | 600 (de unas 4.150 del dataset) |
| Batch size | 3 |
| Épocas completadas | 3 |
| Rango LoRA (`r`) | 6 |
| `lora_alpha` | 6 |
| Learning rate | 1e-4 |
| Resolución | 512 × 512 |
| Capas con LoRA | `to_q`, `to_k`, `to_v`, `to_out.0` (atención de la UNet) |
| Parámetros entrenables | 1.195.776 de 860.716.740 (**0,14 %**) |
| Tiempo | ~4,1 s por paso, ~14 min por época, ~41 min en total |

El entrenamiento se detuvo manualmente al terminar la tercera época por limitaciones de tiempo y de cupo de GPU en Colab. Por eso la celda de entrenamiento del notebook termina con una interrupción (`KeyboardInterrupt`). La LoRA usada es la del final de la tercera época.

## Resultados

### Loss

| Época | Loss media |
|---|---|
| 1 | 0,1451 |
| 2 | 0,1436 |
| 3 | 0,1480 |

![Evolución de la loss](grafica_loss.png)

La loss se mantiene estable. Las diferencias entre épocas (±0,004) son menores que la variación aleatoria esperable: en cada paso el timestep y el ruido se eligen al azar, y la loss de cada paso oscila aproximadamente entre 0,1 y 0,5. En el finetuning de modelos de difusión el efecto del entrenamiento se aprecia sobre todo en las imágenes generadas, no en la loss.

### Antes y después del finetuning (mismo prompt, semilla 42)

![Comparación antes y después](comparacion_antes_despues.png)

### Prompt nuevo, con el modelo cargado desde Hugging Face

Prompt: *"a castle on a hill surrounded by a dark forest"*

![Comparación castillo](comparacion_castillo.png)

## Conclusiones

1. **El finetuning cambia el estilo y respeta la composición.** Con el mismo prompt y la misma semilla, el modelo pasa de imágenes fotográficas a imágenes con estilo de ilustración antigua (tonos terrosos o sepia, trazos visibles, ambientación de época), pero conserva la composición general de la escena.
2. **La LoRA es muy eficiente.** Se ha modificado el comportamiento del modelo entrenando solo el 0,14 % de los parámetros de la UNet, con un archivo final de unos 5 MB, frente a los varios GB del modelo completo.
3. **El efecto depende del tema del prompt.** Es más fuerte en temas habituales de los libros antiguos (un castillo en un bosque, con tramado y tonos sepia típicos del grabado) que en temas que el modelo base asocia mucho con fotografías (una oveja en un prado).
4. **Posibles mejoras.** Con más épocas, más imágenes o un rango `r` mayor, el estilo de grabado en blanco y negro debería marcarse más. También se podría añadir una palabra clave fija a las descripciones (por ejemplo, *"old book illustration"*) para activar el estilo de forma controlada.

## Cómo usar el modelo

Modelo base + LoRA:

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("Roxana80/sd-v1-4-lora-oldbookillustrations")

image = pipe("a castle on a hill surrounded by a dark forest").images[0]
```

Modelo completo (LoRA ya fusionada):

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "Roxana80/sd-v1-4-lora-oldbookillustrations", torch_dtype=torch.float16
).to("cuda")
```
