

# TokenSmith 🔧

> Un kit de herramientas integral para agilizar la edición, búsqueda e inspección de datos para el entrenamiento de modelos de lenguaje a gran escala y la interpretabilidad.

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descripción general

TokenSmith es un paquete de Python potente diseñado para simplificar la gestión de conjuntos de datos para el entrenamiento de modelos de lenguaje de gran escala. Proporciona una interfaz unificada para editar, inspeccionar, buscar, muestrear y exportar conjuntos de datos tokenizados, facilitando el trabajo con datos de entrenamiento a gran escala.

## ✨ Características principales

- **🔍 Búsqueda e Indexación**: Búsqueda rápida de secuencias de tokens con indexación n-grama
- **📊 Inspección de conjuntos de datos**: Examine muestras, lotes y metadatos de documentos  
- **🎯 Muestreo inteligente**: Muestreo flexible con selección basada en políticas
- **✏️ Edición de conjuntos de datos**: Inyecte y modifique muestras de entrenamiento con precisión
- **📤 Utilidades de exportación**: Exporte datos en múltiples formatos
- **📩 Utilidades de ingestión**: Ingestione datos desde múltiples formatos
- **🖥️ Interfaz interactiva**: Interfaz web basada en Streamlit para exploración visual
- **⚡ Eficiente en memoria**: Procesamiento por fragmentos para conjuntos de datos grandes

## 🏗️ Arquitectura

TokenSmith está construido alrededor de un `DatasetManager` central que coordina cinco controladores especializados:

```
DatasetManager
├── SearchHandler    # Búsqueda e indexación de secuencias de tokens
├── InspectHandler   # Examen y visualización del conjunto de datos  
├── SampleHandler    # Estrategias flexibles de muestreo de datos
├── EditHandler      # Modificación e inyección del conjunto de datos
└── ExportHandler    # Exportación de datos en múltiples formatos
└── IngestHandler    # Ingestión de datos en múltiples formatos
```

## 🚀 Inicio rápido

### Instalación

`TokenSmith` se puede instalar de varias maneras según su caso de uso.

Nota: Excepto por la función de búsqueda, todas las características asumen que GPT-NeoX está instalado para usar Megatron. Puede hacerlo simplemente siguiendo los pasos proporcionados [aquí](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#environment-and-dependencies).

## 1. Instalación básica (solo núcleo)

Si solo necesita la **funcionalidad principal** (edición de datos, muestreo, importación, exportación, inspección):

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e .
```

## 2. Con dependencias de documentación

Si planea compilar o servir la documentación localmente:

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e ".[docs]"
```

Una vez instalado, puede compilar y servir la documentación:

```bash
mkdocs serve
```

## 3. Con componentes de interfaz de usuario

Si desea la **interfaz interactiva** para explorar datos:

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e ".[ui]"
```

## 4. Con funciones de búsqueda

Para **búsqueda avanzada a nivel de token y utilidades n-grama**:

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e ".[search]"
```

## 5. Instalación completa (todo)

Para instalar **todas las funciones opcionales** (no incluye la documentación):

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e ".[all]"
```

Esto incluye la documentación, la interfaz de usuario y los extras de búsqueda.

## 6. Instalación para desarrollo

Si está colaborando en `tokensmith`:

```bash
git clone https://github.com/aflah02/tokensmith.git
cd tokensmith
pip install -e ".[all,docs]"
```

Esto configura un entorno local con todos los extras para el desarrollo.

## 🚀 Inicio rápido en Modal

Proporcionamos un proyecto de ejemplo para ayudarle a configurar rápidamente TokenSmith en Modal, una plataforma en la nube sin servidor, utilizando su función de cuadernos. Para comenzar, siga las instrucciones en el directorio `modal_example`.

## 📚 Funcionalidad principal

### 🔍 Operaciones de búsqueda

```python
# Buscar secuencias de tokens
query = [101, 2023, 102]  # IDs de tokens
count = manager.search.count(query)
positions = manager.search.positions(query)
contains = manager.search.contains(query)

# Obtener distribuciones de tokens siguientes
next_tokens = manager.search.count_next(query)
```

### 📊 Inspección de conjuntos de datos

```python
# Inspeccionar muestras individuales
sample = manager.inspect.inspect_sample_by_id(
    sample_id=42,
    return_detokenized=True,
    tokenizer=tokenizer,
    return_doc_details=True
)

# Inspeccionar lotes completos
batch = manager.inspect.inspect_sample_by_batch(
    batch_id=0,
    batch_size=32,
    return_detokenized=True,
    tokenizer=tokenizer
)
```

### 🎯 Muestreo inteligente

```python
# Muestrear por índices específicos
samples = manager.sample.get_samples_by_indices(
    indices=[1, 5, 10, 42],
    return_detokenized=True,
    tokenizer=tokenizer
)

# Muestrear lotes por ID
batches = manager.sample.get_batches_by_ids(
    batch_ids=[0, 1, 2],
    batch_size=32,
    return_detokenized=True,
    tokenizer=tokenizer
)

# Muestreo basado en políticas
def random_policy(n_samples):
    import random
    return random.sample(range(1000), n_samples)

policy_samples = manager.sample.get_samples_by_policy(
    policy_fn=random_policy,
    n_samples=10,
    return_detokenized=True,
    tokenizer=tokenizer
)
```

### ✏️ Edición de conjuntos de datos

```python
# Inyectar texto en ubicaciones específicas
manager.edit.inject_and_preview(
    text="This is injected content",
    tokenizer=tokenizer,
    injection_loc=100,
    injection_type="seq_shuffle",  # o "seq_start"
    dry_run=False
)
```

### 📤 Exportación de datos

```python
# Exportar lotes específicos
manager.export.export_batches(
    batch_ids=[0, 1, 2],
    batch_size=32,
    output_path="exports/batches.jsonl",
    format_type="jsonl",
    return_detokenized=True,
    tokenizer=tokenizer,
    include_doc_details=True
)

# Exportar rangos de secuencias
manager.export.export_sequence_range(
    start_idx=0,
    end_idx=1000,
    output_path="exports/sequences.csv",
    format_type="csv",
    return_detokenized=True,
    tokenizer=tokenizer
)

# Exportar conjunto de datos completo (en fragmentos)
manager.export.export_entire_dataset(
    output_path="exports/full_dataset.jsonl",
    format_type="jsonl",
    return_detokenized=True,
    tokenizer=tokenizer,
    chunk_size=1000
)
```

## 🖥️ Interfaz web interactiva

TokenSmith incluye una interfaz web basada en Streamlit para la exploración visual de conjuntos de datos:

```bash
# Iniciar la interfaz web utilizando el script de conveniencia
cd tokensmith/ui
./run_ui.sh
```

Modifique `run_ui.sh` para cambiar modos y argumentos

La interfaz web proporciona:
- **Página de búsqueda**: Búsqueda interactiva de secuencias de tokens con visualización
- **Página de inspección**: Explore y examine muestras y lotes del conjunto de datos
- **Página ver documentos**: Vea documentos individuales en orden de entrenamiento o de corpus

## 🗂️ Estructura del proyecto

```
tokensmith/
├── manager.py              # Central DatasetManager class
├── utils.py                # Utility functions and classes
├── edit/                   # Dataset editing functionality
│   └── handler.py
├── inspect/                # Dataset inspection tools
│   └── handler.py
├── search/                 # Search and indexing
│   └── handler.py
├── sample/                 # Sampling strategies
│   └── handler.py
├── export/                 # Data export utilities
│   └── handler.py
├── ingest/                 # Data ingestion utilities
│   └── handler.py
└── ui/                     # Streamlit web interface
    ├── app.py
    └── pages/
        ├── search.py
        └── inspect.py
        └── view_documents.py
```

## 📖 Documentación

### Referencia de la API

La documentación completa de la API con docstrings generados automáticamente está disponible en:
**[https://aflah02.github.io/TokenSmith](https://aflah02.github.io/TokenSmith)**

### Tutoriales

Los tutoriales y ejemplos completos están disponibles en el directorio `tutorials/`:

- **[Tutorial de configuración básica](docs/tutorials/01_basic_setup.ipynb)** 
- **[Tutorial de inspección de conjuntos de datos](docs/tutorials/02_inspect_samples.ipynb)** 
- **[Tutorial de muestreo de conjuntos de datos](docs/tutorials/03_sampling_methods.ipynb)**
- **[Tutorial de edición de conjuntos de datos](docs/tutorials/04_dataset_editing_methods.ipynb)**
- **[Tutorial de búsqueda en conjuntos de datos](docs/tutorials/05_search_functionality.ipynb)**


### Compilación de documentación localmente

Para compilar y servir la documentación localmente:

```bash
# Asegúrese de instalar la documentación con el comando apropiado mencionado anteriormente
# Servir localmente (se recarga automáticamente ante cambios)
mkdocs serve
# o use el script de conveniencia
./serve-docs.sh
```

La documentación estará disponible en `http://127.0.0.1:8000`

## 🤝 Contribuir

¡Agradecemos las contribuciones! Consulte nuestras [Líneas directrices para contribuir](CONTRIBUTING.md) para obtener más detalles.

1. Realice un fork del repositorio
2. Cree una rama de características (`git checkout -b feature/amazing-feature`)
3. Confirme sus cambios (`git commit -m 'Agregar función increíble'`)
4. Envíe a la rama (`git push origin feature/amazing-feature`)
5. Abra una solicitud de extracción (Pull Request)

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia Apache 2.0, consulte [aquí](https://www.apache.org/licenses/LICENSE-2.0) para obtener más detalles.

## 🙏 Agradecimientos

- Construido sobre la biblioteca [tokengrams](https://github.com/EleutherAI/tokengrams) para una indexación n-grama eficiente
- Utiliza indexación de conjuntos de datos al estilo Megatron para compatibilidad con las tuberías de entrenamiento existentes

## 📞 Soporte

- 🐛 **Informes de problemas**: [GitHub Issues](https://github.com/aflah02/tokensmith/issues)
- 📖 **Documentación**: [https://aflah02.github.io/TokenSmith](https://aflah02.github.io/TokenSmith)

## ℹ️ Cita

Si encuentra útil esta biblioteca o se basa en ella, recuerde citar nuestro trabajo:

```
@misc{khan2025tokensmithstreamliningdataediting,
      title={TokenSmith: Streamlining Data Editing, Search, and Inspection for Large-Scale Language Model Training and Interpretability}, 
      author={Mohammad Aflah Khan and Ameya Godbole and Johnny Tian-Zheng Wei and Ryan Wang and James Flemings and Krishna Gummadi and Willie Neiswanger and Robin Jia},
      year={2025},
      eprint={2507.19419},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19419}, 
}
```
