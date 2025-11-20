# Repair SQL Module

Este módulo implementa la **Etapa 4** del pipeline LinkAlign: detección y reparación automática de problemas semánticos en consultas SQL generadas.

## 🎯 Propósito

Cuando `gen_sql` genera una consulta sintácticamente correcta que ejecuta sin errores pero devuelve 0 filas, el módulo `repair_sql` analiza la combinación de:
- Pregunta en lenguaje natural
- SQL generada
- Metadata del esquema
- Resultados de ejecución

Para detectar y corregir problemas semánticos comunes como:
- Valores de enum inválidos (ej: "California" → "CA")
- Filtros de año faltantes
- Otros patrones heurísticos

## 📁 Estructura

```
repair_sql/
├── __init__.py          # API pública: should_trigger_repair(), repair_sql()
├── config.py            # RepairConfig con parámetros configurables
├── types.py             # Dataclasses: RepairInput, RepairResult, RepairIssue, etc.
├── detection.py         # Detecta problemas en la SQL
├── mappers.py           # Mapea literales inválidos a valores válidos
├── rules.py             # Reglas heurísticas para generar parches
├── patching.py          # Aplica parches y re-ejecuta
└── signals.py           # Construye señales para el SCM
```

## 🚀 Uso Básico

### 1. Verificar si se debe reparar

```python
from modules.repair_sql import should_trigger_repair

# Señales de gen_sql y exec_sql
gen_signals = {"primary_format_ok": 1.0, "risk_score": 0.3}
exec_signals = {"rows_empty": 1.0, "exec_error": 0.0}

if should_trigger_repair(gen_signals, exec_signals):
    # Proceder con reparación
    pass
```

### 2. Reparar SQL

```python
from modules.repair_sql import repair_sql
from modules.repair_sql.config import RepairConfig

# Configurar exec_runner
def exec_runner(sql: str):
    # Ejecutar SQL y devolver ExecutionResult
    return execute_query(sql, db_config, engine="snowflake")

config = RepairConfig(
    enable_enum_repairs=True,
    enable_year_repairs=True,
    exec_runner=exec_runner,
)

result = repair_sql(
    question="How many babies named John in California in 2000?",
    original_sql='SELECT COUNT(*) FROM usa_names WHERE name = \'John\' AND state = \'California\'',
    expected_shape={"kind": "scalar"},
    schema_context=schema_context,  # De inspect_schema
    gen_signals=gen_signals,
    exec_result=exec_result,
    exec_signals=exec_signals,
    engine="snowflake",
    config=config,
)

if result.applied:
    print(f"Repaired SQL: {result.repaired_sql}")
    print(f"New row count: {result.repaired_exec_result.row_count}")
```

## 🔧 Configuración

### RepairConfig

```python
@dataclass
class RepairConfig:
    enable_enum_repairs: bool = True           # Mapear valores enum inválidos
    enable_year_repairs: bool = True           # Agregar filtros de año faltantes
    enable_llm_enum_mapper: bool = False       # Usar LLM para mapeo de enums
    max_enum_values_per_column: int = 100      # Máx valores para tratar como enum
    llm_mapper_pipeline: Optional[Any] = None  # Pipeline HuggingFace para LLM
    exec_runner: Optional[Callable] = None     # Función para ejecutar SQL
```

## 📊 Señales del Módulo

El módulo genera señales numéricas para el **Strategic Control Module (SCM)**:

- `repair_applied`: 1.0 si se aplicó un parche, 0.0 si no
- `repair_success`: 1.0 si la reparación mejoró el resultado
- `repair_row_count_delta`: Cambio en el número de filas
- `repair_exec_latency_delta_ms`: Cambio en el tiempo de ejecución
- `repair_used_year_rule`: 1.0 si se aplicó regla de año
- `repair_used_enum_rule`: 1.0 si se aplicó regla de enum
- `repair_used_llm_mapping`: 1.0 si se usó LLM para mapeo

## 🧪 Tipos de Reparaciones

### 1. Filtros de Año Faltantes

**Problema detectado:**
- Pregunta menciona un año (ej: "in 2000")
- Existe columna `year` en el esquema
- SQL no filtra por año

**Solución:**
```sql
-- Original
SELECT COUNT(*) FROM usa_names WHERE state = 'CA'

-- Reparada
SELECT COUNT(*) FROM usa_names WHERE state = 'CA' AND "year" = 2000
```

### 2. Valores Enum Inválidos

**Problema detectado:**
- Columna marcada como `safe_for_enum_constraints`
- Literal usado no está en `sample_values`

**Solución (vía diccionario):**
```sql
-- Original
SELECT * FROM usa_names WHERE state = 'California'

-- Reparada
SELECT * FROM usa_names WHERE state = 'CA'
```

**Mapeos de diccionario soportados:**
- **Estados USA**: "California" → "CA", "Texas" → "TX", etc.
- **Género**: "female" → "F", "male" → "M"

**Solución (vía LLM):**
Para columnas desconocidas, el LLM puede sugerir el valor correcto del conjunto `sample_values`.

## 📋 Tipos de Datos

### RepairInput
Entrada completa para el módulo:
```python
@dataclass
class RepairInput:
    question: str                          # Pregunta original
    original_sql: str                      # SQL generada
    expected_shape: Dict[str, Any]         # Shape esperado
    schema_context: SchemaContext          # Metadata del esquema
    gen_signals: Dict[str, float]          # Señales de gen_sql
    exec_result: ExecutionResult           # Resultado de ejecución
    exec_signals: Dict[str, float]         # Señales de exec_sql
    engine: str                            # "snowflake" o "bigquery"
    config: RepairConfig                   # Configuración
```

### RepairResult
Salida del módulo:
```python
@dataclass
class RepairResult:
    applied: bool                                  # ¿Se aplicó reparación?
    original_sql: str                              # SQL original
    repaired_sql: Optional[str]                    # SQL reparada
    original_exec_result: ExecutionResult          # Resultado original
    repaired_exec_result: Optional[ExecutionResult] # Resultado reparado
    issues: List[RepairIssue]                      # Problemas detectados
    patch: Optional[RepairPatch]                   # Parche aplicado
    repair_signals: Dict[str, float]               # Señales para SCM
    debug_info: Dict[str, Any]                     # Info de debug
```

### RepairIssue
Problema detectado:
```python
@dataclass
class RepairIssue:
    issue_type: str                    # "enum_value_mismatch", "missing_year_filter", etc.
    column: Optional[str]              # Columna involucrada
    table: Optional[str]               # Tabla involucrada
    value_used: Optional[str]          # Valor problemático
    suggested_values: List[str]        # Valores válidos sugeridos
    question_value: Optional[Any]      # Valor extraído de la pregunta
    details: Dict[str, Any]            # Info adicional
```

## 🧪 Testing

Ejecutar tests:
```bash
# Todos los tests
pytest link/tests/test_repair_sql.py -v

# Test específico
pytest link/tests/test_repair_sql.py::test_detect_missing_year_filter -v

# Con cobertura
pytest link/tests/test_repair_sql.py --cov=src.modules.repair_sql --cov-report=html
```

## 🎬 Demo

Ejecutar demo completo con reparación:
```bash
# Ejemplo simple
python link/demo_repair_sql_snowflake.py --question "How many babies in California in 2000?"

# Modo interactivo
python link/demo_repair_sql_snowflake.py --interactive

# Con mapeo LLM habilitado
python link/demo_repair_sql_snowflake.py --enable-llm-mapper --question "..."
```

## 🔄 Integración con el Pipeline

```python
# Stage 1: Schema
inspect_result = inspect_schema(question, db_config, llm_pipeline, tokenizer)

# Stage 2: Generation
gen_result = gen_sql(question, inspect_result["schema_context"], llm_pipeline, tokenizer)

# Stage 3: Execution
exec_result = exec_sql(question, gen_result["sql"], gen_result["expected_shape"], db_config)

# Stage 4: Repair (condicional)
if should_trigger_repair(gen_result["gen_signals"], exec_result["exec_signals"]):
    repair_result = repair_sql(
        question=question,
        original_sql=gen_result["sql"],
        expected_shape=gen_result["expected_shape"],
        schema_context=inspect_result["schema_context"],
        gen_signals=gen_result["gen_signals"],
        exec_result=exec_result["result"],
        exec_signals=exec_result["exec_signals"],
        engine="snowflake",
        config=repair_config,
    )
```

## 📝 Metadata Requerida

Para que el módulo funcione correctamente, `inspect_schema` debe proporcionar:

1. **sample_values** en `ColumnMetadata.extra`:
   ```python
   column.extra["sample_values"] = ["CA", "TX", "NY", ...]
   ```

2. **profile** en `ColumnMetadata.extra`:
   ```python
   column.extra["profile"] = ColumnProfile(
       semantic_role="enum",           # "enum", "temporal", "measure", etc.
       safe_for_enum_constraints=True, # Usar sample_values como dominio cerrado
       safe_for_repair_mapping=True,   # Usar en reparaciones
   )
   ```

## 🚧 Limitaciones Actuales

1. **Un parche a la vez**: Solo aplica year O enum, no ambos simultáneamente
2. **Reemplazo simple**: Solo reemplaza primera ocurrencia del literal
3. **Heurísticas básicas**: Reglas limitadas a year y enum
4. **Sin re-generación**: No vuelve a llamar al LLM, solo modifica SQL

## 🔮 Mejoras Futuras

1. **Encadenamiento de parches**: Aplicar múltiples reglas en secuencia
2. **Reglas adicionales**: 
   - Joins faltantes
   - Agregaciones incorrectas
   - Problemas de NULL handling
3. **Feedback al generador**: Usar issues detectados como hints para re-generación
4. **Ranking de parches**: Generar múltiples candidatos y elegir el mejor
5. **Aprendizaje**: Usar éxitos/fallos para refinar heurísticas

## 📚 Referencias

- [LinkAlign Paper](https://arxiv.org/abs/2310.00123) - Sección sobre repair strategies
- [Spider Dataset](https://yale-lily.github.io/spider) - Análisis de errores comunes
- [Enum Detection](./detection.py) - Implementación de detección de enums
- [SCM Integration](../exec_sql/README.md) - Cómo integrar señales con SCM
