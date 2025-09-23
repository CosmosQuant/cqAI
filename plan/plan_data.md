
```python
dm = DataManager({'binance': BinanceDataSource(), 'okx': OKXDataSource()})
df = dm.load_data(source_name='binance')  # Factory selects BinanceDataSource
```

------------------------------------------------------------------------------------------------
## Design Patterns Used

### 🏭 **Factory Pattern (Abstract Factory)**
- **`DataManager`** acts as a factory that creates/selects appropriate data processors
- **`DataSource`** subclasses are concrete factories for different exchanges
- **Benefits**: Decouples object creation from usage, easy to extend with new data sources


### 🔄 **Strategy Pattern**
- **`DataManager`** is the Context
- **Different `DataSource` implementations** are different strategies
- **Benefits**: Runtime switching of data processing strategies
### 🎭 **Facade Pattern**
- **`DataManager`** provides a simple interface to complex data loading operations
- **Benefits**: Hides complexity from client code

### 🔌 **Adapter Pattern**
- **`DataSource.to_standard()`** adapts different data formats to unified schema
- **Column mapping** adapts exchange-specific columns to standard format
- **Benefits**: Makes incompatible interfaces work together
### 🏗️ **Template Method Pattern**
- **`DataSource.to_standard()`** defines the algorithm skeleton
- **Subclasses** can override specific steps if needed
- **Benefits**: Code reuse while allowing customization

### 📋 **Registry Pattern**
- **`DataManager.sources`** maintains a registry of available data sources
- **Benefits**: Dynamic registration and lookup of data processors


------------------------------------------------------------------------------------------------
## Future Extensions

1. **Real-time Data**: Add streaming data support
2. **Database Integration**: Add database data sources  
3. **Cloud Storage**: Add S3/GCS data sources
4. **Data Caching**: Implement intelligent caching
5. **Parallel Processing**: Add multi-threading support
6. **Data Versioning**: Track data lineage and versions

### TODO / Next
- Add more data source implementations (OKX, Bybit, etc.)
- Add data caching mechanisms for faster re-loading
- Performance benchmarking and memory usage optimization - Arrow/Parquet route (fast & memory-light)


### TODO - Factory implementation:
```python
SOURCE_REGISTRY: dict[str, type[DataSource]] = {}

def register_source(name: str):
    def deco(cls):
        SOURCE_REGISTRY[name] = cls
        return cls
    return deco

@register_source("binance")
class BinanceDataSource(DataSource): ...
```

