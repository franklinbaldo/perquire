# PERQUIRE TODO - MAJOR COMPLEXITY ISSUES 🚨

## 🔥 CRITICAL OVERENGINEERING DETECTED

PERQUIRE has severe over-abstraction issues, particularly in database operations that should be simple DuckDB calls.

### 🚨 PRIORITY 1: MASSIVE OVER-ABSTRACTION

#### Database Provider Monolith (832 lines!)
- [ ] **SIMPLIFY**: `src/perquire/database/duckdb_provider.py` - **832 lines!**
  - **Issue**: 832 lines for what should be simple DuckDB operations
  - **Analysis**: This is a classic over-abstraction anti-pattern
  - **Target**: Reduce to 100-150 lines maximum
  - **Action**: Replace complex abstractions with direct DuckDB calls

#### God-Object Investigator (712 lines!)
- [ ] **SPLIT**: `src/perquire/core/investigator.py` - **712 lines!**
  - **Issue**: Single class handling too many responsibilities
  - **Target**: Split into focused modules:
    - `investigator/query_processor.py`
    - `investigator/data_analyzer.py` 
    - `investigator/result_formatter.py`
  - **Goal**: 150-200 lines per module

#### Massive CLI (675 lines!)
- [ ] **SPLIT**: `src/perquire/cli/main.py` - **675 lines!**
  - **Issue**: CLI god-object handling all commands
  - **Target**: Split into command modules:
    - `cli/commands/search.py`
    - `cli/commands/analyze.py`
    - `cli/commands/database.py`
  - **Goal**: 100-150 lines per command module

### 🎯 SPECIFIC SIMPLIFICATION TARGETS

#### Database Provider Over-Abstraction
Current pattern (WRONG):
```python
# 832 lines of abstraction for simple operations
class DuckDBProvider:
    def complex_query_builder(self):  # 50+ lines
    def connection_pooling(self):     # 30+ lines  
    def query_optimization(self):     # 40+ lines
    def result_transformation(self):  # 60+ lines
    # ... hundreds more lines
```

Simple approach (RIGHT):
```python
# Should be ~50 lines maximum
import duckdb

def execute_query(query: str, params: list = None):
    conn = duckdb.connect('perquire.duckdb')
    return conn.execute(query, params or []).fetchall()

def insert_data(table: str, data: dict):
    # Simple insert logic
    
def search_documents(query: str):
    # Direct DuckDB search
```

### 🔢 COMPLEXITY BREAKDOWN

#### Current Situation (EXCESSIVE):
- `duckdb_provider.py`: 832 lines
- `investigator.py`: 712 lines  
- `cli/main.py`: 675 lines
- **Total**: 2,219 lines in just 3 files!

#### Target After Simplification:
- Database operations: ~100 lines (8x reduction)
- Investigator modules: ~450 lines total (1.5x reduction)
- CLI commands: ~400 lines total (1.7x reduction)
- **Total**: ~950 lines (2.3x reduction)

### 🛠️ REFACTORING PLAN

#### Phase 1: Database Simplification (HIGHEST PRIORITY)
- [ ] **Analyze**: What does the 832-line provider actually do?
- [ ] **Extract**: Core DuckDB operations (probably 50-100 lines of actual logic)
- [ ] **Replace**: Complex abstractions with direct DuckDB calls
- [ ] **Test**: Ensure functionality is preserved

#### Phase 2: Investigator Decomposition  
- [ ] **Identify**: Distinct responsibilities within the 712-line god-object
- [ ] **Split**: Into focused modules by functionality
- [ ] **Simplify**: Each module to have single responsibility

#### Phase 3: CLI Restructuring
- [ ] **Create**: `cli/commands/` directory structure
- [ ] **Extract**: Each major command to separate file
- [ ] **Share**: Common utilities in `cli/utils.py`

### 🚩 RED FLAGS FOUND

1. **Over-abstraction**: Database provider more complex than the database itself
2. **God objects**: Single files handling everything
3. **Premature optimization**: Complex connection pooling for single-user tool
4. **Layer explosion**: Multiple abstraction layers for simple operations
5. **Enterprise anti-patterns**: Using enterprise patterns for simple tool

### 🎯 SPECIFIC ANTI-PATTERNS TO REMOVE

#### Database Provider Issues:
- [ ] Remove complex connection pooling (overkill for single-user tool)
- [ ] Remove query builders (use SQL directly)
- [ ] Remove result transformers (return raw results)
- [ ] Remove caching layers (premature optimization)
- [ ] Remove abstraction interfaces (unnecessary complexity)

#### Investigator Issues:
- [ ] Split query processing from data analysis
- [ ] Split result formatting from core logic
- [ ] Remove complex state management
- [ ] Simplify data flow

#### CLI Issues:
- [ ] Remove monolithic command handler
- [ ] Split large command functions
- [ ] Remove excessive argument parsing complexity

### 📊 EXPECTED BENEFITS

1. **60%+ code reduction** in core files
2. **10x easier to understand** database operations
3. **Faster development** with direct DuckDB calls
4. **Easier testing** with focused modules
5. **Better maintainability** with single-responsibility modules

### ⚠️ MIGRATION RISKS

1. **API breaking changes**: Database provider interface will change
2. **Feature regression**: Some complex features might be simplified
3. **Testing needed**: Extensive testing required after refactoring

### 🎉 SUCCESS CRITERIA

- [ ] Database operations under 150 lines
- [ ] No single file over 300 lines  
- [ ] Direct DuckDB calls instead of abstractions
- [ ] All tests passing after refactoring
- [ ] 50%+ reduction in total codebase size

---

**PERQUIRE's main issue**: Solving simple problems with complex enterprise patterns. This tool should be simple and direct, not an enterprise framework!