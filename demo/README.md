This folder contains demo materials.

## run_oracle_query.py

`run_oracle_query.py` demonstrates how to connect to a local Oracle database using the [`oracledb`](https://python-oracledb.readthedocs.io/) driver. The SQL statement to execute is provided as a command line argument.

Environment variables are used for connection details:

- `ORACLE_USER`
- `ORACLE_PASSWORD`
- `ORACLE_HOST` (default `localhost`)
- `ORACLE_PORT` (default `1521`)
- `ORACLE_SERVICE` (default `XE`)

### Example

```bash
python run_oracle_query.py "SELECT * FROM my_table"
```
