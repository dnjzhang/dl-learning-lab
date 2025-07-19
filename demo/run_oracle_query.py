import os
import argparse

# Use the thin client from the `oracledb` package
import oracledb


def get_connection():
    """Create a connection using environment variables."""
    user = os.getenv("ORACLE_USER", "user")
    password = os.getenv("ORACLE_PASSWORD", "password")
    host = os.getenv("ORACLE_HOST", "localhost")
    port = os.getenv("ORACLE_PORT", "1521")
    service = os.getenv("ORACLE_SERVICE", "XE")
    dsn = f"{host}:{port}/{service}"
    return oracledb.connect(user=user, password=password, dsn=dsn)


def main():
    parser = argparse.ArgumentParser(description="Run a SQL statement against a local Oracle DB")
    parser.add_argument("sql", help="SQL query to execute")
    args = parser.parse_args()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            for row in cursor.execute(args.sql):
                print(row)


if __name__ == "__main__":
    main()
