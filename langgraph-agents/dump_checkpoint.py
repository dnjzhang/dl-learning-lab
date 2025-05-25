#!/usr/bin/env python
# coding: utf-8

import sqlite3
import sys

def dump_sqlite(db_path: str):
    # Connect to the database (will create if it doesn’t exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1) Find all the tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        print(f"\n== Table: {table} ==")

        # 2) Print column names
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [col_info[1] for col_info in cursor.fetchall()]
        print(" | ".join(cols))

        # 3) Print all rows
        cursor.execute(f"SELECT * FROM {table};")
        for row in cursor.fetchall():
            print(row)

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dump_checkpoint.py <path/to/your.db>")
    else:
        dump_sqlite(sys.argv[1])
