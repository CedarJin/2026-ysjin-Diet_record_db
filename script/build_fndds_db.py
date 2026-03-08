#!/usr/bin/env python3
"""
Build SQLite database from FNDDS CSV files.

Creates a SQLite database with all 10 FNDDS tables.
"""

import sqlite3
import csv
from pathlib import Path

# Database path
DB_PATH = Path("db/fndds/fndds_2021_2023.db")
CSV_DIR = Path("db/fndds/FNDDS_2021-2023_SAS")

def create_tables(conn):
    """Create all tables with appropriate schema."""
    cursor = conn.cursor()
    
    # 1. nutdesc - Nutrient descriptions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nutdesc (
            Nutrient_code TEXT PRIMARY KEY,
            Nutrient_description TEXT,
            Tagname TEXT,
            Unit TEXT,
            Decimals INTEGER
        )
    """)
    
    # 2. mainfooddesc - Main food descriptions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mainfooddesc (
            Food_code TEXT PRIMARY KEY,
            Main_food_description TEXT,
            WWEIA_Category_number INTEGER,
            WWEIA_Category_description TEXT,
            Start_date DATE,
            End_date DATE
        )
    """)
    
    # 3. addfooddesc - Additional food descriptions (composite PK: Food_code + Seq_num)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS addfooddesc (
            Food_code TEXT,
            Seq_num TEXT,
            Additional_food_description TEXT,
            Start_date DATE,
            End_date DATE,
            PRIMARY KEY (Food_code, Seq_num)
        )
    """)
    
    # 4. fnddsnutval - FNDDS nutrient values (composite PK: Food_code + Nutrient_code)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fnddsnutval (
            Food_code TEXT,
            Nutrient_code TEXT,
            Nutrient_value REAL,
            Start_date DATE,
            End_date DATE,
            PRIMARY KEY (Food_code, Nutrient_code)
        )
    """)
    
    # 5. fnddsingred - FNDDS ingredients (composite PK: Food_code + Seq_num)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fnddsingred (
            Food_code TEXT,
            Seq_num INTEGER,
            Ingredient_code TEXT,
            Ingredient_description TEXT,
            Amount REAL,
            Measure TEXT,
            Portion_code INTEGER,
            Retention_code INTEGER,
            Ingredient_weight REAL,
            Start_date DATE,
            End_date DATE,
            PRIMARY KEY (Food_code, Seq_num)
        )
    """)
    
    # 6. ingrednutval - Ingredient nutrient values (composite PK: Ingredient_code + Nutrient_code)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingrednutval (
            Ingredient_code TEXT,
            Ingredient_description TEXT,
            Nutrient_code TEXT,
            Nutrient_value REAL,
            Nutrient_value_source TEXT,
            FDC_ID TEXT,
            Derivation_code TEXT,
            SR_AddMod_year INTEGER,
            Foundation_year_acquired INTEGER,
            Start_date DATE,
            End_date DATE,
            PRIMARY KEY (Ingredient_code, Nutrient_code)
        )
    """)
    
    # 7. foodweights - Food weights (composite PK: Food_code + Seq_num)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS foodweights (
            Food_code TEXT,
            Seq_num INTEGER,
            Portion_code INTEGER,
            Portion_weight REAL,
            Start_date DATE,
            End_date DATE,
            PRIMARY KEY (Food_code, Seq_num)
        )
    """)
    
    # 8. foodportiondesc - Portion descriptions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS foodportiondesc (
            Portion_code TEXT PRIMARY KEY,
            Portion_description TEXT,
            Start_date DATE,
            End_date DATE
        )
    """)
    
    # 9. moistadjust - Moisture adjustments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moistadjust (
            Food_code TEXT PRIMARY KEY,
            Moisture_change REAL,
            Start_date DATE,
            End_date DATE
        )
    """)
    
    # 10. derivdesc - Derivation descriptions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS derivdesc (
            Derivation_code TEXT PRIMARY KEY,
            Derivation_description TEXT
        )
    """)
    
    conn.commit()
    print("✓ Created all tables")


def import_csv(conn, csv_file, table_name):
    """Import a CSV file into a table."""
    csv_path = CSV_DIR / csv_file
    
    if not csv_path.exists():
        print(f"⚠️  Warning: {csv_file} not found, skipping")
        return 0
    
    cursor = conn.cursor()
    
    # Read CSV and get column names
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if not rows:
            print(f"⚠️  Warning: {csv_file} is empty")
            return 0
        
        # Get column names from first row
        columns = list(rows[0].keys())
        
        # Create placeholders for INSERT
        placeholders = ','.join(['?' for _ in columns])
        columns_str = ','.join(columns)
        
        # Prepare data - handle NULL/NA
        data_to_insert = []
        for row in rows:
            values = []
            for col in columns:
                value = row[col]
                # Handle NULL/NA as None (SQLite will store as NULL)
                if value in ('', 'NA', 'NULL', None):
                    values.append(None)
                # Keep dates as strings (SQLite DATE type accepts text)
                elif 'date' in col.lower():
                    values.append(value if value else None)
                else:
                    values.append(value)
            data_to_insert.append(tuple(values))
        
        # Insert data
        insert_sql = f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        cursor.executemany(insert_sql, data_to_insert)
        
        count = len(data_to_insert)
        conn.commit()
        return count


def main():
    """Main function to build the database."""
    print("=" * 80)
    print("Building FNDDS SQLite Database")
    print("=" * 80)
    
    # Ensure directories exist
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database if it exists
    if DB_PATH.exists():
        print(f"⚠️  Removing existing database: {DB_PATH}")
        DB_PATH.unlink()
    
    # Create database connection
    conn = sqlite3.connect(str(DB_PATH))
    
    try:
        # Create tables
        print("\nCreating tables...")
        create_tables(conn)
        
        # Import CSV files
        print("\nImporting CSV files...")
        print("-" * 80)
        
        imports = [
            ("nutdesc.csv", "nutdesc"),
            ("mainfooddesc.csv", "mainfooddesc"),
            ("addfooddesc.csv", "addfooddesc"),
            ("fnddsnutval.csv", "fnddsnutval"),
            ("fnddsingred.csv", "fnddsingred"),
            ("ingrednutval.csv", "ingrednutval"),
            ("foodweights.csv", "foodweights"),
            ("foodportiondesc.csv", "foodportiondesc"),
            ("moistadjust.csv", "moistadjust"),
            ("derivdesc.csv", "derivdesc"),
        ]
        
        total_rows = 0
        for csv_file, table_name in imports:
            print(f"Importing {csv_file} → {table_name}...", end=" ")
            count = import_csv(conn, csv_file, table_name)
            total_rows += count
            print(f"✓ {count:,} rows")
        
        # Create indexes for better query performance
        print("\nCreating indexes...")
        cursor = conn.cursor()
        
        indexes = [
            ("fnddsnutval", "Nutrient_code"),
            ("fnddsnutval", "Food_code"),
            ("fnddsingred", "Ingredient_code"),
            ("fnddsingred", "Food_code"),
            ("ingrednutval", "Nutrient_code"),
            ("ingrednutval", "Ingredient_code"),
            ("foodweights", "Food_code"),
            ("foodweights", "Portion_code"),
            ("addfooddesc", "Food_code"),
        ]
        
        for table, column in indexes:
            index_name = f"idx_{table}_{column}"
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})")
        
        conn.commit()
        print("✓ Indexes created")
        
        # Print summary
        print("\n" + "=" * 80)
        print("DATABASE SUMMARY")
        print("=" * 80)
        print(f"Database: {DB_PATH}")
        print(f"Total rows imported: {total_rows:,}")
        
        # Show row counts per table
        print("\nTable row counts:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        for (table_name,) in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  {table_name:20s}: {count:>10,} rows")
        
        print("\n✓ Database created successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
