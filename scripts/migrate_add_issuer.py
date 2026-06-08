"""
Database migration script for adding issuer field.

Changes:
1. Add issuer column to document table (VARCHAR 256, nullable)
"""
import sqlite3
from pathlib import Path


def migrate_database(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    changes = []
    
    # Check and add issuer column to document table
    cursor.execute('PRAGMA table_info(document)')
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'issuer' not in columns:
        cursor.execute('ALTER TABLE document ADD COLUMN issuer VARCHAR(256)')
        changes.append('Added issuer column to document table')
        conn.commit()
    else:
        changes.append('issuer column already exists in document table')
    
    conn.close()
    return {'changes': changes}


if __name__ == '__main__':
    db_path = 'data/processed/app.db'
    
    if not Path(db_path).exists():
        print(f'Database not found: {db_path}')
        exit(1)
    
    result = migrate_database(db_path)
    for change in result['changes']:
        print(change)
    
    print('\nMigration completed.')