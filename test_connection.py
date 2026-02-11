# test_connection.py
import mysql.connector
import os

print("🚀 Testing Railway MySQL Connection...")

try:
    # Connect using your environment variables
    conn = mysql.connector.connect(
        host="yamanote.proxy.rlwy.net",
        user="root",
        password="EoJhzIWGIkfIyHVOEnPBrGTMYKMpGyLB",
        database="railway",
        port=55190
    )
    
    print("✅ CONNECTED SUCCESSFULLY!")
    
    # Test query
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    result = cursor.fetchone()
    
    print(f"📊 Found {result[0]} users in database")
    
    # List tables
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    print("📋 Tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    cursor.close()
    conn.close()
    
except mysql.connector.Error as err:
    print(f"❌ DATABASE ERROR: {err}")
except Exception as e:
    print(f"❌ GENERAL ERROR: {e}")