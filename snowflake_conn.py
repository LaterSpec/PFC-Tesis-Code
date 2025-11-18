import json
import snowflake.connector

# Cargar las credenciales desde tu JSON
with open("src/cred/snowflake_credential.json") as f:
    cred = json.load(f)

conn = snowflake.connector.connect(
    user=cred["username"],
    password=cred["password"],
    account=cred["account"],
    role=cred.get("role"),
    warehouse=cred.get("warehouse"),
)

cur = conn.cursor()
# Realizar una consulta de ejemplo
cur.execute("SHOW DATABASES")
for row in cur:
    print(row)
