import json
from pathlib import Path
import snowflake.connector

def run_top5_female_ca_2000(cred_path: str = "src/cred/snowflake_credential.json", max_rows: int = 5):
    # Load credentials
    cred_file = Path(cred_path)
    if not cred_file.exists():
        raise FileNotFoundError(f"Credential file not found: {cred_file}")

    with open(cred_file, "r", encoding="utf-8") as f:
        cred = json.load(f)

    account = cred.get("account")
    user = cred.get("username") or cred.get("user")
    password = cred.get("password")
    warehouse = cred.get("warehouse")
    role = cred.get("role")
    database = cred.get("database", "USA_NAMES")
    schema = cred.get("schema", "USA_NAMES")

    # SQL to run
    sql = """
    SELECT "name", "number"
    FROM USA_NAMES.USA_NAMES.USA_1910_CURRENT
    WHERE "state" = 'CA'
      AND "gender" = 'F'
      AND "year" = 2000
    ORDER BY "number" DESC
    LIMIT 5;
    """

    # Connect and execute
    conn = snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        warehouse=warehouse,
        role=role,
        database=database,
        schema=schema,
    )

    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        cols = [d[0] for d in cur.description] if cur.description else []

        # Print header
        if cols:
            print(" | ".join(cols))
            print("-" * (len(" | ".join(cols)) + 1))

        # Print rows
        for r in rows:
            print(" | ".join(str(v) for v in r))

    finally:
        try:
            cur.close()
        except:
            pass
        conn.close()

if __name__ == "__main__":
    # Ejecutar
    run_top5_female_ca_2000()