import os
from contextlib import contextmanager

try:
    import psycopg2 as pg
except ImportError:
    import psycopg as pg


def connect():
    # The password comes from the PGPASSWORD environment variable
    return pg.connect(host=os.getenv("PGHOST", "localhost"),
                      port=os.getenv("PGPORT", "5432"),
                      dbname=os.getenv("PGDATABASE", "fraud_detection_db"),
                      user=os.getenv("PGUSER", "postgres"))


@contextmanager
def session():
    conn = connect()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
