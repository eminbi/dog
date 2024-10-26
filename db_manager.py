import sqlite3

class DBManager:
    def __init__(self, db_path="data/pet_behavior.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS behaviors (
            id INTEGER PRIMARY KEY,
            behavior TEXT,
            confidence REAL
        );
        """
        self.conn.execute(sql)
        self.conn.commit()
