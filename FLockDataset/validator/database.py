import sqlite3

class ScoreDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with a table to store UID, hotkey, and score."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS miner_scores
                     (uid INTEGER, hotkey TEXT, score REAL, PRIMARY KEY (uid, hotkey))''')
        conn.commit()
        conn.close()

    def insert_or_reset_uid(self, uid: int, hotkey: str):
        """Insert a new UID or reset its score if the hotkey has changed (UID recycled)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT hotkey FROM miner_scores WHERE uid = ?", (uid,))
        result = c.fetchone()
        if result is None or result[0] != hotkey:
            # UID is new or recycled (hotkey changed), reset score to 0.0
            c.execute("INSERT OR REPLACE INTO miner_scores (uid, hotkey, score) VALUES (?, ?, 0.0)", (uid, hotkey))
        conn.commit()
        conn.close()

    def update_score(self, uid: int, delta: float):
        """Update the score for a given UID by adding the delta."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE miner_scores SET score = score + ? WHERE uid = ?", (delta, uid))
        conn.commit()
        conn.close()

    def get_scores(self, uids: list) -> list:
        """Retrieve scores for a list of UIDs, defaulting to 0.0 if not found."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"SELECT uid, score FROM miner_scores WHERE uid IN ({','.join('?'*len(uids))})", uids)
        scores_dict = {uid: score for uid, score in c.fetchall()}
        conn.close()
        return [scores_dict.get(uid, 0.0) for uid in uids]
