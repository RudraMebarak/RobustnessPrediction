PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    original_text TEXT NOT NULL,
    watermarked_text TEXT NOT NULL,
    baseline_z_score REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS features (
    document_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value REAL NOT NULL,
    PRIMARY KEY (document_id, feature_name),
    FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS attacks (
    attack_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    attack_type TEXT NOT NULL,
    attack_config TEXT NOT NULL,
    attacked_text TEXT NOT NULL,
    edit_rate REAL,
    semantic_similarity REAL,
    attacked_z_score REAL,
    watermark_detected INTEGER,
    FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
);
