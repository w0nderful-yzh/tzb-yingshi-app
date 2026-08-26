-- MySQL 8 table initialization for the competition prototype.
-- Execute after 001_create_database.sql.

USE elder_risk_prototype;

CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id VARCHAR(64) NOT NULL,
    mode VARCHAR(16) NOT NULL,
    status VARCHAR(16) NOT NULL,
    device_id VARCHAR(64) NOT NULL,
    enabled_modules JSON NOT NULL,
    started_at DATETIME(3) NOT NULL,
    ended_at DATETIME(3) NULL,
    created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
        ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    CONSTRAINT chk_monitoring_mode
        CHECK (mode IN ('SIMULATION', 'FILE', 'LIVE')),
    CONSTRAINT chk_monitoring_status
        CHECK (status IN ('RUNNING', 'STOPPED', 'ERROR')),
    CONSTRAINT chk_enabled_modules_array
        CHECK (JSON_TYPE(enabled_modules) = 'ARRAY'),
    INDEX idx_session_status (status),
    INDEX idx_session_started_at (started_at)
) ENGINE=InnoDB
  DEFAULT CHARACTER SET=utf8mb4
  COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS risk_events (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    event_id VARCHAR(80) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    device_id VARCHAR(64) NOT NULL,
    module VARCHAR(32) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    risk_score DECIMAL(5,4) NOT NULL,
    risk_level VARCHAR(16) NOT NULL,
    summary VARCHAR(500) NOT NULL,
    evidence_json JSON NOT NULL,
    recommended_action VARCHAR(500) NULL,
    snapshot_path VARCHAR(500) NULL,
    clip_path VARCHAR(500) NULL,
    model_version VARCHAR(64) NOT NULL,
    source VARCHAR(16) NOT NULL,
    status VARCHAR(24) NOT NULL DEFAULT 'PENDING',
    occurred_at DATETIME(3) NOT NULL,
    received_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    handled_at DATETIME(3) NULL,
    handling_note VARCHAR(500) NULL,
    updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
        ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    CONSTRAINT uk_risk_event_id UNIQUE (event_id),
    CONSTRAINT fk_risk_event_session
        FOREIGN KEY (session_id) REFERENCES monitoring_sessions (id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    CONSTRAINT chk_risk_module
        CHECK (module IN ('FALL', 'MENTAL_STATE', 'FRAUD', 'DEVICE')),
    CONSTRAINT chk_risk_score
        CHECK (risk_score >= 0 AND risk_score <= 1),
    CONSTRAINT chk_risk_level
        CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    CONSTRAINT chk_risk_source
        CHECK (source IN ('SIMULATION', 'ALGORITHM', 'EZVIZ')),
    CONSTRAINT chk_risk_status
        CHECK (status IN ('PENDING', 'ACKNOWLEDGED', 'FALSE_ALARM')),
    INDEX idx_event_session_time (session_id, occurred_at),
    INDEX idx_event_module_time (module, occurred_at),
    INDEX idx_event_status_level (status, risk_level),
    INDEX idx_event_occurred_at (occurred_at)
) ENGINE=InnoDB
  DEFAULT CHARACTER SET=utf8mb4
  COLLATE=utf8mb4_0900_ai_ci;
