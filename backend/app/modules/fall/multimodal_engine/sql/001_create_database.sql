-- MySQL 8 database initialization for the competition prototype.
-- Run this file with an administrative account.

CREATE DATABASE IF NOT EXISTS elder_risk_prototype
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_0900_ai_ci;

-- Create the application account separately with a locally chosen password.
-- Do not commit real credentials into this file.
-- Example commands to run manually as an administrator:
-- CREATE USER 'elder_risk_app'@'localhost' IDENTIFIED BY '<LOCAL_PASSWORD>';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON elder_risk_prototype.*
--     TO 'elder_risk_app'@'localhost';
-- FLUSH PRIVILEGES;
