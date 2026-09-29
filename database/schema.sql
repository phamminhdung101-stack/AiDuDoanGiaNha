-- HouseAI Database Schema v2
-- Run: mysql -u root -p < database/schema.sql

CREATE DATABASE IF NOT EXISTS house_prediction
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE house_prediction;

CREATE TABLE IF NOT EXISTS users (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    username    VARCHAR(64) UNIQUE NOT NULL,
    password    VARCHAR(256) NOT NULL COMMENT 'SHA-256 hash',
    fullname    VARCHAR(128) NOT NULL,
    email       VARCHAR(128) UNIQUE NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
);

CREATE TABLE IF NOT EXISTS admins (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    username    VARCHAR(64) UNIQUE NOT NULL,
    password    VARCHAR(256) NOT NULL,
    INDEX idx_username (username)
);

CREATE TABLE IF NOT EXISTS houses (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    title       VARCHAR(256),
    city        VARCHAR(128),
    district    VARCHAR(128),
    street      VARCHAR(256),
    area        FLOAT NOT NULL,
    floors      TINYINT UNSIGNED DEFAULT 1,
    bedrooms    TINYINT UNSIGNED DEFAULT 0,
    bathrooms   TINYINT UNSIGNED DEFAULT 0,
    frontage    FLOAT DEFAULT 0,
    road_width  FLOAT DEFAULT 0,
    house_type  VARCHAR(64),
    direction   VARCHAR(64),
    legal       VARCHAR(64),
    price       BIGINT NOT NULL COMMENT 'Giá VNĐ',
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_city (city),
    INDEX idx_house_type (house_type),
    INDEX idx_price (price)
);

CREATE TABLE IF NOT EXISTS predictions (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT NOT NULL,
    city        VARCHAR(128),
    district    VARCHAR(128),
    street      VARCHAR(256),
    area        FLOAT,
    house_type  VARCHAR(64),
    bedrooms    TINYINT UNSIGNED DEFAULT 0,
    bathrooms   TINYINT UNSIGNED DEFAULT 0,
    floors      TINYINT UNSIGNED DEFAULT 1,
    pred_price  BIGINT,
    min_price   BIGINT,
    max_price   BIGINT,
    confidence  FLOAT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
);

-- Default admin (password: admin123 — đổi ngay trong production!)
INSERT IGNORE INTO admins (username, password) VALUES ('admin', 'admin123');

select *from predictions;
TRUNCATE TABLE houses;