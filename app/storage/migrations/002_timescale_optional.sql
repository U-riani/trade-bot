CREATE EXTENSION IF NOT EXISTS timescaledb;

SELECT create_hypertable('candles', 'open_time', if_not_exists => TRUE);
