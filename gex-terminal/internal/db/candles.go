package db

import "gex-terminal/internal/model"

// SaveCandles bulk-inserts OHLCV candles.
func SaveCandles(symbol, tf string, candles []model.Candle) error {
	if len(candles) == 0 {
		return nil
	}
	tx, err := pool.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(
		`INSERT OR REPLACE INTO btc_candles (symbol, tf, time, open, high, low, close)
		 VALUES (?,?,?,?,?,?,?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, c := range candles {
		_, err = stmt.Exec(symbol, tf, c.Time, c.Open, c.High, c.Low, c.Close)
		if err != nil {
			return err
		}
	}
	return tx.Commit()
}

// GetCandles returns all candles for a symbol/timeframe, oldest first.
func GetCandles(symbol, tf string) ([]model.Candle, error) {
	rows, err := pool.Query(
		`SELECT time, open, high, low, close FROM btc_candles
		 WHERE symbol=? AND tf=? ORDER BY time ASC`, symbol, tf)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []model.Candle
	for rows.Next() {
		var c model.Candle
		if err := rows.Scan(&c.Time, &c.Open, &c.High, &c.Low, &c.Close); err != nil {
			continue
		}
		result = append(result, c)
	}
	return result, nil
}

// GetRecentCandles returns the last N candles, newest first.
func GetRecentCandles(symbol, tf string, limit int) []model.Candle {
	rows, err := pool.Query(
		`SELECT time, open, high, low, close FROM btc_candles
		 WHERE symbol=? AND tf=? ORDER BY time DESC LIMIT ?`, symbol, tf, limit)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var result []model.Candle
	for rows.Next() {
		var c model.Candle
		if err := rows.Scan(&c.Time, &c.Open, &c.High, &c.Low, &c.Close); err != nil {
			continue
		}
		result = append(result, c)
	}
	return result
}

// GetLatestCandleTime returns the most recent candle timestamp for a symbol/tf.
func GetLatestCandleTime(symbol, tf string) (int64, error) {
	var t int64
	err := pool.QueryRow(
		`SELECT COALESCE(MAX(time), 0) FROM btc_candles WHERE symbol=? AND tf=?`,
		symbol, tf,
	).Scan(&t)
	return t, err
}
