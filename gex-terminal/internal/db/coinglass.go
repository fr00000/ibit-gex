package db

import (
	"database/sql"
	"encoding/json"
)

// SaveCoinglassData inserts or replaces a Coinglass metric.
func SaveCoinglassData(date, symbol, metric string, value float64, extraJSON string) error {
	var extra *string
	if extraJSON != "" {
		extra = &extraJSON
	}
	_, err := pool.Exec(
		`INSERT OR REPLACE INTO coinglass_data (date, symbol, metric, value, extra_json)
		 VALUES (?,?,?,?,?)`,
		date, symbol, metric, value, extra,
	)
	return err
}

// GetCoinglassHistory returns the last N rows for a symbol+metric, newest first.
func GetCoinglassHistory(symbol, metric string, limit int) ([]struct {
	Date      string
	Value     float64
	ExtraJSON string
}, error) {
	rows, err := pool.Query(
		`SELECT date, value, extra_json FROM coinglass_data
		 WHERE symbol=? AND metric=? ORDER BY date DESC LIMIT ?`,
		symbol, metric, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []struct {
		Date      string
		Value     float64
		ExtraJSON string
	}
	for rows.Next() {
		var d struct {
			Date      string
			Value     float64
			ExtraJSON string
		}
		var val sql.NullFloat64
		var extra sql.NullString
		if err := rows.Scan(&d.Date, &val, &extra); err != nil {
			continue
		}
		d.Value = val.Float64
		d.ExtraJSON = extra.String
		result = append(result, d)
	}
	return result, nil
}

// GetCoinglassLatest returns the latest value and extra JSON for a metric.
func GetCoinglassLatest(symbol, metric string) (float64, map[string]interface{}, error) {
	var val sql.NullFloat64
	var extra sql.NullString
	err := pool.QueryRow(
		`SELECT value, extra_json FROM coinglass_data
		 WHERE symbol=? AND metric=? ORDER BY date DESC LIMIT 1`,
		symbol, metric,
	).Scan(&val, &extra)
	if err == sql.ErrNoRows {
		return 0, nil, nil
	}
	if err != nil {
		return 0, nil, err
	}

	var extraMap map[string]interface{}
	if extra.Valid && extra.String != "" {
		json.Unmarshal([]byte(extra.String), &extraMap)
	}
	return val.Float64, extraMap, nil
}
