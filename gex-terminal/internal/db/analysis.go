package db

import (
	"database/sql"
	"encoding/json"
)

// GetAnalysisCache returns the cached analysis for a given date (or most recent).
func GetAnalysisCache(ticker string, targetDate string) (string, json.RawMessage, error) {
	var dateStr string
	var analysisJSON string
	var err error

	if targetDate != "" {
		err = pool.QueryRow(
			`SELECT date, analysis_json FROM analysis_cache WHERE ticker=? AND date=?`,
			ticker, targetDate,
		).Scan(&dateStr, &analysisJSON)
	} else {
		err = pool.QueryRow(
			`SELECT date, analysis_json FROM analysis_cache WHERE ticker=? ORDER BY date DESC LIMIT 1`,
			ticker,
		).Scan(&dateStr, &analysisJSON)
	}
	if err == sql.ErrNoRows {
		return "", nil, nil
	}
	if err != nil {
		return "", nil, err
	}
	return dateStr, json.RawMessage(analysisJSON), nil
}

// SetAnalysisCache stores analysis JSON for today.
func SetAnalysisCache(ticker string, data interface{}) error {
	dateStr := nowET().Format("2006-01-02")
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = pool.Exec(
		`INSERT OR REPLACE INTO analysis_cache (date, ticker, analysis_json) VALUES (?,?,?)`,
		dateStr, ticker, string(b),
	)
	return err
}
