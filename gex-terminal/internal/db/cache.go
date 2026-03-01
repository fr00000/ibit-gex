package db

import (
	"database/sql"
	"encoding/json"
)

// GetLatestCache returns the most recent cached data for (ticker, dte).
// If targetDate is non-empty, fetches that specific date.
func GetLatestCache(ticker string, dte int, targetDate string) (string, json.RawMessage, error) {
	var dateStr string
	var dataJSON string

	var err error
	if targetDate != "" {
		err = pool.QueryRow(
			`SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=? AND date=?`,
			ticker, dte, targetDate,
		).Scan(&dateStr, &dataJSON)
	} else {
		err = pool.QueryRow(
			`SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT 1`,
			ticker, dte,
		).Scan(&dateStr, &dataJSON)
	}
	if err == sql.ErrNoRows {
		return "", nil, nil
	}
	if err != nil {
		return "", nil, err
	}
	return dateStr, json.RawMessage(dataJSON), nil
}

// GetPrevCache returns the second-most-recent cache (before today).
func GetPrevCache(ticker string, dte int) (string, json.RawMessage, error) {
	today := nowET().Format("2006-01-02")
	var dateStr string
	var dataJSON string
	err := pool.QueryRow(
		`SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=? AND date<? ORDER BY date DESC LIMIT 1`,
		ticker, dte, today,
	).Scan(&dateStr, &dataJSON)
	if err == sql.ErrNoRows {
		return "", nil, nil
	}
	if err != nil {
		return "", nil, err
	}
	return dateStr, json.RawMessage(dataJSON), nil
}

// SetCachedData stores data in the cache for today.
func SetCachedData(ticker string, dte int, data interface{}) error {
	dateStr := nowET().Format("2006-01-02")
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = pool.Exec(
		`INSERT OR REPLACE INTO data_cache (date, ticker, dte, data_json) VALUES (?,?,?,?)`,
		dateStr, ticker, dte, string(b),
	)
	return err
}

// CacheRow represents a single cache row with date, DTE, and raw JSON.
type CacheRow struct {
	Date     string
	DTE      int
	DataJSON json.RawMessage
}

// GetCacheDatesWithData returns cache rows for a ticker within the last N days.
func GetCacheDatesWithData(ticker string, days int) []CacheRow {
	cutoff := nowET().AddDate(0, 0, -days).Format("2006-01-02")
	rows, err := pool.Query(
		`SELECT date, dte, data_json FROM data_cache
		 WHERE ticker=? AND date >= ? ORDER BY date, dte`,
		ticker, cutoff,
	)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var result []CacheRow
	for rows.Next() {
		var r CacheRow
		var dataJSON string
		if err := rows.Scan(&r.Date, &r.DTE, &dataJSON); err != nil {
			continue
		}
		r.DataJSON = json.RawMessage(dataJSON)
		result = append(result, r)
	}
	return result
}

// GetCacheRowsForDTE returns cache rows for a ticker+DTE, ordered by date DESC.
func GetCacheRowsForDTE(ticker string, dte, limit int) []CacheRow {
	rows, err := pool.Query(
		`SELECT date, dte, data_json FROM data_cache
		 WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT ?`,
		ticker, dte, limit,
	)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var result []CacheRow
	for rows.Next() {
		var r CacheRow
		var dataJSON string
		if err := rows.Scan(&r.Date, &r.DTE, &dataJSON); err != nil {
			continue
		}
		r.DataJSON = json.RawMessage(dataJSON)
		result = append(result, r)
	}
	return result
}

// GetCachedDates returns all dates that have at least minWindows complete DTE caches.
func GetCachedDates(ticker string, minWindows int) ([]string, error) {
	rows, err := pool.Query(
		`SELECT date FROM data_cache WHERE ticker=? GROUP BY date HAVING COUNT(DISTINCT dte) >= ? ORDER BY date DESC`,
		ticker, minWindows,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var dates []string
	for rows.Next() {
		var d string
		if err := rows.Scan(&d); err != nil {
			continue
		}
		dates = append(dates, d)
	}
	return dates, nil
}
