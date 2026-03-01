package db

import (
	"database/sql"
	"encoding/json"
)

// SaveExpiryCache stores per-expiry strike data.
func SaveExpiryCache(date, ticker, expiryDate string, data interface{}) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = pool.Exec(
		`INSERT OR REPLACE INTO expiry_cache (date, ticker, expiry_date, data_json)
		 VALUES (?,?,?,?)`,
		date, ticker, expiryDate, string(b),
	)
	return err
}

// GetExpiryCacheList returns available expiry dates for a ticker+date.
func GetExpiryCacheList(ticker, date string) ([]string, error) {
	var targetDate string
	if date != "" {
		targetDate = date
	} else {
		// Get latest date
		err := pool.QueryRow(
			`SELECT date FROM expiry_cache WHERE ticker=? ORDER BY date DESC LIMIT 1`, ticker,
		).Scan(&targetDate)
		if err == sql.ErrNoRows {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
	}

	rows, err := pool.Query(
		`SELECT expiry_date FROM expiry_cache WHERE ticker=? AND date=? ORDER BY expiry_date ASC`,
		ticker, targetDate,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var expiries []string
	for rows.Next() {
		var e string
		if err := rows.Scan(&e); err != nil {
			continue
		}
		expiries = append(expiries, e)
	}
	return expiries, nil
}

// GetExpiryCacheData returns the raw JSON for a specific expiry.
func GetExpiryCacheData(ticker, date, expiryDate string) (json.RawMessage, error) {
	var dataJSON string
	err := pool.QueryRow(
		`SELECT data_json FROM expiry_cache WHERE ticker=? AND date=? AND expiry_date=?`,
		ticker, date, expiryDate,
	).Scan(&dataJSON)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return json.RawMessage(dataJSON), nil
}

// GetExpiryCacheRange returns raw JSON for all expiries in a date range.
func GetExpiryCacheRange(ticker, cacheDate, fromDate, toDate string) ([]struct {
	ExpiryDate string
	DataJSON   json.RawMessage
}, error) {
	rows, err := pool.Query(
		`SELECT expiry_date, data_json FROM expiry_cache
		 WHERE ticker=? AND date=? AND expiry_date>=? AND expiry_date<=?
		 ORDER BY expiry_date ASC`,
		ticker, cacheDate, fromDate, toDate,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []struct {
		ExpiryDate string
		DataJSON   json.RawMessage
	}
	for rows.Next() {
		var item struct {
			ExpiryDate string
			DataJSON   json.RawMessage
		}
		var dj string
		if err := rows.Scan(&item.ExpiryDate, &dj); err != nil {
			continue
		}
		item.DataJSON = json.RawMessage(dj)
		result = append(result, item)
	}
	return result, nil
}

// CleanExpiryCacheOld deletes rows older than maxDays or with expired expiry dates.
func CleanExpiryCacheOld(maxDays, expiryGraceDays int) error {
	today := nowET()
	oldDate := today.AddDate(0, 0, -maxDays).Format("2006-01-02")
	expiryDate := today.AddDate(0, 0, -expiryGraceDays).Format("2006-01-02")

	_, err := pool.Exec(`DELETE FROM expiry_cache WHERE date < ?`, oldDate)
	if err != nil {
		return err
	}
	_, err = pool.Exec(`DELETE FROM expiry_cache WHERE expiry_date < ?`, expiryDate)
	return err
}
