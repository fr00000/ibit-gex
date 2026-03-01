package db

import (
	"database/sql"

	"gex-terminal/internal/model"
)

// SaveSnapshot inserts or replaces the daily snapshot and strike history.
func SaveSnapshot(ticker string, spot, btcPrice float64, levels *model.Levels, strikes []model.StrikeData) error {
	dateStr := nowET().Format("2006-01-02")
	tx, err := pool.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	_, err = tx.Exec(`INSERT OR REPLACE INTO snapshots
		(date, ticker, spot, btc_price, gamma_flip, call_wall, put_wall, max_pain,
		 regime, net_gex, total_call_oi, total_put_oi, weighted_net_gex)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		dateStr, ticker, spot, btcPrice,
		levels.GammaFlip, levels.CallWall, levels.PutWall, levels.MaxPain,
		levels.Regime, levels.NetGEXTotal, levels.TotalCallOI, levels.TotalPutOI,
		levels.NetGEXTotal,
	)
	if err != nil {
		return err
	}

	stmt, err := tx.Prepare(`INSERT OR REPLACE INTO strike_history
		(date, ticker, strike, call_oi, put_oi, total_oi, net_gex, weighted_net_gex)
		VALUES (?,?,?,?,?,?,?,?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, s := range strikes {
		_, err = stmt.Exec(dateStr, ticker, s.Strike, s.CallOI, s.PutOI,
			s.TotalOI, s.NetGEX, s.NetGEX)
		if err != nil {
			return err
		}
	}

	return tx.Commit()
}

// GetHistory returns the last N daily snapshots for a ticker, newest first.
func GetHistory(ticker string, days int) ([]model.Snapshot, error) {
	rows, err := pool.Query(`SELECT date, spot, btc_price, gamma_flip, call_wall, put_wall,
		max_pain, regime, net_gex, total_call_oi, total_put_oi, weighted_net_gex
		FROM snapshots WHERE ticker=? ORDER BY date DESC LIMIT ?`, ticker, days)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []model.Snapshot
	for rows.Next() {
		var s model.Snapshot
		s.Ticker = ticker
		var spot, btcPrice, gf, cw, pw, mp, ng, wng sql.NullFloat64
		var callOI, putOI sql.NullInt64
		var regime sql.NullString
		err := rows.Scan(&s.Date, &spot, &btcPrice, &gf, &cw, &pw, &mp, &regime, &ng, &callOI, &putOI, &wng)
		if err != nil {
			return nil, err
		}
		s.Spot = spot.Float64
		s.BTCPrice = btcPrice.Float64
		s.GammaFlip = gf.Float64
		s.CallWall = cw.Float64
		s.PutWall = pw.Float64
		s.MaxPain = mp.Float64
		s.Regime = regime.String
		s.NetGEX = ng.Float64
		s.TotalCallOI = callOI.Int64
		s.TotalPutOI = putOI.Int64
		s.WeightedNetGEX = wng.Float64
		result = append(result, s)
	}
	return result, nil
}

// GetPrevStrikes returns the most recent strike history before today.
func GetPrevStrikes(ticker string) (string, map[float64]model.StrikeHistory, error) {
	today := nowET().Format("2006-01-02")

	var prevDate string
	err := pool.QueryRow(
		`SELECT date FROM snapshots WHERE ticker=? AND date<? ORDER BY date DESC LIMIT 1`,
		ticker, today,
	).Scan(&prevDate)
	if err != nil {
		return "", nil, nil // no previous data, not an error
	}

	rows, err := pool.Query(
		`SELECT strike, call_oi, put_oi, total_oi, net_gex, weighted_net_gex
		 FROM strike_history WHERE date=? AND ticker=?`, prevDate, ticker)
	if err != nil {
		return prevDate, nil, err
	}
	defer rows.Close()

	strikes := make(map[float64]model.StrikeHistory)
	for rows.Next() {
		var sh model.StrikeHistory
		var callOI, putOI, totalOI sql.NullInt64
		var ng, wng sql.NullFloat64
		err := rows.Scan(&sh.Strike, &callOI, &putOI, &totalOI, &ng, &wng)
		if err != nil {
			continue
		}
		sh.CallOI = callOI.Int64
		sh.PutOI = putOI.Int64
		sh.TotalOI = totalOI.Int64
		sh.NetGEX = ng.Float64
		sh.WeightedNetGEX = wng.Float64
		strikes[sh.Strike] = sh
	}
	return prevDate, strikes, nil
}

// HeatmapRow is a single row from the strike_history JOIN snapshots query.
type HeatmapRow struct {
	Date     string
	Strike   float64
	TotalOI  int64
	NetGEX   float64
	CallOI   int64
	PutOI    int64
	Spot     float64
	BTCPrice float64
}

// GetStrikeHistoryForHeatmap returns strike data joined with snapshots for heatmap rendering.
func GetStrikeHistoryForHeatmap(ticker string, days int) ([]HeatmapRow, error) {
	cutoff := nowET().AddDate(0, 0, -days).Format("2006-01-02")

	rows, err := pool.Query(`
		SELECT sh.date, sh.strike, sh.total_oi, sh.net_gex, sh.call_oi, sh.put_oi,
		       s.spot, s.btc_price
		FROM strike_history sh
		JOIN snapshots s ON sh.date = s.date AND sh.ticker = s.ticker
		WHERE sh.ticker=? AND sh.date >= ?
		ORDER BY sh.date, sh.strike`, ticker, cutoff)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []HeatmapRow
	for rows.Next() {
		var r HeatmapRow
		var totalOI, callOI, putOI sql.NullInt64
		var netGEX, spot, btcPrice sql.NullFloat64
		err := rows.Scan(&r.Date, &r.Strike, &totalOI, &netGEX, &callOI, &putOI, &spot, &btcPrice)
		if err != nil {
			continue
		}
		r.TotalOI = totalOI.Int64
		r.NetGEX = netGEX.Float64
		r.CallOI = callOI.Int64
		r.PutOI = putOI.Int64
		r.Spot = spot.Float64
		r.BTCPrice = btcPrice.Float64
		result = append(result, r)
	}
	return result, nil
}
