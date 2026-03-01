package db

import (
	"database/sql"
	"gex-terminal/internal/model"
)

// SaveFlow inserts or replaces a single ETF flow row.
func SaveFlow(date, ticker string, dailyFlowDollars, totalBTCETFFlow float64) error {
	_, err := pool.Exec(
		`INSERT OR REPLACE INTO etf_flows
		 (date, ticker, shares_outstanding, aum, nav, daily_flow_shares, daily_flow_dollars, total_btc_etf_flow)
		 VALUES (?,?,NULL,NULL,NULL,0,?,?)`,
		date, ticker, dailyFlowDollars, totalBTCETFFlow,
	)
	return err
}

// DeleteOldFlows removes Farside-less rows (Yahoo-era).
func DeleteOldFlows(ticker string) error {
	_, err := pool.Exec(
		`DELETE FROM etf_flows WHERE ticker=? AND total_btc_etf_flow IS NULL`, ticker)
	return err
}

// FlowRaw holds the raw flow data for macro scoring.
type FlowRaw struct {
	Date             string
	DailyFlowDollars float64
	TotalBTCETFFlow  *float64 // nil if NULL
}

// GetFlowsRaw returns raw flow data for macro scoring, newest first.
func GetFlowsRaw(ticker string, limit int) []FlowRaw {
	rows, err := pool.Query(
		`SELECT date, daily_flow_dollars, total_btc_etf_flow
		 FROM etf_flows WHERE ticker=? ORDER BY date DESC LIMIT ?`, ticker, limit)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var result []FlowRaw
	for rows.Next() {
		var f FlowRaw
		var daily sql.NullFloat64
		var total sql.NullFloat64
		if err := rows.Scan(&f.Date, &daily, &total); err != nil {
			continue
		}
		f.DailyFlowDollars = daily.Float64
		if total.Valid {
			f.TotalBTCETFFlow = &total.Float64
		}
		result = append(result, f)
	}
	return result
}

// GetFlows returns the last N flow rows, newest first.
func GetFlows(ticker string, limit int) ([]model.FlowRow, error) {
	rows, err := pool.Query(
		`SELECT date, daily_flow_dollars, shares_outstanding, aum, nav, total_btc_etf_flow
		 FROM etf_flows WHERE ticker=? ORDER BY date DESC LIMIT ?`, ticker, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []model.FlowRow
	for rows.Next() {
		var f model.FlowRow
		var flow, shares, aum, nav, totalFlow sql.NullFloat64
		if err := rows.Scan(&f.Date, &flow, &shares, &aum, &nav, &totalFlow); err != nil {
			continue
		}
		f.Flow = flow.Float64
		f.Shares = shares.Float64
		f.AUM = aum.Float64
		f.NAV = nav.Float64
		f.TotalFlow = totalFlow.Float64
		result = append(result, f)
	}
	return result, nil
}
