package db

import (
	"database/sql"
)

// PredictionRow represents a scored prediction from the database.
type PredictionRow struct {
	AnalysisDate       string
	ExpiryDate         string
	DTE                int
	DTEWindow          string
	SpotBTC            float64
	CallWallBTC        float64
	PutWallBTC         float64
	GammaFlipBTC       float64
	Regime             string
	BTCCloseOnExpiry   *float64
	BTCHighInWindow    *float64
	BTCLowInWindow     *float64
	CallWallHeld       *bool
	PutWallHeld        *bool
	RangeHeld          *bool
	EMHeld             *bool
	RegimeCorrect      *bool
	CharmCorrect       *bool
	VenueWallsAgree    *bool
	VenueAgreeHeld     *bool
	MaxBreachCallPct   *float64
	MaxBreachPutPct    *float64
	RealizedRangePct   *float64
	CallWallErrorPct   *float64
	PutWallErrorPct    *float64
	AIBottomLine       string
	NetDealerDelta     *float64
	DealerDeltaDir     string
	DeribitAvailable   bool
	IBITCallWallBTC    *float64
	IBITPutWallBTC     *float64
	DeribitCallWallBTC *float64
	DeribitPutWallBTC  *float64
	CallWallHeldT2     *bool
	PutWallHeldT2      *bool
	RangeHeldT2        *bool
}

// SavePrediction inserts a single prediction row.
func SavePrediction(
	analysisDate, ticker, expiryDate string, dte int, dteWindow string,
	spotBTC, callWallBTC, putWallBTC, gammaFlipBTC, maxPainBTC float64,
	regime string, netGEX, netDealerDelta float64,
	dealerDeltaDir, charmDir string, charmNotional float64, charmStrength, vannaStrength string,
	overnightDir string, overnightNotional float64,
	deribitAvail bool,
	ibitCW, ibitPW, deribitCW, deribitPW float64,
	venueAgree bool,
	emUpper, emLower float64,
	aiBottomLine string,
) error {
	deribitInt := 0
	if deribitAvail {
		deribitInt = 1
	}
	venueInt := 0
	if venueAgree {
		venueInt = 1
	}

	_, err := pool.Exec(`INSERT OR REPLACE INTO predictions
		(analysis_date, ticker, expiry_date, dte, dte_window,
		 spot_btc, call_wall_btc, put_wall_btc, gamma_flip_btc, max_pain_btc,
		 regime, net_gex, net_dealer_delta, dealer_delta_direction,
		 charm_direction, charm_notional, charm_strength, vanna_strength,
		 overnight_direction, overnight_notional,
		 deribit_available, ibit_call_wall_btc, ibit_put_wall_btc,
		 deribit_call_wall_btc, deribit_put_wall_btc, venue_walls_agree,
		 em_upper_btc, em_lower_btc, ai_bottom_line)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		analysisDate, ticker, expiryDate, dte, dteWindow,
		spotBTC, callWallBTC, putWallBTC, gammaFlipBTC, maxPainBTC,
		regime, netGEX, netDealerDelta, dealerDeltaDir,
		charmDir, charmNotional, charmStrength, vannaStrength,
		overnightDir, overnightNotional,
		deribitInt, ibitCW, ibitPW, deribitCW, deribitPW, venueInt,
		emUpper, emLower, aiBottomLine,
	)
	return err
}

// GetRecentPredictions returns the last N scored predictions.
func GetRecentPredictions(ticker string, limit int) ([]PredictionRow, error) {
	rows, err := pool.Query(`SELECT
		analysis_date, expiry_date, dte, dte_window,
		spot_btc, call_wall_btc, put_wall_btc, gamma_flip_btc, regime,
		btc_close_on_expiry, btc_high_in_window, btc_low_in_window,
		call_wall_held, put_wall_held, range_held, em_held,
		regime_correct, charm_correct, venue_walls_agree, venue_agree_held,
		max_breach_call_pct, max_breach_put_pct, realized_range_pct,
		call_wall_error_pct, put_wall_error_pct, ai_bottom_line,
		net_dealer_delta, dealer_delta_direction, deribit_available,
		ibit_call_wall_btc, ibit_put_wall_btc, deribit_call_wall_btc, deribit_put_wall_btc,
		call_wall_held_t2, put_wall_held_t2, range_held_t2
		FROM predictions WHERE ticker=? AND scored=1 ORDER BY analysis_date DESC LIMIT ?`,
		ticker, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	return scanPredictions(rows)
}

// GetPredictionsForExpiry returns all predictions for a given expiry date.
func GetPredictionsForExpiry(ticker, expiryDate string) ([]PredictionRow, error) {
	rows, err := pool.Query(`SELECT
		analysis_date, expiry_date, dte, dte_window,
		spot_btc, call_wall_btc, put_wall_btc, gamma_flip_btc, regime,
		btc_close_on_expiry, btc_high_in_window, btc_low_in_window,
		call_wall_held, put_wall_held, range_held, em_held,
		regime_correct, charm_correct, venue_walls_agree, venue_agree_held,
		max_breach_call_pct, max_breach_put_pct, realized_range_pct,
		call_wall_error_pct, put_wall_error_pct, ai_bottom_line,
		net_dealer_delta, dealer_delta_direction, deribit_available,
		ibit_call_wall_btc, ibit_put_wall_btc, deribit_call_wall_btc, deribit_put_wall_btc,
		call_wall_held_t2, put_wall_held_t2, range_held_t2
		FROM predictions WHERE ticker=? AND expiry_date=? AND scored=1 ORDER BY analysis_date ASC`,
		ticker, expiryDate)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	return scanPredictions(rows)
}

func scanPredictions(rows *sql.Rows) ([]PredictionRow, error) {
	var result []PredictionRow
	for rows.Next() {
		var p PredictionRow
		var btcClose, btcHigh, btcLow sql.NullFloat64
		var cwHeld, pwHeld, rHeld, emHeld, regCorr, charmCorr sql.NullInt64
		var venueAgree, venueAgreeHeld sql.NullInt64
		var maxBreachC, maxBreachP, realRange, cwErr, pwErr sql.NullFloat64
		var ndd sql.NullFloat64
		var ibitCW, ibitPW, deribitCW, deribitPW sql.NullFloat64
		var cwHeldT2, pwHeldT2, rHeldT2 sql.NullInt64
		var deribitInt int

		err := rows.Scan(
			&p.AnalysisDate, &p.ExpiryDate, &p.DTE, &p.DTEWindow,
			&p.SpotBTC, &p.CallWallBTC, &p.PutWallBTC, &p.GammaFlipBTC, &p.Regime,
			&btcClose, &btcHigh, &btcLow,
			&cwHeld, &pwHeld, &rHeld, &emHeld,
			&regCorr, &charmCorr, &venueAgree, &venueAgreeHeld,
			&maxBreachC, &maxBreachP, &realRange, &cwErr, &pwErr,
			&p.AIBottomLine, &ndd, &p.DealerDeltaDir, &deribitInt,
			&ibitCW, &ibitPW, &deribitCW, &deribitPW,
			&cwHeldT2, &pwHeldT2, &rHeldT2,
		)
		if err != nil {
			continue
		}

		p.DeribitAvailable = deribitInt == 1
		if btcClose.Valid { v := btcClose.Float64; p.BTCCloseOnExpiry = &v }
		if btcHigh.Valid { v := btcHigh.Float64; p.BTCHighInWindow = &v }
		if btcLow.Valid { v := btcLow.Float64; p.BTCLowInWindow = &v }
		if cwHeld.Valid { v := cwHeld.Int64 == 1; p.CallWallHeld = &v }
		if pwHeld.Valid { v := pwHeld.Int64 == 1; p.PutWallHeld = &v }
		if rHeld.Valid { v := rHeld.Int64 == 1; p.RangeHeld = &v }
		if emHeld.Valid { v := emHeld.Int64 == 1; p.EMHeld = &v }
		if regCorr.Valid { v := regCorr.Int64 == 1; p.RegimeCorrect = &v }
		if charmCorr.Valid { v := charmCorr.Int64 == 1; p.CharmCorrect = &v }
		if venueAgree.Valid { v := venueAgree.Int64 == 1; p.VenueWallsAgree = &v }
		if venueAgreeHeld.Valid { v := venueAgreeHeld.Int64 == 1; p.VenueAgreeHeld = &v }
		if maxBreachC.Valid { p.MaxBreachCallPct = &maxBreachC.Float64 }
		if maxBreachP.Valid { p.MaxBreachPutPct = &maxBreachP.Float64 }
		if realRange.Valid { p.RealizedRangePct = &realRange.Float64 }
		if cwErr.Valid { p.CallWallErrorPct = &cwErr.Float64 }
		if pwErr.Valid { p.PutWallErrorPct = &pwErr.Float64 }
		if ndd.Valid { p.NetDealerDelta = &ndd.Float64 }
		if ibitCW.Valid { p.IBITCallWallBTC = &ibitCW.Float64 }
		if ibitPW.Valid { p.IBITPutWallBTC = &ibitPW.Float64 }
		if deribitCW.Valid { p.DeribitCallWallBTC = &deribitCW.Float64 }
		if deribitPW.Valid { p.DeribitPutWallBTC = &deribitPW.Float64 }
		if cwHeldT2.Valid { v := cwHeldT2.Int64 == 1; p.CallWallHeldT2 = &v }
		if pwHeldT2.Valid { v := pwHeldT2.Int64 == 1; p.PutWallHeldT2 = &v }
		if rHeldT2.Valid { v := rHeldT2.Int64 == 1; p.RangeHeldT2 = &v }

		result = append(result, p)
	}
	return result, nil
}

// GetUnscoredPredictions returns predictions that need scoring (expired and unscored).
func GetUnscoredPredictions(ticker string) ([]PredictionRow, error) {
	today := nowET().Format("2006-01-02")
	rows, err := pool.Query(`SELECT
		analysis_date, expiry_date, dte, dte_window,
		spot_btc, call_wall_btc, put_wall_btc, gamma_flip_btc, regime,
		NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, venue_walls_agree, NULL,
		NULL, NULL, NULL, NULL, NULL, ai_bottom_line,
		net_dealer_delta, dealer_delta_direction, deribit_available,
		ibit_call_wall_btc, ibit_put_wall_btc, deribit_call_wall_btc, deribit_put_wall_btc,
		NULL, NULL, NULL
		FROM predictions WHERE ticker=? AND scored=0 AND expiry_date<? ORDER BY expiry_date ASC`,
		ticker, today)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	return scanPredictions(rows)
}

// ScorePrediction updates a prediction with scoring results.
func ScorePrediction(
	analysisDate, ticker, expiryDate, scoredDate string,
	expHigh, expLow, expClose float64,
	winHigh, winLow float64,
	callWallHeld, putWallHeld, rangeHeld, emHeld *int,
	regimeCorrect, charmCorrect, venueAgreeHeld *int,
	maxBreachCall, maxBreachPut, realizedRangePct float64,
	callWallError, putWallError *float64,
	cwHeldT2, pwHeldT2, rangeHeldT2 *int,
) error {
	_, err := pool.Exec(`UPDATE predictions SET
		scored=1, scored_date=?,
		btc_high_on_expiry=?, btc_low_on_expiry=?, btc_close_on_expiry=?,
		btc_high_in_window=?, btc_low_in_window=?,
		call_wall_held=?, put_wall_held=?, range_held=?, em_held=?,
		regime_correct=?, charm_correct=?, venue_agree_held=?,
		max_breach_call_pct=?, max_breach_put_pct=?, realized_range_pct=?,
		call_wall_error_pct=?, put_wall_error_pct=?,
		call_wall_held_t2=?, put_wall_held_t2=?, range_held_t2=?
		WHERE analysis_date=? AND ticker=? AND expiry_date=? AND scored=0`,
		scoredDate,
		expHigh, expLow, expClose,
		winHigh, winLow,
		callWallHeld, putWallHeld, rangeHeld, emHeld,
		regimeCorrect, charmCorrect, venueAgreeHeld,
		maxBreachCall, maxBreachPut, realizedRangePct,
		callWallError, putWallError,
		cwHeldT2, pwHeldT2, rangeHeldT2,
		analysisDate, ticker, expiryDate,
	)
	return err
}
