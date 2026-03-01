package trends

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"gex-terminal/internal/db"
	"gex-terminal/internal/model"
)

// HistoryTrends is the compressed 30-day snapshot trend for AI analysis.
type HistoryTrends struct {
	RegimeStreak   *RegimeStreak          `json:"regime_streak"`
	LevelMigration map[string]*Migration  `json:"level_migration"`
	GEXTrend       *GEXTrend              `json:"gex_trend"`
	OITrend        *OITrend               `json:"oi_trend"`
	RangeEvolution *RangeEvolution        `json:"range_evolution"`
	Narrative      string                 `json:"narrative"`
}

// RegimeStreak tracks consecutive regime days.
type RegimeStreak struct {
	CurrentRegime  string          `json:"current_regime"`
	StreakDays     int             `json:"streak_days"`
	PriorRegime    *string         `json:"prior_regime"`
	RegimeHistory  []string        `json:"regime_history"`
	Persistence30D *Persistence30D `json:"persistence_30d"`
}

// Persistence30D measures regime dominance over 30 days.
type Persistence30D struct {
	PositiveDays    int     `json:"positive_days"`
	NegativeDays    int     `json:"negative_days"`
	PersistencePct  int     `json:"persistence_pct"`
	SignFlips       int     `json:"sign_flips"`
	AvgGEXMagnitude int     `json:"avg_gex_magnitude"`
	IsPersistent    bool    `json:"is_persistent"`
}

// Migration tracks a single level's directional trend.
type Migration struct {
	CurrentBTC           int     `json:"current_btc"`
	Direction            string  `json:"direction"` // "rising", "falling", "stable"
	Change3DBTC          *int    `json:"change_3d_btc"`
	Change5DBTC          *int    `json:"change_5d_btc"`
	HeldAtBTC            *int    `json:"held_at_btc"`
	ConsecutiveDirection int     `json:"consecutive_direction"`
}

// GEXTrend tracks net GEX direction.
type GEXTrend struct {
	NetGEXDirection    string   `json:"net_gex_direction"` // "building", "decaying", "flat"
	NetGEX3DChangePct  *float64 `json:"net_gex_3d_change_pct"`
}

// OITrend tracks open interest trends.
type OITrend struct {
	TotalCallOIDirection string   `json:"total_call_oi_direction"`
	TotalPutOIDirection  string   `json:"total_put_oi_direction"`
	CallOI3DChangePct    *float64 `json:"call_oi_3d_change_pct"`
	PutOI3DChangePct     *float64 `json:"put_oi_3d_change_pct"`
	PCRTrend             string   `json:"pcr_trend"`
}

// RangeEvolution tracks call_wall - put_wall spread changes.
type RangeEvolution struct {
	RangeWidthCurrentBTC *int    `json:"range_width_current_btc"`
	RangeWidth3DAgoBTC   *int    `json:"range_width_3d_ago_btc"`
	RangeTrend           string  `json:"range_trend"` // "expanding", "contracting", "stable"
	RangeBias            string  `json:"range_bias"`
}

// SummarizeHistoryTrends compresses 30 days of snapshots into trend signals.
func SummarizeHistoryTrends(ticker string, btcPerShare float64, currentLevels map[string]float64) *HistoryTrends {
	history, err := db.GetHistory(ticker, 30)
	if err != nil || len(history) < 2 {
		return nil
	}

	n := len(history)

	// ── Regime streak ──
	currentRegime := history[0].Regime
	streak := 1
	for i := 1; i < n; i++ {
		if history[i].Regime == currentRegime {
			streak++
		} else {
			break
		}
	}
	var priorRegime *string
	if streak < n {
		pr := history[streak].Regime
		priorRegime = &pr
	}

	regimeAbbrev := make([]string, n)
	for i, h := range history {
		if h.Regime == "positive_gamma" {
			regimeAbbrev[i] = "pos"
		} else {
			regimeAbbrev[i] = "neg"
		}
	}

	// Persistence 30d
	nPositive := 0
	for _, h := range history {
		if h.Regime == "positive_gamma" {
			nPositive++
		}
	}
	nNegative := n - nPositive
	dominantCount := nPositive
	if nNegative > nPositive {
		dominantCount = nNegative
	}
	persistencePct := 0
	if n > 0 {
		persistencePct = int(math.Round(float64(dominantCount) / float64(n) * 100))
	}
	signFlips := 0
	for i := 1; i < n; i++ {
		if history[i].Regime != history[i-1].Regime {
			signFlips++
		}
	}

	// Average GEX magnitude
	var sumGEX float64
	var gexCount int
	for _, h := range history {
		sumGEX += math.Abs(h.NetGEX)
		gexCount++
	}
	avgGEXMag := 0
	if gexCount > 0 {
		avgGEXMag = int(math.Round(sumGEX / float64(gexCount)))
	}

	regimeStreak := &RegimeStreak{
		CurrentRegime: currentRegime,
		StreakDays:    streak,
		PriorRegime:   priorRegime,
		RegimeHistory: regimeAbbrev,
		Persistence30D: &Persistence30D{
			PositiveDays:    nPositive,
			NegativeDays:    nNegative,
			PersistencePct:  persistencePct,
			SignFlips:       signFlips,
			AvgGEXMagnitude: avgGEXMag,
			IsPersistent:    persistencePct >= 70 && signFlips <= 5,
		},
	}

	// ── Level migration ──
	currentBPS := btcPerShare
	levelMigration := map[string]*Migration{}

	type levelDef struct {
		name string
		getCurrent func() float64
		getHist func(h model.Snapshot) float64
	}
	levelDefs := []levelDef{
		{"call_wall", func() float64 { return currentLevels["call_wall"] }, func(h model.Snapshot) float64 { return h.CallWall }},
		{"put_wall", func() float64 { return currentLevels["put_wall"] }, func(h model.Snapshot) float64 { return h.PutWall }},
		{"gamma_flip", func() float64 { return currentLevels["gamma_flip"] }, func(h model.Snapshot) float64 { return h.GammaFlip }},
	}

	for _, ld := range levelDefs {
		curIBIT := ld.getCurrent()
		curBTC := 0
		if currentBPS > 0 {
			curBTC = int(math.Round(curIBIT / currentBPS))
		}

		// Convert each day's value to BTC
		var valid []int
		for _, h := range history {
			bps := rowBPS(h, btcPerShare)
			val := ld.getHist(h)
			if val > 0 && bps > 0 {
				valid = append(valid, int(math.Round(val/bps)))
			}
		}

		if len(valid) < 2 {
			levelMigration[ld.name] = &Migration{
				CurrentBTC: curBTC,
				Direction:  "stable",
				HeldAtBTC:  &curBTC,
			}
			continue
		}

		direction, heldAt := classifyDirection(valid, curBTC)
		consec := consecutiveDir(valid)

		var change3d, change5d *int
		if len(valid) >= 2 {
			idx := min(2, len(valid)-1)
			v := valid[0] - valid[idx]
			change3d = &v
		}
		if len(valid) >= 5 {
			idx := min(4, len(valid)-1)
			v := valid[0] - valid[idx]
			change5d = &v
		}

		mig := &Migration{
			CurrentBTC:           curBTC,
			Direction:            direction,
			Change3DBTC:          change3d,
			Change5DBTC:          change5d,
			ConsecutiveDirection: consec,
		}
		if direction == "stable" && heldAt != nil {
			mig.HeldAtBTC = heldAt
		}
		levelMigration[ld.name] = mig
	}

	// ── GEX trend ──
	var netGEXVals []float64
	for _, h := range history {
		netGEXVals = append(netGEXVals, h.NetGEX)
	}
	netGEX3DPct := pctChangeND(netGEXVals, min(3, len(netGEXVals)-1))
	gexTrend := &GEXTrend{
		NetGEXDirection:   gexDirection(netGEX3DPct),
		NetGEX3DChangePct: netGEX3DPct,
	}

	// ── OI trend ──
	var callOIVals, putOIVals []float64
	for _, h := range history {
		callOIVals = append(callOIVals, float64(h.TotalCallOI))
		putOIVals = append(putOIVals, float64(h.TotalPutOI))
	}
	call3DPct := pctChangeND(callOIVals, min(3, len(callOIVals)-1))
	put3DPct := pctChangeND(putOIVals, min(3, len(putOIVals)-1))

	// PCR trend
	var pcrVals []float64
	for _, h := range history {
		if h.TotalCallOI > 0 && h.TotalPutOI > 0 {
			pcrVals = append(pcrVals, float64(h.TotalPutOI)/float64(h.TotalCallOI))
		}
	}
	pcrTrend := "stable"
	if len(pcrVals) >= 2 {
		diff := pcrVals[0] - pcrVals[len(pcrVals)-1]
		if diff > 0.05 {
			pcrTrend = "rising"
		} else if diff < -0.05 {
			pcrTrend = "falling"
		}
	}

	oiTrend := &OITrend{
		TotalCallOIDirection: gexDirection(call3DPct),
		TotalPutOIDirection:  gexDirection(put3DPct),
		CallOI3DChangePct:    call3DPct,
		PutOI3DChangePct:     put3DPct,
		PCRTrend:             pcrTrend,
	}

	// ── Range evolution ──
	var cwBTCVals, pwBTCVals []int
	for _, h := range history {
		bps := rowBPS(h, btcPerShare)
		if h.CallWall > 0 && h.PutWall > 0 && bps > 0 {
			cwBTCVals = append(cwBTCVals, int(math.Round(h.CallWall/bps)))
			pwBTCVals = append(pwBTCVals, int(math.Round(h.PutWall/bps)))
		}
	}

	var rangeCurrent, range3DAgo *int
	if len(cwBTCVals) > 0 && len(pwBTCVals) > 0 {
		rc := cwBTCVals[0] - pwBTCVals[0]
		rangeCurrent = &rc
	}
	if len(cwBTCVals) >= 2 && len(pwBTCVals) >= 2 {
		idx := min(2, len(cwBTCVals)-1)
		ra := cwBTCVals[idx] - pwBTCVals[idx]
		range3DAgo = &ra
	}

	rangeTrend := "stable"
	if rangeCurrent != nil && range3DAgo != nil {
		diff := *rangeCurrent - *range3DAgo
		threshold := float64(*rangeCurrent) * 0.1
		if float64(diff) > threshold {
			rangeTrend = "expanding"
		} else if float64(diff) < -threshold {
			rangeTrend = "contracting"
		}
	}

	rangeBias := computeRangeBias(cwBTCVals, pwBTCVals)

	rangeEvolution := &RangeEvolution{
		RangeWidthCurrentBTC: rangeCurrent,
		RangeWidth3DAgoBTC:   range3DAgo,
		RangeTrend:           rangeTrend,
		RangeBias:            rangeBias,
	}

	// ── Narrative ──
	narrative := buildNarrative(currentRegime, streak, levelMigration, oiTrend, rangeTrend)

	return &HistoryTrends{
		RegimeStreak:   regimeStreak,
		LevelMigration: levelMigration,
		GEXTrend:       gexTrend,
		OITrend:        oiTrend,
		RangeEvolution: rangeEvolution,
		Narrative:      narrative,
	}
}

// helpers

func rowBPS(h model.Snapshot, defaultBPS float64) float64 {
	if h.BTCPrice > 0 && h.Spot > 0 {
		return h.Spot / h.BTCPrice
	}
	return defaultBPS
}

func classifyDirection(valuesNewestFirst []int, currentPriceBTC int) (string, *int) {
	if len(valuesNewestFirst) < 2 {
		return "stable", nil
	}
	end := min(3, len(valuesNewestFirst))
	recent := valuesNewestFirst[:end]
	netChange := recent[0] - recent[len(recent)-1]
	threshold := float64(currentPriceBTC) * 0.01
	if math.Abs(float64(netChange)) <= threshold {
		sum := 0
		for _, v := range recent {
			sum += v
		}
		avg := sum / len(recent)
		return "stable", &avg
	} else if netChange > 0 {
		return "rising", nil
	}
	return "falling", nil
}

func consecutiveDir(valuesNewestFirst []int) int {
	if len(valuesNewestFirst) < 2 {
		return 0
	}
	firstDir := valuesNewestFirst[0] - valuesNewestFirst[1]
	if firstDir == 0 {
		return 0
	}
	count := 1
	for i := 1; i < len(valuesNewestFirst)-1; i++ {
		diff := valuesNewestFirst[i] - valuesNewestFirst[i+1]
		if (diff > 0 && firstDir > 0) || (diff < 0 && firstDir < 0) {
			count++
		} else {
			break
		}
	}
	return count
}

func pctChangeND(values []float64, nd int) *float64 {
	if len(values) <= nd || nd <= 0 {
		return nil
	}
	old := values[nd]
	if old != 0 {
		pct := math.Round(((values[0]-old)/math.Abs(old))*1000) / 10
		return &pct
	}
	return nil
}

func gexDirection(pct *float64) string {
	if pct == nil {
		return "flat"
	}
	if *pct > 10 {
		return "building"
	} else if *pct < -10 {
		return "decaying"
	}
	return "flat"
}

func computeRangeBias(cwBTCVals, pwBTCVals []int) string {
	if len(cwBTCVals) < 2 || len(pwBTCVals) < 2 {
		return "stable"
	}
	cwChg := cwBTCVals[0] - cwBTCVals[len(cwBTCVals)-1]
	pwChg := pwBTCVals[0] - pwBTCVals[len(pwBTCVals)-1]
	cwThresh := float64(cwBTCVals[0]) * 0.01
	pwThresh := float64(pwBTCVals[0]) * 0.01

	cwRising := float64(cwChg) > cwThresh
	cwFalling := float64(cwChg) < -cwThresh
	pwRising := float64(pwChg) > pwThresh
	pwFalling := float64(pwChg) < -pwThresh

	switch {
	case cwRising && pwRising:
		return "upward_shift"
	case cwFalling && pwFalling:
		return "downward_shift"
	case cwRising && pwFalling:
		return "expanding_symmetric"
	case cwFalling && pwRising:
		return "contracting"
	case cwRising && !pwRising && !pwFalling:
		return "expanding_upward"
	case cwFalling && !pwRising && !pwFalling:
		return "contracting_from_above"
	case pwFalling && !cwRising && !cwFalling:
		return "expanding_downward"
	case pwRising && !cwRising && !cwFalling:
		return "contracting_from_below"
	default:
		return "stable"
	}
}

func buildNarrative(currentRegime string, streak int, levelMigration map[string]*Migration, oiTrend *OITrend, rangeTrend string) string {
	regimeStr := "Positive gamma"
	if currentRegime != "positive_gamma" {
		regimeStr = "Negative gamma"
	}
	suffix := "s"
	if streak == 1 {
		suffix = ""
	}
	parts := []string{fmt.Sprintf("%s regime for %d day%s", regimeStr, streak, suffix)}

	cwMig := levelMigration["call_wall"]
	pwMig := levelMigration["put_wall"]

	if cwMig != nil && pwMig != nil {
		if cwMig.Direction == "rising" && pwMig.Direction == "rising" {
			var chgStr string
			if cwMig.Change5DBTC != nil {
				chgStr = fmt.Sprintf(" +$%d over 5d", *cwMig.Change5DBTC)
			} else if cwMig.Change3DBTC != nil {
				chgStr = fmt.Sprintf(" +$%d over 3d", *cwMig.Change3DBTC)
			}
			parts = append(parts, "walls migrating upward"+chgStr)
		} else if cwMig.Direction == "falling" && pwMig.Direction == "falling" {
			var chgStr string
			if cwMig.Change5DBTC != nil {
				chgStr = fmt.Sprintf(" %d over 5d", *cwMig.Change5DBTC)
			} else if cwMig.Change3DBTC != nil {
				chgStr = fmt.Sprintf(" %d over 3d", *cwMig.Change3DBTC)
			}
			parts = append(parts, "walls migrating downward"+chgStr)
		} else if rangeTrend == "contracting" {
			parts = append(parts, "range contracting")
		} else if rangeTrend == "expanding" {
			parts = append(parts, "range expanding")
		} else {
			var moves []string
			if cwMig.Direction == "rising" {
				moves = append(moves, "call wall rising")
			} else if cwMig.Direction == "falling" {
				moves = append(moves, "call wall decaying")
			}
			if pwMig.Direction == "rising" {
				moves = append(moves, "put wall rising")
			} else if pwMig.Direction == "falling" {
				moves = append(moves, "put wall falling")
			}
			if len(moves) > 0 {
				parts = append(parts, strings.Join(moves, ", "))
			}
		}
	}

	// OI context
	if oiTrend.TotalCallOIDirection == "building" && oiTrend.TotalPutOIDirection == "building" {
		parts = append(parts, "OI building on both sides")
	} else if oiTrend.TotalCallOIDirection == "decaying" && oiTrend.TotalPutOIDirection == "decaying" {
		parts = append(parts, "OI unwinding")
	} else if oiTrend.TotalCallOIDirection == "building" {
		parts = append(parts, "call OI building")
	} else if oiTrend.TotalPutOIDirection == "building" {
		parts = append(parts, "put OI building")
	}

	maxParts := 3
	if len(parts) > maxParts {
		parts = parts[:maxParts]
	}
	return strings.Join(parts, " \u2014 ")
}

// StructureTrends is the per-DTE-window trend analysis.
type StructureTrends struct {
	WindowTrends map[string]*WindowTrend `json:"window_trends"`
	Convergence  map[string]string       `json:"convergence"`
}

// WindowTrend tracks a single DTE window's level migration.
type WindowTrend struct {
	Days          int          `json:"days"`
	Trend         string       `json:"trend,omitempty"`
	CallWall      *TrendEntry  `json:"call_wall,omitempty"`
	PutWall       *TrendEntry  `json:"put_wall,omitempty"`
	GammaFlip     *TrendEntry  `json:"gamma_flip,omitempty"`
	RegimeChanges int          `json:"regime_changes,omitempty"`
}

// TrendEntry is a first/last/direction for one metric.
type TrendEntry struct {
	First     *float64 `json:"first"`
	Last      *float64 `json:"last"`
	Direction string   `json:"direction"`
}

// SummarizeStructureTrends analyzes per-DTE-window level migration.
func SummarizeStructureTrends(ticker string, days int) *StructureTrends {
	if days <= 0 {
		days = 7
	}

	dteToWindow := map[int]string{3: "0-3", 7: "4-7", 14: "8-14", 30: "15-30", 45: "31-45"}
	windowOrder := []string{"0-3", "4-7", "8-14", "15-30", "31-45"}

	cacheRows := db.GetCacheDatesWithData(ticker, days)
	if len(cacheRows) == 0 {
		return nil
	}

	type windowEntry struct {
		Date      string
		CallWall  *float64
		PutWall   *float64
		GammaFlip *float64
		Regime    string
	}

	windowHistory := map[string][]windowEntry{}

	for _, row := range cacheRows {
		window, ok := dteToWindow[row.DTE]
		if !ok {
			continue
		}

		var d struct {
			BTCPerShare    float64                `json:"btc_per_share"`
			Levels         map[string]interface{} `json:"levels"`
			CombinedBTC    map[string]interface{} `json:"combined_levels_btc"`
		}
		if err := json.Unmarshal(row.DataJSON, &d); err != nil {
			continue
		}

		var cw, pw, gf *float64
		var regime string

		if d.CombinedBTC != nil {
			cw = jsonFloat(d.CombinedBTC, "call_wall")
			pw = jsonFloat(d.CombinedBTC, "put_wall")
			gf = jsonFloat(d.CombinedBTC, "gamma_flip")
			regime = jsonString(d.CombinedBTC, "regime")
			if regime == "" {
				regime = jsonString(d.Levels, "regime")
			}
		} else if d.Levels != nil {
			bps := d.BTCPerShare
			if bps <= 0 {
				bps = 1
			}
			if v := jsonFloat(d.Levels, "call_wall"); v != nil {
				val := *v / bps
				cw = &val
			}
			if v := jsonFloat(d.Levels, "put_wall"); v != nil {
				val := *v / bps
				pw = &val
			}
			if v := jsonFloat(d.Levels, "gamma_flip"); v != nil {
				val := *v / bps
				gf = &val
			}
			regime = jsonString(d.Levels, "regime")
		}

		windowHistory[window] = append(windowHistory[window], windowEntry{
			Date:      row.Date,
			CallWall:  cw,
			PutWall:   pw,
			GammaFlip: gf,
			Regime:    regime,
		})
	}

	if len(windowHistory) == 0 {
		return nil
	}

	trends := map[string]*WindowTrend{}
	for window, entries := range windowHistory {
		if len(entries) < 2 {
			trends[window] = &WindowTrend{Days: len(entries), Trend: "insufficient_data"}
			continue
		}

		first := entries[0]
		last := entries[len(entries)-1]

		regimeChanges := 0
		for i := 1; i < len(entries); i++ {
			if entries[i].Regime != entries[i-1].Regime {
				regimeChanges++
			}
		}

		trends[window] = &WindowTrend{
			Days:          len(entries),
			CallWall:      makeTrendEntry(first.CallWall, last.CallWall),
			PutWall:       makeTrendEntry(first.PutWall, last.PutWall),
			GammaFlip:     makeTrendEntry(first.GammaFlip, last.GammaFlip),
			RegimeChanges: regimeChanges,
		}
	}

	// Cross-window convergence
	convergence := map[string]string{}
	for _, metric := range []string{"call_wall", "put_wall"} {
		var valuesFirst, valuesLast []float64
		for _, w := range windowOrder {
			entries := windowHistory[w]
			if len(entries) == 0 {
				continue
			}
			var first, last *float64
			switch metric {
			case "call_wall":
				first = entries[0].CallWall
				if len(entries) > 1 {
					last = entries[len(entries)-1].CallWall
				}
			case "put_wall":
				first = entries[0].PutWall
				if len(entries) > 1 {
					last = entries[len(entries)-1].PutWall
				}
			}
			if first != nil {
				valuesFirst = append(valuesFirst, *first)
			}
			if last != nil {
				valuesLast = append(valuesLast, *last)
			}
		}

		if len(valuesFirst) >= 2 && len(valuesLast) >= 2 {
			spreadFirst := maxFloat(valuesFirst) - minFloat(valuesFirst)
			spreadLast := maxFloat(valuesLast) - minFloat(valuesLast)
			if spreadFirst > 0 {
				change := (spreadLast - spreadFirst) / spreadFirst * 100
				if change < -10 {
					convergence[metric] = "converging"
				} else if change > 10 {
					convergence[metric] = "diverging"
				} else {
					convergence[metric] = "stable"
				}
			}
		}
	}

	return &StructureTrends{
		WindowTrends: trends,
		Convergence:  convergence,
	}
}

func makeTrendEntry(first, last *float64) *TrendEntry {
	if first == nil || last == nil {
		return nil
	}
	return &TrendEntry{
		First:     first,
		Last:      last,
		Direction: trendDirection(*first, *last, 3),
	}
}

func trendDirection(old, new_ float64, thresholdPct float64) string {
	if old == 0 {
		return "unknown"
	}
	pct := (new_ - old) / old * 100
	if pct > thresholdPct {
		return "rising"
	} else if pct < -thresholdPct {
		return "falling"
	}
	return "stable"
}

func jsonFloat(m map[string]interface{}, key string) *float64 {
	if m == nil {
		return nil
	}
	v, ok := m[key]
	if !ok || v == nil {
		return nil
	}
	switch val := v.(type) {
	case float64:
		if val == 0 {
			return nil
		}
		return &val
	case json.Number:
		f, err := val.Float64()
		if err != nil || f == 0 {
			return nil
		}
		return &f
	}
	return nil
}

func jsonString(m map[string]interface{}, key string) string {
	if m == nil {
		return ""
	}
	v, ok := m[key]
	if !ok || v == nil {
		return ""
	}
	s, _ := v.(string)
	return s
}

func maxFloat(vals []float64) float64 {
	m := vals[0]
	for _, v := range vals[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func minFloat(vals []float64) float64 {
	m := vals[0]
	for _, v := range vals[1:] {
		if v < m {
			m = v
		}
	}
	return m
}
