package analysis

import (
	"encoding/json"
	"fmt"
	"math"
	"sync"

	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
	"gex-terminal/internal/gex"
	"gex-terminal/internal/macro"
	"gex-terminal/internal/model"
	"gex-terminal/internal/trends"
)

// BuildAnalysisData constructs the full data blob sent to the AI for analysis.
// This mirrors the Python build_analysis_data function.
func BuildAnalysisData(ticker string) (map[string]interface{}, error) {
	cfg := config.TickerConfig[ticker]
	if cfg == nil {
		return nil, nil
	}

	dtes := config.DTEWindows
	results := map[int]*model.DataBlob{}

	// Fetch all DTE windows in parallel
	var mu sync.Mutex
	var wg sync.WaitGroup
	var firstErr error

	for _, w := range dtes {
		w := w
		wg.Add(1)
		go func() {
			defer wg.Done()
			data, err := gex.FetchWithCache(ticker, w.MaxDTE, w.MinDTE, false)
			mu.Lock()
			defer mu.Unlock()
			if err != nil && firstErr == nil {
				firstErr = err
			}
			if data != nil {
				results[w.Label] = data
			}
		}()
	}
	wg.Wait()

	if firstErr != nil && len(results) == 0 {
		return nil, firstErr
	}

	summaries := map[string]interface{}{}

	// History trends
	firstData := results[dtes[0].Label]
	if firstData != nil {
		ht := trends.SummarizeHistoryTrends(ticker, firstData.BTCPerShare, map[string]float64{
			"call_wall":  firstData.Levels.CallWall,
			"put_wall":   firstData.Levels.PutWall,
			"gamma_flip": firstData.Levels.GammaFlip,
		})
		if ht != nil {
			summaries["_history_trends"] = ht
		}
	}

	// Structure trends
	st := trends.SummarizeStructureTrends(ticker, 7)
	if st != nil {
		summaries["_structure_trends"] = st
	}

	// Macro regime
	macroResult := macro.ComputeMacroRegime(ticker, 30)
	if macroResult != nil {
		summaries["_macro_regime"] = map[string]interface{}{
			"score":               macroResult.Score,
			"bias":                macroResult.Bias,
			"swing_signal":        macroResult.SwingSignal,
			"high_conviction":     macroResult.HighConviction,
			"signals":             macroResult.Signals,
			"structural_entry":    macroResult.StructuralEntry,
			"invalidation":        macroResult.Invalidation,
			"coinglass_available": macroResult.CoinglassAvail,
		}
	}

	// Per-DTE-window summaries
	for _, w := range dtes {
		d := results[w.Label]
		if d == nil {
			continue
		}

		key := w.WindowLabel()
		bps := d.BTCPerShare
		ibitLvl := d.Levels
		deribitAvail := d.DeribitAvail
		combinedLvl := d.CombinedLevelsBTC
		deribitLvl := d.DeribitLevelsBTC
		currentBTC := d.BTCSpot

		// Use combined levels as primary when Deribit is available
		var primaryLvlBTC map[string]interface{}
		var primaryNetGEX, primaryActiveGEX float64
		var primaryRegime string
		var primaryActiveCW, primaryActivePW float64
		var primaryConfidence int
		var primaryWarnings []string
		var sourceLabel string

		if deribitAvail && combinedLvl != nil {
			primaryLvlBTC = map[string]interface{}{
				"call_wall":  math.Round(combinedLvl.CallWall),
				"put_wall":   math.Round(combinedLvl.PutWall),
				"gamma_flip": math.Round(combinedLvl.GammaFlip),
				"max_pain":   math.Round(combinedLvl.MaxPain),
				"resistance": roundSlice(combinedLvl.Resistance),
				"support":    roundSlice(combinedLvl.Support),
			}
			primaryNetGEX = combinedLvl.NetGEXTotal
			primaryActiveGEX = combinedLvl.ActiveGEXTotal
			primaryRegime = combinedLvl.Regime
			if primaryRegime == "" {
				primaryRegime = ibitLvl.Regime
			}
			primaryActiveCW = math.Round(orFloat(combinedLvl.ActiveCallWall, combinedLvl.CallWall))
			primaryActivePW = math.Round(orFloat(combinedLvl.ActivePutWall, combinedLvl.PutWall))
			primaryConfidence = combinedLvl.PositioningConfidence
			primaryWarnings = combinedLvl.PositioningWarnings
			sourceLabel = "IBIT + Deribit (~91% of BTC options OI)"
		} else if ibitLvl != nil && bps > 0 {
			primaryLvlBTC = map[string]interface{}{
				"call_wall":  math.Round(ibitLvl.CallWall / bps),
				"put_wall":   math.Round(ibitLvl.PutWall / bps),
				"gamma_flip": math.Round(ibitLvl.GammaFlip / bps),
				"max_pain":   math.Round(ibitLvl.MaxPain / bps),
				"resistance": divSlice(ibitLvl.Resistance, bps),
				"support":    divSlice(ibitLvl.Support, bps),
			}
			primaryNetGEX = ibitLvl.NetGEXTotal
			primaryActiveGEX = ibitLvl.ActiveGEXTotal
			primaryRegime = ibitLvl.Regime
			primaryActiveCW = math.Round(orFloat(ibitLvl.ActiveCallWall, ibitLvl.CallWall) / bps)
			primaryActivePW = math.Round(orFloat(ibitLvl.ActivePutWall, ibitLvl.PutWall) / bps)
			primaryConfidence = ibitLvl.PositioningConfidence
			primaryWarnings = ibitLvl.PositioningWarnings
			sourceLabel = "IBIT only (~52% of BTC options OI)"
		} else {
			continue
		}

		windowSummary := map[string]interface{}{
			"spot_btc":                currentBTC,
			"spot_ibit":              d.Spot,
			"btc_per_share":          bps,
			"source":                 sourceLabel,
			"levels_btc":             primaryLvlBTC,
			"regime":                 primaryRegime,
			"active_gex_total":       primaryActiveGEX,
			"net_gex_total":          primaryNetGEX,
			"level_trajectory":       ibitLvl.LevelTrajectory,
			"active_call_wall_btc":   primaryActiveCW,
			"active_put_wall_btc":    primaryActivePW,
			"positioning_confidence": primaryConfidence,
			"positioning_warnings":   primaryWarnings,
			"volume_gex_total":       ibitLvl.VolumeGEXTotal,
			"gex_activity_ratio":     ibitLvl.GEXActivityRatio,
			"expected_move":          d.ExpectedMove,
			"oi_changes":             d.OIChanges,
			"data_freshness":         d.DataFreshness,
		}

		// Breakout
		if d.Breakout != nil {
			windowSummary["breakout"] = map[string]interface{}{
				"up_signals":   d.Breakout.UpSignals,
				"down_signals": d.Breakout.DownSignals,
				"up_score":     d.Breakout.UpScore,
				"down_score":   d.Breakout.DownScore,
				"bias":         d.Breakout.Bias,
			}
		}

		// Flow forecast
		if d.FlowForecast != nil {
			windowSummary["flow_forecast"] = d.FlowForecast
		}

		// ETF flows
		if d.ETFFlows != nil {
			windowSummary["etf_flows"] = map[string]interface{}{
				"daily_flow_millions":          math.Round(d.ETFFlows.DailyFlowDollars/1e6*10) / 10,
				"total_btc_etf_flow_millions":  math.Round(d.ETFFlows.TotalBTCETFFlow/1e6*10) / 10,
				"direction":                    d.ETFFlows.Direction,
				"strength":                     d.ETFFlows.Strength,
				"streak":                       d.ETFFlows.Streak,
				"avg_5d_millions":              math.Round(d.ETFFlows.AvgFlow5D/1e6*10) / 10,
			}
		}

		// Significant levels (top 8, omit greeks_note)
		if d.SignificantLevels != nil {
			limit := min(8, len(d.SignificantLevels))
			sigLevels := make([]map[string]interface{}, limit)
			for i := 0; i < limit; i++ {
				sl := d.SignificantLevels[i]
				sigLevels[i] = map[string]interface{}{
					"strike":    sl.Strike,
					"btc":       sl.BTC,
					"strike_btc": sl.StrikeBTC,
					"type":      sl.Type,
					"call_oi":   sl.CallOI,
					"put_oi":    sl.PutOI,
					"total_oi":  sl.TotalOI,
					"net_gex":   sl.NetGEX,
					"net_vanna": sl.NetVanna,
					"net_charm": sl.NetCharm,
					"dist_pct":  sl.DistPct,
					"is_major":  sl.IsMajor,
					"behavior":  sl.Behavior,
				}
				if sl.OIDelta != nil {
					sigLevels[i]["oi_delta"] = sl.OIDelta
				}
			}
			windowSummary["significant_levels"] = sigLevels
		}

		// Dealer delta briefing
		if d.DealerDeltaBriefing != nil {
			windowSummary["dealer_delta_briefing"] = d.DealerDeltaBriefing
		}

		// Delta flip points
		if d.DeltaFlipPoints != nil {
			fps := make([]map[string]interface{}, len(d.DeltaFlipPoints))
			for i, fp := range d.DeltaFlipPoints {
				distPct := 0.0
				if d.Spot > 0 {
					distPct = math.Round(((fp.PriceIBIT-d.Spot)/d.Spot)*1000) / 10
				}
				fps[i] = map[string]interface{}{
					"price_btc":      fp.PriceBTC,
					"distance_pct":   distPct,
					"from_direction": fp.FromDirection,
					"to_direction":   fp.ToDirection,
				}
			}
			windowSummary["delta_flip_points"] = fps
		}

		// Key level dealer deltas
		if d.DealerDeltaBriefing != nil && d.DealerDeltaBriefing.KeyLevelDeltas != nil {
			klds := map[string]interface{}{}
			for name, delta := range d.DealerDeltaBriefing.KeyLevelDeltas {
				klds[name] = math.Round(delta)
			}
			windowSummary["key_level_dealer_delta"] = klds
		}

		// IV term structure
		if d.IVTermStructure != nil {
			windowSummary["iv_term_structure"] = d.IVTermStructure
		}

		// Per-venue breakdown
		if deribitAvail && deribitLvl != nil && ibitLvl != nil && bps > 0 {
			windowSummary["venue_breakdown"] = map[string]interface{}{
				"ibit": map[string]interface{}{
					"oi_contracts":   ibitLvl.TotalCallOI + ibitLvl.TotalPutOI,
					"net_gex":        ibitLvl.NetGEXTotal,
					"call_wall_btc":  math.Round(ibitLvl.CallWall / bps),
					"put_wall_btc":   math.Round(ibitLvl.PutWall / bps),
					"gamma_flip_btc": math.Round(ibitLvl.GammaFlip / bps),
					"regime":         ibitLvl.Regime,
					"net_vanna":      ibitLvl.NetVanna,
					"net_charm":      ibitLvl.NetCharm,
				},
				"deribit": map[string]interface{}{
					"oi_btc":         d.DeribitOIBTC,
					"net_gex":        d.DeribitNetGEX,
					"call_wall_btc":  math.Round(deribitLvl.CallWall),
					"put_wall_btc":   math.Round(deribitLvl.PutWall),
					"gamma_flip_btc": math.Round(deribitLvl.GammaFlip),
					"regime":         deribitLvl.Regime,
					"net_vanna":      deribitLvl.NetVanna,
					"net_charm":      deribitLvl.NetCharm,
				},
			}
		}

		// GEX distribution (top 20 strikes by abs GEX, sorted by BTC price)
		if d.GEXChart != nil {
			type gexSortable struct {
				idx    int
				absGEX float64
			}
			sortable := make([]gexSortable, len(d.GEXChart))
			for i, s := range d.GEXChart {
				sortable[i] = gexSortable{i, math.Abs(s.NetGEX)}
			}
			// Sort by abs GEX desc
			for i := 0; i < len(sortable)-1; i++ {
				for j := i + 1; j < len(sortable); j++ {
					if sortable[j].absGEX > sortable[i].absGEX {
						sortable[i], sortable[j] = sortable[j], sortable[i]
					}
				}
			}
			top := min(20, len(sortable))
			topIdxs := make([]int, top)
			for i := 0; i < top; i++ {
				topIdxs[i] = sortable[i].idx
			}
			// Sort by BTC price
			for i := 0; i < len(topIdxs)-1; i++ {
				for j := i + 1; j < len(topIdxs); j++ {
					if d.GEXChart[topIdxs[j]].BTC < d.GEXChart[topIdxs[i]].BTC {
						topIdxs[i], topIdxs[j] = topIdxs[j], topIdxs[i]
					}
				}
			}
			gexDist := make([]map[string]interface{}, top)
			for i, idx := range topIdxs {
				s := d.GEXChart[idx]
				gexDist[i] = map[string]interface{}{
					"btc":          math.Round(s.BTC),
					"net_gex":      math.Round(s.NetGEX),
					"ibit_gex":     math.Round(s.IBITGEX),
					"deribit_gex":  math.Round(s.DeribitGEX),
					"call_oi":      s.CallOI,
					"put_oi":       s.PutOI,
					"total_volume": s.TotalVolume,
				}
			}
			windowSummary["gex_distribution"] = gexDist
		}

		// Day-over-day changes from previous cache
		prevDate, prevRaw, _ := db.GetPrevCache(ticker, w.Label)
		if prevRaw != nil {
			var prevD model.DataBlob
			if err := json.Unmarshal(prevRaw, &prevD); err == nil && prevD.Levels != nil {
				prevBPS := prevD.BTCPerShare
				if prevBPS <= 0 {
					prevBPS = bps
				}
				windowSummary["changes_vs_prev"] = map[string]interface{}{
					"prev_date":            prevDate,
					"spot_btc_prev":        math.Round(prevD.BTCSpot),
					"spot_btc_change":      math.Round(d.BTCSpot - prevD.BTCSpot),
					"regime_prev":          prevD.Levels.Regime,
					"regime_changed":       prevD.Levels.Regime != ibitLvl.Regime,
					"call_wall_btc_prev":   math.Round(prevD.Levels.CallWall / prevBPS),
					"call_wall_btc_change": math.Round(ibitLvl.CallWall/bps - prevD.Levels.CallWall/prevBPS),
					"put_wall_btc_prev":    math.Round(prevD.Levels.PutWall / prevBPS),
					"put_wall_btc_change":  math.Round(ibitLvl.PutWall/bps - prevD.Levels.PutWall/prevBPS),
					"gamma_flip_btc_prev":  math.Round(prevD.Levels.GammaFlip / prevBPS),
					"gamma_flip_btc_change": math.Round(ibitLvl.GammaFlip/bps - prevD.Levels.GammaFlip/prevBPS),
					"net_gex_prev":         prevD.Levels.NetGEXTotal,
					"net_gex_change":       ibitLvl.NetGEXTotal - prevD.Levels.NetGEXTotal,
					"pcr_prev":             prevD.Levels.PCR,
					"pcr_change":           math.Round((ibitLvl.PCR-prevD.Levels.PCR)*1000) / 1000,
				}
			}
		}

		// Structural pattern detection
		if primaryLvlBTC != nil {
			patterns := detectStructuralPatterns(primaryLvlBTC, primaryRegime, primaryNetGEX, ibitLvl, currentBTC,
				windowSummary["venue_breakdown"], windowSummary["changes_vs_prev"])
			if len(patterns) > 0 {
				windowSummary["detected_patterns"] = patterns
			}
		}

		summaries[key] = windowSummary
	}

	return summaries, nil
}

// detectStructuralPatterns identifies rule-based structural patterns near key levels.
func detectStructuralPatterns(lvlBTC map[string]interface{}, regime string, netGEX float64,
	ibitLvl *model.Levels, spotBTC float64,
	venueBreakdown interface{}, changesVsPrev interface{}) []map[string]interface{} {

	var patterns []map[string]interface{}
	gammaFlip, _ := lvlBTC["gamma_flip"].(float64)
	callWall, _ := lvlBTC["call_wall"].(float64)
	putWall, _ := lvlBTC["put_wall"].(float64)
	pcr := 0.0
	activity := 0.0
	if ibitLvl != nil {
		pcr = ibitLvl.PCR
		activity = ibitLvl.GEXActivityRatio
	}

	// Pattern 1: Gamma squeeze setup
	if gammaFlip > 0 && spotBTC > 0 {
		flipDistPct := math.Abs(spotBTC-gammaFlip) / spotBTC * 100
		if regime == "negative_gamma" && flipDistPct < 2.0 && pcr < 0.7 {
			patterns = append(patterns, map[string]interface{}{
				"pattern":           "gamma_squeeze_setup",
				"signal":            "Spot within 2% of gamma flip in negative gamma with call-heavy flow",
				"flip_distance_pct": math.Round(flipDistPct*100) / 100,
			})
		}
	}

	// Pattern 2: Wall pinning
	if callWall > 0 && putWall > 0 && spotBTC > 0 {
		cwDist := math.Abs(spotBTC-callWall) / spotBTC * 100
		pwDist := math.Abs(spotBTC-putWall) / spotBTC * 100
		nearestWall := math.Min(cwDist, pwDist)
		if nearestWall < 0.5 {
			pinnedTo := "call_wall"
			if pwDist < cwDist {
				pinnedTo = "put_wall"
			}
			patterns = append(patterns, map[string]interface{}{
				"pattern":      "wall_pinning",
				"signal":       "Spot within 0.5% of " + pinnedTo + " — gravitational pull likely",
				"distance_pct": math.Round(nearestWall*100) / 100,
				"pinned_to":    pinnedTo,
			})
		}
	}

	// Pattern 3: Venue convergence
	if vb, ok := venueBreakdown.(map[string]interface{}); ok {
		ibit, _ := vb["ibit"].(map[string]interface{})
		deribit, _ := vb["deribit"].(map[string]interface{})
		if ibit != nil && deribit != nil && spotBTC > 0 {
			ibitCW, _ := ibit["call_wall_btc"].(float64)
			deribitCW, _ := deribit["call_wall_btc"].(float64)
			if ibitCW > 0 && deribitCW > 0 {
				cwDiff := math.Abs(ibitCW-deribitCW) / spotBTC * 100
				if cwDiff < 1.0 {
					patterns = append(patterns, map[string]interface{}{
						"pattern":    "venue_convergence_resistance",
						"signal":     "IBIT + Deribit call walls within " + fmtF(cwDiff, 1) + "% — high conviction resistance",
						"ibit_cw":    ibitCW,
						"deribit_cw": deribitCW,
					})
				}
			}
			ibitPW, _ := ibit["put_wall_btc"].(float64)
			deribitPW, _ := deribit["put_wall_btc"].(float64)
			if ibitPW > 0 && deribitPW > 0 {
				pwDiff := math.Abs(ibitPW-deribitPW) / spotBTC * 100
				if pwDiff < 1.0 {
					patterns = append(patterns, map[string]interface{}{
						"pattern":    "venue_convergence_support",
						"signal":     "IBIT + Deribit put walls within " + fmtF(pwDiff, 1) + "% — high conviction support",
						"ibit_pw":    ibitPW,
						"deribit_pw": deribitPW,
					})
				}
			}
		}
	}

	// Pattern 4: Regime transition
	if cvp, ok := changesVsPrev.(map[string]interface{}); ok {
		if changed, ok := cvp["regime_changed"].(bool); ok && changed {
			prevRegime, _ := cvp["regime_prev"].(string)
			patterns = append(patterns, map[string]interface{}{
				"pattern":      "regime_transition",
				"signal":       "Regime flipped from " + prevRegime + " to " + regime + " — mixed signals, reduce sizing",
				"prev_regime":  prevRegime,
				"new_regime":   regime,
			})
		}
	}

	// Pattern 5: High activity ratio
	if activity > 1.5 {
		patterns = append(patterns, map[string]interface{}{
			"pattern":        "high_hedging_activity",
			"signal":         "Activity ratio " + fmtF(activity, 1) + "x — dealers actively hedging, levels are being tested today",
			"activity_ratio": activity,
		})
	}

	return patterns
}

func roundSlice(s []float64) []float64 {
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = math.Round(v)
	}
	return out
}

func divSlice(s []float64, d float64) []float64 {
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = math.Round(v / d)
	}
	return out
}

func orFloat(a, b float64) float64 {
	if a != 0 {
		return a
	}
	return b
}

func fmtF(v float64, prec int) string {
	return fmt.Sprintf("%.*f", prec, v)
}
