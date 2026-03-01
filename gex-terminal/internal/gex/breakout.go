package gex

import (
	"fmt"
	"math"
	"sort"

	"gex-terminal/internal/model"
)

// ComputeBreakout generates bullish/bearish breakout signals.
func ComputeBreakout(
	strikes []model.StrikeData,
	spot float64,
	levels *model.Levels,
	expectedMove *model.ExpectedMove,
	prevStrikes map[float64]model.StrikeHistory,
	refPerShare float64,
	etfFlows *model.ETFFlows,
	flowForecast *model.FlowForecast,
	deribitLevelsBTC *model.Levels,
	btcPerShare, btcSpot float64,
) *model.BreakoutAnalysis {
	result := &model.BreakoutAnalysis{
		UpSignals:   []string{},
		DownSignals: []string{},
		UpTargets:   []model.BreakoutTarget{},
		DownTargets: []model.BreakoutTarget{},
	}

	if len(strikes) == 0 || spot <= 0 || levels.CallWall <= 0 || levels.PutWall <= 0 {
		result.Bias = "balanced"
		return result
	}

	// ── 1. Wall Asymmetry ──
	cwGEX := 0.0
	pwGEX := 0.0
	for _, s := range strikes {
		if s.Strike == levels.CallWall {
			cwGEX = s.NetGEX
		}
		if s.Strike == levels.PutWall {
			pwGEX = math.Abs(s.NetGEX)
		}
	}
	if pwGEX > 0 {
		ratio := cwGEX / pwGEX
		if ratio < 0.5 {
			result.UpSignals = append(result.UpSignals, fmt.Sprintf("Weak ceiling: call wall GEX is %.1fx put wall", ratio))
		} else if ratio > 2.0 && cwGEX > 0 {
			result.DownSignals = append(result.DownSignals, fmt.Sprintf("Weak floor: put wall GEX is %.1fx call wall", 1/ratio))
		}
	}

	// ── 2. Wall OI Decay/Build ──
	if prevStrikes != nil {
		ois := make([]float64, 0, len(prevStrikes))
		for _, ps := range prevStrikes {
			ois = append(ois, float64(ps.TotalOI))
		}
		sort.Float64s(ois)
		p80 := percentile(ois, 80)

		for _, s := range strikes {
			prev, ok := prevStrikes[s.Strike]
			if !ok || float64(prev.TotalOI) < p80 {
				continue
			}
			if prev.TotalOI <= 0 {
				continue
			}
			pctChg := float64(s.TotalOI-prev.TotalOI) / float64(prev.TotalOI) * 100

			if s.Strike == levels.CallWall {
				if pctChg < -20 {
					result.UpSignals = append(result.UpSignals,
						fmt.Sprintf("Call wall DECAYING (%.0f%% OI drop)", math.Abs(pctChg)))
				} else if pctChg > 25 {
					result.DownSignals = append(result.DownSignals,
						fmt.Sprintf("Call wall BUILDING (+%.0f%% OI)", pctChg))
				}
			}
			if s.Strike == levels.PutWall {
				if pctChg < -20 {
					result.DownSignals = append(result.DownSignals,
						fmt.Sprintf("Put wall DECAYING (%.0f%% OI drop)", math.Abs(pctChg)))
				} else if pctChg > 25 {
					result.UpSignals = append(result.UpSignals,
						fmt.Sprintf("Put wall BUILDING (+%.0f%% OI)", pctChg))
				}
			}
		}
	}

	// ── 3. Expected Move vs Range ──
	rangePct := (levels.CallWall - levels.PutWall) / spot * 100
	if expectedMove != nil && rangePct > 0.5 {
		emWidth := expectedMove.Pct * 2 * 100
		if emWidth > rangePct {
			note := fmt.Sprintf("Expected move (%.1f%%) > range (%.1f%%) — breakout likely", emWidth, rangePct)
			result.EMNote = &note
		}
	}

	// ── 4. Negative Gamma Near Wall ──
	if levels.Regime == "negative_gamma" {
		threshold := math.Max(rangePct*0.3, 0.5) / 100.0 * spot
		cwDist := math.Abs(spot - levels.CallWall)
		pwDist := math.Abs(spot - levels.PutWall)

		if cwDist < threshold && spot <= levels.CallWall {
			result.UpSignals = append(result.UpSignals, "Negative gamma near call wall — gamma squeeze potential")
		}
		if pwDist < threshold && spot >= levels.PutWall {
			result.DownSignals = append(result.DownSignals, "Negative gamma near put wall — waterfall risk")
		}
	}

	// ── 5. OI Beyond Walls ──
	if levels.TotalCallOI > 0 {
		var aboveCW int64
		for _, s := range strikes {
			if s.Strike > levels.CallWall {
				aboveCW += s.CallOI
			}
		}
		pct := float64(aboveCW) / float64(levels.TotalCallOI) * 100
		if pct > 65 {
			result.UpSignals = append(result.UpSignals, fmt.Sprintf("%.0f%% of call OI above call wall", pct))
		}
	}
	if levels.TotalPutOI > 0 {
		var belowPW int64
		for _, s := range strikes {
			if s.Strike < levels.PutWall {
				belowPW += s.PutOI
			}
		}
		pct := float64(belowPW) / float64(levels.TotalPutOI) * 100
		if pct > 65 {
			result.DownSignals = append(result.DownSignals, fmt.Sprintf("%.0f%% of put OI below put wall", pct))
		}
	}

	// ── 6. P/C Ratio ──
	pcr := levels.PCR
	if pcr > 1.5 {
		result.UpSignals = append(result.UpSignals, fmt.Sprintf("P/C ratio %.2f — short squeeze fuel", pcr))
		result.DownSignals = append(result.DownSignals, fmt.Sprintf("P/C ratio %.2f — heavy put skew", pcr))
	} else if pcr < 0.6 {
		result.UpSignals = append(result.UpSignals, fmt.Sprintf("P/C ratio %.2f — aggressive call skew", pcr))
	}

	// ── 7. ETF Flow Signals ──
	if etfFlows != nil {
		if etfFlows.Streak >= 3 && (etfFlows.Strength == "moderate" || etfFlows.Strength == "strong") {
			result.UpSignals = append(result.UpSignals,
				fmt.Sprintf("Institutional accumulation: %d-day inflow streak", etfFlows.Streak))
		}
		if etfFlows.Streak <= -3 && (etfFlows.Strength == "moderate" || etfFlows.Strength == "strong") {
			result.DownSignals = append(result.DownSignals,
				fmt.Sprintf("Institutional distribution: %d-day outflow streak", -etfFlows.Streak))
		}
		if etfFlows.AvgFlow5D > 100e6 {
			result.UpSignals = append(result.UpSignals, "Sustained buying pressure ($100M+ avg 5d inflow)")
		}
		if etfFlows.AvgFlow5D < -100e6 {
			result.DownSignals = append(result.DownSignals, "Sustained selling pressure ($100M+ avg 5d outflow)")
		}
	}

	// ── 8. Venue Wall Convergence ──
	if deribitLevelsBTC != nil && btcPerShare > 0 && btcSpot > 0 {
		ibitCW := levels.CallWall / btcPerShare
		ibitPW := levels.PutWall / btcPerShare
		deribitCW := deribitLevelsBTC.CallWall
		deribitPW := deribitLevelsBTC.PutWall

		if ibitCW > 0 && deribitCW > 0 {
			cwDiff := math.Abs(deribitCW-ibitCW) / ibitCW * 100
			if cwDiff < 2 {
				result.DownSignals = append(result.DownSignals, "Venue-reinforced call wall (IBIT + Deribit within 2%)")
			} else if cwDiff > 5 && deribitCW > ibitCW {
				result.UpSignals = append(result.UpSignals,
					fmt.Sprintf("Deribit call wall %.0f%% above IBIT — room to run", cwDiff))
			}
		}
		if ibitPW > 0 && deribitPW > 0 {
			pwDiff := math.Abs(deribitPW-ibitPW) / ibitPW * 100
			if pwDiff < 2 {
				result.UpSignals = append(result.UpSignals, "Venue-reinforced put wall (IBIT + Deribit within 2%)")
			} else if pwDiff > 5 && deribitPW < ibitPW {
				result.DownSignals = append(result.DownSignals,
					fmt.Sprintf("Deribit put wall %.0f%% below IBIT — room to fall", pwDiff))
			}
		}
	}

	// ── 9. Overnight Charm Flow ──
	if flowForecast != nil && flowForecast.Charm != nil {
		str := flowForecast.Charm.Strength
		if str == "moderate" || str == "significant" {
			if flowForecast.Charm.Direction == "buy" {
				result.UpSignals = append(result.UpSignals,
					fmt.Sprintf("Overnight charm flow: dealers BUY (%s)", str))
			} else {
				result.DownSignals = append(result.DownSignals,
					fmt.Sprintf("Overnight charm flow: dealers SELL (%s)", str))
			}
		}
	}

	// ── Targets ──
	// Up targets: top 2 positive-GEX strikes above call wall
	for _, s := range strikes {
		if s.Strike > levels.CallWall && s.NetGEX > 0 {
			btc := s.Strike
			if btcPerShare > 0 {
				btc = s.Strike / btcPerShare
			}
			result.UpTargets = append(result.UpTargets, model.BreakoutTarget{
				Strike:  s.Strike,
				BTC:     btc,
				TotalOI: s.TotalOI,
				NetGEX:  s.NetGEX,
			})
		}
	}
	sort.Slice(result.UpTargets, func(i, j int) bool { return result.UpTargets[i].NetGEX > result.UpTargets[j].NetGEX })
	if len(result.UpTargets) > 2 {
		result.UpTargets = result.UpTargets[:2]
	}

	// Down targets: top 2 negative-GEX strikes below put wall
	for _, s := range strikes {
		if s.Strike < levels.PutWall && s.NetGEX < 0 {
			btc := s.Strike
			if btcPerShare > 0 {
				btc = s.Strike / btcPerShare
			}
			result.DownTargets = append(result.DownTargets, model.BreakoutTarget{
				Strike:  s.Strike,
				BTC:     btc,
				TotalOI: s.TotalOI,
				NetGEX:  s.NetGEX,
			})
		}
	}
	sort.Slice(result.DownTargets, func(i, j int) bool { return result.DownTargets[i].NetGEX < result.DownTargets[j].NetGEX })
	if len(result.DownTargets) > 2 {
		result.DownTargets = result.DownTargets[:2]
	}

	// ── Bias ──
	result.UpScore = len(result.UpSignals)
	result.DownScore = len(result.DownSignals)
	if result.UpScore >= result.DownScore+2 {
		result.Bias = "upside"
	} else if result.DownScore >= result.UpScore+2 {
		result.Bias = "downside"
	} else {
		result.Bias = "balanced"
	}

	return result
}
