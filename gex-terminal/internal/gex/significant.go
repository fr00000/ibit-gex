package gex

import (
	"fmt"
	"math"
	"sort"

	"gex-terminal/internal/model"
)

// ComputeSignificantLevels finds major positioning levels with regime-adjusted behavior.
func ComputeSignificantLevels(
	strikes []model.StrikeData,
	spot float64,
	levels *model.Levels,
	prevStrikes map[float64]model.StrikeHistory,
	isCrypto bool,
	refPerShare float64,
	etfFlows *model.ETFFlows,
) []model.SignificantLevel {
	if len(strikes) == 0 || spot <= 0 {
		return nil
	}

	// Calculate OI percentiles
	ois := make([]float64, len(strikes))
	for i, s := range strikes {
		ois[i] = float64(s.TotalOI)
	}
	sort.Float64s(ois)

	p75 := percentile(ois, 75)
	p90 := percentile(ois, 90)

	var result []model.SignificantLevel

	for _, s := range strikes {
		totalOI := float64(s.TotalOI)
		if totalOI < p75 {
			continue
		}

		// Within 25% of spot
		distPct := (s.Strike - spot) / spot * 100
		if math.Abs(distPct) > 25 {
			continue
		}

		// Classify
		var lvlType string
		if float64(s.PutOI) > float64(s.CallOI)*1.5 && s.NetGEX < 0 {
			lvlType = "put_wall"
		} else if float64(s.CallOI) > float64(s.PutOI)*1.5 && s.NetGEX > 0 {
			lvlType = "call_wall"
		} else if totalOI > p90 {
			lvlType = "oi_magnet"
		} else {
			continue
		}

		isMajor := totalOI > p90

		// Behavior description
		behavior := computeBehavior(lvlType, isMajor, levels.Regime, etfFlows)

		// Greeks note
		var greeksNote *string
		if note := computeGreeksNote(lvlType, s.NetVanna, s.NetCharm); note != "" {
			greeksNote = &note
		}

		// OI delta vs yesterday
		var oiDelta *model.OIDelta
		if prevStrikes != nil {
			if prev, ok := prevStrikes[s.Strike]; ok {
				delta := s.TotalOI - prev.TotalOI
				var pctChg float64
				if prev.TotalOI > 0 {
					pctChg = float64(delta) / float64(prev.TotalOI) * 100
				}

				status := "stable"
				if delta > 0 && math.Abs(pctChg) > 10 {
					status = "BUILDING"
				} else if delta < 0 && math.Abs(pctChg) > 10 {
					status = "DECAYING"
				}

				oiDelta = &model.OIDelta{
					Delta:  delta,
					Pct:    pctChg,
					Status: status,
				}
			}
		}

		btcPrice := s.Strike
		if !isCrypto && refPerShare > 0 {
			btcPrice = s.Strike / refPerShare
		}

		result = append(result, model.SignificantLevel{
			Strike:    s.Strike,
			BTC:       btcPrice,
			StrikeBTC: btcPrice,
			Type:      lvlType,
			CallOI:    s.CallOI,
			PutOI:     s.PutOI,
			TotalOI:   s.TotalOI,
			NetGEX:    s.NetGEX,
			NetVanna:  s.NetVanna,
			NetCharm:  s.NetCharm,
			DistPct:   distPct,
			IsMajor:   isMajor,
			Behavior:  behavior,
			GreeksNote: greeksNote,
			OIDelta:   oiDelta,
		})
	}

	sort.Slice(result, func(i, j int) bool { return result[i].Strike < result[j].Strike })
	return result
}

func computeBehavior(lvlType string, isMajor bool, regime string, etfFlows *model.ETFFlows) string {
	var behavior string

	if regime == "negative_gamma" {
		switch lvlType {
		case "put_wall":
			if isMajor {
				behavior = "acceleration zone — dealers sell + gamma amplifies"
			} else {
				behavior = "minor acceleration zone"
			}
		case "call_wall":
			if isMajor {
				behavior = "mechanical resistance — dealers sell + gamma stabilizes"
			} else {
				behavior = "minor resistance"
			}
		case "oi_magnet":
			behavior = "OI magnet — gravitational pull near expiry"
		}
	} else {
		switch lvlType {
		case "put_wall":
			if isMajor {
				behavior = "mechanical support — dealers buy + gamma stabilizes"
			} else {
				behavior = "minor support"
			}
		case "call_wall":
			if isMajor {
				behavior = "mechanical resistance — dealers sell + gamma stabilizes"
			} else {
				behavior = "minor resistance"
			}
		case "oi_magnet":
			behavior = "OI magnet — gravitational pull near expiry"
		}
	}

	// ETF flow modifier
	if etfFlows != nil {
		switch lvlType {
		case "put_wall":
			if etfFlows.Direction == "inflow" {
				behavior += " + inflows reinforce"
			} else if etfFlows.Direction == "outflow" && (etfFlows.Strength == "moderate" || etfFlows.Strength == "strong") {
				behavior += " — outflows weaken"
			}
		case "call_wall":
			if etfFlows.Direction == "outflow" {
				behavior += " + outflows reinforce"
			} else if etfFlows.Direction == "inflow" && (etfFlows.Strength == "moderate" || etfFlows.Strength == "strong") {
				behavior += " — inflows pressure"
			}
		}
	}

	return behavior
}

func computeGreeksNote(lvlType string, netVanna, netCharm float64) string {
	var parts []string

	if math.Abs(netVanna) > 500 {
		switch lvlType {
		case "put_wall":
			if netVanna > 0 {
				parts = append(parts, "Vol spike weakens support")
			} else {
				parts = append(parts, "Vol spike strengthens support")
			}
		case "call_wall":
			if netVanna > 0 {
				parts = append(parts, "Vol crush strengthens ceiling")
			} else {
				parts = append(parts, "Vol crush weakens ceiling")
			}
		}
	}

	if math.Abs(netCharm) > 500 {
		if netCharm > 0 {
			parts = append(parts, "Charm: dealers buy overnight")
		} else {
			parts = append(parts, "Charm: dealers sell overnight")
		}
	}

	if len(parts) == 0 {
		return ""
	}
	result := parts[0]
	for _, p := range parts[1:] {
		result += "; " + p
	}
	return result
}

func percentile(sorted []float64, pct float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := pct / 100.0 * float64(len(sorted)-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi || hi >= len(sorted) {
		return sorted[lo]
	}
	frac := idx - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

// helper used for breakout
func formatBTC(price float64) string {
	return fmt.Sprintf("$%.0f", price)
}
