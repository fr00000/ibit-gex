package background

import (
	"fmt"
	"log/slog"
	"math"
	"strings"
	"time"

	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
	"gex-terminal/internal/model"
)

// SavePredictions saves structured predictions for each expiry in each DTE window.
func SavePredictions(ticker string, results map[int]*model.DataBlob, analysisMap map[string]interface{}) {
	today := nowET().Format("2006-01-02")

	for _, w := range config.DTEWindows {
		d := results[w.Label]
		if d == nil {
			continue
		}

		windowKey := fmt.Sprintf("%d-%d", w.MinDTE, w.MaxDTE)
		bps := d.BTCPerShare
		lvl := d.Levels
		combinedLvl := d.CombinedLevelsBTC
		deribitLvl := d.DeribitLevelsBTC
		flow := d.FlowForecast
		em := d.ExpectedMove
		deribitAvail := d.DeribitAvail

		// Use combined levels when available, else IBIT-only converted to BTC
		var cwBTC, pwBTC, gfBTC, mpBTC, netGEX, netDD float64
		var regime, ddDir string

		if deribitAvail && combinedLvl != nil {
			cwBTC = combinedLvl.CallWall
			pwBTC = combinedLvl.PutWall
			gfBTC = combinedLvl.GammaFlip
			mpBTC = combinedLvl.MaxPain
			regime = combinedLvl.Regime
			netGEX = combinedLvl.NetGEXTotal
			netDD = combinedLvl.NetDealerDelta
		} else if lvl != nil && bps > 0 {
			cwBTC = lvl.CallWall / bps
			pwBTC = lvl.PutWall / bps
			gfBTC = lvl.GammaFlip / bps
			mpBTC = lvl.MaxPain / bps
			regime = lvl.Regime
			netGEX = lvl.NetGEXTotal
			netDD = lvl.NetDealerDelta
		}

		if netDD < 0 {
			ddDir = "short"
		} else {
			ddDir = "long"
		}

		// Per-venue walls in BTC
		var ibitCW, ibitPW, deribitCW, deribitPW float64
		if lvl != nil && bps > 0 {
			ibitCW = lvl.CallWall / bps
			ibitPW = lvl.PutWall / bps
		}
		if deribitLvl != nil {
			deribitCW = deribitLvl.CallWall
			deribitPW = deribitLvl.PutWall
		}

		// Venue wall agreement: walls within 3%
		venueAgree := false
		if deribitCW > 0 && ibitCW > 0 {
			cwDiff := math.Abs(deribitCW-ibitCW) / ibitCW
			pwDiff := 1.0
			if ibitPW > 0 && deribitPW > 0 {
				pwDiff = math.Abs(deribitPW-ibitPW) / ibitPW
			}
			venueAgree = cwDiff < 0.03 && pwDiff < 0.03
		}

		// Flow forecast fields
		var charmDir string
		var charmNotional float64
		var charmStrength, vannaStrength string
		var overnightDir string
		var overnightNotional float64

		if flow != nil {
			if flow.Charm != nil {
				charmDir = flow.Charm.Direction
				charmNotional = flow.Charm.NetNotional
				charmStrength = flow.Charm.Strength
			}
			if flow.Vanna != nil {
				vannaStrength = flow.Vanna.Strength
			}
			if flow.Overnight != nil {
				overnightDir = flow.Overnight.Direction
				overnightNotional = flow.Overnight.NetNotional
			}
		}

		// Expected move bounds
		var emUpper, emLower float64
		if em != nil {
			emUpper = em.UpperBTC
			emLower = em.LowerBTC
		}

		// AI bottom line extraction
		aiBL := ""
		if analysisMap != nil {
			textKey := fmt.Sprintf("%d-%dd", w.MinDTE, w.MaxDTE)
			if text, ok := analysisMap[textKey].(string); ok {
				if idx := strings.Index(text, "BOTTOM LINE:"); idx >= 0 {
					endIdx := strings.Index(text[idx:], "\n")
					if endIdx > 0 {
						aiBL = strings.TrimSpace(text[idx : idx+endIdx])
					} else if len(text)-idx < 300 {
						aiBL = strings.TrimSpace(text[idx:])
					}
				}
			}
		}

		// Save one row per expiry date in this window
		spotBTC := d.BTCSpot

		for _, expDateStr := range d.Expirations {
			expDate, err := time.Parse("2006-01-02", expDateStr)
			if err != nil {
				continue
			}
			analysisDate, _ := time.Parse("2006-01-02", today)
			dte := int(expDate.Sub(analysisDate).Hours() / 24)
			if dte < 0 {
				continue
			}

			if err := db.SavePrediction(
				today, ticker, expDateStr, dte, windowKey,
				spotBTC, cwBTC, pwBTC, gfBTC, mpBTC,
				regime, netGEX, netDD, ddDir,
				charmDir, charmNotional, charmStrength, vannaStrength,
				overnightDir, overnightNotional,
				deribitAvail, ibitCW, ibitPW, deribitCW, deribitPW, venueAgree,
				emUpper, emLower, aiBL,
			); err != nil {
				slog.Error("[predictions] save error", "ticker", ticker, "expiry", expDateStr, "err", err)
			}
		}
	}
	slog.Info("[predictions] saved predictions", "ticker", ticker)
}

// ScoreExpiredPredictions scores predictions whose expiry date has passed.
func ScoreExpiredPredictions(ticker string) (int, error) {
	cfg := config.TickerConfig[ticker]
	if cfg == nil {
		return 0, nil
	}
	candleSymbol := cfg.BinanceSymbol

	// Get unscored expired predictions
	unscored, err := db.GetUnscoredPredictions(ticker)
	if err != nil {
		return 0, err
	}
	if len(unscored) == 0 {
		return 0, nil
	}

	// Group by expiry date
	byExpiry := map[string][]db.PredictionRow{}
	for _, p := range unscored {
		byExpiry[p.ExpiryDate] = append(byExpiry[p.ExpiryDate], p)
	}

	scoredCount := 0
	today := nowET().Format("2006-01-02")

	for expDate, preds := range byExpiry {
		expTime, err := time.Parse("2006-01-02", expDate)
		if err != nil {
			continue
		}
		expTS := expTime.Unix()

		// Get the close candle on expiry day
		expCandles := db.GetRecentCandles(candleSymbol, "1d", 0)
		var expHigh, expLow, expClose float64
		foundExpiry := false
		for _, c := range expCandles {
			if c.Time >= expTS && c.Time < expTS+86400 {
				expHigh = c.High
				expLow = c.Low
				expClose = c.Close
				foundExpiry = true
				break
			}
		}
		if !foundExpiry {
			continue
		}

		for _, p := range preds {
			if p.SpotBTC == 0 {
				continue
			}

			analysisTime, err := time.Parse("2006-01-02", p.AnalysisDate)
			if err != nil {
				continue
			}
			analysisTS := analysisTime.Unix()

			// Get window candles (from analysis date to expiry)
			allCandles := db.GetRecentCandles(candleSymbol, "1d", 0)
			var winHigh, winLow float64
			first := true
			for _, c := range allCandles {
				if c.Time >= analysisTS && c.Time <= expTS+86400 {
					if first {
						winHigh = c.High
						winLow = c.Low
						first = true
					}
					if c.High > winHigh {
						winHigh = c.High
					}
					if c.Low < winLow || first {
						winLow = c.Low
					}
					first = false
				}
			}
			if first {
				continue // no candles found
			}

			cw := p.CallWallBTC
			pw := p.PutWallBTC

			// Binary scores
			var callWallHeld, putWallHeld, rangeHeld *int
			if cw > 0 {
				v := boolToInt(winHigh <= cw)
				callWallHeld = &v
			}
			if pw > 0 {
				v := boolToInt(winLow >= pw)
				putWallHeld = &v
			}
			if callWallHeld != nil && putWallHeld != nil {
				v := boolToInt(*callWallHeld == 1 && *putWallHeld == 1)
				rangeHeld = &v
			}

			// Expected move held
			var emHeld *int
			if p.CallWallBTC > 0 && p.PutWallBTC > 0 {
				// Use the EM from predictions if available
				// For now use expiry day candle
				v := boolToInt(expHigh <= cw && expLow >= pw)
				emHeld = &v
			}

			// Regime correctness
			dteDays := int(expTime.Sub(analysisTime).Hours()/24)
			if dteDays < 1 {
				dteDays = 1
			}
			realizedRangePct := (winHigh - winLow) / p.SpotBTC * 100
			dailyRange := realizedRangePct / float64(dteDays)

			var regimeCorrect *int
			if p.Regime == "positive_gamma" {
				v := boolToInt(dailyRange < 2.0)
				regimeCorrect = &v
			} else if p.Regime == "negative_gamma" {
				v := boolToInt(dailyRange > 1.0)
				regimeCorrect = &v
			}

			// Charm correctness
			var charmCorrect *int
			if p.DealerDeltaDir != "" && p.SpotBTC > 0 {
				// Find next day's open
				for _, c := range allCandles {
					if c.Time > analysisTS {
						if p.DealerDeltaDir == "long" {
							v := boolToInt(c.Open > p.SpotBTC)
							charmCorrect = &v
						} else {
							v := boolToInt(c.Open < p.SpotBTC)
							charmCorrect = &v
						}
						break
					}
				}
			}

			// Breach and error metrics
			var maxBreachCall, maxBreachPut float64
			if cw > 0 && winHigh > cw {
				maxBreachCall = (winHigh - cw) / cw * 100
			}
			if pw > 0 && winLow < pw {
				maxBreachPut = (pw - winLow) / pw * 100
			}

			var callWallError, putWallError *float64
			if cw > 0 {
				v := (winHigh - cw) / p.SpotBTC * 100
				callWallError = &v
			}
			if pw > 0 {
				v := (pw - winLow) / p.SpotBTC * 100
				putWallError = &v
			}

			// Venue agree held
			var venueAgreeHeld *int
			if p.VenueWallsAgree != nil && *p.VenueWallsAgree && rangeHeld != nil {
				venueAgreeHeld = rangeHeld
			}

			// T+2 scoring
			t2TS := expTS + 86400
			var cwHeldT2, pwHeldT2, rangeHeldT2 *int
			for _, c := range allCandles {
				if c.Time >= t2TS && c.Time < t2TS+86400 {
					// Found T+2 candle — compute extended window
					var t2WinHigh, t2WinLow float64
					t2First := true
					for _, tc := range allCandles {
						if tc.Time >= analysisTS && tc.Time <= t2TS+86400 {
							if t2First {
								t2WinHigh = tc.High
								t2WinLow = tc.Low
								t2First = false
							}
							if tc.High > t2WinHigh {
								t2WinHigh = tc.High
							}
							if tc.Low < t2WinLow {
								t2WinLow = tc.Low
							}
						}
					}
					if !t2First {
						if cw > 0 {
							v := boolToInt(t2WinHigh <= cw*1.01)
							cwHeldT2 = &v
						}
						if pw > 0 {
							v := boolToInt(t2WinLow >= pw*0.99)
							pwHeldT2 = &v
						}
						if cwHeldT2 != nil && pwHeldT2 != nil {
							v := boolToInt(*cwHeldT2 == 1 && *pwHeldT2 == 1)
							rangeHeldT2 = &v
						}
					}
					break
				}
			}

			// Update the prediction in DB
			if err := db.ScorePrediction(
				p.AnalysisDate, ticker, p.ExpiryDate, today,
				expHigh, expLow, expClose,
				winHigh, winLow,
				callWallHeld, putWallHeld, rangeHeld, emHeld,
				regimeCorrect, charmCorrect, venueAgreeHeld,
				maxBreachCall, maxBreachPut, realizedRangePct,
				callWallError, putWallError,
				cwHeldT2, pwHeldT2, rangeHeldT2,
			); err != nil {
				slog.Error("[predictions] score update error", "err", err)
			} else {
				scoredCount++
			}
		}
	}

	return scoredCount, nil
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
