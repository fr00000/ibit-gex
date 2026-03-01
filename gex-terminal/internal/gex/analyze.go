package gex

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"sync"
	"time"

	"gex-terminal/internal/bs"
	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
	"gex-terminal/internal/fetch"
	"gex-terminal/internal/model"
)

var etLoc *time.Location

func init() {
	var err error
	etLoc, err = time.LoadLocation("America/New_York")
	if err != nil {
		panic(err)
	}
}

func nowET() time.Time { return time.Now().In(etLoc) }

// FetchAndAnalyze is the main data pipeline. It fetches options from Yahoo + Deribit,
// computes all Greeks and GEX, and returns a comprehensive DataBlob.
func FetchAndAnalyze(ticker string, maxDTE, minDTE int) (*model.DataBlob, error) {
	tc, ok := config.TickerConfig[ticker]
	if !ok {
		return nil, fmt.Errorf("unknown ticker: %s", ticker)
	}

	// ── Phase 1: Spot prices ──
	spot, err := fetchSpot(ticker)
	if err != nil {
		return nil, fmt.Errorf("fetch spot: %w", err)
	}

	refSpot, err := fetchRefSpot(tc.RefTicker)
	if err != nil {
		slog.Warn("ref spot failed, using default", "ref", tc.RefTicker, "err", err)
		refSpot = spot / tc.PerShareDefault
	}

	btcPerShare := spot / refSpot
	if btcPerShare <= 0 {
		btcPerShare = tc.PerShareDefault
	}

	rfr := config.RiskFreeRateDefault
	if rate, err := fetch.FetchRiskFreeRate(); err == nil && rate > 0 {
		rfr = rate
		slog.Debug("risk-free rate", "rate", rfr)
	}

	// ── Phase 2: Options chains from Yahoo ──
	expirations, err := fetch.FetchYahooExpirations(ticker)
	if err != nil {
		return nil, fmt.Errorf("fetch expirations: %w", err)
	}

	// Filter to DTE window
	now := time.Now().UTC()
	cutoffMin := now.AddDate(0, 0, minDTE)
	cutoffMax := now.AddDate(0, 0, maxDTE)

	var selectedExpiries []time.Time
	for _, exp := range expirations {
		if !exp.Before(cutoffMin) && !exp.After(cutoffMax) {
			selectedExpiries = append(selectedExpiries, exp)
		}
	}

	// Fallback: nearest future expiry
	if len(selectedExpiries) == 0 {
		for _, exp := range expirations {
			if exp.After(now) {
				selectedExpiries = append(selectedExpiries, exp)
				break
			}
		}
	}
	if len(selectedExpiries) == 0 && len(expirations) > 0 {
		selectedExpiries = append(selectedExpiries, expirations[len(expirations)-1])
	}

	slog.Info("selected expirations", "ticker", ticker, "count", len(selectedExpiries),
		"dte_range", fmt.Sprintf("%d-%d", minDTE, maxDTE))

	// Get previous strikes for delta tracking
	_, prevStrikes, _ := db.GetPrevStrikes(ticker)

	// Get ETF flows
	var etfFlows *model.ETFFlows
	if ticker == "IBIT" {
		etfFlows, _ = fetch.FetchFarsideFlows()
	}

	// ── Process IBIT chains ──
	strikeMap := map[float64]*model.StrikeData{}
	var cachedChains []CachedChain
	var ivTermEntries []model.IVTermEntry
	var expiryStrs []string

	for _, exp := range selectedExpiries {
		chain, err := fetch.FetchYahooOptions(ticker, exp.Unix())
		if err != nil {
			slog.Warn("skip expiry", "date", exp.Format("2006-01-02"), "err", err)
			continue
		}

		dte := exp.Sub(now).Hours() / 24.0
		T := math.Max(dte/365.0, 0.5/365.0)
		expiryStr := exp.Format("2006-01-02")
		expiryStrs = append(expiryStrs, expiryStr)

		var chainOpts []ChainOption

		// ATM IV tracking for term structure
		var ivWeightedSum, ivWeightTotal float64

		processOptions := func(opts []fetch.YahooOption, optType string) {
			sign := 1.0
			if optType == "put" {
				sign = -1.0
			}

			for _, o := range opts {
				if o.OpenInterest <= 0 || o.ImpliedVol <= 0 {
					continue
				}
				if math.Abs(o.Strike-spot)/spot > config.StrikeRangePct {
					continue
				}

				gamma := bs.Gamma(spot, o.Strike, T, rfr, o.ImpliedVol)
				delta := bs.Delta(spot, o.Strike, T, rfr, o.ImpliedVol, optType)
				vanna := bs.Vanna(spot, o.Strike, T, rfr, o.ImpliedVol)
				charm := bs.Charm(spot, o.Strike, T, rfr, o.ImpliedVol, optType)

				oi := float64(o.OpenInterest)
				vol := float64(o.Volume)
				mult := 100.0

				gex := sign * gamma * oi * mult * spot * spot * 0.01
				volGEX := sign * gamma * vol * mult * spot * spot * 0.01
				dealerDelta := -delta * oi * mult * spot
				dealerVanna := -vanna * oi * mult * spot * 0.01
				dealerCharm := -charm * oi * mult / 365.0 * spot

				sd, exists := strikeMap[o.Strike]
				if !exists {
					sd = &model.StrikeData{
						Strike:          o.Strike,
						BTC:             math.Round(o.Strike / btcPerShare),
						ExpiryBreakdown: map[string]float64{},
					}
					strikeMap[o.Strike] = sd
				}

				sd.NetGEX += gex
				sd.IBITGEX += gex
				sd.NetDealerDelta += dealerDelta
				sd.NetVanna += dealerVanna
				sd.NetCharm += dealerCharm
				sd.TotalOI += o.OpenInterest
				sd.TotalVolume += o.Volume

				if optType == "call" {
					sd.CallGEX += gex
					sd.CallOI += o.OpenInterest
					sd.CallVolume += o.Volume
					sd.CallVolGEX += volGEX
					sd.CallVanna += dealerVanna
					sd.CallCharm += dealerCharm
				} else {
					sd.PutGEX += gex
					sd.PutOI += o.OpenInterest
					sd.PutVolume += o.Volume
					sd.PutVolGEX += volGEX
					sd.PutVanna += dealerVanna
					sd.PutCharm += dealerCharm
				}

				sd.ExpiryBreakdown[expiryStr] += gex

				// ATM IV tracking (within 2% of spot)
				if math.Abs(o.Strike-spot)/spot < 0.02 {
					ivWeightedSum += o.ImpliedVol * oi
					ivWeightTotal += oi
				}

				chainOpts = append(chainOpts, ChainOption{
					Strike:     o.Strike,
					OI:         oi,
					IV:         o.ImpliedVol,
					OptionType: optType,
					Multiplier: mult,
				})
			}
		}

		processOptions(chain.Calls, "call")
		processOptions(chain.Puts, "put")

		// Store chain for delta repricing
		cachedChains = append(cachedChains, CachedChain{
			ExpiryStr: expiryStr,
			DTE:       dte,
			Options:   chainOpts,
		})

		// IV term structure entry
		if ivWeightTotal > 0 {
			ivTermEntries = append(ivTermEntries, model.IVTermEntry{
				Expiry: expiryStr,
				DTE:    int(math.Round(dte)),
				ATMIV:  ivWeightedSum / ivWeightTotal,
				Source: "ibit",
			})
		}
	}

	// Compute active_gex (new OI weighting)
	for strike, sd := range strikeMap {
		if prev, ok := prevStrikes[strike]; ok && prev.TotalOI > 0 {
			newOI := float64(sd.TotalOI - prev.TotalOI)
			if newOI > 0 {
				frac := newOI / float64(sd.TotalOI)
				sd.ActiveGEX = sd.NetGEX * (1 + frac)
			} else {
				sd.ActiveGEX = sd.NetGEX
			}
		} else {
			sd.ActiveGEX = sd.NetGEX
		}
	}

	// ── Phase 3: Deribit integration ──
	var deribitOpts []model.DeribitOption
	var deribitAvail bool
	var deribitOIBTC, deribitNetGEX float64
	var deribitLevelsBTC *model.Levels
	deribitStrikeMap := map[float64]*model.StrikeData{}

	if tc.DeribitCurrency != "" {
		opts, err := fetch.FetchDeribitOptions(minDTE, maxDTE, tc.DeribitCurrency)
		if err == nil && len(opts) > 0 {
			deribitAvail = true
			deribitOpts = opts

			for _, o := range opts {
				T := math.Max(o.DTE/365.0, 0.5/365.0)
				iv := o.IV / 100.0 // percentage → decimal
				btcSpot := o.UnderlyingPrice

				sign := 1.0
				if o.OptionType == "put" {
					sign = -1.0
				}

				gamma := bs.Gamma(btcSpot, o.Strike, T, rfr, iv)
				gex := sign * gamma * o.OI * 1.0 * btcSpot * btcSpot * 0.01

				sd, exists := deribitStrikeMap[o.Strike]
				if !exists {
					sd = &model.StrikeData{
						Strike: o.Strike,
						BTC:    math.Round(o.Strike),
					}
					deribitStrikeMap[o.Strike] = sd
				}

				sd.NetGEX += gex
				sd.DeribitGEX += gex
				deribitNetGEX += gex

				if o.OptionType == "call" {
					sd.CallGEX += gex
					sd.CallOI += int64(o.OI)
				} else {
					sd.PutGEX += gex
					sd.PutOI += int64(o.OI)
				}
				sd.TotalOI = sd.CallOI + sd.PutOI
				deribitOIBTC += o.OI

				// Deribit IV term structure
				if math.Abs(o.Strike-btcSpot)/btcSpot < 0.02 {
					// Will aggregate below
				}
			}

			// Compute Deribit-only levels
			deribitStrikes := mapToSlice(deribitStrikeMap)
			if len(deribitStrikes) > 0 && deribitOpts[0].UnderlyingPrice > 0 {
				deribitLevelsBTC = ComputeLevels(deribitStrikes, deribitOpts[0].UnderlyingPrice, nil)
			}
		}
	}

	// ── Phase 4: Compute levels ──
	ibitStrikes := mapToSlice(strikeMap)
	levels := ComputeLevels(ibitStrikes, spot, etfFlows)

	// Compute combined levels in BTC space
	var combinedLevelsBTC *model.Levels
	if deribitAvail && refSpot > 0 {
		combinedMap := map[float64]*model.StrikeData{}
		// Add IBIT strikes (converted to BTC)
		for _, sd := range strikeMap {
			btc := math.Round(sd.Strike / btcPerShare)
			if existing, ok := combinedMap[btc]; ok {
				existing.NetGEX += sd.NetGEX
				existing.IBITGEX += sd.IBITGEX
				existing.CallOI += sd.CallOI
				existing.PutOI += sd.PutOI
				existing.TotalOI += sd.TotalOI
				existing.NetVanna += sd.NetVanna
				existing.NetCharm += sd.NetCharm
			} else {
				combinedMap[btc] = &model.StrikeData{
					Strike:   btc,
					BTC:      btc,
					NetGEX:   sd.NetGEX,
					IBITGEX:  sd.IBITGEX,
					CallOI:   sd.CallOI,
					PutOI:    sd.PutOI,
					TotalOI:  sd.TotalOI,
					NetVanna: sd.NetVanna,
					NetCharm: sd.NetCharm,
				}
			}
		}
		// Add Deribit strikes
		for _, sd := range deribitStrikeMap {
			btc := math.Round(sd.Strike)
			if existing, ok := combinedMap[btc]; ok {
				existing.NetGEX += sd.NetGEX
				existing.DeribitGEX += sd.DeribitGEX
				existing.CallOI += sd.CallOI
				existing.PutOI += sd.PutOI
				existing.TotalOI += sd.TotalOI
			} else {
				combinedMap[btc] = &model.StrikeData{
					Strike:     btc,
					BTC:        btc,
					NetGEX:     sd.NetGEX,
					DeribitGEX: sd.DeribitGEX,
					CallOI:     sd.CallOI,
					PutOI:      sd.PutOI,
					TotalOI:    sd.TotalOI,
				}
			}
		}

		combinedStrikes := mapToSlice(combinedMap)
		combinedLevelsBTC = ComputeLevels(combinedStrikes, refSpot, etfFlows)
	}

	// ── Phase 5: Expected move (ATM straddle) ──
	var expectedMove *model.ExpectedMove
	for _, chain := range cachedChains {
		for _, opt := range chain.Options {
			if opt.OptionType != "call" {
				continue
			}
			// Find ATM call closest to spot
			if math.Abs(opt.Strike-spot)/spot < 0.02 && opt.OI > 10 {
				straddle := opt.IV * spot * math.Sqrt(chain.DTE/365.0) // Simplified
				emPct := straddle / spot
				expectedMove = &model.ExpectedMove{
					UpperBTC:  (spot + straddle) / btcPerShare,
					LowerBTC:  (spot - straddle) / btcPerShare,
					UpperIBIT: spot + straddle,
					LowerIBIT: spot - straddle,
					Pct:       emPct,
					DTE:       chain.DTE,
				}
				break
			}
		}
		if expectedMove != nil {
			break
		}
	}

	// ── Phase 6: Analysis layers ──
	flowForecast := ComputeFlowForecast(ibitStrikes, spot, levels, true)
	significantLevels := ComputeSignificantLevels(ibitStrikes, spot, levels, prevStrikes, true, btcPerShare, etfFlows)
	breakout := ComputeBreakout(ibitStrikes, spot, levels, expectedMove, prevStrikes, btcPerShare,
		etfFlows, flowForecast, deribitLevelsBTC, btcPerShare, refSpot)
	dealerScenarios := ComputeDealerDeltaScenarios(cachedChains, spot, levels, expectedMove,
		rfr, btcPerShare, true, deribitOpts)
	dealerBriefing := GenerateDealerDeltaBriefing(dealerScenarios, spot, levels, btcPerShare, true)

	// ── Phase 7: Build GEX chart ──
	gexChart := buildGEXChart(strikeMap, deribitStrikeMap, btcPerShare)

	// ── Phase 8: OI changes vs yesterday ──
	var oiChanges *model.OIChanges
	if prevStrikes != nil {
		var prevCallOI, prevPutOI int64
		var prevNetGEX float64
		for _, ps := range prevStrikes {
			prevCallOI += ps.CallOI
			prevPutOI += ps.PutOI
			prevNetGEX += ps.NetGEX
		}
		oiChanges = &model.OIChanges{
			CallOIChange: levels.TotalCallOI - prevCallOI,
			PutOIChange:  levels.TotalPutOI - prevPutOI,
			NetGEXChange: levels.NetGEXTotal - prevNetGEX,
		}
	}

	// ── Phase 9: Save snapshot ──
	db.SaveSnapshot(ticker, spot, refSpot, levels, ibitStrikes)

	// ── Build DataBlob ──
	blob := &model.DataBlob{
		Ticker:     ticker,
		AssetLabel: tc.AssetLabel,
		DTEKey:     maxDTE,
		Spot:       spot,
		BTCSpot:    refSpot,
		BTCPerShare: btcPerShare,
		IsBTC:      tc.AssetLabel == "BTC",
		Timestamp:  nowET().Format("2006-01-02 15:04"),
		Levels:     levels,
		CombinedLevelsBTC: combinedLevelsBTC,
		DeribitLevelsBTC:  deribitLevelsBTC,
		ExpectedMove:      expectedMove,
		GEXChart:          gexChart,
		SignificantLevels: significantLevels,
		Breakout:          breakout,
		ETFFlows:          etfFlows,
		OIChanges:         oiChanges,
		FlowForecast:      flowForecast,
		DealerDeltaProfile: dealerScenarios.Profile,
		DealerDeltaBriefing: dealerBriefing,
		DeltaFlipPoints:    dealerScenarios.FlipPoints,
		IVTermStructure:    ivTermEntries,
		Expirations:        expiryStrs,
		DeribitAvail:       deribitAvail,
		DeribitOIBTC:       deribitOIBTC,
		DeribitNetGEX:      deribitNetGEX,
	}

	blob.DTEWindow.Min = minDTE
	blob.DTEWindow.Max = maxDTE

	return blob, nil
}

func fetchSpot(ticker string) (float64, error) {
	q, err := fetch.FetchYahooQuote(ticker)
	if err != nil {
		// Fallback to v8 chart
		price, err2 := fetch.FetchYahooV8Price(ticker)
		if err2 != nil {
			return 0, fmt.Errorf("both quote methods failed: %v; %v", err, err2)
		}
		return price, nil
	}
	return q.RegularMarketPrice, nil
}

func fetchRefSpot(refTicker string) (float64, error) {
	q, err := fetch.FetchYahooQuote(refTicker)
	if err != nil {
		price, err2 := fetch.FetchYahooV8Price(refTicker)
		if err2 != nil {
			return 0, fmt.Errorf("both ref methods failed: %v; %v", err, err2)
		}
		return price, nil
	}
	return q.RegularMarketPrice, nil
}

func mapToSlice(m map[float64]*model.StrikeData) []model.StrikeData {
	result := make([]model.StrikeData, 0, len(m))
	for _, sd := range m {
		result = append(result, *sd)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Strike < result[j].Strike })
	return result
}

func buildGEXChart(ibitMap, deribitMap map[float64]*model.StrikeData, btcPerShare float64) []model.StrikeData {
	merged := map[float64]*model.StrikeData{}

	for _, sd := range ibitMap {
		btc := math.Round(sd.Strike / btcPerShare)
		if existing, ok := merged[btc]; ok {
			existing.IBITGEX += sd.IBITGEX
			existing.NetGEX += sd.NetGEX
			existing.CallOI += sd.CallOI
			existing.PutOI += sd.PutOI
			existing.TotalOI += sd.TotalOI
			existing.NetVanna += sd.NetVanna
			existing.NetCharm += sd.NetCharm
			existing.CallVolume += sd.CallVolume
			existing.PutVolume += sd.PutVolume
			existing.TotalVolume += sd.TotalVolume
			existing.NetDealerDelta += sd.NetDealerDelta
		} else {
			merged[btc] = &model.StrikeData{
				Strike:         sd.Strike,
				BTC:            btc,
				IBITGEX:        sd.IBITGEX,
				NetGEX:         sd.NetGEX,
				CallOI:         sd.CallOI,
				PutOI:          sd.PutOI,
				TotalOI:        sd.TotalOI,
				NetVanna:       sd.NetVanna,
				NetCharm:       sd.NetCharm,
				CallVolume:     sd.CallVolume,
				PutVolume:      sd.PutVolume,
				TotalVolume:    sd.TotalVolume,
				NetDealerDelta: sd.NetDealerDelta,
			}
		}
	}

	for _, sd := range deribitMap {
		btc := math.Round(sd.Strike)
		if existing, ok := merged[btc]; ok {
			existing.DeribitGEX += sd.DeribitGEX
			existing.NetGEX += sd.DeribitGEX
			existing.CallOI += sd.CallOI
			existing.PutOI += sd.PutOI
			existing.TotalOI += sd.TotalOI
		} else {
			merged[btc] = &model.StrikeData{
				BTC:        btc,
				DeribitGEX: sd.DeribitGEX,
				NetGEX:     sd.DeribitGEX,
				CallOI:     sd.CallOI,
				PutOI:      sd.PutOI,
				TotalOI:    sd.TotalOI,
			}
		}
	}

	result := make([]model.StrikeData, 0, len(merged))
	for _, sd := range merged {
		result = append(result, *sd)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].BTC < result[j].BTC })
	return result
}

// ── Cache Layer ──

var (
	yahooCheckTimes = map[string]time.Time{}
	yahooCheckMu    sync.Mutex
)

// FetchWithCache wraps FetchAndAnalyze with caching logic.
func FetchWithCache(ticker string, dte, minDTE int, forceRefresh bool) (*model.DataBlob, error) {
	today := nowET().Format("2006-01-02")

	// Check cache
	cacheDate, raw, err := db.GetLatestCache(ticker, dte, "")
	if err != nil {
		slog.Warn("cache read error", "err", err)
	}

	if !forceRefresh && raw != nil && cacheDate == today {
		var blob model.DataBlob
		if err := unmarshalBlob(raw, &blob); err == nil {
			if blob.DTEWindow.Min == minDTE {
				return &blob, nil
			}
		}
	}

	// Throttle Yahoo checks
	checkKey := fmt.Sprintf("%s-%d-%d", ticker, dte, minDTE)
	yahooCheckMu.Lock()
	lastCheck := yahooCheckTimes[checkKey]
	yahooCheckMu.Unlock()

	if !forceRefresh && raw != nil && time.Since(lastCheck).Seconds() < float64(config.YahooCheckInterval) {
		var blob model.DataBlob
		if err := unmarshalBlob(raw, &blob); err == nil {
			return &blob, nil
		}
	}

	// Fetch fresh
	blob, err := FetchAndAnalyze(ticker, dte, minDTE)
	if err != nil {
		if raw != nil {
			slog.Warn("fetch failed, returning cached", "err", err)
			var cached model.DataBlob
			if uerr := unmarshalBlob(raw, &cached); uerr == nil {
				return &cached, nil
			}
		}
		return nil, err
	}

	yahooCheckMu.Lock()
	yahooCheckTimes[checkKey] = nowET()
	yahooCheckMu.Unlock()

	// OI comparison (skip save if identical)
	if !forceRefresh && raw != nil {
		var cached model.DataBlob
		if uerr := unmarshalBlob(raw, &cached); uerr == nil {
			if cached.Levels != nil && blob.Levels != nil {
				if cached.Levels.TotalCallOI == blob.Levels.TotalCallOI &&
					cached.Levels.TotalPutOI == blob.Levels.TotalPutOI {
					return &cached, nil
				}
			}
		}
	}

	// Save to cache
	if err := db.SetCachedData(ticker, dte, blob); err != nil {
		slog.Warn("cache write error", "err", err)
	}

	return blob, nil
}

func unmarshalBlob(raw []byte, blob *model.DataBlob) error {
	return json.Unmarshal(raw, blob)
}
