package fetch

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
)

var (
	coinglassMu        sync.Mutex
	coinglassLastFetch time.Time
)

const coinglassBase = "https://open-api-v4.coinglass.com/api"

// FetchCoinglassData fetches all metrics from Coinglass API and stores in DB.
// Requires COINGLASS_API_KEY to be set.
func FetchCoinglassData() error {
	if config.CoinglassAPIKey == "" {
		return nil // Graceful degradation
	}

	coinglassMu.Lock()
	defer coinglassMu.Unlock()

	// Only fetch once per day
	today := nowET().Format("2006-01-02")
	if coinglassLastFetch.Format("2006-01-02") == today {
		return nil
	}

	slog.Info("fetching coinglass data")

	var errs []error
	if err := fetchFundingRates(); err != nil {
		errs = append(errs, fmt.Errorf("funding: %w", err))
	}
	if err := fetchAggregateOI(); err != nil {
		errs = append(errs, fmt.Errorf("oi: %w", err))
	}
	if err := fetchLiquidationHistory(); err != nil {
		errs = append(errs, fmt.Errorf("liquidation-history: %w", err))
	}
	if err := fetchLiquidationSnapshot(); err != nil {
		errs = append(errs, fmt.Errorf("liquidation-snapshot: %w", err))
	}

	coinglassLastFetch = time.Now()

	if len(errs) > 0 {
		slog.Warn("coinglass partial errors", "errors", errs)
	}
	return nil
}

func coinglassGet(path string) (json.RawMessage, error) {
	url := coinglassBase + path
	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("CG-API-KEY", config.CoinglassAPIKey)

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Code string          `json:"code"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if result.Code != "0" {
		return nil, fmt.Errorf("coinglass code: %s", result.Code)
	}
	return result.Data, nil
}

func fetchFundingRates() error {
	data, err := coinglassGet("/futures/funding-rate/oi-weight-history?symbol=BTC&interval=h8&limit=90")
	if err != nil {
		return err
	}

	var entries []struct {
		Time  int64   `json:"time"`
		Close float64 `json:"close"`
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}

	// Group by date, average funding rates per day
	dayRates := map[string][]float64{}
	for _, e := range entries {
		t := time.UnixMilli(e.Time).UTC()
		day := t.Format("2006-01-02")
		dayRates[day] = append(dayRates[day], e.Close)
	}

	for day, rates := range dayRates {
		sum := 0.0
		for _, r := range rates {
			sum += r
		}
		avg := sum / float64(len(rates))
		extra, _ := json.Marshal(map[string]int{"samples": len(rates)})
		if err := db.SaveCoinglassData(day, "BTC", "avg_funding_rate", avg, string(extra)); err != nil {
			slog.Warn("save funding", "day", day, "err", err)
		}
	}

	slog.Info("coinglass funding rates saved", "days", len(dayRates))
	return nil
}

func fetchAggregateOI() error {
	data, err := coinglassGet("/futures/open-interest/aggregated-history?symbol=BTC&interval=1d&limit=90")
	if err != nil {
		return err
	}

	var entries []struct {
		Time  int64   `json:"time"`
		Close float64 `json:"close"`
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}

	for _, e := range entries {
		day := time.UnixMilli(e.Time).UTC().Format("2006-01-02")
		if err := db.SaveCoinglassData(day, "BTC", "total_oi_usd", e.Close, ""); err != nil {
			slog.Warn("save oi", "day", day, "err", err)
		}
	}

	slog.Info("coinglass OI saved", "entries", len(entries))
	return nil
}

func fetchLiquidationHistory() error {
	data, err := coinglassGet("/futures/liquidation/aggregated-history?symbol=BTC&interval=1d&limit=90&exchange_list=Binance,OKX,Bybit,Bitget,BingX,dYdX,CoinEx,Kraken,Bitmex")
	if err != nil {
		return err
	}

	var entries []struct {
		Time              int64   `json:"time"`
		AggLongLiqUSD     float64 `json:"aggregated_long_liquidation_usd"`
		AggShortLiqUSD    float64 `json:"aggregated_short_liquidation_usd"`
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}

	for _, e := range entries {
		day := time.UnixMilli(e.Time).UTC().Format("2006-01-02")
		total := e.AggLongLiqUSD + e.AggShortLiqUSD
		if err := db.SaveCoinglassData(day, "BTC", "total_liquidation_usd", total, ""); err != nil {
			slog.Warn("save liq", "day", day, "err", err)
		}
	}

	slog.Info("coinglass liquidation history saved", "entries", len(entries))
	return nil
}

func fetchLiquidationSnapshot() error {
	data, err := coinglassGet("/futures/liquidation/coin-list")
	if err != nil {
		return err
	}

	var coins []struct {
		Symbol           string  `json:"symbol"`
		LongLiqUSD24h    float64 `json:"long_liquidation_usd_24h"`
		ShortLiqUSD24h   float64 `json:"short_liquidation_usd_24h"`
		LongLiqUSD12h    float64 `json:"long_liquidation_usd_12h"`
		ShortLiqUSD12h   float64 `json:"short_liquidation_usd_12h"`
		LongLiqUSD4h     float64 `json:"long_liquidation_usd_4h"`
		ShortLiqUSD4h    float64 `json:"short_liquidation_usd_4h"`
		LongLiqUSD1h     float64 `json:"long_liquidation_usd_1h"`
		ShortLiqUSD1h    float64 `json:"short_liquidation_usd_1h"`
	}
	if err := json.Unmarshal(data, &coins); err != nil {
		return err
	}

	for _, c := range coins {
		if c.Symbol != "BTC" {
			continue
		}

		total24h := c.LongLiqUSD24h + c.ShortLiqUSD24h
		extra, _ := json.Marshal(map[string]float64{
			"long_24h":  c.LongLiqUSD24h,
			"short_24h": c.ShortLiqUSD24h,
			"long_12h":  c.LongLiqUSD12h,
			"short_12h": c.ShortLiqUSD12h,
			"long_4h":   c.LongLiqUSD4h,
			"short_4h":  c.ShortLiqUSD4h,
			"long_1h":   c.LongLiqUSD1h,
			"short_1h":  c.ShortLiqUSD1h,
		})

		today := nowET().Format("2006-01-02")
		if err := db.SaveCoinglassData(today, "BTC", "liquidation_snapshot", total24h, string(extra)); err != nil {
			return err
		}
		slog.Info("coinglass liquidation snapshot saved", "total_24h", total24h)
		break
	}

	return nil
}

func nowET() time.Time {
	loc, _ := time.LoadLocation("America/New_York")
	return time.Now().In(loc)
}
