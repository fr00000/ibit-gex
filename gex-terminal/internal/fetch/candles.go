package fetch

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"gex-terminal/internal/db"
	"gex-terminal/internal/model"
)

// TF durations in milliseconds.
var tfDurations = map[string]int64{
	"15m": 15 * 60 * 1000,
	"1h":  60 * 60 * 1000,
	"4h":  4 * 60 * 60 * 1000,
	"1d":  24 * 60 * 60 * 1000,
}

// BackfillCandles fetches historical candles and stores them in the DB.
func BackfillCandles(symbol, tf string, days int) error {
	dur, ok := tfDurations[tf]
	if !ok {
		return fmt.Errorf("unknown timeframe: %s", tf)
	}

	now := time.Now().UnixMilli()
	start := now - int64(days)*24*60*60*1000
	total := 0

	for cursor := start; cursor < now; {
		klines, err := fetchBinanceKlines(symbol, tf, cursor, now, 1000)
		if err != nil {
			slog.Warn("kline fetch error", "err", err)
			break
		}
		if len(klines) == 0 {
			break
		}

		candles := make([]model.Candle, 0, len(klines))
		var lastTs int64
		for _, k := range klines {
			candles = append(candles, model.Candle{
				Time:  k.Time / 1000, // ms → seconds
				Open:  k.Open,
				High:  k.High,
				Low:   k.Low,
				Close: k.Close,
			})
			lastTs = k.Time
		}

		if err := db.SaveCandles(symbol, tf, candles); err != nil {
			return fmt.Errorf("save candles: %w", err)
		}

		total += len(candles)
		cursor = lastTs + dur

		if len(klines) < 1000 {
			break
		}
		time.Sleep(200 * time.Millisecond)
	}

	slog.Info("backfill done", "symbol", symbol, "tf", tf, "candles", total)
	return nil
}

// UpdateCandles fetches recent candles since the last stored timestamp.
func UpdateCandles(symbol, tf string) error {
	latest, err := db.GetLatestCandleTime(symbol, tf)
	if err != nil {
		return err
	}

	if latest == 0 {
		return BackfillCandles(symbol, tf, 90)
	}

	startMs := latest * 1000 // seconds → ms
	now := time.Now().UnixMilli()

	klines, err := fetchBinanceKlines(symbol, tf, startMs, now, 1000)
	if err != nil {
		return err
	}

	candles := make([]model.Candle, 0, len(klines))
	for _, k := range klines {
		candles = append(candles, model.Candle{
			Time:  k.Time / 1000,
			Open:  k.Open,
			High:  k.High,
			Low:   k.Low,
			Close: k.Close,
		})
	}

	return db.SaveCandles(symbol, tf, candles)
}

type rawKline struct {
	Time  int64
	Open  float64
	High  float64
	Low   float64
	Close float64
}

func fetchBinanceKlines(symbol, tf string, startMs, endMs int64, limit int) ([]rawKline, error) {
	client := &http.Client{Timeout: 10 * time.Second}

	// Try Binance US, Binance Global, then OKX
	endpoints := []struct {
		name string
		url  string
		okx  bool
	}{
		{"binance-us", fmt.Sprintf("https://api.binance.us/api/v3/klines?symbol=%s&interval=%s&startTime=%d&endTime=%d&limit=%d",
			symbol, tf, startMs, endMs, limit), false},
		{"binance", fmt.Sprintf("https://api.binance.com/api/v3/klines?symbol=%s&interval=%s&startTime=%d&endTime=%d&limit=%d",
			symbol, tf, startMs, endMs, limit), false},
		{"okx", buildOKXURL(symbol, tf, startMs, limit), true},
	}

	for _, ep := range endpoints {
		if ep.url == "" {
			continue
		}

		resp, err := client.Get(ep.url)
		if err != nil {
			continue
		}

		var klines []rawKline
		if ep.okx {
			klines, err = parseOKX(resp)
		} else {
			klines, err = parseBinance(resp)
		}
		resp.Body.Close()

		if err != nil {
			slog.Debug("kline parse failed", "source", ep.name, "err", err)
			continue
		}
		if len(klines) > 0 {
			return klines, nil
		}
	}

	return nil, fmt.Errorf("all kline sources failed for %s %s", symbol, tf)
}

func buildOKXURL(symbol, tf string, startMs int64, limit int) string {
	// Convert symbol: BTCUSDT → BTC-USDT
	instID := ""
	if len(symbol) > 4 && symbol[len(symbol)-4:] == "USDT" {
		instID = symbol[:len(symbol)-4] + "-USDT"
	} else {
		return ""
	}

	// Convert tf: 1h → 1H, 4h → 4H, 1d → 1D
	okxTF := tf
	switch tf {
	case "1h":
		okxTF = "1H"
	case "4h":
		okxTF = "4H"
	case "1d":
		okxTF = "1D"
	case "15m":
		okxTF = "15m"
	}

	return fmt.Sprintf("https://www.okx.com/api/v5/market/history-candles?instId=%s&bar=%s&after=%d&limit=%d",
		instID, okxTF, startMs, limit)
}

func parseBinance(resp *http.Response) ([]rawKline, error) {
	var raw [][]json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}

	klines := make([]rawKline, 0, len(raw))
	for _, r := range raw {
		if len(r) < 5 {
			continue
		}
		var ts int64
		json.Unmarshal(r[0], &ts)
		o := parseFloat(r[1])
		h := parseFloat(r[2])
		l := parseFloat(r[3])
		c := parseFloat(r[4])
		klines = append(klines, rawKline{Time: ts, Open: o, High: h, Low: l, Close: c})
	}
	return klines, nil
}

func parseOKX(resp *http.Response) ([]rawKline, error) {
	var result struct {
		Code string          `json:"code"`
		Data [][]json.RawMessage `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if result.Code != "0" {
		return nil, fmt.Errorf("okx code: %s", result.Code)
	}

	klines := make([]rawKline, 0, len(result.Data))
	for _, r := range result.Data {
		if len(r) < 5 {
			continue
		}
		ts := parseIntFromRaw(r[0])
		o := parseFloat(r[1])
		h := parseFloat(r[2])
		l := parseFloat(r[3])
		c := parseFloat(r[4])
		klines = append(klines, rawKline{Time: ts, Open: o, High: h, Low: l, Close: c})
	}

	// OKX returns in reverse chronological order
	for i, j := 0, len(klines)-1; i < j; i, j = i+1, j-1 {
		klines[i], klines[j] = klines[j], klines[i]
	}
	return klines, nil
}

func parseFloat(raw json.RawMessage) float64 {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		f, _ := strconv.ParseFloat(s, 64)
		return f
	}
	var f float64
	json.Unmarshal(raw, &f)
	return f
}

func parseIntFromRaw(raw json.RawMessage) int64 {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		i, _ := strconv.ParseInt(s, 10, 64)
		return i
	}
	var i int64
	json.Unmarshal(raw, &i)
	return i
}
