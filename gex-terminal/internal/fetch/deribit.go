package fetch

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"gex-terminal/internal/config"
	"gex-terminal/internal/model"
)

var (
	deribitCaches = map[string]*deribitCache{}
	deribitMu     sync.Mutex
)

type deribitCache struct {
	data    []deribitInstrument
	fetched time.Time
}

type deribitInstrument struct {
	InstrumentName  string  `json:"instrument_name"`
	OpenInterest    float64 `json:"open_interest"`
	MarkIV          float64 `json:"mark_iv"`
	UnderlyingPrice float64 `json:"underlying_price"`
}

// FetchDeribitOptions fetches options from Deribit public API.
// Returns options filtered to [minDTE, maxDTE].
func FetchDeribitOptions(minDTE, maxDTE int, currency string) ([]model.DeribitOption, error) {
	deribitMu.Lock()
	defer deribitMu.Unlock()

	cache, ok := deribitCaches[currency]
	if ok && time.Since(cache.fetched).Seconds() < float64(config.DeribitCacheSeconds) {
		return filterDeribit(cache.data, minDTE, maxDTE)
	}

	url := fmt.Sprintf("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?kind=option&currency=%s", currency)

	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "ibit-gex/1.0")

	resp, err := client.Do(req)
	if err != nil {
		// Return cached data if available
		if cache != nil {
			slog.Warn("deribit fetch failed, using cache", "err", err)
			return filterDeribit(cache.data, minDTE, maxDTE)
		}
		return nil, fmt.Errorf("deribit fetch: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Result []deribitInstrument `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		if cache != nil {
			return filterDeribit(cache.data, minDTE, maxDTE)
		}
		return nil, fmt.Errorf("deribit decode: %w", err)
	}

	deribitCaches[currency] = &deribitCache{
		data:    result.Result,
		fetched: time.Now(),
	}

	slog.Info("deribit fetched", "currency", currency, "instruments", len(result.Result))
	return filterDeribit(result.Result, minDTE, maxDTE)
}

// ResetDeribitCache clears the cache for a currency.
func ResetDeribitCache(currency string) {
	deribitMu.Lock()
	defer deribitMu.Unlock()
	delete(deribitCaches, currency)
}

// DeribitFreshness returns the age of the Deribit cache for a currency.
func DeribitFreshness(currency string) (ageMinutes float64, ok bool) {
	deribitMu.Lock()
	defer deribitMu.Unlock()
	cache, exists := deribitCaches[currency]
	if !exists {
		return 0, false
	}
	return time.Since(cache.fetched).Minutes(), true
}

func filterDeribit(instruments []deribitInstrument, minDTE, maxDTE int) ([]model.DeribitOption, error) {
	now := time.Now().UTC()
	var result []model.DeribitOption

	for _, inst := range instruments {
		if inst.OpenInterest <= 10 || inst.MarkIV <= 0 || inst.UnderlyingPrice <= 0 {
			continue
		}

		// Parse instrument name: BTC-27MAR26-100000-C
		parts := strings.Split(inst.InstrumentName, "-")
		if len(parts) != 4 {
			continue
		}

		dateStr := parts[1]
		strikeStr := parts[2]
		optTypeStr := parts[3]

		// Parse expiry date
		expiry, err := time.Parse("2Jan06", dateStr)
		if err != nil {
			continue
		}

		// Calculate DTE
		dte := expiry.Sub(now).Hours() / 24.0
		if dte < float64(minDTE) || dte > float64(maxDTE) {
			continue
		}

		// Parse strike
		var strike float64
		fmt.Sscanf(strikeStr, "%f", &strike)
		if strike <= 0 {
			continue
		}

		// Parse option type
		var optType string
		switch optTypeStr {
		case "C":
			optType = "call"
		case "P":
			optType = "put"
		default:
			continue
		}

		result = append(result, model.DeribitOption{
			Strike:          strike,
			ExpiryStr:       dateStr,
			ExpiryDate:      expiry.Unix(),
			DTE:             dte,
			OptionType:      optType,
			OI:              inst.OpenInterest,
			IV:              inst.MarkIV, // percentage (65.0)
			UnderlyingPrice: inst.UnderlyingPrice,
		})
	}

	return result, nil
}
