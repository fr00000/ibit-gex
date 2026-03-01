package fetch

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/cookiejar"
	"regexp"
	"strings"
	"sync"
	"time"
)

var (
	yahooMu     sync.Mutex
	yahooCrumb  string
	yahooClient *http.Client
	yahooCrumbExpiry time.Time
)

const (
	yahooUserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
	yahooBase      = "https://query2.finance.yahoo.com"
)

// YahooQuote holds basic price data from Yahoo Finance.
type YahooQuote struct {
	RegularMarketPrice float64
	Symbol             string
}

// YahooOptionChain holds the full options chain for one expiration.
type YahooOptionChain struct {
	ExpiryDate   time.Time
	ExpiryStr    string // "2006-01-02"
	Calls        []YahooOption
	Puts         []YahooOption
}

// YahooOption is a single option contract.
type YahooOption struct {
	Strike          float64
	LastPrice       float64
	Bid             float64
	Ask             float64
	Volume          int64
	OpenInterest    int64
	ImpliedVol      float64 // decimal (0.65)
	InTheMoney      bool
}

// ensureCrumb establishes the Yahoo Finance cookie/crumb session.
func ensureCrumb() error {
	yahooMu.Lock()
	defer yahooMu.Unlock()

	if yahooCrumb != "" && time.Now().Before(yahooCrumbExpiry) {
		return nil
	}

	jar, _ := cookiejar.New(nil)
	yahooClient = &http.Client{
		Timeout: 15 * time.Second,
		Jar:     jar,
	}

	// Step 1: Visit Yahoo Finance to get cookies
	req, _ := http.NewRequest("GET", "https://fc.yahoo.com/", nil)
	req.Header.Set("User-Agent", yahooUserAgent)
	resp, err := yahooClient.Do(req)
	if err != nil {
		return fmt.Errorf("yahoo cookie: %w", err)
	}
	resp.Body.Close()

	// Step 2: Get crumb
	req, _ = http.NewRequest("GET", "https://query2.finance.yahoo.com/v1/test/getcrumb", nil)
	req.Header.Set("User-Agent", yahooUserAgent)
	resp, err = yahooClient.Do(req)
	if err != nil {
		return fmt.Errorf("yahoo crumb: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("yahoo crumb read: %w", err)
	}

	crumb := strings.TrimSpace(string(body))
	if crumb == "" || len(crumb) > 50 {
		return fmt.Errorf("yahoo crumb invalid: %q", crumb)
	}

	yahooCrumb = crumb
	yahooCrumbExpiry = time.Now().Add(30 * time.Minute)
	slog.Info("yahoo crumb obtained", "crumb_len", len(crumb))
	return nil
}

// FetchYahooQuote gets current price for a ticker symbol.
func FetchYahooQuote(symbol string) (*YahooQuote, error) {
	if err := ensureCrumb(); err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/v10/finance/quoteSummary/%s?modules=price&crumb=%s",
		yahooBase, symbol, yahooCrumb)

	yahooMu.Lock()
	client := yahooClient
	yahooMu.Unlock()

	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("User-Agent", yahooUserAgent)

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("yahoo quote: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		QuoteSummary struct {
			Result []struct {
				Price struct {
					RegularMarketPrice struct {
						Raw float64 `json:"raw"`
					} `json:"regularMarketPrice"`
					Symbol string `json:"symbol"`
				} `json:"price"`
			} `json:"result"`
			Error *struct {
				Code string `json:"code"`
			} `json:"error"`
		} `json:"quoteSummary"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if result.QuoteSummary.Error != nil {
		// Crumb expired — reset and retry once
		yahooCrumb = ""
		return nil, fmt.Errorf("yahoo API error: %s", result.QuoteSummary.Error.Code)
	}

	if len(result.QuoteSummary.Result) == 0 {
		return nil, fmt.Errorf("yahoo: no result for %s", symbol)
	}

	price := result.QuoteSummary.Result[0].Price
	return &YahooQuote{
		RegularMarketPrice: price.RegularMarketPrice.Raw,
		Symbol:             price.Symbol,
	}, nil
}

// FetchYahooExpirations returns available option expiration dates for a ticker.
func FetchYahooExpirations(symbol string) ([]time.Time, error) {
	if err := ensureCrumb(); err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/v7/finance/options/%s?crumb=%s", yahooBase, symbol, yahooCrumb)

	yahooMu.Lock()
	client := yahooClient
	yahooMu.Unlock()

	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("User-Agent", yahooUserAgent)

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		OptionChain struct {
			Result []struct {
				ExpirationDates []int64 `json:"expirationDates"`
			} `json:"result"`
		} `json:"optionChain"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if len(result.OptionChain.Result) == 0 {
		return nil, fmt.Errorf("yahoo: no options for %s", symbol)
	}

	timestamps := result.OptionChain.Result[0].ExpirationDates
	dates := make([]time.Time, len(timestamps))
	for i, ts := range timestamps {
		dates[i] = time.Unix(ts, 0).UTC()
	}

	return dates, nil
}

// FetchYahooOptions fetches the full options chain for a specific expiration.
func FetchYahooOptions(symbol string, expiryTS int64) (*YahooOptionChain, error) {
	if err := ensureCrumb(); err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/v7/finance/options/%s?date=%d&crumb=%s",
		yahooBase, symbol, expiryTS, yahooCrumb)

	yahooMu.Lock()
	client := yahooClient
	yahooMu.Unlock()

	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("User-Agent", yahooUserAgent)

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		OptionChain struct {
			Result []struct {
				Options []struct {
					ExpirationDate int64 `json:"expirationDate"`
					Calls          []yahooOptionJSON `json:"calls"`
					Puts           []yahooOptionJSON `json:"puts"`
				} `json:"options"`
			} `json:"result"`
		} `json:"optionChain"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if len(result.OptionChain.Result) == 0 || len(result.OptionChain.Result[0].Options) == 0 {
		return nil, fmt.Errorf("yahoo: no options data for %s at %d", symbol, expiryTS)
	}

	opts := result.OptionChain.Result[0].Options[0]
	expiry := time.Unix(opts.ExpirationDate, 0).UTC()

	chain := &YahooOptionChain{
		ExpiryDate: expiry,
		ExpiryStr:  expiry.Format("2006-01-02"),
		Calls:      convertOptions(opts.Calls),
		Puts:       convertOptions(opts.Puts),
	}

	return chain, nil
}

// FetchRiskFreeRate fetches the 13-week T-bill rate from Yahoo Finance.
func FetchRiskFreeRate() (float64, error) {
	quote, err := FetchYahooQuote("^IRX")
	if err != nil {
		return 0, err
	}
	if quote.RegularMarketPrice <= 0 {
		return 0, fmt.Errorf("invalid IRX rate: %f", quote.RegularMarketPrice)
	}
	return quote.RegularMarketPrice / 100.0, nil
}

type yahooOptionJSON struct {
	Strike struct {
		Raw float64 `json:"raw"`
	} `json:"strike"`
	LastPrice struct {
		Raw float64 `json:"raw"`
	} `json:"lastPrice"`
	Bid struct {
		Raw float64 `json:"raw"`
	} `json:"bid"`
	Ask struct {
		Raw float64 `json:"raw"`
	} `json:"ask"`
	Volume struct {
		Raw int64 `json:"raw"`
	} `json:"volume"`
	OpenInterest struct {
		Raw int64 `json:"raw"`
	} `json:"openInterest"`
	ImpliedVolatility struct {
		Raw float64 `json:"raw"`
	} `json:"impliedVolatility"`
	InTheMoney bool `json:"inTheMoney"`
}

func convertOptions(raw []yahooOptionJSON) []YahooOption {
	result := make([]YahooOption, 0, len(raw))
	for _, r := range raw {
		result = append(result, YahooOption{
			Strike:       r.Strike.Raw,
			LastPrice:    r.LastPrice.Raw,
			Bid:          r.Bid.Raw,
			Ask:          r.Ask.Raw,
			Volume:       r.Volume.Raw,
			OpenInterest: r.OpenInterest.Raw,
			ImpliedVol:   r.ImpliedVolatility.Raw,
			InTheMoney:   r.InTheMoney,
		})
	}
	return result
}

// yahooV8 is a helper to try the v8 chart endpoint as fallback for price.
func FetchYahooV8Price(symbol string) (float64, error) {
	url := fmt.Sprintf("https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=1d&range=1d", symbol)

	client := &http.Client{Timeout: 10 * time.Second}
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("User-Agent", yahooUserAgent)

	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	// Quick regex for regularMarketPrice
	re := regexp.MustCompile(`"regularMarketPrice":(\d+\.?\d*)`)
	matches := re.FindSubmatch(body)
	if len(matches) < 2 {
		return 0, fmt.Errorf("v8 price not found for %s", symbol)
	}

	var price float64
	fmt.Sscanf(string(matches[1]), "%f", &price)
	return price, nil
}
