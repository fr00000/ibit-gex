package fetch

import (
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/html"

	"gex-terminal/internal/db"
	"gex-terminal/internal/model"
)

var (
	farsideMu      sync.Mutex
	farsideLastFetch time.Time
)

// FetchFarsideFlows fetches ETF fund flow data from Farside Investors.
// Returns the most recent flow data for IBIT, or nil if unavailable.
func FetchFarsideFlows() (*model.ETFFlows, error) {
	farsideMu.Lock()
	defer farsideMu.Unlock()

	if time.Since(farsideLastFetch).Seconds() < 3600 && !farsideLastFetch.IsZero() {
		// Return from DB instead of refetching
		return getLatestFlowFromDB()
	}

	url := "https://farside.co.uk/bitcoin-etf-flow-all-data/"
	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; gex-terminal/1.0)")

	resp, err := client.Do(req)
	if err != nil {
		slog.Warn("farside fetch failed", "err", err)
		return getLatestFlowFromDB()
	}
	defer resp.Body.Close()

	doc, err := html.Parse(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("farside parse: %w", err)
	}

	// Find the table and extract data
	rows := extractTableRows(doc)
	if len(rows) == 0 {
		slog.Warn("farside: no table rows found")
		return getLatestFlowFromDB()
	}

	// Find IBIT and Total column indices from header
	if len(rows) < 2 {
		return nil, fmt.Errorf("farside: too few rows")
	}

	header := rows[0]
	ibitIdx := -1
	totalIdx := -1
	for i, cell := range header {
		cl := strings.ToUpper(strings.TrimSpace(cell))
		if cl == "IBIT" {
			ibitIdx = i
		}
		if cl == "TOTAL" {
			totalIdx = i
		}
	}

	if ibitIdx < 0 {
		return nil, fmt.Errorf("farside: IBIT column not found in header: %v", header)
	}

	// Delete old Yahoo-era rows
	db.DeleteOldFlows("IBIT")

	// Parse data rows
	dateRe := regexp.MustCompile(`^\d{1,2}\s+\w{3}\s+\d{4}$`)
	count := 0

	for _, row := range rows[1:] {
		if len(row) <= ibitIdx {
			continue
		}

		// First cell should be a date
		dateStr := strings.TrimSpace(row[0])
		if !dateRe.MatchString(dateStr) {
			continue
		}

		// Parse date: "28 Feb 2026"
		t, err := time.Parse("2 Jan 2006", dateStr)
		if err != nil {
			continue
		}
		isoDate := t.Format("2006-01-02")

		ibitFlow := parseFarsideValue(row[ibitIdx])
		var totalFlow float64
		if totalIdx >= 0 && totalIdx < len(row) {
			totalFlow = parseFarsideValue(row[totalIdx])
		}

		// Flows are in millions
		if err := db.SaveFlow(isoDate, "IBIT", ibitFlow*1e6, totalFlow*1e6); err != nil {
			slog.Warn("save flow error", "date", isoDate, "err", err)
		}
		count++
	}

	farsideLastFetch = time.Now()
	slog.Info("farside flows parsed", "rows", count)

	return getLatestFlowFromDB()
}

func getLatestFlowFromDB() (*model.ETFFlows, error) {
	flows, err := db.GetFlows("IBIT", 10)
	if err != nil || len(flows) == 0 {
		return nil, err
	}

	latest := flows[0]
	result := &model.ETFFlows{
		DailyFlowDollars: latest.Flow,
		TotalBTCETFFlow:  latest.TotalFlow,
	}

	// Direction
	if latest.Flow >= 0 {
		result.Direction = "inflow"
	} else {
		result.Direction = "outflow"
	}

	// Strength
	absFlow := math.Abs(latest.Flow)
	switch {
	case absFlow < 10e6:
		result.Strength = "negligible"
	case absFlow < 50e6:
		result.Strength = "minor"
	case absFlow < 200e6:
		result.Strength = "moderate"
	default:
		result.Strength = "strong"
	}

	// Streak
	streak := 0
	if len(flows) > 0 {
		dir := flows[0].Flow >= 0
		for _, f := range flows {
			if (f.Flow >= 0) == dir {
				if dir {
					streak++
				} else {
					streak--
				}
			} else {
				break
			}
		}
	}
	result.Streak = streak

	// 5-day average
	sum := 0.0
	n := 0
	for i, f := range flows {
		if i >= 5 {
			break
		}
		sum += f.Flow
		n++
	}
	if n > 0 {
		result.AvgFlow5D = sum / float64(n)
	}

	return result, nil
}

// parseFarsideValue parses Farside flow values like "1,113.7", "(157.6)", "-", "".
func parseFarsideValue(s string) float64 {
	s = strings.TrimSpace(s)
	if s == "" || s == "-" {
		return 0
	}

	negative := false
	if strings.HasPrefix(s, "(") && strings.HasSuffix(s, ")") {
		negative = true
		s = s[1 : len(s)-1]
	}

	s = strings.ReplaceAll(s, ",", "")
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}

	if negative {
		return -f
	}
	return f
}

// extractTableRows parses HTML and returns all table rows as string slices.
func extractTableRows(doc *html.Node) [][]string {
	var rows [][]string
	var findTable func(*html.Node)

	findTable = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "table" {
			// Check if this is the ETF table
			rows = parseTable(n)
			if len(rows) > 5 { // Likely the right table
				return
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			findTable(c)
			if len(rows) > 5 {
				return
			}
		}
	}

	findTable(doc)
	return rows
}

func parseTable(table *html.Node) [][]string {
	var rows [][]string
	var walk func(*html.Node)

	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && (n.Data == "tr") {
			var cells []string
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				if c.Type == html.ElementNode && (c.Data == "td" || c.Data == "th") {
					cells = append(cells, textContent(c))
				}
			}
			if len(cells) > 0 {
				rows = append(rows, cells)
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}

	walk(table)
	return rows
}

func textContent(n *html.Node) string {
	if n.Type == html.TextNode {
		return n.Data
	}
	var sb strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		sb.WriteString(textContent(c))
	}
	return sb.String()
}
