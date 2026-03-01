package server

import (
	"encoding/json"
	"math"
	"net/http"
	"sort"
	"strconv"

	"gex-terminal/internal/config"
	"gex-terminal/internal/db"
	"gex-terminal/internal/gex"
)

func (s *Server) handleSnapshotDates(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	if _, ok := config.TickerConfig[ticker]; !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	dates, err := db.GetCachedDates(ticker, 4)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	if dates == nil {
		dates = []string{}
	}
	writeJSON(w, 200, map[string]interface{}{"dates": dates})
}

func (s *Server) handleData(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	if _, ok := config.TickerConfig[ticker]; !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	dteStr := r.URL.Query().Get("dte")
	dte := 3
	if dteStr != "" {
		if v, err := strconv.Atoi(dteStr); err == nil && v > 0 && v <= 90 {
			dte = v
		}
	}

	dateParam := r.URL.Query().Get("date")

	// Historical mode: serve from cache
	if dateParam != "" {
		cacheDate, raw, err := db.GetLatestCache(ticker, dte, dateParam)
		if err != nil {
			writeError(w, 500, err.Error())
			return
		}
		if raw == nil {
			writeError(w, 404, "no data for that date")
			return
		}
		// Inject snapshot_date into the response
		var blob map[string]interface{}
		json.Unmarshal(raw, &blob)
		blob["snapshot_date"] = cacheDate
		writeJSON(w, 200, blob)
		return
	}

	// Live mode: fetch through cache layer (checks freshness, fetches if stale)
	minDTE := 0
	for _, dw := range config.DTEWindows {
		if dw.MaxDTE == dte {
			minDTE = dw.MinDTE
			break
		}
	}

	data, err := gex.FetchWithCache(ticker, dte, minDTE, false)
	if err != nil {
		// Fallback to raw cache if fetch fails
		_, raw, cacheErr := db.GetLatestCache(ticker, dte, "")
		if cacheErr == nil && raw != nil {
			writeRawJSON(w, 200, raw)
			return
		}
		writeError(w, 500, err.Error())
		return
	}
	if data != nil {
		writeJSON(w, 200, data)
		return
	}

	// Fallback to DTE=3
	if dte != 3 {
		data, err = gex.FetchWithCache(ticker, 3, 0, false)
		if err == nil && data != nil {
			data.Stale = true
			fb := strconv.Itoa(dte)
			data.FallbackFrom = &fb
			writeJSON(w, 200, data)
			return
		}
	}

	writeJSON(w, 200, map[string]string{"status": "no data yet"})
}

func (s *Server) handleOutlook(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	if _, ok := config.TickerConfig[ticker]; !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	dateParam := r.URL.Query().Get("date")

	var spot float64
	windows := make([]map[string]interface{}, 0, len(config.DTEWindows))

	for _, dw := range config.DTEWindows {
		_, raw, err := db.GetLatestCache(ticker, dw.MaxDTE, dateParam)
		if err != nil || raw == nil {
			continue
		}
		var blob map[string]interface{}
		json.Unmarshal(raw, &blob)

		if s, ok := blob["btc_spot"].(float64); ok && s > 0 {
			spot = s
		}

		win := map[string]interface{}{
			"label":   dw.WindowLabel(),
			"min_dte": dw.MinDTE,
			"max_dte": dw.MaxDTE,
		}

		// Extract levels from combined_levels_btc, falling back to levels
		if cl, ok := blob["combined_levels_btc"].(map[string]interface{}); ok {
			win["call_wall"] = cl["call_wall"]
			win["put_wall"] = cl["put_wall"]
			win["gamma_flip"] = cl["gamma_flip"]
			win["regime"] = cl["regime"]
			win["net_gex"] = cl["net_gex_total"]
		} else if l, ok := blob["levels"].(map[string]interface{}); ok {
			win["call_wall"] = l["call_wall"]
			win["put_wall"] = l["put_wall"]
			win["gamma_flip"] = l["gamma_flip"]
			win["regime"] = l["regime"]
			win["net_gex"] = l["net_gex_total"]
		}

		// Expected move
		if em, ok := blob["expected_move"].(map[string]interface{}); ok {
			win["em_upper"] = em["upper_btc"]
			win["em_lower"] = em["lower_btc"]
		}

		// Venue data
		if dl, ok := blob["deribit_levels_btc"].(map[string]interface{}); ok {
			win["deribit_cw"] = dl["call_wall"]
			win["deribit_pw"] = dl["put_wall"]
		}

		// Delta flip
		if fps, ok := blob["delta_flip_points"].([]interface{}); ok && len(fps) > 0 {
			if fp, ok := fps[0].(map[string]interface{}); ok {
				if v, ok := fp["price_btc"].(float64); ok {
					win["delta_flip_btc"] = v
				}
			}
		}

		windows = append(windows, win)
	}

	writeJSON(w, 200, map[string]interface{}{
		"spot":    spot,
		"windows": windows,
	})
}

func (s *Server) handleRangeCone(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	if _, ok := config.TickerConfig[ticker]; !ok {
		writeError(w, 400, "unknown ticker")
		return
	}

	dateParam := r.URL.Query().Get("date")

	var spot float64
	windows := make([]map[string]interface{}, 0, len(config.DTEWindows))

	for _, dw := range config.DTEWindows {
		_, raw, err := db.GetLatestCache(ticker, dw.MaxDTE, dateParam)
		if err != nil || raw == nil {
			continue
		}
		var blob map[string]interface{}
		json.Unmarshal(raw, &blob)

		if s, ok := blob["btc_spot"].(float64); ok && s > 0 {
			spot = s
		}

		win := map[string]interface{}{
			"label":   dw.WindowLabel(),
			"min_dte": dw.MinDTE,
			"max_dte": dw.MaxDTE,
		}

		if em, ok := blob["expected_move"].(map[string]interface{}); ok {
			win["em_upper"] = em["upper_btc"]
			win["em_lower"] = em["lower_btc"]
		}

		// GEX zones: top 25 by magnitude from gex_chart
		if gc, ok := blob["gex_chart"].([]interface{}); ok {
			zones := make([]map[string]interface{}, 0, len(gc))
			for _, g := range gc {
				if gm, ok := g.(map[string]interface{}); ok {
					zones = append(zones, map[string]interface{}{
						"btc": gm["btc"],
						"gex": gm["net_gex"],
					})
				}
			}
			// Sort by |gex| descending, take top 25
			sort.Slice(zones, func(i, j int) bool {
				gi, _ := zones[i]["gex"].(float64)
				gj, _ := zones[j]["gex"].(float64)
				return math.Abs(gi) > math.Abs(gj)
			})
			if len(zones) > 25 {
				zones = zones[:25]
			}
			// Re-sort by BTC price ascending
			sort.Slice(zones, func(i, j int) bool {
				bi, _ := zones[i]["btc"].(float64)
				bj, _ := zones[j]["btc"].(float64)
				return bi < bj
			})
			win["gex_zones"] = zones
		}

		if cl, ok := blob["combined_levels_btc"].(map[string]interface{}); ok {
			win["gamma_flip"] = cl["gamma_flip"]
			win["regime"] = cl["regime"]
		}

		windows = append(windows, win)
	}

	writeJSON(w, 200, map[string]interface{}{
		"spot":    spot,
		"windows": windows,
	})
}

func (s *Server) handleStructure(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	daysStr := r.URL.Query().Get("days")
	days := 30
	if daysStr != "" {
		if v, err := strconv.Atoi(daysStr); err == nil && v > 0 {
			days = v
		}
	}

	// Build structure from data_cache history
	result := make(map[string]interface{})

	// Get all cached dates for this ticker
	dates, err := db.GetCachedDates(ticker, 1)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}

	// Limit to requested days
	if len(dates) > days {
		dates = dates[:days]
	}

	for _, d := range dates {
		dayData := map[string]interface{}{
			"windows": map[string]interface{}{},
		}
		for _, dw := range config.DTEWindows {
			_, raw, err := db.GetLatestCache(ticker, dw.MaxDTE, d)
			if err != nil || raw == nil {
				continue
			}
			var blob map[string]interface{}
			json.Unmarshal(raw, &blob)

			if s, ok := blob["btc_spot"].(float64); ok {
				dayData["spot"] = s
			}

			winKey := dw.WindowLabel()
			winKey = winKey[:len(winKey)-1] // Remove 'd' suffix for "0-3" format
			winData := map[string]interface{}{}

			if cl, ok := blob["combined_levels_btc"].(map[string]interface{}); ok {
				winData["call_wall"] = cl["call_wall"]
				winData["put_wall"] = cl["put_wall"]
				winData["gamma_flip"] = cl["gamma_flip"]
				winData["regime"] = cl["regime"]
			}
			if dl, ok := blob["deribit_levels_btc"].(map[string]interface{}); ok {
				winData["deribit_cw"] = dl["call_wall"]
				winData["deribit_pw"] = dl["put_wall"]
			}

			winsMap := dayData["windows"].(map[string]interface{})
			winsMap[winKey] = winData
		}
		result[d] = dayData
	}

	writeJSON(w, 200, result)
}

func (s *Server) handleStructureHeatmap(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = "IBIT"
	}
	days := 30
	if v, err := strconv.Atoi(r.URL.Query().Get("days")); err == nil && v > 0 {
		days = v
	}
	bucketSize := 1000
	if v, err := strconv.Atoi(r.URL.Query().Get("bucket")); err == nil && v > 0 {
		bucketSize = v
	}

	rows, err := db.GetStrikeHistoryForHeatmap(ticker, days)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}

	heatmap := map[string]map[int]map[string]float64{}
	allBuckets := map[int]bool{}

	for _, row := range rows {
		if row.Spot == 0 || row.BTCPrice == 0 {
			continue
		}
		bps := row.BTCPrice / row.Spot
		strikeBTC := row.Strike * bps
		bucket := int(math.Round(strikeBTC/float64(bucketSize))) * bucketSize
		allBuckets[bucket] = true

		if _, ok := heatmap[row.Date]; !ok {
			heatmap[row.Date] = map[int]map[string]float64{}
		}
		if _, ok := heatmap[row.Date][bucket]; !ok {
			heatmap[row.Date][bucket] = map[string]float64{
				"total_oi": 0, "net_gex": 0, "call_oi": 0, "put_oi": 0,
			}
		}
		heatmap[row.Date][bucket]["total_oi"] += float64(row.TotalOI)
		heatmap[row.Date][bucket]["net_gex"] += row.NetGEX
		heatmap[row.Date][bucket]["call_oi"] += float64(row.CallOI)
		heatmap[row.Date][bucket]["put_oi"] += float64(row.PutOI)
	}

	dates := make([]string, 0, len(heatmap))
	for d := range heatmap {
		dates = append(dates, d)
	}
	sort.Strings(dates)

	buckets := make([]int, 0, len(allBuckets))
	for b := range allBuckets {
		buckets = append(buckets, b)
	}
	sort.Ints(buckets)

	data := map[string][]map[string]interface{}{}
	for _, date := range dates {
		var cells []map[string]interface{}
		for _, bucket := range buckets {
			vals := heatmap[date][bucket]
			if vals != nil && (vals["total_oi"] > 0 || vals["net_gex"] != 0) {
				cells = append(cells, map[string]interface{}{
					"price":    bucket,
					"total_oi": vals["total_oi"],
					"net_gex":  math.Round(vals["net_gex"]*100) / 100,
					"call_oi":  vals["call_oi"],
					"put_oi":   vals["put_oi"],
				})
			}
		}
		data[date] = cells
	}

	writeJSON(w, 200, map[string]interface{}{
		"dates":       dates,
		"buckets":     buckets,
		"bucket_size": bucketSize,
		"data":        data,
	})
}
