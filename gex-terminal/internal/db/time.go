package db

import "time"

var etLoc *time.Location

func init() {
	var err error
	etLoc, err = time.LoadLocation("America/New_York")
	if err != nil {
		panic("cannot load America/New_York timezone: " + err.Error())
	}
}

// nowET returns the current time in Eastern Time.
func nowET() time.Time {
	return time.Now().In(etLoc)
}
