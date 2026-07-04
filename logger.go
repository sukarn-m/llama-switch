package main

import (
	"fmt"
	"io"
	"os"
	"sync"
)

// CondLogger is a logger that writes to stdout. It is goroutine-safe.
type CondLogger struct {
	out io.Writer
	mu  sync.Mutex
}

func NewLogger() *CondLogger {
	return &CondLogger{out: os.Stdout}
}

func (l *CondLogger) Printf(format string, args ...any) {
	l.mu.Lock()
	defer l.mu.Unlock()
	fmt.Fprintf(l.out, format+"\n", args...)
}
