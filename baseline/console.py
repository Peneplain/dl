"""A single-command console: input arriving while execution is busy is discarded."""

from contextlib import contextmanager
import os
import select
import termios


class ManualConsole:
    def __init__(self, stream):
        self.fd = stream.fileno()
        self.tty = os.isatty(self.fd)
        self.pending = b""
        self.eof = False
        self.discarded = 0

    def read_line(self, idle=lambda: None):
        while True:
            if b"\n" in self.pending:
                line, self.pending = self.pending.split(b"\n", 1)
                return line.decode("utf-8")
            if self.eof:
                if self.pending:
                    line, self.pending = self.pending, b""
                    return line.decode("utf-8")
                return None
            idle()
            ready, _, _ = select.select([self.fd], [], [], .05)
            if ready:
                chunk = os.read(self.fd, 4096)
                if chunk:
                    self.pending += chunk
                else:
                    self.eof = True

    def discard(self):
        self.discarded += len(self.pending)
        self.pending = b""
        while not self.eof and select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 4096)
            if not chunk:
                self.eof = True
                break
            self.discarded += len(chunk)

    @contextmanager
    def busy(self):
        original = None
        self.discarded = 0
        if self.tty:
            original = termios.tcgetattr(self.fd)
            flags = termios.tcgetattr(self.fd)
            flags[3] &= ~(termios.ECHO | termios.ICANON)
            flags[6][termios.VMIN] = 0
            flags[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, flags)
        self.discard()
        try:
            yield self.discard
        finally:
            self.discard()
            if original is not None:
                termios.tcflush(self.fd, termios.TCIFLUSH)
                termios.tcsetattr(self.fd, termios.TCSANOW, original)
