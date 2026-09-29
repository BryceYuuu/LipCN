"""Live transcript as Server-Sent Events on http://127.0.0.1:8765/events.

    curl -N http://127.0.0.1:8765/events

Events (JSON in `data:`):
  start    {"id"}                          you pressed the key
  partial  {"id", "text"}                  live lip-read guess, updates ~2x a second
  final    {"id", "text", "raw", "latency"} cleaned sentence after you let go
  cancel   {"id"}
Local only (bound to 127.0.0.1).
"""
from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = b"""<!doctype html><meta charset=utf-8><title>Lipflow live</title>
<style>body{font:20px system-ui;background:#111;color:#eee;margin:40px}#p{color:#f5708a}.f{margin:6px 0}</style>
<h3>Lipflow live</h3><div id=log></div><div id=p></div><script>
const es=new EventSource('/events'),p=document.getElementById('p'),log=document.getElementById('log');
es.addEventListener('partial',e=>p.textContent=JSON.parse(e.data).text+' \\u2026');
es.addEventListener('final',e=>{p.textContent='';const d=document.createElement('div');d.className='f';
d.textContent=JSON.parse(e.data).text;log.appendChild(d)});
es.addEventListener('cancel',()=>p.textContent='');</script>"""


class Broadcaster:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def publish(self, event: str, **data):
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
        with self._lock:
            for q in self._subs:
                q.put(msg)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.remove(q)


def serve(bus: Broadcaster, port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/events":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            q = bus.subscribe()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                    except queue.Empty:
                        msg = b": ping\n\n"
                    self.wfile.write(msg)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                bus.unsubscribe(q)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True, name="lipflow-sse").start()
    return httpd
