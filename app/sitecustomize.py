"""Blocks all outbound network calls from Python processes by default."""
import os, socket, sys

if os.environ.get("ALLOW_NET", "0") not in ("1", "true", "True"):
    _real_create_connection = socket.create_connection
    _real_getaddrinfo = socket.getaddrinfo

    def _deny_conn(*args, **kwargs):
        raise OSError("Outbound network disabled by sitecustomize.py")

    def _deny_dns(*args, **kwargs):
        raise OSError("DNS lookups disabled by sitecustomize.py")

    socket.create_connection = _deny_conn
    socket.getaddrinfo = _deny_dns

    # Optional: shout once so you know the guard loaded
    sys.stderr.write("[netguard] Python network disabled (set ALLOW_NET=1 to enable)\n")