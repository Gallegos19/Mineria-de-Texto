# Configuración de Gunicorn para EC2
import multiprocessing

# Server socket
bind = "0.0.0.0:8080"
backlog = 2048

# Worker processes
workers = 1  # Para aplicaciones con ML, mejor 1 worker con más memoria
worker_class = "gthread"
threads = 2
worker_connections = 1000
max_requests = 100
max_requests_jitter = 10

# Timeouts
timeout = 300  # 5 minutos para carga de BERT
keepalive = 2
graceful_timeout = 30

# Memory management
preload_app = True
max_worker_memory = 2048000  # 2GB por worker

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "text_mining_api"

# Server mechanics
daemon = False
pidfile = "/var/run/gunicorn/text_mining_api.pid"
user = "ubuntu"
group = "ubuntu"
tmp_upload_dir = None

# SSL (si necesitas HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"