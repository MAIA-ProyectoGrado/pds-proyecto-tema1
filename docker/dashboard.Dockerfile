# Tablero: archivos estáticos servidos por nginx, que además hace de proxy hacia el
# servicio de inferencia para que el navegador use un único origen.
FROM nginx:1.27-alpine

COPY src/dashboard/ /usr/share/nginx/html/
# Sustituye la configuración de desarrollo (apunta a 127.0.0.1:8000) por la del contenedor
COPY docker/config.docker.js /usr/share/nginx/html/config.js
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# serve.py es solo para desarrollo local sin Docker
RUN rm -f /usr/share/nginx/html/serve.py

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=4s --start-period=10s --retries=3 \
  CMD wget -q --spider http://127.0.0.1:8080/ || exit 1
