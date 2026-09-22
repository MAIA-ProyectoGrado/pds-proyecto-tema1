// Configuración del tablero dentro del contenedor.
// nginx sirve el tablero y hace proxy de /api/ al servicio de inferencia, así que
// el navegador usa el mismo origen de la página: funciona igual en localhost que
// en la IP pública de la EC2, sin reconstruir la imagen.
window.SCIF_API_URL = window.location.origin + "/api";
