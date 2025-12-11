from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import numpy as np
from PIL import Image, ImageOps, ImageChops

# --- LÓGICA DE PROCESAMIENTO ---

def generar_semitono(image, lpi=45, angle=45):
    """
    Genera un semitono real rotado matemáticamente.
    """
    img = image.convert('L')  # Escala de grises
    
    # 1. Rotar la imagen según el ángulo deseado (para evitar moiré)
    img_rotated = img.rotate(angle, expand=1)
    
    # 2. Calcular el tamaño de la trama
    width, height = img_rotated.size
    
    # Truco "Pro": Aumentar contraste antes de tramar para definir puntos duros
    img_rotated = ImageOps.autocontrast(img_rotated, cutoff=10)
    
    # Convertir a 1 bit (Blanco y Negro puro)
    # Dither.FLOYDSTEINBERG crea un efecto más suave y orgánico
    halftone_rotated = img_rotated.convert('1', dither=Image.Dither.FLOYDSTEINBERG)
    
    # 3. Rotar de regreso a la posición original
    halftone_final = halftone_rotated.rotate(-angle, expand=1)
    
    # Recortar al tamaño original (al rotar crece el lienzo)
    w_orig, h_orig = image.size
    w_new, h_new = halftone_final.size
    left = (w_new - w_orig) / 2
    top = (h_new - h_orig) / 2
    
    halftone_final = halftone_final.crop((left, top, left + w_orig, top + h_orig))
    
    return halftone_final

def procesar_dtf_pro(image_bytes, lpi):
    """
    Simula separación de canales CMYK y aplica semitonos a cada uno.
    """
    # Abrir imagen desde bytes
    original = Image.open(io.BytesIO(image_bytes)).convert('CMYK')
    c, m, y, k = original.split()
    
    # Ángulos estándar de imprenta para evitar Moiré
    # C: 15, M: 75, Y: 90, K: 45
    c_halftone = generar_semitono(c, lpi, angle=15)
    m_halftone = generar_semitono(m, lpi, angle=75)
    y_halftone = generar_semitono(y, lpi, angle=90)
    k_halftone = generar_semitono(k, lpi, angle=45)
    
    # Recombinar canales
    final_image = Image.merge('CMYK', (c_halftone, m_halftone, y_halftone, k_halftone))
    
    return final_image.convert('RGB') # Convertir a RGB para que el navegador lo entienda

# --- MANEJADOR DEL SERVIDOR (VERCEL) ---

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    # NUEVO: Esto arregla el error 501 en el navegador
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write("¡El servidor DTF está en línea y funcionando! Usa esta URL en Shopify.".encode('utf-8'))

    def do_POST(self):
        try:
            # 1. Leer el cuerpo de la petición (JSON)
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # 2. Extraer imagen Base64 y parámetros
            image_data = data.get('image') # String base64
            lpi = int(data.get('lpi', 45)) # Lineatura
            
            if not image_data:
                raise ValueError("No se envió imagen")

            # Limpiar cabecera base64 si existe
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            
            # Decodificar
            image_bytes = base64.b64decode(image_data)
            
            # 3. PROCESAR IMAGEN
            imagen_procesada = procesar_dtf_pro(image_bytes, lpi)
            
            # 4. Convertir resultado a Base64 para devolver
            buffered = io.BytesIO()
            imagen_procesada.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 5. Responder al Frontend
            response = {
                "status": "success",
                "message": "Procesamiento completado",
                "image": f"data:image/png;base64,{img_str}"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
