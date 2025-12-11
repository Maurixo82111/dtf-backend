from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import numpy as np
from PIL import Image, ImageOps

# --- LÓGICA DE PROCESAMIENTO ---

def generar_semitono(image, lpi=45, angle=45):
    """
    Genera un semitono y lo devuelve en modo 'L' (Grises) para evitar errores de mezcla.
    """
    # Forzar entrada a escala de grises
    img = image.convert('L')
    
    # 1. Rotar
    img_rotated = img.rotate(angle, expand=1)
    
    # 2. Contraste
    img_rotated = ImageOps.autocontrast(img_rotated, cutoff=10)
    
    # 3. Convertir a Trama (Blanco y Negro puro - 1 bit)
    halftone_rotated = img_rotated.convert('1', dither=Image.Dither.FLOYDSTEINBERG)
    
    # 4. Rotar de regreso
    halftone_final = halftone_rotated.rotate(-angle, expand=1)
    
    # 5. Recortar
    w_orig, h_orig = image.size
    w_new, h_new = halftone_final.size
    left = (w_new - w_orig) / 2
    top = (h_new - h_orig) / 2
    
    halftone_final = halftone_final.crop((left, top, left + w_orig, top + h_orig))
    
    # CRÍTICO: Convertir a 'L' para que coincida con los canales CMYK y arreglar 'mode mismatch'
    return halftone_final.convert('L')

def procesar_dtf_pro(image_bytes, lpi):
    # Abrir y convertir a CMYK
    original = Image.open(io.BytesIO(image_bytes)).convert('CMYK')
    c, m, y, k = original.split()
    
    # Aplicar semitonos con ángulos correctos
    c_halftone = generar_semitono(c, lpi, angle=15)
    m_halftone = generar_semitono(m, lpi, angle=75)
    y_halftone = generar_semitono(y, lpi, angle=90)
    k_halftone = generar_semitono(k, lpi, angle=45)
    
    # Mezclar canales (Ahora todos son modo 'L', no fallará)
    final_image = Image.merge('CMYK', (c_halftone, m_halftone, y_halftone, k_halftone))
    
    return final_image.convert('RGB')

# --- MANEJADOR DEL SERVIDOR ---

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write("Servidor DTF V3 (Fix Mode Mismatch) - Online".encode('utf-8'))

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self.send_response(400)
                self.end_headers()
                return

            post_data = self.rfile.read(length)
            data = json.loads(post_data.decode('utf-8'))
            
            image_data = data.get('image')
            lpi = int(data.get('lpi', 45))
            
            if not image_data:
                raise ValueError("Falta la imagen")

            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Procesar
            imagen_procesada = procesar_dtf_pro(image_bytes, lpi)
            
            # Guardar y responder
            buffered = io.BytesIO()
            imagen_procesada.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            response = {
                "status": "success",
                "image": f"data:image/png;base64,{img_str}"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

        except Exception as e:
            print(f"Error: {str(e)}") # Log en consola de Vercel
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
