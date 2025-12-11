from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import numpy as np
from PIL import Image, ImageOps, ImageChops

# --- MOTORES DE TRAMADO ---

def generar_patron_matematico(image, shape='round', lpi=45, angle=45):
    """
    Genera tramas geométricas (Líneas o Círculos) usando matemáticas de matrices.
    Es más preciso para DTF que la difusión de error.
    """
    # 1. Preparar imagen
    img_gray = image.convert('L')
    
    # 2. Rotar para el ángulo (Evitar Moiré)
    img_rotated = img_gray.rotate(angle, resample=Image.BICUBIC, expand=1)
    w, h = img_rotated.size
    
    # 3. Crear malla de coordenadas (Grid)
    # Factor de escala para simular LPI (Puntos por pulgada)
    # Asumimos 300 DPI base para el cálculo
    scale = (300 / lpi) * 0.5 
    
    y, x = np.ogrid[:h, :w]
    
    # 4. Generar el patrón matemático (Threshold Map)
    if shape == 'line':
        # Patrón de Líneas: Seno de la posición Y
        pattern = np.sin(y / scale * np.pi)
    else: # round / ellipse
        # Patrón de Círculos: Seno de X * Seno de Y
        pattern = np.sin(x / scale * np.pi) * np.sin(y / scale * np.pi)
    
    # Normalizar patrón de -1..1 a 0..255
    pattern = (pattern + 1) * 127.5
    threshold_map = pattern.astype(np.uint8)
    
    # 5. Aplicar el umbral (Thresholding)
    img_array = np.array(img_rotated)
    # Si el pixel es más oscuro que el patrón -> Negro (Tinta), si no -> Blanco
    # En DTF: Blanco es transparente, Negro es tinta (en máscara)
    halftone_array = (img_array > threshold_map) * 255
    halftone_img = Image.fromarray(halftone_array.astype(np.uint8)).convert('1')
    
    # 6. Rotar de regreso
    halftone_final = halftone_img.rotate(-angle, expand=1)
    
    # 7. Recortar (Crop)
    w_orig, h_orig = image.size
    w_new, h_new = halftone_final.size
    left = (w_new - w_orig) / 2
    top = (h_new - h_orig) / 2
    
    return halftone_final.crop((left, top, left + w_orig, top + h_orig)).convert('L')

def generar_difusion(image, lpi=45):
    """ Trama clásica Floyd-Steinberg (Puntos orgánicos dispersos) """
    img = image.convert('L')
    img = ImageOps.autocontrast(img, cutoff=5)
    return img.convert('1', dither=Image.Dither.FLOYDSTEINBERG).convert('L')

# --- LÓGICA DE PROCESAMIENTO PRINCIPAL ---

def procesar_dtf_avanzado(image_bytes, config):
    # 1. Cargar imagen
    img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
    
    # Configuración
    bg_color_hex = config.get('bgColor', '#000000') # Color a eliminar
    tolerance = int(config.get('tolerance', 50))    # Rango de "cercanía"
    lpi = int(config.get('lpi', 45))
    angle = int(config.get('angle', 22))
    shape = config.get('shape', 'round') # round, line, diffusion
    
    # 2. ELIMINACIÓN DE FONDO INTELIGENTE (Knockout)
    if bg_color_hex:
        # Convertir Hex a RGB
        bg_color_hex = bg_color_hex.lstrip('#')
        target_rgb = tuple(int(bg_color_hex[i:i+2], 16) for i in (0, 2, 4))
        
        # Calcular distancia de colores usando NumPy (Muy rápido)
        arr = np.array(img)
        r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
        
        # Distancia Euclidiana del color actual al color de fondo
        diff = np.sqrt(
            (r - target_rgb[0])**2 + 
            (g - target_rgb[1])**2 + 
            (b - target_rgb[2])**2
        )
        
        # --- LÓGICA DE MÁSCARA DTF ---
        # Si la distancia es 0 (Mismo color) -> Transparente (Alpha 0)
        # Si la distancia es < tolerance -> Semitransparente (Rasterizar)
        # Si la distancia es > tolerance -> Sólido (Alpha 255)
        
        # Crear nueva capa Alpha basada en la "diferencia"
        # Escalamos la diferencia para que sea el nuevo Alpha
        # Pixeles lejanos al fondo (d > tol) serán opacos.
        # Pixeles cercanos (d < tol) serán tenues.
        new_alpha = np.clip(diff * (255.0 / tolerance), 0, 255).astype(np.uint8)
        
        # Combinar con el Alpha original de la imagen
        final_alpha = np.minimum(a, new_alpha)
        
        # Actualizar el canal Alpha de la imagen
        arr[:,:,3] = final_alpha
        img = Image.fromarray(arr)

    # 3. APLICAR TRAMA SOLO A LAS TRANSPARENCIAS
    # Separamos canales
    r, g, b, a = img.split()
    
    # Generamos la trama basada en el canal Alpha (la transparencia dicta el punto)
    if shape == 'diffusion':
        halftone_mask = generar_difusion(a, lpi)
    else:
        halftone_mask = generar_patron_matematico(a, shape, lpi, angle)
    
    # 4. COMPOSICIÓN FINAL
    # Usamos la máscara de semitono como el nuevo canal Alpha definitivo.
    # Esto hace "agujeros" reales en la imagen.
    final_img = Image.merge('RGBA', (r, g, b, halftone_mask))
    
    return final_img

# --- SERVER HANDLER ---

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write("DTF Engine V4 (Knockout + Shapes) Online".encode())

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(length)
            data = json.loads(post_data.decode('utf-8'))
            
            image_data = data.get('image')
            if not image_data: raise ValueError("No image")
            if "base64," in image_data: image_data = image_data.split("base64,")[1]
            
            # Procesar
            result_img = procesar_dtf_avanzado(base64.b64decode(image_data), data)
            
            # Guardar
            buffered = io.BytesIO()
            result_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "image": f"data:image/png;base64,{img_str}"}).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
