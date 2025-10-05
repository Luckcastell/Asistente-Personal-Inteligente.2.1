import customtkinter as ctk
import requests
import os

# --- Configuración y Constantes ---
# La URL de tu backend de FastAPI, donde debe estar corriendo el main.py
URL_BASE_API = "http://127.0.0.1:8000"

# Establece el tema de la aplicación para una estética moderna
ctk.set_appearance_mode("System") 
ctk.set_default_color_theme("blue")

class SurielApp(ctk.CTk):
    """
    Clase principal que define la ventana de la aplicación Suriel (GUI de escritorio).
    """
    def __init__(self):
        super().__init__()

        # --- Configuración de la Ventana ---
        self.title("🤖 Suriel | Asistente Personal RAG")
        self.geometry("600x650")

        # Configuración del Grid para hacerla responsive
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) # La caja de chat tomará la mayor parte del espacio

        # --- 1. Área de Carga de Conocimiento ---
        self.frame_subida = ctk.CTkFrame(self)
        self.frame_subida.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        
        self.label_titulo = ctk.CTkLabel(self.frame_subida, text="Cargar Base de Conocimiento (PDF)", font=ctk.CTkFont(size=14, weight="bold"))
        self.label_titulo.pack(pady=(10, 5))

        self.boton_subir = ctk.CTkButton(self.frame_subida, text="Seleccionar y Subir PDF", command=self.subir_pdf)
        self.boton_subir.pack(pady=5, padx=20, fill="x")

        self.estado_subida_label = ctk.CTkLabel(self.frame_subida, text="")
        self.estado_subida_label.pack(pady=(0, 10))

        # --- 2. Área de Chat ---
        # El CTkTextbox funciona como la caja de historial
        self.caja_chat = ctk.CTkTextbox(self, state="disabled")
        self.caja_chat.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        
        # Configurar etiquetas de estilo (opcional, pero ayuda a la legibilidad)
        self.caja_chat.tag_config("suriel", foreground="#2ECC71") # Verde
        self.caja_chat.tag_config("usuario", foreground="#3498DB") # Azul
        self.caja_chat.tag_config("error", foreground="#E74C3C") # Rojo

        # Mensaje inicial
        self.mostrar_mensaje_suriel("¡Hola! Soy Suriel, tu asistente personal. Cargá tus documentos PDF para que pueda empezar a responder.")


        # --- 3. Área de Entrada de Usuario ---
        self.entrada_usuario = ctk.CTkEntry(self, placeholder_text="Hacé una pregunta sobre tus documentos...", width=400)
        self.entrada_usuario.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        # Atar la tecla Enter para enviar el mensaje
        self.entrada_usuario.bind("<Return>", lambda event=None: self.enviar_mensaje())
        
        self.boton_enviar = ctk.CTkButton(self, text="Enviar", command=self.enviar_mensaje)
        # Posicionamos el botón de enviar a la derecha del campo de texto
        self.boton_enviar.grid(row=3, column=0, padx=(420, 20), pady=(0, 20), sticky="e") 
        self.entrada_usuario.focus()

    # --- Funcionalidades ---

    def mostrar_mensaje_suriel(self, texto: str, es_error: bool = False):
        """Muestra un mensaje de Suriel en la caja de chat con coloración."""
        tag = "error" if es_error else "suriel"
        
        self.caja_chat.configure(state="normal")
        self.caja_chat.insert(ctk.END, f"\nSuriel: {texto}\n", tag)
        self.caja_chat.configure(state="disabled")
        self.caja_chat.see(ctk.END) # Scroll automático al final

    def mostrar_mensaje_usuario(self, texto: str):
        """Muestra un mensaje del Usuario en la caja de chat con coloración."""
        self.caja_chat.configure(state="normal")
        self.caja_chat.insert(ctk.END, f"\nUsuario: {texto}\n", "usuario")
        self.caja_chat.configure(state="disabled")
        self.caja_chat.see(ctk.END)

    def subir_pdf(self):
        """Maneja la selección del archivo y llama al endpoint /upload del backend."""
        
        # Abre el diálogo de selección de archivo
        ruta_archivo = ctk.filedialog.askopenfilename(
            title="Seleccionar archivo PDF para indexar",
            filetypes=[("Archivos PDF", "*.pdf")]
        )

        if not ruta_archivo:
            self.estado_subida_label.configure(text="⚠️ Subida cancelada.", text_color="yellow")
            return

        nombre_archivo = os.path.basename(ruta_archivo)
        self.estado_subida_label.configure(text=f"Subiendo e indexando '{nombre_archivo}'... ⏳", text_color="orange")
        self.boton_subir.configure(state="disabled")

        try:
            # Petición con 'requests' para enviar el archivo
            with open(ruta_archivo, "rb") as f:
                archivos = {"file": (nombre_archivo, f, "application/pdf")}
                
                respuesta = requests.post(f"{URL_BASE_API}/upload", files=archivos, timeout=120)

            if respuesta.status_code == 200:
                self.estado_subida_label.configure(text=f"✅ '{nombre_archivo}' indexado con éxito!", text_color="green")
            else:
                datos_error = respuesta.json().get("detail", "Error desconocido en el servidor.")
                self.estado_subida_label.configure(text=f"❌ Error al indexar: {datos_error}", text_color="red")

        except requests.exceptions.ConnectionError:
            self.estado_subida_label.configure(text="❌ Error de conexión. ¿Está corriendo el servidor de FastAPI?", text_color="red")
        except Exception as e:
            self.estado_subida_label.configure(text=f"❌ Ocurrió un error inesperado: {e}", text_color="red")
        finally:
            self.boton_subir.configure(state="normal")

    def enviar_mensaje(self):
        """Maneja el envío del mensaje y llama al endpoint /chat del backend."""
        mensaje_usuario = self.entrada_usuario.get()
        if not mensaje_usuario.strip():
            return
        
        self.mostrar_mensaje_usuario(mensaje_usuario)
        self.entrada_usuario.delete(0, ctk.END)
        
        # Bloquear controles para feedback visual
        self.entrada_usuario.configure(state="disabled")
        self.boton_enviar.configure(state="disabled")
        
        # Indicador de carga
        self.mostrar_mensaje_suriel("Pensando... 🧠 (cargando respuesta)")

        # La conexión y procesamiento se debe hacer en un hilo separado en una aplicación GUI real
        # para no bloquear la interfaz, pero para simplificar, lo dejamos síncrono por ahora.
        try:
            respuesta = requests.post(
                f"{URL_BASE_API}/chat", 
                json={"mensaje": mensaje_usuario}, 
                timeout=120
            )

            # Reemplazar el mensaje de "Pensando..."
            # Borramos las últimas 3 lineas (dos saltos de línea + el mensaje de Suriel)
            self.caja_chat.configure(state="normal")
            self.caja_chat.delete("end-3l", "end-1l") 
            self.caja_chat.configure(state="disabled")

            if respuesta.status_code == 200:
                respuesta_llm = respuesta.json().get("respuesta", "Suriel no pudo generar una respuesta.")
                self.mostrar_mensaje_suriel(respuesta_llm)
                
            else:
                datos_error = respuesta.json().get("detail", "Error del servidor.")
                self.mostrar_mensaje_suriel(f"❌ Error del servidor: {datos_error}", es_error=True)

        except requests.exceptions.ConnectionError:
            self.mostrar_mensaje_suriel("❌ Error de conexión. Asegurate de que el servidor (backend) esté corriendo.", es_error=True)
        except Exception as e:
            self.mostrar_mensaje_suriel(f"❌ Ocurrió un error inesperado: {e}", es_error=True)
        finally:
            # Restaurar controles
            self.entrada_usuario.configure(state="normal")
            self.boton_enviar.configure(state="normal")
            self.entrada_usuario.focus()


# --- Punto de Ejecución ---
if __name__ == "__main__":
    
    # ⚠️ Recordatorio importante para la ejecución
    print(f"===============================================================")
    print(f"Iniciando GUI. ASEGURATE que el backend de FastAPI esté corriendo en:")
    print(f"         {URL_BASE_API}")
    print(f"===============================================================")
    
    app = SurielApp()
    app.mainloop() # Inicia el bucle principal de la aplicación de escritorio