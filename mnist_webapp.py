import gradio as gr
from PIL import Image
import requests
import io

def recognize_digit(image):
    # Convertir en image PIL (si nécessaire)
    image = Image.fromarray(image.astype('uint8'))

    # Convertir l'image en objet binaire pour l'envoi
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    img_binary.seek(0)  # Repositionner le pointeur au début du fichier

    # Envoyer une requête POST avec l'image sous forme de fichier
    files = {'file': ('image.png', img_binary, 'image/png')}
    response = requests.post("http://127.0.0.1:5000/predict", files=files)

    # Vérifier la réponse
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        return "Error: Could not get prediction"

# Interface Gradio
if __name__ == '__main__':
    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True)
