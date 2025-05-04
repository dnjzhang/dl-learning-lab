import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io

def plot_graph(image_data):
    img = PILImage.open(io.BytesIO(image_data))
    plt.imshow(img)
    plt.axis("off")
    plt.show()