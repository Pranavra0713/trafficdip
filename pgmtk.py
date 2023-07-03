import cv2
import tkinter as tk
from PIL import ImageTk, Image

cascade_src = 'cars.xml'

# List of image file paths
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']

car_cascade = cv2.CascadeClassifier(cascade_src)

max_cars = 0
max_cars_image = ''

processed_images = set()

# Create the main Tkinter window
window = tk.Tk()
window.title("Car Detection")

# Create a frame to hold the image and label widgets
frame = tk.Frame(window)

for i, image_path in enumerate(image_paths):
    image_path = f'dataset/{image_path}'
    if image_path in processed_images:
        continue

    img = cv2.imread(image_path)

    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30), maxSize=(200, 200))

        # Count the number of cars detected
        num_cars = len(cars)

        if num_cars > max_cars:
            max_cars = num_cars
            max_cars_image = image_path

        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Convert the image from OpenCV format to PIL format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # Resize the image to fit in the display
        max_width = 400
        max_height = 300
        pil_image.thumbnail((max_width, max_height), Image.ANTIALIAS)

        # Create a Tkinter PhotoImage from the PIL image
        photo_image = ImageTk.PhotoImage(pil_image)

        # Create a label to display the image
        label = tk.Label(frame, image=photo_image)
        label.image = photo_image
        label.grid(row=(i // 2) * 2, column=i % 2, padx=10, pady=10)

        # Create a label to display the image name
        name_label = tk.Label(frame, text=f"Image {i + 1}", font=("Arial", 12))
        name_label.grid(row=(i // 2) * 2 + 1, column=i % 2, padx=10, pady=0)

        # Add the image to the set of processed images
        processed_images.add(image_path)

    else:
        print(f"Failed to load the image: {image_path}. Please check the file path and file name.")

# Pack the frame into the window
frame.pack()

# Create a label to display the output statement
output_label = tk.Label(window, text=f"Green light provided to {max_cars_image} with {max_cars} car(s) detected.",
                        font=("Arial", 14), pady=10)
output_label.pack()

# Start the Tkinter event loop
window.mainloop()
