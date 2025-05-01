from locust import HttpUser, TaskSet, task, between, os
import uuid, base64, random
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGE_FOLDER = "inputfolder"  # Folder containing images
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Client', IMAGE_FOLDER)
# print("Image path: ", image_path)

images = []


for filename in os.listdir(image_path):
    # print("Filename: ", filename)
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(image_path, filename)
        with open(path, "rb") as img_file:
            img_data = img_file.read()
            images.append(img_data)

if not images:
    raise ValueError("No images found in the folder!")

class HelloWorldUser(HttpUser):
    # wait_time = between(1, 5)

    #@task
    #def hello_world(self):
    #    self.client.get("/")
    host = "http://localhost:8000"

    @task
    def test_keypoints(self):
        def prepare_data(image_bytes):
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_id = str(uuid.uuid5(uuid.NAMESPACE_OID, image_base64))
            return {
                "id": image_id,
                "image": image_base64
            }
        
        def send_request(image):
            data = prepare_data(image)
            with self.client.post("/keypoints", json=data, catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"Failed! Status: {response.status_code}")

        image = random.choice(images)
        send_request(image)
            #for future in as_completed(futures):
            #    try:
            #        future.result()
            #    except Exception as e:
            #        print(f"Exception: {e}")