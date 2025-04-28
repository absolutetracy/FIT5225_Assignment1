from locust import HttpUser, TaskSet, task, between, LoadTestShape
import uuid, base64, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gevent


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

def monitor_failures(environment):
    FAILURE_THRESHOLD = 0.0
    CHECK_INTERVAL = 1
    while True:
        gevent.sleep(CHECK_INTERVAL)
        fail_ratio = environment.runner.stats.total.fail_ratio
        if fail_ratio > FAILURE_THRESHOLD:
            current_users = environment.runner.user_count
            print(f"Failure ratio exceeded: {fail_ratio:.2f} > {FAILURE_THRESHOLD}")
            print(f"The number of users: {current_users}")
            with open("failure_log.txt", "a") as log_file:
                log_file.write(f"Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Failure ratio exceeded: {fail_ratio:.2f} > {FAILURE_THRESHOLD}\n")
                log_file.write(f"Failure at {current_users} users\n")

            environment.runner.quit()
            break

class LinearLoadShape(LoadTestShape):
    max_users = 1000
    spawn_rate = 1

    def tick(self):
        run_time = self.get_run_time()
        if run_time * self.spawn_rate >= self.max_users:
            return None
        total_users = int(run_time * self.spawn_rate)
        return (total_users, self.spawn_rate)


class HelloWorldUser(HttpUser):
    # wait_time = between(1, 5)

    #@task
    #def hello_world(self):
    #    self.client.get("/")
    # host = "http://localhost:8000"

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

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(send_request, image) for image in images]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception: {e}")

    

    def on_start(self):
        # This method is called when a simulated user starts executing
        # You can initialize any state or perform setup here
        

        gevent.spawn(monitor_failures, self.environment)

    