from locust import HttpUser, TaskSet, task, between, LoadTestShape
import uuid, base64, os, time, random, argparse, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import gevent

parser = argparse.ArgumentParser(description="Load test script")
parser.add_argument("--muser", type=int, default=1000, help="Max number of users")
parser.add_argument("--rate", type=int, default=1, help="Spawn rate")
parser.add_argument("--user", type=int, default=1, help="Initial number of users")
parser.add_argument("--tick", type=int, default=1, help="Tick time in seconds")
args, _ = parser.parse_known_args()



print("Number of users: ", args.muser)
print("Spawn rate: ", args.rate)
print("Initial number of users: ", args.user)
print("Tick time: ", args.tick)

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
            avg_response_time = environment.runner.stats.total.avg_response_time
            print(f"Average response time: {avg_response_time:.2f} ms")
            print(f"Failure ratio exceeded: {fail_ratio:.2f} > {FAILURE_THRESHOLD}")
            print(f"The number of users: {current_users}")
            with open("failure_log.txt", "a") as log_file:
                log_file.write(f"Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Failure ratio exceeded: {fail_ratio:.2f} > {FAILURE_THRESHOLD}\n")
                log_file.write(f"Failure at {current_users} users\n")
                log_file.write(f"Average response time: {avg_response_time:.2f} ms\n")

            environment.runner.quit()
            break

class LinearLoadShape(LoadTestShape):
    max_users = args.muser
    spawn_rate = args.rate
    initial_users = args.user
    step_time = args.tick  # seconds

    def tick(self):
        run_time = self.get_run_time()
        ramp_users = int(run_time * self.spawn_rate)
        total_users = self.initial_users + ramp_users
        if total_users >= self.max_users:
            return None
        # total_users = int(run_time * self.spawn_rate)
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

        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(send_request, image) for image in images]
        #     for future in as_completed(futures):
        #         try:
        #             future.result()
        #         except Exception as e:
        #             print(f"Exception: {e}")
        for image in images:
            send_request(image)
        #send_request(random.choice(images))
    

    def on_start(self):
        # This method is called when a simulated user starts executing
        # You can initialize any state or perform setup here
        

        gevent.spawn(monitor_failures, self.environment)

    