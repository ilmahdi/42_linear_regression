import socket
import threading
import json


theta0 = 0
theta1 = 0


def update_thetas(host="127.0.0.1", port=65431):
    """Receive updated theta values from the trainer in real-time."""
    global theta0, theta1
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    theta0, theta1 = map(float, data.decode().strip().split(","))
                    print(f"Updated: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")


def main():
    with open(".training_results.json", "r") as file:
        tr = json.load(file)
        theta0 = tr["theta0"]
        theta1 = tr["theta1"]
    listener_thread = threading.Thread(target=update_thetas, daemon=True)
    listener_thread.start()
    while True:
        try:
            mileage = int(input("Enter the mileage: "))
            print(
                f"the estimated price for this mileag is: {theta0 + (theta1 * mileage)}"
            )
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
