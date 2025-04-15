import requests

class PPOClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.task_id = None

    def attach(self):
        res = requests.post(f"{self.base_url}/attach")
        res.raise_for_status()
        self.task_id = res.json()["task_id"]
        print(f"[CLIENT] Attached new task: {self.task_id}")
        return self.task_id

    def detach(self):
        if not self.task_id:
            raise ValueError("No task_id to detach.")
        res = requests.post(f"{self.base_url}/detach", params={"task_id": self.task_id})
        res.raise_for_status()
        print(f"[CLIENT] Detached task: {self.task_id}")
        self.task_id = None

    def step(self, observation: dict):
        if not self.task_id:
            raise ValueError("Task not attached.")
        payload = {
            "task_id": self.task_id,
            "obs": observation
        }
        res = requests.post(f"{self.base_url}/step", json=payload)
        res.raise_for_status()
        return res.json()

    def feedback(self, reward: float, done: bool, next_obs: dict = None):
        if not self.task_id:
            raise ValueError("Task not attached.")
        payload = {
            "task_id": self.task_id,
            "reward": reward,
            "done": done,
            "next_obs": next_obs or {}
        }
        res = requests.post(f"{self.base_url}/feedback", json=payload)
        res.raise_for_status()
        return res.json()

    def generate(self, prompt: str, **kwargs):
        if not self.task_id:
            raise ValueError("Task not attached.")
        payload = {
            "task_id": self.task_id,
            "prompt": prompt,
            "max_new_tokens": kwargs.get("max_new_tokens", 100),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("do_sample", True),
        }
        res = requests.post(f"{self.base_url}/generate", json=payload)
        res.raise_for_status()
        return res.json()
