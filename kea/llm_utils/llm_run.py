# ppo_run_inference.py
import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from llm_policy import LLMAgent
from ppo_env import EnvRunner


def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment and system arguments
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True, nargs="?", const=True,
                        help="if toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    # Environment and minimal algorithm parameters for inference
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=32,
                        help="the number of steps to run in each environment per rollout")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Whether to print debug information and render")
    # Model loading parameters
    parser.add_argument("--load-8bit", type=bool, default=False,
                        help="Whether to convert model to 8bits")
    parser.add_argument("--resume", type=bool, default=False,
                        help="Whether to resume from a previous checkpoint")
    parser.add_argument("--load-path", type=str, default="saved_models",
                        help="The path to load the checkpoint")
    # Logging parameters
    parser.add_argument("--record-path", type=str, default="llm5_runs",
                        help="The path to save the tensorboard results")
    parser.add_argument("--normalization-mode", type=str, default="token",
                        help="The normalization mode for dealing with token logits")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    return args


if __name__ == "__main__":
    args = parse_args()
    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time_str}"

    # Set up TensorBoard writer
    writer = SummaryWriter(f"{args.record_path}/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n" +
                    "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env_params = {"seed": args.seed, "debug": args.debug}

    # Create the environment runner (fully decoupled)
    env_runner = EnvRunner(args, run_name, env_params, device)

    # Initialize the agent (load from checkpoint if resume==True)
    if args.resume:
        agent = LLMAgent(normalization_mode=args.normalization_mode, load_path=args.load_path, load_8bit=args.load_8bit)
    else:
        agent = LLMAgent(normalization_mode=args.normalization_mode, load_8bit=args.load_8bit)

    # Inference loop: run a fixed number of episodes
    global_step = 0
    start_time = time.time()
    next_obs = env_runner.reset()
    next_done = torch.zeros(args.num_envs).to(device)

    num_episodes = 10  # change this value as desired
    episode_count = 0

    while episode_count < num_episodes:
        for step in range(args.num_steps):
            global_step += args.num_envs
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            next_obs, reward, next_done, info = env_runner.step(action)

            # Check for episode termination in info dictionary
            for item in info:
                if "episode" in item.keys():
                    print(
                        f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    episode_count += 1
                    if episode_count >= num_episodes:
                        break

        # Log inference speed
        writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)

    env_runner.close()
    writer.close()
