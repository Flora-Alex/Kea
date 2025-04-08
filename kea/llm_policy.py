import sys

import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, TaskType

import os
import torch.nn as nn
import numpy as np

from torch.distributions.categorical import Categorical
import copy

root = os.path.dirname(os.path.abspath(__file__))

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LLMAgent(nn.Module):
    def __init__(self, normalization_mode='token', load_path=None, load_8bit=False):
        super().__init__()

        self.load_8bit = load_8bit
        self.base_model = 'Qwen/Qwen2.5-1.5B'
        self.lora_r = 8
        self.lora_alpha = 16
        # self.lora_dropout = 0.05
        self.lora_dropout = 0
        self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        assert (
            self.base_model
        ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
        except:  # noqa: E722
            pass

        self.normalization_mode = normalization_mode

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

        self.llm = self._init_llm()

        if load_path:
            self.load(load_path)
        else:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)

    def _init_llm(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            load_in_8bit=self.load_8bit,
            device_map="auto",
            cache_dir=os.path.join(root, f'weights/{self.base_model}')
        )

        if not self.load_8bit:
            model.half().to(self.device)
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,)

        return model

    def _init_actor(self, lora_weights=None):
        if lora_weights is None:
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(self.llm, config)

            model.print_trainable_parameters()

            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))
        else:
            model = PeftModel.from_pretrained(
                self.llm,
                lora_weights,
                torch_dtype=torch.float16,
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        if not self.load_8bit:
            model.half()
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        return model

    def _init_critic(self, critic_weights=None):
        critic = Critic(self.actor, self.tokenizer)
        if critic_weights is not None:
            critic.v_head.load_state_dict(torch.load(critic_weights, map_location="cpu"))
        return critic

    def save(self, epoch, exp_path):
        print("save model")
        exp_path = os.path.join(exp_path, "epoch_{:04d}".format(epoch))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        self.actor.save_pretrained(exp_path)
        # save critic
        # torch.save(self.critic.v_head.state_dict(), os.path.join(exp_path, "critic.pth"))

    def load(self, exp_path):
        print("load model")
        lora_weights = exp_path
        # critic_weights = os.path.join(exp_path, "critic.pth")
        self.actor = self._init_actor(lora_weights).to(self.device)
        # self.critic = self._init_critic(critic_weights).to(self.device)

    def get_value(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with self.actor.disable_adapter():
            value = self.critic(input_ids, attention_mask=attention_mask)
        return value

    def get_action_and_value(self, prompt, action_list, action=None, is_warmup=False, return_value=True):
        prompt_num = len(prompt)
        action_num = len(action_list)

        sequence = [f"{p} {a}" for p, a in zip(prompt, action_list)]

        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)

        attention_mask = inputs["attention_mask"].to(self.device)
        if is_warmup:
            with torch.no_grad():
                outputs = self.actor(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.actor(input_ids, attention_mask=attention_mask)

        self.action_list_ids = self.tokenizer(action_list, return_tensors="pt", padding=True)

        self.action_list_length = torch.sum(self.action_list_ids["attention_mask"], dim=-1) - 1  # delete first token
        sequence_length = torch.sum(attention_mask, dim=-1)
        action_index = [[end - start, end] for start, end in zip(self.action_list_length, sequence_length)]

        # maybe no need to use it, directly use logits
        logits = torch.log_softmax(outputs.logits, dim=-1)

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)

        slices = [gen_logits[i, start - 1:end - 1] for i, (start, end) in enumerate(action_index)]

        action_logits = torch.stack([torch.sum(s) for s in slices])
        if self.normalization_mode == 'token':
            action_logits = action_logits / self.action_list_length.to(self.device)
        elif self.normalization_mode == 'word':
            action_word_num = torch.tensor([len(action.split()) for action in action_list]).to(self.device)
            action_logits = action_logits / action_word_num
        elif self.normalization_mode == 'sum':
            action_logits = action_logits
        else:
            assert 1 == 2

        action_logits = action_logits.reshape(-1, action_num).float()

        probs = Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()

        if return_value:
            return action, probs.log_prob(action), probs.entropy(), self.get_value(prompt)
        else:
            return action, probs.log_prob(action), probs.entropy(), None


## Note that the following code is modified from
## https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py
class Critic(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        # `OPT` models use word_embed_proj_dim as final output
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(
             self.config, "hidden_size") else self.config.word_embed_proj_dim if hasattr(
             self.config, "word_embed_proj_dim") else self.config.n_embd
        self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False)
        self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
        self.v_head_mlp3 = nn.Linear(512, 1, bias=False)
        self.relu = nn.ReLU()
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      head_mask=None,
                      inputs_embeds=None,
                      use_cache=False):
        with torch.no_grad():
            transformer_outputs = self.rwtranrsformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True)

        hidden_states = transformer_outputs[1][-1][:, -1, :].float()

        x = self.relu(self.v_head_mlp1(hidden_states))
        x = self.relu(self.v_head_mlp2(x))
        values = self.v_head_mlp3(x).squeeze(-1)
        return values