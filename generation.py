#!/usr/bin/env python
from typing import List
import torch

from llama.model import Transformer
from llama.tokenizer import Tokenizer

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: int = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory with checkpoints.
            tokenizer_path (str): Path to the tokenizer.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int]): Number of model shards. If not provided, it is inferred from the checkpoint.
            seed (int): Seed for RNG.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.
        """
        prev_time = time.time()
        if local_rank == 0:
            print("Loading")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        # Initialize model parallel before importing modules that might use distributed
        # initialization.
        model_parallel.initialize_model_parallel(model_parallel_size)

        # Load the parameters of the model from the checkpoint directory.
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"No checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Number of checkpoints ({len(checkpoints)}) should match model_parallel_size ({model_parallel_size})"

        ckpt_path = checkpoints[local_rank]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        if local_rank == 0:
            print(f"Loaded in {time.time() - prev_time:.2f} seconds")
        return cls(model, tokenizer, model_args)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):
        """
        Perform text completion on a batch of prompts.

        Args:
            prompts (List[str]): List of input prompts.
            temperature (float): Temperature for sampling. Lower values make the output more deterministic.
            top_p (float): Top-p sampling threshold.
            max_gen_len (Optional[int]): Maximum length of generated text. If not provided, defaults to the model's maximum.

        Returns:
            List[str]: List of generated texts.
        """
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # ... (rest of method)
        return generated

    def chat_completion(
        self,
        dialogs: List[List[Tuple[str, str]]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):
        """
        Generate chat completions for a list of dialogues.

        Args:
            dialogs (List[List[Tuple[str, str]]]): List of dialogues, each being a list of (role, message) pairs.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling threshold.
            max_gen_len (Optional[int]): Maximum length of generated response.

        Returns:
            List[Dict]: List of results with generated responses.
        """
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1
        prompt_tokens = []
        for dialog in dialogs:
            tokenized_dialog = self.tokenizer.encode_dialog(dialog)
            prompt_tokens.append(tokenized_dialog)
        # ... (rest of method)
        return results
