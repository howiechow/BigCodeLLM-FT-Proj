#!/usr/bin/env python
from llama import Llama
import fire

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    dialogs = [
        ["Who are you?", "I am Llama, a large language model."]
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_seq_len,
        temperature=temperature,
        top_p=top_p,
    )
    for result in results:
        print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print('\n==================================\n')

if __name__ == "__main__":
    fire.Fire(main)
