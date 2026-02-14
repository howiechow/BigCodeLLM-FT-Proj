from llama import Llama
import fire

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    instructions = [
        {"role": "user", "content": "In Bash, how do I list all text files in the current directory (and subdirectories) that have been modified in the last month?"},
        {"role": "user", "content": "What is the difference between list and tuple in Python?"},
        {"role": "user", "content": "Write a Python function to find the longest common subsequence of two strings."},
    ]
    
    results = generator.chat_completion(
        instructions,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    
    for instruction, result in zip(instructions, results):
        for msg in result:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)