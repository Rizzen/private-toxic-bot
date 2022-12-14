import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def init_model(checkpoint: str):
    tokenizer =  AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    history = torch.zeros((1, 0), dtype=torch.int)
    return (model, tokenizer, history)


def predict2(model, tokenizer, messages_history):
    chat_history_ids = torch.zeros((1, 0), dtype=torch.int)
    N = 5
    messages = messages_history[-N:]
    print(messages)
    for i in messages:
        new_user_input_ids = tokenizer.encode(f"|0|{get_length_param(i, tokenizer)}|" \
                                              + i + tokenizer.eos_token, return_tensors="pt")
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    next_len = "-"  #input("Exp. len?(-/1/2/3): ")
    # encode the new user input, add parameters and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(f"|1|{next_len}|", return_tensors="pt")
    chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1).to("cpu")
    input_len = chat_history_ids.shape[-1]
    print("input len = " + str(input_len))
    chat_history_ids = model.generate(
        chat_history_ids,
        num_return_sequences=1,                     # use for more variants, but have to print [i]
        max_length=512,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.9,
        temperature = 0.6,                        # 0 for greedy
        # mask_token_id=tokenizer.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # unk_token_id=tokenizer.unk_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # device='cpu'
    )

    # pretty print last ouput tokens from bot
    decoded = tokenizer.decode(chat_history_ids[:, input_len:][0], skip_special_tokens=True)
    # print(f"===> SlavaGPT-2:  {decoded}")

    return decoded


def get_length_param(text: str, tokenizer) -> str:
    """Maps text to 1 of 4 buckets based on length after encoding.

    Parameters
    ----------
    text: str
        The text to be given 1 of 4 length parameters.

    tokenizer: HuggingFace tokenizer
        Tokenizer that used to compute the length of the text after encoding.
        For more info ee https://huggingface.co/transformers/main_classes/tokenizer.html

    Returns
    -------
    len_param: str
        One of four buckets:
        '1' for short, '2' for medium, '3' for long texts and '-' for all others.
    """
    tokens_count = len(tokenizer.encode(text))
    if tokens_count <= 15:
        len_param = '1'
    elif tokens_count <= 50:
        len_param = '2'
    elif tokens_count <= 256:
        len_param = '3'
    else:
        len_param = '-'
    return len_param