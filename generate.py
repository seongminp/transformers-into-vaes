import argparse

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate(model, starter_tokens, **kwargs):

    # encoder_outputs = self.t5.model.run_encoder(
    #    kwargs.get("input_ids"),
    #    kwargs.get("attention_mask"),
    #    kwargs.get("head_mask"),
    # kwargs.get("inputs_embeds"),
    # kwargs.get("output_attentions"),
    # kwargs.get("output_hidden_states"),
    # kwargs.get("return_dict"),
    # )
    # pooled = self.bart.model.pool(encoder_outputs.hidden_states)
    # past, z, mu, logvar = self.bart.model.build_past(encoder_outputs, pooled)

    # new_encoder_hidden_states = torch.zeros((1, 1)).to(pooled.device)
    # new_attention_mask = torch.ones((1, 1)).to(pooled.device)

    # generated = torch.tensor(
    #     [self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]
    # 3).unsqueeze(0)
    generated = torch.tensor(starter_tokens).unsqueeze(0).to(model.device)

    output, encoder_outputs = None, None
    while generated.shape[1] < 1000:

        # decoder_inputs = self.t5.prepare_inputs_for_generation(generated, past=past)

        sampled_z = kwargs.get("sampled_z") if output is None else None

        with torch.no_grad():
            output = model.t5(
                input_ids=kwargs.get("input_ids"),
                attention_mask=kwargs.get("attention_mask"),
                # attention_mask=torch.ones((generated.shape[0], generated.shape[1] + 1)),
                # encoder_hidden_states=None, #new_encoder_hidden_states,  # Modified.
                # encoder_attention_mask=None, #new_attention_mask,  # Modified.
                # attention_mask=encoder_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=generated[:, -1].unsqueeze(0),
                # encoder_hidden_states=encoder_outputs[0],  # Modified.
                # encoder_attention_mask=attention_mask,  # Modified.
                # head_mask=kwargs.get("decoder_head_mask"),
                # cross_attn_head_mask=kwargs.get("cross_attn_head_mask"),
                past_key_values=output.past_key_values if output else None,
                # inputs_embeds=decoder_inputs_embeds,
                use_cache=True,
                # output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                sampled_z=sampled_z,
            )

        temperature = kwargs.get("temperature") if "temperature" in kwargs else 1.0
        top_k = kwargs.get("top_k") if "top_k" in kwargs else 0
        top_p = kwargs.get("top_p") if "top_p" in kwargs else 0

        logits = output.logits[0, -1, :] / temperature
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, 1)
        # next_token_id = torch.argmax(filtered_logits, dim=-1).unsqueeze(0)

        generated = torch.cat((generated, next_token_id.unsqueeze(0)), dim=1)
        past = output.past_key_values
        encoder_outputs = BaseModelOutput(
            last_hidden_state=output.encoder_last_hidden_state,
            hidden_states=output.encoder_hidden_states,
            attentions=output.encoder_attentions,
        )
        if next_token_id == model.tokenizer.eos_token_id:
            break

    return generated


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument.")
    args = parser.parse_args()
