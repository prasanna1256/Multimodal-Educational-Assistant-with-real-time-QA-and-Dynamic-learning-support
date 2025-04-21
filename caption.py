import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
import torch.nn as nn
import torch

# Load the pre-trained model and processor
MODEL_DIRECTORY = "blip-image-captioning-base"
MODEL_PATH = os.path.join(MODEL_DIRECTORY)
PROCESSOR_PATH = os.path.join(MODEL_DIRECTORY)

processor = BlipProcessor.from_pretrained(PROCESSOR_PATH)
config = BlipConfig.from_pretrained(MODEL_PATH)

# Define a custom model architecture with two additional layers
class CustomBlipForConditionalGeneration(BlipForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # Define additional layers
        self.additional_layer1 = nn.Linear(768, 768)  # Adjust the size as needed
        self.additional_layer2 = nn.Linear(768, 768)  # Adjust the size as needed
        # Initialize additional layers with Xavier initialization
        nn.init.xavier_uniform_(self.additional_layer1.weight)
        nn.init.zeros_(self.additional_layer1.bias)
        nn.init.xavier_uniform_(self.additional_layer2.weight)
        nn.init.zeros_(self.additional_layer2.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        # Apply additional layers
        sequence_output = outputs[0]
        sequence_output = self.additional_layer1(sequence_output)
        sequence_output = nn.functional.relu(sequence_output)
        sequence_output = self.additional_layer2(sequence_output)
        return sequence_output, outputs[1:]

model = CustomBlipForConditionalGeneration.from_pretrained(MODEL_PATH)

def generate_caption(image, max_new_tokens=1000):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


