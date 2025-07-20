from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import inferless
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = '1'

@inferless.request
class RequestObjects(BaseModel):
    audio_path: str = Field(default="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3")
    text_prompt: str = Field(default="What can you tell me about the audio?")
    max_new_tokens: Optional[int] = 500
    temperature: Optional[float] = 1.0
    do_sample: Optional[bool] = True
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50

@inferless.response
class ResponseObjects(BaseModel):
    generated_text: str = Field(default="Test Output")

class InferlessPythonModel:
    def initialize(self):
        self.repo_id = "mistralai/Voxtral-Mini-3B-2507"
        self.processor = AutoProcessor.from_pretrained(self.repo_id)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.repo_id, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )

    def _create_conversation(self, audio_path: str, text_prompt: str) -> List[Dict[str, Any]]:
        """Create conversation format for the model"""
        content = []
        
        # Add audio file
        content.append({
                "type": "audio",
                "path": audio_path
            })
        
        # Add text prompt
        content.append({
            "type": "text", 
            "text": text_prompt
        })
        
        conversation = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return conversation

    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        # Create conversation format
        conversation = self._create_conversation(inputs.audio_path, inputs.text_prompt)
        
        print(f"Processing {len(inputs.audio_paths)} audio file(s)...")
        
        # Apply chat template
        model_inputs = self.processor.apply_chat_template(conversation)
        model_inputs = model_inputs.to(self.device, dtype=torch.bfloat16)
        
        # Generate response
        generation_kwargs = {
            "max_new_tokens": inputs.max_new_tokens,
            "do_sample": inputs.do_sample,
            "temperature": inputs.temperature,
            "top_p": inputs.top_p,
            "top_k": inputs.top_k,
        }
        
        print("Generating response...")
        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, **generation_kwargs)
        
        # Decode the generated tokens
        decoded_outputs = self.processor.batch_decode(
            outputs[:, model_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        generated_text = decoded_outputs[0].strip()
        
        print("Response generated successfully!")
        
        return ResponseObjects(
            generated_text=generated_text
        )

    def finalize(self):
        self.model = None
