from transformers import AutoModel
import os

model_path = os.path.join('/Users/tod/PretrainedLLM', 'ModernBERT-base')
model = AutoModel.from_pretrained(model_path, local_files_only=True)

print("Model type:", type(model).__name__)
print("\nTop-level attributes:")
for attr in dir(model):
    if not attr.startswith("_"):
        print(f"  {attr}")

print("\nFinding layers:")
# Check if model has a layers attribute
if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
    print("  Traditional BERT structure with model.encoder.layer")
    print(f"  Number of layers: {len(model.encoder.layer)}")
elif hasattr(model, "layers"):
    print("  Modern structure with model.layers")
    print(f"  Number of layers: {len(model.layers)}")
else:
    print("  Searching for layer attributes...")
    for attr in dir(model):
        if not attr.startswith("_"):
            attr_value = getattr(model, attr)
            if hasattr(attr_value, "__len__") and not isinstance(attr_value, (str, dict, set)):
                try:
                    length = len(attr_value)
                    print(f"  Found potential layers in {attr} with length {length}")
                except:
                    pass

# Print model details
print("\nModel config:")
for key, value in model.config.to_dict().items():
    print(f"  {key}: {value}")