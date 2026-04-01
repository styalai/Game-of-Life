uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu126

[tool.uv]
override-dependencies = [
  "torch==2.10.0+cu126",
]
uv pip install tilelang