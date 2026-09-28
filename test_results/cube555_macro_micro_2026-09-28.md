# Cube555 outer/model batch correction

User target: outer8192, independent model microbatch, requested beam29,360,128;
reuse existing Transformer pipeline profiles. Python Cube555 regression: 16 passed.
Native C++ launcher previously ignored the separate microbatch. Now it processes
contiguous model chunks and preserves score-ring offsets and stream dependency.
Native Kaggle build, replay smoke and large allocation measurements are pending.
