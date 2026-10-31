# Cube555 native multi-GPU beam search

[Open the Kaggle notebook](https://www.kaggle.com/code/trydotatwo/cube555-native-2xt4-blend/181).

Choose GPU T4 x2, enable Internet, edit the first cell if needed, then click **Run All**.

The notebook loads the Cube555 graph and puzzle states, then calls CayleyPy's `graph.beam_search` with the native multi-GPU backend. In the first cell, choose the puzzle range, model checkpoints and Transformer/MLP blend, beam width, collect mode, BFS radius, reflection, GPU count, output folder and result publication.

Set `TRANSFORMER_WEIGHT` to `1.0` for Transformer only or `0.0` for MLP only. Intermediate values blend the two models. Profile selection is automatic: a matching GitHub profile is reused; otherwise the library calibrates the current GPU setup.

Solutions are replay-checked and saved locally. On Kaggle, the notebook can also publish them to the results catalog. For another graph or model, use the general `multigpubeamsearch` API and provide a supported model exporter.
