# CayleyPy native beam search (Cube555 default)

[Open the Kaggle notebook](https://www.kaggle.com/code/trydotatwo/cube555-native-2xt4-blend/180).

Edit the first visible cell and choose **Run All**. Select GPU T4 x2 and enable Internet on Kaggle.

The first cell accepts a CayleyPy `GRAPH`, a `PREDICTOR`, and a start state directly. The same `graph.beam_search` call then runs on any native-supported permutation graph and registered model family. The bundled Cube555 Transformer/MLP model remains the default example. Beam width, collect, BFS, reflection, GPU count and result output stay configurable in that cell.

Native search requires permutation generators, states of up to 150 entries, and a registered model exporter (or a matching `NativeModel` artifact). New model architectures need an exporter. Local solutions are replay-checked and saved. Publication requires the catalog competition and puzzle type; the catalog also enforces its model and two-GPU schema. The GitHub profile registry contains the verified 2×T4 profile for the bundled Cube555 default beam of 1,048,576; a profile is reused only for a matching graph, model, beam and GPU configuration.
