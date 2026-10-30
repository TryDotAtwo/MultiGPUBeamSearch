# Cube555 blend with CayleyPy

[Open the Kaggle notebook](https://www.kaggle.com/code/trydotatwo/cube555-native-2xt4-blend/178).

Edit the first visible cell and choose **Run All**. Select GPU T4 x2 and enable Internet on Kaggle.

The notebook uses the CayleyPy `graph.beam_search` API with the Transformer/MLP blend. Set the puzzle, beam, model weight, collect/BFS/reflection options and result publishing in the first cell. The GitHub source includes a verified 2×T4 profile for the default beam of 1,048,576. The library loads exact matches and checks them against the current GPUs; if none matches, it tunes for the requested beam. Solutions are replayed, saved and published from a two-GPU Kaggle run.

