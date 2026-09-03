# Third-party notices

The AIDE architecture in `detectors/aide/upstream/` is derived from
[shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) and retains its MIT
license in `detectors/aide/upstream/LICENSE`. Local changes remove training
code, keep only inference components, and preserve gradients through the
frozen ConvNeXt branch.

`detectors/npr.py` is an independent, checkpoint-compatible implementation of
the architecture described by
[NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection).
The upstream repository did not include an explicit software license when this
submission was prepared, so its source files are not redistributed here.
