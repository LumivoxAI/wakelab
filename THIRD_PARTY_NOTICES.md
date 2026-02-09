# Third-Party Notices

## Silero VAD

Wakelab's ONNX adapter implements the state and context handling required by the
Silero VAD model. The model and the corresponding algorithm are provided by the
Silero Team under the MIT License:

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## openWakeWord

Wakelab's ONNX wake-word adapter reimplements the narrow streaming feature
and inference contract from openWakeWord. The corresponding openWakeWord
inference code is provided under the Apache License 2.0:

Copyright 2022 David Scripka

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at

<https://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

The separately acquired `melspectrogram.onnx` and `embedding_model.onnx`
feature artifacts are pinned to the files published with openWakeWord release
`v0.5.1`. They are not included in Wakelab packages. Upstream bundled
wake-word classifiers are licensed under CC BY-NC-SA 4.0 and are neither
registered, downloaded, nor redistributed by Wakelab. Caller-supplied
classifiers remain caller-owned trusted inputs; Wakelab makes no claim about
their provenance or license.
