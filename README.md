KERAS **Signature Classification** on ICDAR 2011 Signature Dataset using **Siamese** CNN.

**Backbone models**: custom CNN and pretrained CNNs on ImageNet including Xception, InceptionV3, ResNet50, and MobileNetV2
  

# **Import Packages**


```python
import os
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow import keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from google.colab import drive
```

# **Parameters**


```python
# -------------- Model ----------------
# backbone_model = 'custom_cnn'
## CNN models (pretrained on ImageNet)
# backbone_model = 'Xception'
# backbone_model = 'InceptionV3'
# backbone_model = 'ResNet50'
backbone_model = 'MobileNetV2'

# freeze convolutional layers:
freeze_conv_layers = True

# data:
img_size = 224        # image size

# training parameters:
batch_size = 64       # batch size
learning_rate = 1e-2  # learning rate
num_epoches = 5      # maximum number of epoches
steps_per_epoch = 100 # itration steps in each epoch

data_path = "./sign_data_models"
```

# **Dataset**

**Downloading Signature_Verification_Dataset from www.kaggle.com.**

You can manually download and use signature-verification-dataset.zip file from this url:

https://www.kaggle.com/datasets/robinreni/signature-verification-dataset/data


```python
! pip install -q kaggle
from google.colab import files


if not os.path.exists("kaggle.json"):
  # Choose and upload Kaggle's API token file: kaggle.json
  # (in your kaggle profile, and create new token file and download kaggle.json)
  # Choose the kaggle.json file that you downloaded
  files.upload()

# Make directory named kaggle and copy kaggle.json file there.
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/

# Change the permissions of the file.
! chmod 600 ~/.kaggle/kaggle.json

if not os.path.exists("signature-verification-dataset.zip"):
  # download dataset
  ! kaggle datasets download -d robinreni/signature-verification-dataset


```



     <input type="file" id="files-848dcfc2-2dc6-4663-a9df-8f56c176fd5a" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-848dcfc2-2dc6-4663-a9df-8f56c176fd5a">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving kaggle.json to kaggle.json
    Downloading signature-verification-dataset.zip to /content
     99% 596M/601M [00:18<00:00, 39.6MB/s]
    100% 601M/601M [00:18<00:00, 34.0MB/s]


unzip the dataset:


```python
# unzip dataset
!unzip "signature-verification-dataset.zip"
```

    Archive:  signature-verification-dataset.zip
      inflating: sign_data/sign_data/test/049/01_049.png  
      inflating: sign_data/sign_data/test/049/02_049.png  
      inflating: sign_data/sign_data/test/049/03_049.png  
      inflating: sign_data/sign_data/test/049/04_049.png  
      inflating: sign_data/sign_data/test/049/05_049.png  
      inflating: sign_data/sign_data/test/049/06_049.png  
      inflating: sign_data/sign_data/test/049/07_049.png  
      inflating: sign_data/sign_data/test/049/08_049.png  
      inflating: sign_data/sign_data/test/049/09_049.png  
      inflating: sign_data/sign_data/test/049/10_049.png  
      inflating: sign_data/sign_data/test/049/11_049.png  
      inflating: sign_data/sign_data/test/049/12_049.png  
      inflating: sign_data/sign_data/test/049_forg/01_0114049.PNG  
      inflating: sign_data/sign_data/test/049_forg/01_0206049.PNG  
      inflating: sign_data/sign_data/test/049_forg/01_0210049.PNG  
      inflating: sign_data/sign_data/test/049_forg/02_0114049.PNG  
      inflating: sign_data/sign_data/test/049_forg/02_0206049.PNG  
      inflating: sign_data/sign_data/test/049_forg/02_0210049.PNG  
      inflating: sign_data/sign_data/test/049_forg/03_0114049.PNG  
      inflating: sign_data/sign_data/test/049_forg/03_0206049.PNG  
      inflating: sign_data/sign_data/test/049_forg/03_0210049.PNG  
      inflating: sign_data/sign_data/test/049_forg/04_0114049.PNG  
      inflating: sign_data/sign_data/test/049_forg/04_0206049.PNG  
      inflating: sign_data/sign_data/test/049_forg/04_0210049.PNG  
      inflating: sign_data/sign_data/test/050/01_050.png  
      inflating: sign_data/sign_data/test/050/02_050.png  
      inflating: sign_data/sign_data/test/050/03_050.png  
      inflating: sign_data/sign_data/test/050/04_050.png  
      inflating: sign_data/sign_data/test/050/05_050.png  
      inflating: sign_data/sign_data/test/050/06_050.png  
      inflating: sign_data/sign_data/test/050/07_050.png  
      inflating: sign_data/sign_data/test/050/08_050.png  
      inflating: sign_data/sign_data/test/050/09_050.png  
      inflating: sign_data/sign_data/test/050/10_050.png  
      inflating: sign_data/sign_data/test/050/11_050.png  
      inflating: sign_data/sign_data/test/050/12_050.png  
      inflating: sign_data/sign_data/test/050_forg/01_0125050.PNG  
      inflating: sign_data/sign_data/test/050_forg/01_0126050.PNG  
      inflating: sign_data/sign_data/test/050_forg/01_0204050.PNG  
      inflating: sign_data/sign_data/test/050_forg/02_0125050.PNG  
      inflating: sign_data/sign_data/test/050_forg/02_0126050.PNG  
      inflating: sign_data/sign_data/test/050_forg/02_0204050.PNG  
      inflating: sign_data/sign_data/test/050_forg/03_0125050.PNG  
      inflating: sign_data/sign_data/test/050_forg/03_0126050.PNG  
      inflating: sign_data/sign_data/test/050_forg/03_0204050.PNG  
      inflating: sign_data/sign_data/test/050_forg/04_0125050.PNG  
      inflating: sign_data/sign_data/test/050_forg/04_0126050.PNG  
      inflating: sign_data/sign_data/test/050_forg/04_0204050.PNG  
      inflating: sign_data/sign_data/test/051/01_051.png  
      inflating: sign_data/sign_data/test/051/02_051.png  
      inflating: sign_data/sign_data/test/051/03_051.png  
      inflating: sign_data/sign_data/test/051/04_051.png  
      inflating: sign_data/sign_data/test/051/05_051.png  
      inflating: sign_data/sign_data/test/051/06_051.png  
      inflating: sign_data/sign_data/test/051/07_051.png  
      inflating: sign_data/sign_data/test/051/08_051.png  
      inflating: sign_data/sign_data/test/051/09_051.png  
      inflating: sign_data/sign_data/test/051/10_051.png  
      inflating: sign_data/sign_data/test/051/11_051.png  
      inflating: sign_data/sign_data/test/051/12_051.png  
      inflating: sign_data/sign_data/test/051_forg/01_0104051.PNG  
      inflating: sign_data/sign_data/test/051_forg/01_0120051.PNG  
      inflating: sign_data/sign_data/test/051_forg/02_0104051.PNG  
      inflating: sign_data/sign_data/test/051_forg/02_0120051.PNG  
      inflating: sign_data/sign_data/test/051_forg/03_0104051.PNG  
      inflating: sign_data/sign_data/test/051_forg/03_0120051.PNG  
      inflating: sign_data/sign_data/test/051_forg/04_0104051.PNG  
      inflating: sign_data/sign_data/test/051_forg/04_0120051.PNG  
      inflating: sign_data/sign_data/test/052/01_052.png  
      inflating: sign_data/sign_data/test/052/02_052.png  
      inflating: sign_data/sign_data/test/052/03_052.png  
      inflating: sign_data/sign_data/test/052/04_052.png  
      inflating: sign_data/sign_data/test/052/05_052.png  
      inflating: sign_data/sign_data/test/052/06_052.png  
      inflating: sign_data/sign_data/test/052/07_052.png  
      inflating: sign_data/sign_data/test/052/08_052.png  
      inflating: sign_data/sign_data/test/052/09_052.png  
      inflating: sign_data/sign_data/test/052/10_052.png  
      inflating: sign_data/sign_data/test/052/11_052.png  
      inflating: sign_data/sign_data/test/052/12_052.png  
      inflating: sign_data/sign_data/test/052_forg/01_0106052.PNG  
      inflating: sign_data/sign_data/test/052_forg/01_0109052.PNG  
      inflating: sign_data/sign_data/test/052_forg/01_0207052.PNG  
      inflating: sign_data/sign_data/test/052_forg/01_0210052.PNG  
      inflating: sign_data/sign_data/test/052_forg/02_0106052.PNG  
      inflating: sign_data/sign_data/test/052_forg/02_0109052.PNG  
      inflating: sign_data/sign_data/test/052_forg/02_0207052.PNG  
      inflating: sign_data/sign_data/test/052_forg/02_0210052.PNG  
      inflating: sign_data/sign_data/test/052_forg/03_0106052.PNG  
      inflating: sign_data/sign_data/test/052_forg/03_0109052.PNG  
      inflating: sign_data/sign_data/test/052_forg/03_0207052.PNG  
      inflating: sign_data/sign_data/test/052_forg/03_0210052.PNG  
      inflating: sign_data/sign_data/test/052_forg/04_0106052.PNG  
      inflating: sign_data/sign_data/test/052_forg/04_0109052.PNG  
      inflating: sign_data/sign_data/test/052_forg/04_0207052.PNG  
      inflating: sign_data/sign_data/test/052_forg/04_0210052.PNG  
      inflating: sign_data/sign_data/test/053/01_053.png  
      inflating: sign_data/sign_data/test/053/02_053.png  
      inflating: sign_data/sign_data/test/053/03_053.png  
      inflating: sign_data/sign_data/test/053/04_053.png  
      inflating: sign_data/sign_data/test/053/05_053.png  
      inflating: sign_data/sign_data/test/053/06_053.png  
      inflating: sign_data/sign_data/test/053/07_053.png  
      inflating: sign_data/sign_data/test/053/08_053.png  
      inflating: sign_data/sign_data/test/053/09_053.png  
      inflating: sign_data/sign_data/test/053/10_053.png  
      inflating: sign_data/sign_data/test/053/11_053.png  
      inflating: sign_data/sign_data/test/053/12_053.png  
      inflating: sign_data/sign_data/test/053_forg/01_0107053.PNG  
      inflating: sign_data/sign_data/test/053_forg/01_0115053.PNG  
      inflating: sign_data/sign_data/test/053_forg/01_0202053.PNG  
      inflating: sign_data/sign_data/test/053_forg/01_0207053.PNG  
      inflating: sign_data/sign_data/test/053_forg/02_0107053.PNG  
      inflating: sign_data/sign_data/test/053_forg/02_0115053.PNG  
      inflating: sign_data/sign_data/test/053_forg/02_0202053.PNG  
      inflating: sign_data/sign_data/test/053_forg/02_0207053.PNG  
      inflating: sign_data/sign_data/test/053_forg/03_0107053.PNG  
      inflating: sign_data/sign_data/test/053_forg/03_0115053.PNG  
      inflating: sign_data/sign_data/test/053_forg/03_0202053.PNG  
      inflating: sign_data/sign_data/test/053_forg/03_0207053.PNG  
      inflating: sign_data/sign_data/test/053_forg/04_0107053.PNG  
      inflating: sign_data/sign_data/test/053_forg/04_0115053.PNG  
      inflating: sign_data/sign_data/test/053_forg/04_0202053.PNG  
      inflating: sign_data/sign_data/test/053_forg/04_0207053.PNG  
      inflating: sign_data/sign_data/test/054/01_054.png  
      inflating: sign_data/sign_data/test/054/02_054.png  
      inflating: sign_data/sign_data/test/054/03_054.png  
      inflating: sign_data/sign_data/test/054/04_054.png  
      inflating: sign_data/sign_data/test/054/05_054.png  
      inflating: sign_data/sign_data/test/054/06_054.png  
      inflating: sign_data/sign_data/test/054/07_054.png  
      inflating: sign_data/sign_data/test/054/08_054.png  
      inflating: sign_data/sign_data/test/054/09_054.png  
      inflating: sign_data/sign_data/test/054/10_054.png  
      inflating: sign_data/sign_data/test/054/11_054.png  
      inflating: sign_data/sign_data/test/054/12_054.png  
      inflating: sign_data/sign_data/test/054_forg/01_0102054.PNG  
      inflating: sign_data/sign_data/test/054_forg/01_0124054.PNG  
      inflating: sign_data/sign_data/test/054_forg/01_0207054.PNG  
      inflating: sign_data/sign_data/test/054_forg/01_0208054.PNG  
      inflating: sign_data/sign_data/test/054_forg/01_0214054.PNG  
      inflating: sign_data/sign_data/test/054_forg/02_0102054.PNG  
      inflating: sign_data/sign_data/test/054_forg/02_0124054.PNG  
      inflating: sign_data/sign_data/test/054_forg/02_0207054.PNG  
      inflating: sign_data/sign_data/test/054_forg/02_0208054.PNG  
      inflating: sign_data/sign_data/test/054_forg/02_0214054.PNG  
      inflating: sign_data/sign_data/test/054_forg/03_0102054.PNG  
      inflating: sign_data/sign_data/test/054_forg/03_0124054.PNG  
      inflating: sign_data/sign_data/test/054_forg/03_0207054.PNG  
      inflating: sign_data/sign_data/test/054_forg/03_0208054.PNG  
      inflating: sign_data/sign_data/test/054_forg/03_0214054.PNG  
      inflating: sign_data/sign_data/test/054_forg/04_0102054.PNG  
      inflating: sign_data/sign_data/test/054_forg/04_0124054.PNG  
      inflating: sign_data/sign_data/test/054_forg/04_0207054.PNG  
      inflating: sign_data/sign_data/test/054_forg/04_0208054.PNG  
      inflating: sign_data/sign_data/test/054_forg/04_0214054.PNG  
      inflating: sign_data/sign_data/test/055/01_055.png  
      inflating: sign_data/sign_data/test/055/02_055.png  
      inflating: sign_data/sign_data/test/055/03_055.png  
      inflating: sign_data/sign_data/test/055/04_055.png  
      inflating: sign_data/sign_data/test/055/05_055.png  
      inflating: sign_data/sign_data/test/055/06_055.png  
      inflating: sign_data/sign_data/test/055/07_055.png  
      inflating: sign_data/sign_data/test/055/08_055.png  
      inflating: sign_data/sign_data/test/055/09_055.png  
      inflating: sign_data/sign_data/test/055/10_055.png  
      inflating: sign_data/sign_data/test/055/11_055.png  
      inflating: sign_data/sign_data/test/055/12_055.png  
      inflating: sign_data/sign_data/test/055_forg/01_0118055.PNG  
      inflating: sign_data/sign_data/test/055_forg/01_0120055.PNG  
      inflating: sign_data/sign_data/test/055_forg/01_0202055.PNG  
      inflating: sign_data/sign_data/test/055_forg/02_0118055.PNG  
      inflating: sign_data/sign_data/test/055_forg/02_0120055.PNG  
      inflating: sign_data/sign_data/test/055_forg/02_0202055.PNG  
      inflating: sign_data/sign_data/test/055_forg/03_0118055.PNG  
      inflating: sign_data/sign_data/test/055_forg/03_0120055.PNG  
      inflating: sign_data/sign_data/test/055_forg/03_0202055.PNG  
      inflating: sign_data/sign_data/test/055_forg/04_0118055.PNG  
      inflating: sign_data/sign_data/test/055_forg/04_0120055.PNG  
      inflating: sign_data/sign_data/test/055_forg/04_0202055.PNG  
      inflating: sign_data/sign_data/test/056/01_056.png  
      inflating: sign_data/sign_data/test/056/02_056.png  
      inflating: sign_data/sign_data/test/056/03_056.png  
      inflating: sign_data/sign_data/test/056/04_056.png  
      inflating: sign_data/sign_data/test/056/05_056.png  
      inflating: sign_data/sign_data/test/056/06_056.png  
      inflating: sign_data/sign_data/test/056/07_056.png  
      inflating: sign_data/sign_data/test/056/08_056.png  
      inflating: sign_data/sign_data/test/056/09_056.png  
      inflating: sign_data/sign_data/test/056/10_056.png  
      inflating: sign_data/sign_data/test/056/11_056.png  
      inflating: sign_data/sign_data/test/056/12_056.png  
      inflating: sign_data/sign_data/test/056_forg/01_0105056.PNG  
      inflating: sign_data/sign_data/test/056_forg/01_0115056.PNG  
      inflating: sign_data/sign_data/test/056_forg/02_0105056.PNG  
      inflating: sign_data/sign_data/test/056_forg/02_0115056.PNG  
      inflating: sign_data/sign_data/test/056_forg/03_0105056.PNG  
      inflating: sign_data/sign_data/test/056_forg/03_0115056.PNG  
      inflating: sign_data/sign_data/test/056_forg/04_0105056.PNG  
      inflating: sign_data/sign_data/test/056_forg/04_0115056.PNG  
      inflating: sign_data/sign_data/test/057/01_057.png  
      inflating: sign_data/sign_data/test/057/02_057.png  
      inflating: sign_data/sign_data/test/057/03_057.png  
      inflating: sign_data/sign_data/test/057/04_057.png  
      inflating: sign_data/sign_data/test/057/05_057.png  
      inflating: sign_data/sign_data/test/057/06_057.png  
      inflating: sign_data/sign_data/test/057/07_057.png  
      inflating: sign_data/sign_data/test/057/08_057.png  
      inflating: sign_data/sign_data/test/057/09_057.png  
      inflating: sign_data/sign_data/test/057/10_057.png  
      inflating: sign_data/sign_data/test/057/11_057.png  
      inflating: sign_data/sign_data/test/057/12_057.png  
      inflating: sign_data/sign_data/test/057_forg/01_0117057.PNG  
      inflating: sign_data/sign_data/test/057_forg/01_0208057.PNG  
      inflating: sign_data/sign_data/test/057_forg/01_0210057.PNG  
      inflating: sign_data/sign_data/test/057_forg/02_0117057.PNG  
      inflating: sign_data/sign_data/test/057_forg/02_0208057.PNG  
      inflating: sign_data/sign_data/test/057_forg/02_0210057.PNG  
      inflating: sign_data/sign_data/test/057_forg/03_0117057.PNG  
      inflating: sign_data/sign_data/test/057_forg/03_0208057.PNG  
      inflating: sign_data/sign_data/test/057_forg/03_0210057.PNG  
      inflating: sign_data/sign_data/test/057_forg/04_0117057.PNG  
      inflating: sign_data/sign_data/test/057_forg/04_0208057.PNG  
      inflating: sign_data/sign_data/test/057_forg/04_0210057.PNG  
      inflating: sign_data/sign_data/test/058/01_058.png  
      inflating: sign_data/sign_data/test/058/02_058.png  
      inflating: sign_data/sign_data/test/058/03_058.png  
      inflating: sign_data/sign_data/test/058/04_058.png  
      inflating: sign_data/sign_data/test/058/05_058.png  
      inflating: sign_data/sign_data/test/058/06_058.png  
      inflating: sign_data/sign_data/test/058/07_058.png  
      inflating: sign_data/sign_data/test/058/08_058.png  
      inflating: sign_data/sign_data/test/058/09_058.png  
      inflating: sign_data/sign_data/test/058/10_058.png  
      inflating: sign_data/sign_data/test/058/11_058.png  
      inflating: sign_data/sign_data/test/058/12_058.png  
      inflating: sign_data/sign_data/test/058_forg/01_0109058.PNG  
      inflating: sign_data/sign_data/test/058_forg/01_0110058.PNG  
      inflating: sign_data/sign_data/test/058_forg/01_0125058.PNG  
      inflating: sign_data/sign_data/test/058_forg/01_0127058.PNG  
      inflating: sign_data/sign_data/test/058_forg/02_0109058.PNG  
      inflating: sign_data/sign_data/test/058_forg/02_0110058.PNG  
      inflating: sign_data/sign_data/test/058_forg/02_0125058.PNG  
      inflating: sign_data/sign_data/test/058_forg/02_0127058.PNG  
      inflating: sign_data/sign_data/test/058_forg/03_0109058.PNG  
      inflating: sign_data/sign_data/test/058_forg/03_0110058.PNG  
      inflating: sign_data/sign_data/test/058_forg/03_0125058.PNG  
      inflating: sign_data/sign_data/test/058_forg/03_0127058.PNG  
      inflating: sign_data/sign_data/test/058_forg/04_0109058.PNG  
      inflating: sign_data/sign_data/test/058_forg/04_0110058.PNG  
      inflating: sign_data/sign_data/test/058_forg/04_0125058.PNG  
      inflating: sign_data/sign_data/test/058_forg/04_0127058.PNG  
      inflating: sign_data/sign_data/test/059/01_059.png  
      inflating: sign_data/sign_data/test/059/02_059.png  
      inflating: sign_data/sign_data/test/059/03_059.png  
      inflating: sign_data/sign_data/test/059/04_059.png  
      inflating: sign_data/sign_data/test/059/05_059.png  
      inflating: sign_data/sign_data/test/059/06_059.png  
      inflating: sign_data/sign_data/test/059/07_059.png  
      inflating: sign_data/sign_data/test/059/08_059.png  
      inflating: sign_data/sign_data/test/059/09_059.png  
      inflating: sign_data/sign_data/test/059/10_059.png  
      inflating: sign_data/sign_data/test/059/11_059.png  
      inflating: sign_data/sign_data/test/059/12_059.png  
      inflating: sign_data/sign_data/test/059_forg/01_0104059.PNG  
      inflating: sign_data/sign_data/test/059_forg/01_0125059.PNG  
      inflating: sign_data/sign_data/test/059_forg/02_0104059.PNG  
      inflating: sign_data/sign_data/test/059_forg/02_0125059.PNG  
      inflating: sign_data/sign_data/test/059_forg/03_0104059.PNG  
      inflating: sign_data/sign_data/test/059_forg/03_0125059.PNG  
      inflating: sign_data/sign_data/test/059_forg/04_0104059.PNG  
      inflating: sign_data/sign_data/test/059_forg/04_0125059.PNG  
      inflating: sign_data/sign_data/test/060/01_060.png  
      inflating: sign_data/sign_data/test/060/02_060.png  
      inflating: sign_data/sign_data/test/060/03_060.png  
      inflating: sign_data/sign_data/test/060/04_060.png  
      inflating: sign_data/sign_data/test/060/05_060.png  
      inflating: sign_data/sign_data/test/060/06_060.png  
      inflating: sign_data/sign_data/test/060/07_060.png  
      inflating: sign_data/sign_data/test/060/08_060.png  
      inflating: sign_data/sign_data/test/060/09_060.png  
      inflating: sign_data/sign_data/test/060/10_060.png  
      inflating: sign_data/sign_data/test/060/11_060.png  
      inflating: sign_data/sign_data/test/060/12_060.png  
      inflating: sign_data/sign_data/test/060_forg/01_0111060.PNG  
      inflating: sign_data/sign_data/test/060_forg/01_0121060.PNG  
      inflating: sign_data/sign_data/test/060_forg/01_0126060.PNG  
      inflating: sign_data/sign_data/test/060_forg/02_0111060.PNG  
      inflating: sign_data/sign_data/test/060_forg/02_0121060.PNG  
      inflating: sign_data/sign_data/test/060_forg/02_0126060.PNG  
      inflating: sign_data/sign_data/test/060_forg/03_0111060.PNG  
      inflating: sign_data/sign_data/test/060_forg/03_0121060.PNG  
      inflating: sign_data/sign_data/test/060_forg/03_0126060.PNG  
      inflating: sign_data/sign_data/test/060_forg/04_0111060.PNG  
      inflating: sign_data/sign_data/test/060_forg/04_0121060.PNG  
      inflating: sign_data/sign_data/test/060_forg/04_0126060.PNG  
      inflating: sign_data/sign_data/test/061/01_061.png  
      inflating: sign_data/sign_data/test/061/02_061.png  
      inflating: sign_data/sign_data/test/061/03_061.png  
      inflating: sign_data/sign_data/test/061/04_061.png  
      inflating: sign_data/sign_data/test/061/05_061.png  
      inflating: sign_data/sign_data/test/061/06_061.png  
      inflating: sign_data/sign_data/test/061/07_061.png  
      inflating: sign_data/sign_data/test/061/08_061.png  
      inflating: sign_data/sign_data/test/061/09_061.png  
      inflating: sign_data/sign_data/test/061/10_061.png  
      inflating: sign_data/sign_data/test/061/11_061.png  
      inflating: sign_data/sign_data/test/061/12_061.png  
      inflating: sign_data/sign_data/test/061_forg/01_0102061.PNG  
      inflating: sign_data/sign_data/test/061_forg/01_0112061.PNG  
      inflating: sign_data/sign_data/test/061_forg/01_0206061.PNG  
      inflating: sign_data/sign_data/test/061_forg/02_0102061.PNG  
      inflating: sign_data/sign_data/test/061_forg/02_0112061.PNG  
      inflating: sign_data/sign_data/test/061_forg/02_0206061.PNG  
      inflating: sign_data/sign_data/test/061_forg/03_0102061.PNG  
      inflating: sign_data/sign_data/test/061_forg/03_0112061.PNG  
      inflating: sign_data/sign_data/test/061_forg/03_0206061.PNG  
      inflating: sign_data/sign_data/test/061_forg/04_0102061.PNG  
      inflating: sign_data/sign_data/test/061_forg/04_0112061.PNG  
      inflating: sign_data/sign_data/test/061_forg/04_0206061.PNG  
      inflating: sign_data/sign_data/test/062/01_062.png  
      inflating: sign_data/sign_data/test/062/02_062.png  
      inflating: sign_data/sign_data/test/062/03_062.png  
      inflating: sign_data/sign_data/test/062/04_062.png  
      inflating: sign_data/sign_data/test/062/05_062.png  
      inflating: sign_data/sign_data/test/062/06_062.png  
      inflating: sign_data/sign_data/test/062/07_062.png  
      inflating: sign_data/sign_data/test/062/08_062.png  
      inflating: sign_data/sign_data/test/062/09_062.png  
      inflating: sign_data/sign_data/test/062/10_062.png  
      inflating: sign_data/sign_data/test/062/11_062.png  
      inflating: sign_data/sign_data/test/062/12_062.png  
      inflating: sign_data/sign_data/test/062_forg/01_0109062.PNG  
      inflating: sign_data/sign_data/test/062_forg/01_0116062.PNG  
      inflating: sign_data/sign_data/test/062_forg/01_0201062.PNG  
      inflating: sign_data/sign_data/test/062_forg/02_0109062.PNG  
      inflating: sign_data/sign_data/test/062_forg/02_0116062.PNG  
      inflating: sign_data/sign_data/test/062_forg/02_0201062.PNG  
      inflating: sign_data/sign_data/test/062_forg/03_0109062.PNG  
      inflating: sign_data/sign_data/test/062_forg/03_0116062.PNG  
      inflating: sign_data/sign_data/test/062_forg/03_0201062.PNG  
      inflating: sign_data/sign_data/test/062_forg/04_0109062.PNG  
      inflating: sign_data/sign_data/test/062_forg/04_0116062.PNG  
      inflating: sign_data/sign_data/test/062_forg/04_0201062.PNG  
      inflating: sign_data/sign_data/test/063/01_063.png  
      inflating: sign_data/sign_data/test/063/02_063.png  
      inflating: sign_data/sign_data/test/063/03_063.png  
      inflating: sign_data/sign_data/test/063/04_063.png  
      inflating: sign_data/sign_data/test/063/05_063.png  
      inflating: sign_data/sign_data/test/063/06_063.png  
      inflating: sign_data/sign_data/test/063/07_063.png  
      inflating: sign_data/sign_data/test/063/08_063.png  
      inflating: sign_data/sign_data/test/063/09_063.png  
      inflating: sign_data/sign_data/test/063/10_063.png  
      inflating: sign_data/sign_data/test/063/11_063.png  
      inflating: sign_data/sign_data/test/063/12_063.png  
      inflating: sign_data/sign_data/test/063_forg/01_0104063.PNG  
      inflating: sign_data/sign_data/test/063_forg/01_0108063.PNG  
      inflating: sign_data/sign_data/test/063_forg/01_0119063.PNG  
      inflating: sign_data/sign_data/test/063_forg/02_0104063.PNG  
      inflating: sign_data/sign_data/test/063_forg/02_0108063.PNG  
      inflating: sign_data/sign_data/test/063_forg/02_0119063.PNG  
      inflating: sign_data/sign_data/test/063_forg/03_0104063.PNG  
      inflating: sign_data/sign_data/test/063_forg/03_0108063.PNG  
      inflating: sign_data/sign_data/test/063_forg/03_0119063.PNG  
      inflating: sign_data/sign_data/test/063_forg/04_0104063.PNG  
      inflating: sign_data/sign_data/test/063_forg/04_0108063.PNG  
      inflating: sign_data/sign_data/test/063_forg/04_0119063.PNG  
      inflating: sign_data/sign_data/test/064/01_064.png  
      inflating: sign_data/sign_data/test/064/02_064.png  
      inflating: sign_data/sign_data/test/064/03_064.png  
      inflating: sign_data/sign_data/test/064/04_064.png  
      inflating: sign_data/sign_data/test/064/05_064.png  
      inflating: sign_data/sign_data/test/064/06_064.png  
      inflating: sign_data/sign_data/test/064/07_064.png  
      inflating: sign_data/sign_data/test/064/08_064.png  
      inflating: sign_data/sign_data/test/064/09_064.png  
      inflating: sign_data/sign_data/test/064/10_064.png  
      inflating: sign_data/sign_data/test/064/11_064.png  
      inflating: sign_data/sign_data/test/064/12_064.png  
      inflating: sign_data/sign_data/test/064_forg/01_0105064.PNG  
      inflating: sign_data/sign_data/test/064_forg/01_0203064.PNG  
      inflating: sign_data/sign_data/test/064_forg/02_0105064.PNG  
      inflating: sign_data/sign_data/test/064_forg/02_0203064.PNG  
      inflating: sign_data/sign_data/test/064_forg/03_0105064.PNG  
      inflating: sign_data/sign_data/test/064_forg/03_0203064.PNG  
      inflating: sign_data/sign_data/test/064_forg/04_0105064.PNG  
      inflating: sign_data/sign_data/test/064_forg/04_0203064.PNG  
      inflating: sign_data/sign_data/test/065/01_065.png  
      inflating: sign_data/sign_data/test/065/02_065.png  
      inflating: sign_data/sign_data/test/065/03_065.png  
      inflating: sign_data/sign_data/test/065/04_065.png  
      inflating: sign_data/sign_data/test/065/05_065.png  
      inflating: sign_data/sign_data/test/065/06_065.png  
      inflating: sign_data/sign_data/test/065/07_065.png  
      inflating: sign_data/sign_data/test/065/08_065.png  
      inflating: sign_data/sign_data/test/065/09_065.png  
      inflating: sign_data/sign_data/test/065/10_065.png  
      inflating: sign_data/sign_data/test/065/11_065.png  
      inflating: sign_data/sign_data/test/065/12_065.png  
      inflating: sign_data/sign_data/test/065_forg/01_0118065.PNG  
      inflating: sign_data/sign_data/test/065_forg/01_0206065.PNG  
      inflating: sign_data/sign_data/test/065_forg/02_0118065.PNG  
      inflating: sign_data/sign_data/test/065_forg/02_0206065.PNG  
      inflating: sign_data/sign_data/test/065_forg/03_0118065.PNG  
      inflating: sign_data/sign_data/test/065_forg/03_0206065.PNG  
      inflating: sign_data/sign_data/test/065_forg/04_0118065.PNG  
      inflating: sign_data/sign_data/test/065_forg/04_0206065.PNG  
      inflating: sign_data/sign_data/test/066/01_066.png  
      inflating: sign_data/sign_data/test/066/02_066.png  
      inflating: sign_data/sign_data/test/066/03_066.png  
      inflating: sign_data/sign_data/test/066/04_066.png  
      inflating: sign_data/sign_data/test/066/05_066.png  
      inflating: sign_data/sign_data/test/066/06_066.png  
      inflating: sign_data/sign_data/test/066/07_066.png  
      inflating: sign_data/sign_data/test/066/08_066.png  
      inflating: sign_data/sign_data/test/066/09_066.png  
      inflating: sign_data/sign_data/test/066/10_066.png  
      inflating: sign_data/sign_data/test/066/11_066.png  
      inflating: sign_data/sign_data/test/066/12_066.png  
      inflating: sign_data/sign_data/test/066_forg/01_0101066.PNG  
      inflating: sign_data/sign_data/test/066_forg/01_0127066.PNG  
      inflating: sign_data/sign_data/test/066_forg/01_0211066.PNG  
      inflating: sign_data/sign_data/test/066_forg/01_0212066.PNG  
      inflating: sign_data/sign_data/test/066_forg/02_0101066.PNG  
      inflating: sign_data/sign_data/test/066_forg/02_0127066.PNG  
      inflating: sign_data/sign_data/test/066_forg/02_0211066.PNG  
      inflating: sign_data/sign_data/test/066_forg/02_0212066.PNG  
      inflating: sign_data/sign_data/test/066_forg/03_0101066.PNG  
      inflating: sign_data/sign_data/test/066_forg/03_0127066.PNG  
      inflating: sign_data/sign_data/test/066_forg/03_0211066.PNG  
      inflating: sign_data/sign_data/test/066_forg/03_0212066.PNG  
      inflating: sign_data/sign_data/test/066_forg/04_0101066.PNG  
      inflating: sign_data/sign_data/test/066_forg/04_0127066.PNG  
      inflating: sign_data/sign_data/test/066_forg/04_0211066.PNG  
      inflating: sign_data/sign_data/test/066_forg/04_0212066.PNG  
      inflating: sign_data/sign_data/test/067/01_067.png  
      inflating: sign_data/sign_data/test/067/02_067.png  
      inflating: sign_data/sign_data/test/067/03_067.png  
      inflating: sign_data/sign_data/test/067/04_067.png  
      inflating: sign_data/sign_data/test/067/05_067.png  
      inflating: sign_data/sign_data/test/067/06_067.png  
      inflating: sign_data/sign_data/test/067/07_067.png  
      inflating: sign_data/sign_data/test/067/08_067.png  
      inflating: sign_data/sign_data/test/067/09_067.png  
      inflating: sign_data/sign_data/test/067/10_067.png  
      inflating: sign_data/sign_data/test/067/11_067.png  
      inflating: sign_data/sign_data/test/067/12_067.png  
      inflating: sign_data/sign_data/test/067_forg/01_0205067.PNG  
      inflating: sign_data/sign_data/test/067_forg/01_0212067.PNG  
      inflating: sign_data/sign_data/test/067_forg/02_0205067.PNG  
      inflating: sign_data/sign_data/test/067_forg/02_0212067.PNG  
      inflating: sign_data/sign_data/test/067_forg/03_0205067.PNG  
      inflating: sign_data/sign_data/test/067_forg/03_0212067.PNG  
      inflating: sign_data/sign_data/test/067_forg/04_0205067.PNG  
      inflating: sign_data/sign_data/test/067_forg/04_0212067.PNG  
      inflating: sign_data/sign_data/test/068/01_068.png  
      inflating: sign_data/sign_data/test/068/02_068.png  
      inflating: sign_data/sign_data/test/068/03_068.png  
      inflating: sign_data/sign_data/test/068/04_068.png  
      inflating: sign_data/sign_data/test/068/05_068.png  
      inflating: sign_data/sign_data/test/068/06_068.png  
      inflating: sign_data/sign_data/test/068/07_068.png  
      inflating: sign_data/sign_data/test/068/08_068.png  
      inflating: sign_data/sign_data/test/068/09_068.png  
      inflating: sign_data/sign_data/test/068/10_068.png  
      inflating: sign_data/sign_data/test/068/11_068.png  
      inflating: sign_data/sign_data/test/068/12_068.png  
      inflating: sign_data/sign_data/test/068_forg/01_0113068.PNG  
      inflating: sign_data/sign_data/test/068_forg/01_0124068.PNG  
      inflating: sign_data/sign_data/test/068_forg/02_0113068.PNG  
      inflating: sign_data/sign_data/test/068_forg/02_0124068.PNG  
      inflating: sign_data/sign_data/test/068_forg/03_0113068.PNG  
      inflating: sign_data/sign_data/test/068_forg/03_0124068.PNG  
      inflating: sign_data/sign_data/test/068_forg/04_0113068.PNG  
      inflating: sign_data/sign_data/test/068_forg/04_0124068.PNG  
      inflating: sign_data/sign_data/test/069/01_069.png  
      inflating: sign_data/sign_data/test/069/02_069.png  
      inflating: sign_data/sign_data/test/069/03_069.png  
      inflating: sign_data/sign_data/test/069/04_069.png  
      inflating: sign_data/sign_data/test/069/05_069.png  
      inflating: sign_data/sign_data/test/069/06_069.png  
      inflating: sign_data/sign_data/test/069/07_069.png  
      inflating: sign_data/sign_data/test/069/08_069.png  
      inflating: sign_data/sign_data/test/069/09_069.png  
      inflating: sign_data/sign_data/test/069/10_069.png  
      inflating: sign_data/sign_data/test/069/11_069.png  
      inflating: sign_data/sign_data/test/069/12_069.png  
      inflating: sign_data/sign_data/test/069_forg/01_0106069.PNG  
      inflating: sign_data/sign_data/test/069_forg/01_0108069.PNG  
      inflating: sign_data/sign_data/test/069_forg/01_0111069.PNG  
      inflating: sign_data/sign_data/test/069_forg/02_0106069.PNG  
      inflating: sign_data/sign_data/test/069_forg/02_0108069.PNG  
      inflating: sign_data/sign_data/test/069_forg/02_0111069.PNG  
      inflating: sign_data/sign_data/test/069_forg/03_0106069.PNG  
      inflating: sign_data/sign_data/test/069_forg/03_0108069.PNG  
      inflating: sign_data/sign_data/test/069_forg/03_0111069.PNG  
      inflating: sign_data/sign_data/test/069_forg/04_0106069.PNG  
      inflating: sign_data/sign_data/test/069_forg/04_0108069.PNG  
      inflating: sign_data/sign_data/test/069_forg/04_0111069.PNG  
      inflating: sign_data/sign_data/test_data.csv  
      inflating: sign_data/sign_data/train/001/001_01.PNG  
      inflating: sign_data/sign_data/train/001/001_02.PNG  
      inflating: sign_data/sign_data/train/001/001_03.PNG  
      inflating: sign_data/sign_data/train/001/001_04.PNG  
      inflating: sign_data/sign_data/train/001/001_05.PNG  
      inflating: sign_data/sign_data/train/001/001_06.PNG  
      inflating: sign_data/sign_data/train/001/001_07.PNG  
      inflating: sign_data/sign_data/train/001/001_08.PNG  
      inflating: sign_data/sign_data/train/001/001_09.PNG  
      inflating: sign_data/sign_data/train/001/001_10.PNG  
      inflating: sign_data/sign_data/train/001/001_11.PNG  
      inflating: sign_data/sign_data/train/001/001_12.PNG  
      inflating: sign_data/sign_data/train/001/001_13.PNG  
      inflating: sign_data/sign_data/train/001/001_14.PNG  
      inflating: sign_data/sign_data/train/001/001_15.PNG  
      inflating: sign_data/sign_data/train/001/001_16.PNG  
      inflating: sign_data/sign_data/train/001/001_17.PNG  
      inflating: sign_data/sign_data/train/001/001_18.PNG  
      inflating: sign_data/sign_data/train/001/001_19.PNG  
      inflating: sign_data/sign_data/train/001/001_20.PNG  
      inflating: sign_data/sign_data/train/001/001_21.PNG  
      inflating: sign_data/sign_data/train/001/001_22.PNG  
      inflating: sign_data/sign_data/train/001/001_23.PNG  
      inflating: sign_data/sign_data/train/001/001_24.PNG  
      inflating: sign_data/sign_data/train/001_forg/0119001_01.png  
      inflating: sign_data/sign_data/train/001_forg/0119001_02.png  
      inflating: sign_data/sign_data/train/001_forg/0119001_03.png  
      inflating: sign_data/sign_data/train/001_forg/0119001_04.png  
      inflating: sign_data/sign_data/train/001_forg/0201001_01.png  
      inflating: sign_data/sign_data/train/001_forg/0201001_02.png  
      inflating: sign_data/sign_data/train/001_forg/0201001_03.png  
      inflating: sign_data/sign_data/train/001_forg/0201001_04.png  
      inflating: sign_data/sign_data/train/002/002_01.PNG  
      inflating: sign_data/sign_data/train/002/002_02.PNG  
      inflating: sign_data/sign_data/train/002/002_03.PNG  
      inflating: sign_data/sign_data/train/002/002_04.PNG  
      inflating: sign_data/sign_data/train/002/002_05.PNG  
      inflating: sign_data/sign_data/train/002/002_06.PNG  
      inflating: sign_data/sign_data/train/002/002_07.PNG  
      inflating: sign_data/sign_data/train/002/002_08.PNG  
      inflating: sign_data/sign_data/train/002/002_09.PNG  
      inflating: sign_data/sign_data/train/002/002_10.PNG  
      inflating: sign_data/sign_data/train/002/002_11.PNG  
      inflating: sign_data/sign_data/train/002/002_12.PNG  
      inflating: sign_data/sign_data/train/002/002_13.PNG  
      inflating: sign_data/sign_data/train/002/002_14.PNG  
      inflating: sign_data/sign_data/train/002/002_15.PNG  
      inflating: sign_data/sign_data/train/002/002_16.PNG  
      inflating: sign_data/sign_data/train/002/002_17.PNG  
      inflating: sign_data/sign_data/train/002/002_18.PNG  
      inflating: sign_data/sign_data/train/002/002_19.PNG  
      inflating: sign_data/sign_data/train/002/002_20.PNG  
      inflating: sign_data/sign_data/train/002/002_21.PNG  
      inflating: sign_data/sign_data/train/002/002_22.PNG  
      inflating: sign_data/sign_data/train/002/002_23.PNG  
      inflating: sign_data/sign_data/train/002/002_24.PNG  
      inflating: sign_data/sign_data/train/002_forg/0108002_01.png  
      inflating: sign_data/sign_data/train/002_forg/0108002_02.png  
      inflating: sign_data/sign_data/train/002_forg/0108002_03.png  
      inflating: sign_data/sign_data/train/002_forg/0108002_04.png  
      inflating: sign_data/sign_data/train/002_forg/0110002_01.png  
      inflating: sign_data/sign_data/train/002_forg/0110002_02.png  
      inflating: sign_data/sign_data/train/002_forg/0110002_03.png  
      inflating: sign_data/sign_data/train/002_forg/0110002_04.png  
      inflating: sign_data/sign_data/train/002_forg/0118002_01.png  
      inflating: sign_data/sign_data/train/002_forg/0118002_02.png  
      inflating: sign_data/sign_data/train/002_forg/0118002_03.png  
      inflating: sign_data/sign_data/train/002_forg/0118002_04.png  
      inflating: sign_data/sign_data/train/003/003_01.PNG  
      inflating: sign_data/sign_data/train/003/003_02.PNG  
      inflating: sign_data/sign_data/train/003/003_03.PNG  
      inflating: sign_data/sign_data/train/003/003_04.PNG  
      inflating: sign_data/sign_data/train/003/003_05.PNG  
      inflating: sign_data/sign_data/train/003/003_06.PNG  
      inflating: sign_data/sign_data/train/003/003_07.PNG  
      inflating: sign_data/sign_data/train/003/003_08.PNG  
      inflating: sign_data/sign_data/train/003/003_09.PNG  
      inflating: sign_data/sign_data/train/003/003_10.PNG  
      inflating: sign_data/sign_data/train/003/003_11.PNG  
      inflating: sign_data/sign_data/train/003/003_12.PNG  
      inflating: sign_data/sign_data/train/003/003_13.PNG  
      inflating: sign_data/sign_data/train/003/003_14.PNG  
      inflating: sign_data/sign_data/train/003/003_15.PNG  
      inflating: sign_data/sign_data/train/003/003_16.PNG  
      inflating: sign_data/sign_data/train/003/003_17.PNG  
      inflating: sign_data/sign_data/train/003/003_18.PNG  
      inflating: sign_data/sign_data/train/003/003_19.PNG  
      inflating: sign_data/sign_data/train/003/003_20.PNG  
      inflating: sign_data/sign_data/train/003/003_21.PNG  
      inflating: sign_data/sign_data/train/003/003_22.PNG  
      inflating: sign_data/sign_data/train/003/003_23.PNG  
      inflating: sign_data/sign_data/train/003/003_24.PNG  
      inflating: sign_data/sign_data/train/003_forg/0121003_01.png  
      inflating: sign_data/sign_data/train/003_forg/0121003_02.png  
      inflating: sign_data/sign_data/train/003_forg/0121003_03.png  
      inflating: sign_data/sign_data/train/003_forg/0121003_04.png  
      inflating: sign_data/sign_data/train/003_forg/0126003_01.png  
      inflating: sign_data/sign_data/train/003_forg/0126003_02.png  
      inflating: sign_data/sign_data/train/003_forg/0126003_03.png  
      inflating: sign_data/sign_data/train/003_forg/0126003_04.png  
      inflating: sign_data/sign_data/train/003_forg/0206003_01.png  
      inflating: sign_data/sign_data/train/003_forg/0206003_02.png  
      inflating: sign_data/sign_data/train/003_forg/0206003_03.png  
      inflating: sign_data/sign_data/train/003_forg/0206003_04.png  
      inflating: sign_data/sign_data/train/004/004_01.PNG  
      inflating: sign_data/sign_data/train/004/004_02.PNG  
      inflating: sign_data/sign_data/train/004/004_03.PNG  
      inflating: sign_data/sign_data/train/004/004_04.PNG  
      inflating: sign_data/sign_data/train/004/004_05.PNG  
      inflating: sign_data/sign_data/train/004/004_06.PNG  
      inflating: sign_data/sign_data/train/004/004_07.PNG  
      inflating: sign_data/sign_data/train/004/004_08.PNG  
      inflating: sign_data/sign_data/train/004/004_09.PNG  
      inflating: sign_data/sign_data/train/004/004_10.PNG  
      inflating: sign_data/sign_data/train/004/004_11.PNG  
      inflating: sign_data/sign_data/train/004/004_12.PNG  
      inflating: sign_data/sign_data/train/004/004_13.PNG  
      inflating: sign_data/sign_data/train/004/004_14.PNG  
      inflating: sign_data/sign_data/train/004/004_15.PNG  
      inflating: sign_data/sign_data/train/004/004_16.PNG  
      inflating: sign_data/sign_data/train/004/004_17.PNG  
      inflating: sign_data/sign_data/train/004/004_18.PNG  
      inflating: sign_data/sign_data/train/004/004_19.PNG  
      inflating: sign_data/sign_data/train/004/004_20.PNG  
      inflating: sign_data/sign_data/train/004/004_21.PNG  
      inflating: sign_data/sign_data/train/004/004_22.PNG  
      inflating: sign_data/sign_data/train/004/004_23.PNG  
      inflating: sign_data/sign_data/train/004/004_24.PNG  
      inflating: sign_data/sign_data/train/004_forg/0103004_02.png  
      inflating: sign_data/sign_data/train/004_forg/0103004_03.png  
      inflating: sign_data/sign_data/train/004_forg/0103004_04.png  
      inflating: sign_data/sign_data/train/004_forg/0105004_01.png  
      inflating: sign_data/sign_data/train/004_forg/0105004_02.png  
      inflating: sign_data/sign_data/train/004_forg/0105004_03.png  
      inflating: sign_data/sign_data/train/004_forg/0105004_04.png  
      inflating: sign_data/sign_data/train/004_forg/0124004_01.png  
      inflating: sign_data/sign_data/train/004_forg/0124004_02.png  
      inflating: sign_data/sign_data/train/004_forg/0124004_03.png  
      inflating: sign_data/sign_data/train/004_forg/0124004_04.png  
      inflating: sign_data/sign_data/train/006/006_01.PNG  
      inflating: sign_data/sign_data/train/006/006_02.PNG  
      inflating: sign_data/sign_data/train/006/006_03.PNG  
      inflating: sign_data/sign_data/train/006/006_04.PNG  
      inflating: sign_data/sign_data/train/006/006_05.PNG  
      inflating: sign_data/sign_data/train/006/006_06.PNG  
      inflating: sign_data/sign_data/train/006/006_07.PNG  
      inflating: sign_data/sign_data/train/006/006_08.PNG  
      inflating: sign_data/sign_data/train/006/006_09.PNG  
      inflating: sign_data/sign_data/train/006/006_10.PNG  
      inflating: sign_data/sign_data/train/006/006_11.PNG  
      inflating: sign_data/sign_data/train/006/006_12.PNG  
      inflating: sign_data/sign_data/train/006/006_13.PNG  
      inflating: sign_data/sign_data/train/006/006_14.PNG  
      inflating: sign_data/sign_data/train/006/006_15.PNG  
      inflating: sign_data/sign_data/train/006/006_16.PNG  
      inflating: sign_data/sign_data/train/006/006_17.PNG  
      inflating: sign_data/sign_data/train/006/006_18.PNG  
      inflating: sign_data/sign_data/train/006/006_19.PNG  
      inflating: sign_data/sign_data/train/006/006_20.PNG  
      inflating: sign_data/sign_data/train/006/006_21.PNG  
      inflating: sign_data/sign_data/train/006/006_22.PNG  
      inflating: sign_data/sign_data/train/006/006_23.PNG  
      inflating: sign_data/sign_data/train/006/006_24.PNG  
      inflating: sign_data/sign_data/train/006_forg/0111006_01.png  
      inflating: sign_data/sign_data/train/006_forg/0111006_02.png  
      inflating: sign_data/sign_data/train/006_forg/0111006_03.png  
      inflating: sign_data/sign_data/train/006_forg/0111006_04.png  
      inflating: sign_data/sign_data/train/006_forg/0202006_01.png  
      inflating: sign_data/sign_data/train/006_forg/0202006_02.png  
      inflating: sign_data/sign_data/train/006_forg/0202006_03.png  
      inflating: sign_data/sign_data/train/006_forg/0202006_04.png  
      inflating: sign_data/sign_data/train/006_forg/0205006_01.png  
      inflating: sign_data/sign_data/train/006_forg/0205006_02.png  
      inflating: sign_data/sign_data/train/006_forg/0205006_03.png  
      inflating: sign_data/sign_data/train/006_forg/0205006_04.png  
      inflating: sign_data/sign_data/train/009/009_01.PNG  
      inflating: sign_data/sign_data/train/009/009_02.PNG  
      inflating: sign_data/sign_data/train/009/009_03.PNG  
      inflating: sign_data/sign_data/train/009/009_04.PNG  
      inflating: sign_data/sign_data/train/009/009_05.PNG  
      inflating: sign_data/sign_data/train/009/009_06.PNG  
      inflating: sign_data/sign_data/train/009/009_07.PNG  
      inflating: sign_data/sign_data/train/009/009_08.PNG  
      inflating: sign_data/sign_data/train/009/009_09.PNG  
      inflating: sign_data/sign_data/train/009/009_10.PNG  
      inflating: sign_data/sign_data/train/009/009_11.PNG  
      inflating: sign_data/sign_data/train/009/009_12.PNG  
      inflating: sign_data/sign_data/train/009/009_13.PNG  
      inflating: sign_data/sign_data/train/009/009_14.PNG  
      inflating: sign_data/sign_data/train/009/009_15.PNG  
      inflating: sign_data/sign_data/train/009/009_16.PNG  
      inflating: sign_data/sign_data/train/009/009_17.PNG  
      inflating: sign_data/sign_data/train/009/009_18.PNG  
      inflating: sign_data/sign_data/train/009/009_19.PNG  
      inflating: sign_data/sign_data/train/009/009_20.PNG  
      inflating: sign_data/sign_data/train/009/009_21.PNG  
      inflating: sign_data/sign_data/train/009/009_22.PNG  
      inflating: sign_data/sign_data/train/009/009_23.PNG  
      inflating: sign_data/sign_data/train/009/009_24.PNG  
      inflating: sign_data/sign_data/train/009_forg/0117009_01.png  
      inflating: sign_data/sign_data/train/009_forg/0117009_02.png  
      inflating: sign_data/sign_data/train/009_forg/0117009_03.png  
      inflating: sign_data/sign_data/train/009_forg/0117009_04.png  
      inflating: sign_data/sign_data/train/009_forg/0123009_01.png  
      inflating: sign_data/sign_data/train/009_forg/0123009_02.png  
      inflating: sign_data/sign_data/train/009_forg/0123009_03.png  
      inflating: sign_data/sign_data/train/009_forg/0123009_04.png  
      inflating: sign_data/sign_data/train/009_forg/0201009_01.png  
      inflating: sign_data/sign_data/train/009_forg/0201009_02.png  
      inflating: sign_data/sign_data/train/009_forg/0201009_03.png  
      inflating: sign_data/sign_data/train/009_forg/0201009_04.png  
      inflating: sign_data/sign_data/train/012/012_01.PNG  
      inflating: sign_data/sign_data/train/012/012_02.PNG  
      inflating: sign_data/sign_data/train/012/012_03.PNG  
      inflating: sign_data/sign_data/train/012/012_04.PNG  
      inflating: sign_data/sign_data/train/012/012_05.PNG  
      inflating: sign_data/sign_data/train/012/012_06.PNG  
      inflating: sign_data/sign_data/train/012/012_07.PNG  
      inflating: sign_data/sign_data/train/012/012_08.PNG  
      inflating: sign_data/sign_data/train/012/012_09.PNG  
      inflating: sign_data/sign_data/train/012/012_10.PNG  
      inflating: sign_data/sign_data/train/012/012_11.PNG  
      inflating: sign_data/sign_data/train/012/012_12.PNG  
      inflating: sign_data/sign_data/train/012/012_13.PNG  
      inflating: sign_data/sign_data/train/012/012_14.PNG  
      inflating: sign_data/sign_data/train/012/012_15.PNG  
      inflating: sign_data/sign_data/train/012/012_16.PNG  
      inflating: sign_data/sign_data/train/012/012_17.PNG  
      inflating: sign_data/sign_data/train/012/012_18.PNG  
      inflating: sign_data/sign_data/train/012/012_19.PNG  
      inflating: sign_data/sign_data/train/012/012_20.PNG  
      inflating: sign_data/sign_data/train/012/012_21.PNG  
      inflating: sign_data/sign_data/train/012/012_22.PNG  
      inflating: sign_data/sign_data/train/012/012_23.PNG  
      inflating: sign_data/sign_data/train/012/012_24.PNG  
      inflating: sign_data/sign_data/train/012_forg/0113012_01.png  
      inflating: sign_data/sign_data/train/012_forg/0113012_02.png  
      inflating: sign_data/sign_data/train/012_forg/0113012_03.png  
      inflating: sign_data/sign_data/train/012_forg/0113012_04.png  
      inflating: sign_data/sign_data/train/012_forg/0206012_01.png  
      inflating: sign_data/sign_data/train/012_forg/0206012_02.png  
      inflating: sign_data/sign_data/train/012_forg/0206012_03.png  
      inflating: sign_data/sign_data/train/012_forg/0206012_04.png  
      inflating: sign_data/sign_data/train/012_forg/0210012_01.png  
      inflating: sign_data/sign_data/train/012_forg/0210012_02.png  
      inflating: sign_data/sign_data/train/012_forg/0210012_03.png  
      inflating: sign_data/sign_data/train/012_forg/0210012_04.png  
      inflating: sign_data/sign_data/train/013/01_013.png  
      inflating: sign_data/sign_data/train/013/02_013.png  
      inflating: sign_data/sign_data/train/013/03_013.png  
      inflating: sign_data/sign_data/train/013/04_013.png  
      inflating: sign_data/sign_data/train/013/05_013.png  
      inflating: sign_data/sign_data/train/013/06_013.png  
      inflating: sign_data/sign_data/train/013/07_013.png  
      inflating: sign_data/sign_data/train/013/08_013.png  
      inflating: sign_data/sign_data/train/013/09_013.png  
      inflating: sign_data/sign_data/train/013/10_013.png  
      inflating: sign_data/sign_data/train/013/11_013.png  
      inflating: sign_data/sign_data/train/013/12_013.png  
      inflating: sign_data/sign_data/train/013_forg/01_0113013.PNG  
      inflating: sign_data/sign_data/train/013_forg/01_0203013.PNG  
      inflating: sign_data/sign_data/train/013_forg/01_0204013.PNG  
      inflating: sign_data/sign_data/train/013_forg/02_0113013.PNG  
      inflating: sign_data/sign_data/train/013_forg/02_0203013.PNG  
      inflating: sign_data/sign_data/train/013_forg/02_0204013.PNG  
      inflating: sign_data/sign_data/train/013_forg/03_0113013.PNG  
      inflating: sign_data/sign_data/train/013_forg/03_0203013.PNG  
      inflating: sign_data/sign_data/train/013_forg/03_0204013.PNG  
      inflating: sign_data/sign_data/train/013_forg/04_0113013.PNG  
      inflating: sign_data/sign_data/train/013_forg/04_0203013.PNG  
      inflating: sign_data/sign_data/train/013_forg/04_0204013.PNG  
      inflating: sign_data/sign_data/train/014/014_01.PNG  
      inflating: sign_data/sign_data/train/014/014_02.PNG  
      inflating: sign_data/sign_data/train/014/014_03.PNG  
      inflating: sign_data/sign_data/train/014/014_04.PNG  
      inflating: sign_data/sign_data/train/014/014_05.PNG  
      inflating: sign_data/sign_data/train/014/014_06.PNG  
      inflating: sign_data/sign_data/train/014/014_07.PNG  
      inflating: sign_data/sign_data/train/014/014_08.PNG  
      inflating: sign_data/sign_data/train/014/014_09.PNG  
      inflating: sign_data/sign_data/train/014/014_10.PNG  
      inflating: sign_data/sign_data/train/014/014_11.PNG  
      inflating: sign_data/sign_data/train/014/014_12.PNG  
      inflating: sign_data/sign_data/train/014/014_13.PNG  
      inflating: sign_data/sign_data/train/014/014_14.PNG  
      inflating: sign_data/sign_data/train/014/014_15.PNG  
      inflating: sign_data/sign_data/train/014/014_16.PNG  
      inflating: sign_data/sign_data/train/014/014_17.PNG  
      inflating: sign_data/sign_data/train/014/014_18.PNG  
      inflating: sign_data/sign_data/train/014/014_19.PNG  
      inflating: sign_data/sign_data/train/014/014_20.PNG  
      inflating: sign_data/sign_data/train/014/014_21.PNG  
      inflating: sign_data/sign_data/train/014/014_22.PNG  
      inflating: sign_data/sign_data/train/014/014_23.PNG  
      inflating: sign_data/sign_data/train/014/014_24.PNG  
      inflating: sign_data/sign_data/train/014_forg/0102014_01.png  
      inflating: sign_data/sign_data/train/014_forg/0102014_02.png  
      inflating: sign_data/sign_data/train/014_forg/0102014_03.png  
      inflating: sign_data/sign_data/train/014_forg/0102014_04.png  
      inflating: sign_data/sign_data/train/014_forg/0104014_01.png  
      inflating: sign_data/sign_data/train/014_forg/0104014_02.png  
      inflating: sign_data/sign_data/train/014_forg/0104014_03.png  
      inflating: sign_data/sign_data/train/014_forg/0104014_04.png  
      inflating: sign_data/sign_data/train/014_forg/0208014_01.png  
      inflating: sign_data/sign_data/train/014_forg/0208014_02.png  
      inflating: sign_data/sign_data/train/014_forg/0208014_03.png  
      inflating: sign_data/sign_data/train/014_forg/0208014_04.png  
      inflating: sign_data/sign_data/train/014_forg/0214014_01.png  
      inflating: sign_data/sign_data/train/014_forg/0214014_02.png  
      inflating: sign_data/sign_data/train/014_forg/0214014_03.png  
      inflating: sign_data/sign_data/train/014_forg/0214014_04.png  
      inflating: sign_data/sign_data/train/015/015_01.PNG  
      inflating: sign_data/sign_data/train/015/015_02.PNG  
      inflating: sign_data/sign_data/train/015/015_03.PNG  
      inflating: sign_data/sign_data/train/015/015_04.PNG  
      inflating: sign_data/sign_data/train/015/015_05.PNG  
      inflating: sign_data/sign_data/train/015/015_06.PNG  
      inflating: sign_data/sign_data/train/015/015_07.PNG  
      inflating: sign_data/sign_data/train/015/015_08.PNG  
      inflating: sign_data/sign_data/train/015/015_09.PNG  
      inflating: sign_data/sign_data/train/015/015_10.PNG  
      inflating: sign_data/sign_data/train/015/015_11.PNG  
      inflating: sign_data/sign_data/train/015/015_12.PNG  
      inflating: sign_data/sign_data/train/015/015_13.PNG  
      inflating: sign_data/sign_data/train/015/015_14.PNG  
      inflating: sign_data/sign_data/train/015/015_15.PNG  
      inflating: sign_data/sign_data/train/015/015_16.PNG  
      inflating: sign_data/sign_data/train/015/015_17.PNG  
      inflating: sign_data/sign_data/train/015/015_18.PNG  
      inflating: sign_data/sign_data/train/015/015_19.PNG  
      inflating: sign_data/sign_data/train/015/015_20.PNG  
      inflating: sign_data/sign_data/train/015/015_21.PNG  
      inflating: sign_data/sign_data/train/015/015_22.PNG  
      inflating: sign_data/sign_data/train/015/015_23.PNG  
      inflating: sign_data/sign_data/train/015/015_24.PNG  
      inflating: sign_data/sign_data/train/015_forg/0106015_01.png  
      inflating: sign_data/sign_data/train/015_forg/0106015_02.png  
      inflating: sign_data/sign_data/train/015_forg/0106015_03.png  
      inflating: sign_data/sign_data/train/015_forg/0106015_04.png  
      inflating: sign_data/sign_data/train/015_forg/0210015_01.png  
      inflating: sign_data/sign_data/train/015_forg/0210015_02.png  
      inflating: sign_data/sign_data/train/015_forg/0210015_03.png  
      inflating: sign_data/sign_data/train/015_forg/0210015_04.png  
      inflating: sign_data/sign_data/train/015_forg/0213015_01.png  
      inflating: sign_data/sign_data/train/015_forg/0213015_02.png  
      inflating: sign_data/sign_data/train/015_forg/0213015_03.png  
      inflating: sign_data/sign_data/train/015_forg/0213015_04.png  
      inflating: sign_data/sign_data/train/016/016_01.PNG  
      inflating: sign_data/sign_data/train/016/016_02.PNG  
      inflating: sign_data/sign_data/train/016/016_03.PNG  
      inflating: sign_data/sign_data/train/016/016_04.PNG  
      inflating: sign_data/sign_data/train/016/016_05.PNG  
      inflating: sign_data/sign_data/train/016/016_06.PNG  
      inflating: sign_data/sign_data/train/016/016_07.PNG  
      inflating: sign_data/sign_data/train/016/016_08.PNG  
      inflating: sign_data/sign_data/train/016/016_09.PNG  
      inflating: sign_data/sign_data/train/016/016_10.PNG  
      inflating: sign_data/sign_data/train/016/016_11.PNG  
      inflating: sign_data/sign_data/train/016/016_12.PNG  
      inflating: sign_data/sign_data/train/016/016_13.PNG  
      inflating: sign_data/sign_data/train/016/016_14.PNG  
      inflating: sign_data/sign_data/train/016/016_15.PNG  
      inflating: sign_data/sign_data/train/016/016_16.PNG  
      inflating: sign_data/sign_data/train/016/016_17.PNG  
      inflating: sign_data/sign_data/train/016/016_18.PNG  
      inflating: sign_data/sign_data/train/016/016_20.PNG  
      inflating: sign_data/sign_data/train/016/016_21.PNG  
      inflating: sign_data/sign_data/train/016/016_22.PNG  
      inflating: sign_data/sign_data/train/016/016_23.PNG  
      inflating: sign_data/sign_data/train/016/016_24.PNG  
      inflating: sign_data/sign_data/train/016_forg/0107016_01.png  
      inflating: sign_data/sign_data/train/016_forg/0107016_02.png  
      inflating: sign_data/sign_data/train/016_forg/0107016_03.png  
      inflating: sign_data/sign_data/train/016_forg/0107016_04.png  
      inflating: sign_data/sign_data/train/016_forg/0110016_01.png  
      inflating: sign_data/sign_data/train/016_forg/0110016_02.png  
      inflating: sign_data/sign_data/train/016_forg/0110016_03.png  
      inflating: sign_data/sign_data/train/016_forg/0110016_04.png  
      inflating: sign_data/sign_data/train/016_forg/0127016_01.png  
      inflating: sign_data/sign_data/train/016_forg/0127016_02.png  
      inflating: sign_data/sign_data/train/016_forg/0127016_03.png  
      inflating: sign_data/sign_data/train/016_forg/0127016_04.png  
      inflating: sign_data/sign_data/train/016_forg/0202016_01.png  
      inflating: sign_data/sign_data/train/016_forg/0202016_02.png  
      inflating: sign_data/sign_data/train/016_forg/0202016_03.png  
      inflating: sign_data/sign_data/train/016_forg/0202016_04.png  
      inflating: sign_data/sign_data/train/017/01_017.png  
      inflating: sign_data/sign_data/train/017/02_017.png  
      inflating: sign_data/sign_data/train/017/03_017.png  
      inflating: sign_data/sign_data/train/017/04_017.png  
      inflating: sign_data/sign_data/train/017/05_017.png  
      inflating: sign_data/sign_data/train/017/06_017.png  
      inflating: sign_data/sign_data/train/017/07_017.png  
      inflating: sign_data/sign_data/train/017/08_017.png  
      inflating: sign_data/sign_data/train/017/09_017.png  
      inflating: sign_data/sign_data/train/017/10_017.png  
      inflating: sign_data/sign_data/train/017/11_017.png  
      inflating: sign_data/sign_data/train/017/12_017.png  
      inflating: sign_data/sign_data/train/017_forg/01_0107017.PNG  
      inflating: sign_data/sign_data/train/017_forg/01_0124017.PNG  
      inflating: sign_data/sign_data/train/017_forg/01_0211017.PNG  
      inflating: sign_data/sign_data/train/017_forg/02_0107017.PNG  
      inflating: sign_data/sign_data/train/017_forg/02_0124017.PNG  
      inflating: sign_data/sign_data/train/017_forg/02_0211017.PNG  
      inflating: sign_data/sign_data/train/017_forg/03_0107017.PNG  
      inflating: sign_data/sign_data/train/017_forg/03_0124017.PNG  
      inflating: sign_data/sign_data/train/017_forg/03_0211017.PNG  
      inflating: sign_data/sign_data/train/017_forg/04_0107017.PNG  
      inflating: sign_data/sign_data/train/017_forg/04_0124017.PNG  
      inflating: sign_data/sign_data/train/017_forg/04_0211017.PNG  
      inflating: sign_data/sign_data/train/018/01_018.png  
      inflating: sign_data/sign_data/train/018/02_018.png  
      inflating: sign_data/sign_data/train/018/03_018.png  
      inflating: sign_data/sign_data/train/018/04_018.png  
      inflating: sign_data/sign_data/train/018/05_018.png  
      inflating: sign_data/sign_data/train/018/06_018.png  
      inflating: sign_data/sign_data/train/018/07_018.png  
      inflating: sign_data/sign_data/train/018/08_018.png  
      inflating: sign_data/sign_data/train/018/09_018.png  
      inflating: sign_data/sign_data/train/018/10_018.png  
      inflating: sign_data/sign_data/train/018/11_018.png  
      inflating: sign_data/sign_data/train/018/12_018.png  
      inflating: sign_data/sign_data/train/018_forg/01_0106018.PNG  
      inflating: sign_data/sign_data/train/018_forg/01_0112018.PNG  
      inflating: sign_data/sign_data/train/018_forg/01_0202018.PNG  
      inflating: sign_data/sign_data/train/018_forg/02_0106018.PNG  
      inflating: sign_data/sign_data/train/018_forg/02_0112018.PNG  
      inflating: sign_data/sign_data/train/018_forg/02_0202018.PNG  
      inflating: sign_data/sign_data/train/018_forg/03_0106018.PNG  
      inflating: sign_data/sign_data/train/018_forg/03_0112018.PNG  
      inflating: sign_data/sign_data/train/018_forg/03_0202018.PNG  
      inflating: sign_data/sign_data/train/018_forg/04_0106018.PNG  
      inflating: sign_data/sign_data/train/018_forg/04_0112018.PNG  
      inflating: sign_data/sign_data/train/018_forg/04_0202018.PNG  
      inflating: sign_data/sign_data/train/019/01_019.png  
      inflating: sign_data/sign_data/train/019/02_019.png  
      inflating: sign_data/sign_data/train/019/03_019.png  
      inflating: sign_data/sign_data/train/019/04_019.png  
      inflating: sign_data/sign_data/train/019/05_019.png  
      inflating: sign_data/sign_data/train/019/06_019.png  
      inflating: sign_data/sign_data/train/019/07_019.png  
      inflating: sign_data/sign_data/train/019/08_019.png  
      inflating: sign_data/sign_data/train/019/09_019.png  
      inflating: sign_data/sign_data/train/019/10_019.png  
      inflating: sign_data/sign_data/train/019/11_019.png  
      inflating: sign_data/sign_data/train/019/12_019.png  
      inflating: sign_data/sign_data/train/019_forg/01_0115019.PNG  
      inflating: sign_data/sign_data/train/019_forg/01_0116019.PNG  
      inflating: sign_data/sign_data/train/019_forg/01_0119019.PNG  
      inflating: sign_data/sign_data/train/019_forg/02_0115019.PNG  
      inflating: sign_data/sign_data/train/019_forg/02_0116019.PNG  
      inflating: sign_data/sign_data/train/019_forg/02_0119019.PNG  
      inflating: sign_data/sign_data/train/019_forg/03_0115019.PNG  
      inflating: sign_data/sign_data/train/019_forg/03_0116019.PNG  
      inflating: sign_data/sign_data/train/019_forg/03_0119019.PNG  
      inflating: sign_data/sign_data/train/019_forg/04_0115019.PNG  
      inflating: sign_data/sign_data/train/019_forg/04_0116019.PNG  
      inflating: sign_data/sign_data/train/019_forg/04_0119019.PNG  
      inflating: sign_data/sign_data/train/020/01_020.png  
      inflating: sign_data/sign_data/train/020/02_020.png  
      inflating: sign_data/sign_data/train/020/03_020.png  
      inflating: sign_data/sign_data/train/020/04_020.png  
      inflating: sign_data/sign_data/train/020/05_020.png  
      inflating: sign_data/sign_data/train/020/06_020.png  
      inflating: sign_data/sign_data/train/020/07_020.png  
      inflating: sign_data/sign_data/train/020/08_020.png  
      inflating: sign_data/sign_data/train/020/09_020.png  
      inflating: sign_data/sign_data/train/020/10_020.png  
      inflating: sign_data/sign_data/train/020/11_020.png  
      inflating: sign_data/sign_data/train/020/12_020.png  
      inflating: sign_data/sign_data/train/020_forg/01_0105020.PNG  
      inflating: sign_data/sign_data/train/020_forg/01_0117020.PNG  
      inflating: sign_data/sign_data/train/020_forg/01_0127020.PNG  
      inflating: sign_data/sign_data/train/020_forg/01_0213020.PNG  
      inflating: sign_data/sign_data/train/020_forg/02_0101020.PNG  
      inflating: sign_data/sign_data/train/020_forg/02_0105020.PNG  
      inflating: sign_data/sign_data/train/020_forg/02_0117020.PNG  
      inflating: sign_data/sign_data/train/020_forg/02_0127020.PNG  
      inflating: sign_data/sign_data/train/020_forg/02_0213020.PNG  
      inflating: sign_data/sign_data/train/020_forg/03_0101020.PNG  
      inflating: sign_data/sign_data/train/020_forg/03_0105020.PNG  
      inflating: sign_data/sign_data/train/020_forg/03_0117020.PNG  
      inflating: sign_data/sign_data/train/020_forg/03_0127020.PNG  
      inflating: sign_data/sign_data/train/020_forg/03_0213020.PNG  
      inflating: sign_data/sign_data/train/020_forg/04_0101020.PNG  
      inflating: sign_data/sign_data/train/020_forg/04_0105020.PNG  
      inflating: sign_data/sign_data/train/020_forg/04_0117020.PNG  
      inflating: sign_data/sign_data/train/020_forg/04_0127020.PNG  
      inflating: sign_data/sign_data/train/020_forg/04_0213020.PNG  
      inflating: sign_data/sign_data/train/021/01_021.png  
      inflating: sign_data/sign_data/train/021/02_021.png  
      inflating: sign_data/sign_data/train/021/03_021.png  
      inflating: sign_data/sign_data/train/021/04_021.png  
      inflating: sign_data/sign_data/train/021/05_021.png  
      inflating: sign_data/sign_data/train/021/06_021.png  
      inflating: sign_data/sign_data/train/021/07_021.png  
      inflating: sign_data/sign_data/train/021/08_021.png  
      inflating: sign_data/sign_data/train/021/09_021.png  
      inflating: sign_data/sign_data/train/021/10_021.png  
      inflating: sign_data/sign_data/train/021/11_021.png  
      inflating: sign_data/sign_data/train/021/12_021.png  
      inflating: sign_data/sign_data/train/021_forg/01_0110021.PNG  
      inflating: sign_data/sign_data/train/021_forg/01_0204021.PNG  
      inflating: sign_data/sign_data/train/021_forg/01_0211021.PNG  
      inflating: sign_data/sign_data/train/021_forg/02_0110021.PNG  
      inflating: sign_data/sign_data/train/021_forg/02_0204021.PNG  
      inflating: sign_data/sign_data/train/021_forg/02_0211021.PNG  
      inflating: sign_data/sign_data/train/021_forg/03_0110021.PNG  
      inflating: sign_data/sign_data/train/021_forg/03_0204021.PNG  
      inflating: sign_data/sign_data/train/021_forg/03_0211021.PNG  
      inflating: sign_data/sign_data/train/021_forg/04_0110021.PNG  
      inflating: sign_data/sign_data/train/021_forg/04_0204021.PNG  
      inflating: sign_data/sign_data/train/021_forg/04_0211021.PNG  
      inflating: sign_data/sign_data/train/022/01_022.png  
      inflating: sign_data/sign_data/train/022/02_022.png  
      inflating: sign_data/sign_data/train/022/03_022.png  
      inflating: sign_data/sign_data/train/022/04_022.png  
      inflating: sign_data/sign_data/train/022/05_022.png  
      inflating: sign_data/sign_data/train/022/06_022.png  
      inflating: sign_data/sign_data/train/022/07_022.png  
      inflating: sign_data/sign_data/train/022/08_022.png  
      inflating: sign_data/sign_data/train/022/09_022.png  
      inflating: sign_data/sign_data/train/022/10_022.png  
      inflating: sign_data/sign_data/train/022/11_022.png  
      inflating: sign_data/sign_data/train/022/12_022.png  
      inflating: sign_data/sign_data/train/022_forg/01_0125022.PNG  
      inflating: sign_data/sign_data/train/022_forg/01_0127022.PNG  
      inflating: sign_data/sign_data/train/022_forg/01_0208022.PNG  
      inflating: sign_data/sign_data/train/022_forg/01_0214022.PNG  
      inflating: sign_data/sign_data/train/022_forg/02_0125022.PNG  
      inflating: sign_data/sign_data/train/022_forg/02_0127022.PNG  
      inflating: sign_data/sign_data/train/022_forg/02_0208022.PNG  
      inflating: sign_data/sign_data/train/022_forg/02_0214022.PNG  
      inflating: sign_data/sign_data/train/022_forg/03_0125022.PNG  
      inflating: sign_data/sign_data/train/022_forg/03_0127022.PNG  
      inflating: sign_data/sign_data/train/022_forg/03_0208022.PNG  
      inflating: sign_data/sign_data/train/022_forg/03_0214022.PNG  
      inflating: sign_data/sign_data/train/022_forg/04_0125022.PNG  
      inflating: sign_data/sign_data/train/022_forg/04_0127022.PNG  
      inflating: sign_data/sign_data/train/022_forg/04_0208022.PNG  
      inflating: sign_data/sign_data/train/022_forg/04_0214022.PNG  
      inflating: sign_data/sign_data/train/023/01_023.png  
      inflating: sign_data/sign_data/train/023/02_023.png  
      inflating: sign_data/sign_data/train/023/03_023.png  
      inflating: sign_data/sign_data/train/023/04_023.png  
      inflating: sign_data/sign_data/train/023/05_023.png  
      inflating: sign_data/sign_data/train/023/06_023.png  
      inflating: sign_data/sign_data/train/023/07_023.png  
      inflating: sign_data/sign_data/train/023/08_023.png  
      inflating: sign_data/sign_data/train/023/09_023.png  
      inflating: sign_data/sign_data/train/023/10_023.png  
      inflating: sign_data/sign_data/train/023/11_023.png  
      inflating: sign_data/sign_data/train/023/12_023.png  
      inflating: sign_data/sign_data/train/023_forg/01_0126023.PNG  
      inflating: sign_data/sign_data/train/023_forg/01_0203023.PNG  
      inflating: sign_data/sign_data/train/023_forg/02_0126023.PNG  
      inflating: sign_data/sign_data/train/023_forg/02_0203023.PNG  
      inflating: sign_data/sign_data/train/023_forg/03_0126023.PNG  
      inflating: sign_data/sign_data/train/023_forg/03_0203023.PNG  
      inflating: sign_data/sign_data/train/023_forg/04_0126023.PNG  
      inflating: sign_data/sign_data/train/023_forg/04_0203023.PNG  
      inflating: sign_data/sign_data/train/024/01_024.png  
      inflating: sign_data/sign_data/train/024/02_024.png  
      inflating: sign_data/sign_data/train/024/03_024.png  
      inflating: sign_data/sign_data/train/024/04_024.png  
      inflating: sign_data/sign_data/train/024/05_024.png  
      inflating: sign_data/sign_data/train/024/06_024.png  
      inflating: sign_data/sign_data/train/024/07_024.png  
      inflating: sign_data/sign_data/train/024/08_024.png  
      inflating: sign_data/sign_data/train/024/09_024.png  
      inflating: sign_data/sign_data/train/024/10_024.png  
      inflating: sign_data/sign_data/train/024/11_024.png  
      inflating: sign_data/sign_data/train/024/12_024.png  
      inflating: sign_data/sign_data/train/024_forg/01_0102024.PNG  
      inflating: sign_data/sign_data/train/024_forg/01_0119024.PNG  
      inflating: sign_data/sign_data/train/024_forg/01_0120024.PNG  
      inflating: sign_data/sign_data/train/024_forg/02_0102024.PNG  
      inflating: sign_data/sign_data/train/024_forg/02_0119024.PNG  
      inflating: sign_data/sign_data/train/024_forg/02_0120024.PNG  
      inflating: sign_data/sign_data/train/024_forg/03_0102024.PNG  
      inflating: sign_data/sign_data/train/024_forg/03_0119024.PNG  
      inflating: sign_data/sign_data/train/024_forg/03_0120024.PNG  
      inflating: sign_data/sign_data/train/024_forg/04_0102024.PNG  
      inflating: sign_data/sign_data/train/024_forg/04_0119024.PNG  
      inflating: sign_data/sign_data/train/024_forg/04_0120024.PNG  
      inflating: sign_data/sign_data/train/025/01_025.png  
      inflating: sign_data/sign_data/train/025/02_025.png  
      inflating: sign_data/sign_data/train/025/03_025.png  
      inflating: sign_data/sign_data/train/025/04_025.png  
      inflating: sign_data/sign_data/train/025/05_025.png  
      inflating: sign_data/sign_data/train/025/06_025.png  
      inflating: sign_data/sign_data/train/025/07_025.png  
      inflating: sign_data/sign_data/train/025/08_025.png  
      inflating: sign_data/sign_data/train/025/09_025.png  
      inflating: sign_data/sign_data/train/025/10_025.png  
      inflating: sign_data/sign_data/train/025/11_025.png  
      inflating: sign_data/sign_data/train/025/12_025.png  
      inflating: sign_data/sign_data/train/025_forg/01_0116025.PNG  
      inflating: sign_data/sign_data/train/025_forg/01_0121025.PNG  
      inflating: sign_data/sign_data/train/025_forg/02_0116025.PNG  
      inflating: sign_data/sign_data/train/025_forg/02_0121025.PNG  
      inflating: sign_data/sign_data/train/025_forg/03_0116025.PNG  
      inflating: sign_data/sign_data/train/025_forg/03_0121025.PNG  
      inflating: sign_data/sign_data/train/025_forg/04_0116025.PNG  
      inflating: sign_data/sign_data/train/025_forg/04_0121025.PNG  
      inflating: sign_data/sign_data/train/026/01_026.png  
      inflating: sign_data/sign_data/train/026/02_026.png  
      inflating: sign_data/sign_data/train/026/03_026.png  
      inflating: sign_data/sign_data/train/026/04_026.png  
      inflating: sign_data/sign_data/train/026/05_026.png  
      inflating: sign_data/sign_data/train/026/06_026.png  
      inflating: sign_data/sign_data/train/026/07_026.png  
      inflating: sign_data/sign_data/train/026/08_026.png  
      inflating: sign_data/sign_data/train/026/09_026.png  
      inflating: sign_data/sign_data/train/026/10_026.png  
      inflating: sign_data/sign_data/train/026/11_026.png  
      inflating: sign_data/sign_data/train/026/12_026.png  
      inflating: sign_data/sign_data/train/026_forg/01_0119026.PNG  
      inflating: sign_data/sign_data/train/026_forg/01_0123026.PNG  
      inflating: sign_data/sign_data/train/026_forg/01_0125026.PNG  
      inflating: sign_data/sign_data/train/026_forg/02_0119026.PNG  
      inflating: sign_data/sign_data/train/026_forg/02_0123026.PNG  
      inflating: sign_data/sign_data/train/026_forg/02_0125026.PNG  
      inflating: sign_data/sign_data/train/026_forg/03_0119026.PNG  
      inflating: sign_data/sign_data/train/026_forg/03_0123026.PNG  
      inflating: sign_data/sign_data/train/026_forg/03_0125026.PNG  
      inflating: sign_data/sign_data/train/026_forg/04_0119026.PNG  
      inflating: sign_data/sign_data/train/026_forg/04_0123026.PNG  
      inflating: sign_data/sign_data/train/026_forg/04_0125026.PNG  
      inflating: sign_data/sign_data/train/027/01_027.png  
      inflating: sign_data/sign_data/train/027/02_027.png  
      inflating: sign_data/sign_data/train/027/03_027.png  
      inflating: sign_data/sign_data/train/027/04_027.png  
      inflating: sign_data/sign_data/train/027/05_027.png  
      inflating: sign_data/sign_data/train/027/06_027.png  
      inflating: sign_data/sign_data/train/027/07_027.png  
      inflating: sign_data/sign_data/train/027/08_027.png  
      inflating: sign_data/sign_data/train/027/09_027.png  
      inflating: sign_data/sign_data/train/027/10_027.png  
      inflating: sign_data/sign_data/train/027/11_027.png  
      inflating: sign_data/sign_data/train/027/12_027.png  
      inflating: sign_data/sign_data/train/027_forg/01_0101027.PNG  
      inflating: sign_data/sign_data/train/027_forg/01_0212027.PNG  
      inflating: sign_data/sign_data/train/027_forg/02_0101027.PNG  
      inflating: sign_data/sign_data/train/027_forg/02_0212027.PNG  
      inflating: sign_data/sign_data/train/027_forg/03_0101027.PNG  
      inflating: sign_data/sign_data/train/027_forg/03_0212027.PNG  
      inflating: sign_data/sign_data/train/027_forg/04_0101027.PNG  
      inflating: sign_data/sign_data/train/027_forg/04_0212027.PNG  
      inflating: sign_data/sign_data/train/028/01_028.png  
      inflating: sign_data/sign_data/train/028/02_028.png  
      inflating: sign_data/sign_data/train/028/03_028.png  
      inflating: sign_data/sign_data/train/028/04_028.png  
      inflating: sign_data/sign_data/train/028/05_028.png  
      inflating: sign_data/sign_data/train/028/06_028.png  
      inflating: sign_data/sign_data/train/028/07_028.png  
      inflating: sign_data/sign_data/train/028/08_028.png  
      inflating: sign_data/sign_data/train/028/09_028.png  
      inflating: sign_data/sign_data/train/028/10_028.png  
      inflating: sign_data/sign_data/train/028/11_028.png  
      inflating: sign_data/sign_data/train/028/12_028.png  
      inflating: sign_data/sign_data/train/028_forg/01_0126028.PNG  
      inflating: sign_data/sign_data/train/028_forg/01_0205028.PNG  
      inflating: sign_data/sign_data/train/028_forg/01_0212028.PNG  
      inflating: sign_data/sign_data/train/028_forg/02_0126028.PNG  
      inflating: sign_data/sign_data/train/028_forg/02_0205028.PNG  
      inflating: sign_data/sign_data/train/028_forg/02_0212028.PNG  
      inflating: sign_data/sign_data/train/028_forg/03_0126028.PNG  
      inflating: sign_data/sign_data/train/028_forg/03_0205028.PNG  
      inflating: sign_data/sign_data/train/028_forg/03_0212028.PNG  
      inflating: sign_data/sign_data/train/028_forg/04_0126028.PNG  
      inflating: sign_data/sign_data/train/028_forg/04_0205028.PNG  
      inflating: sign_data/sign_data/train/028_forg/04_0212028.PNG  
      inflating: sign_data/sign_data/train/029/01_029.png  
      inflating: sign_data/sign_data/train/029/02_029.png  
      inflating: sign_data/sign_data/train/029/03_029.png  
      inflating: sign_data/sign_data/train/029/04_029.png  
      inflating: sign_data/sign_data/train/029/05_029.png  
      inflating: sign_data/sign_data/train/029/06_029.png  
      inflating: sign_data/sign_data/train/029/07_029.png  
      inflating: sign_data/sign_data/train/029/08_029.png  
      inflating: sign_data/sign_data/train/029/09_029.png  
      inflating: sign_data/sign_data/train/029/10_029.png  
      inflating: sign_data/sign_data/train/029/11_029.png  
      inflating: sign_data/sign_data/train/029/12_029.png  
      inflating: sign_data/sign_data/train/029_forg/01_0104029.PNG  
      inflating: sign_data/sign_data/train/029_forg/01_0115029.PNG  
      inflating: sign_data/sign_data/train/029_forg/01_0203029.PNG  
      inflating: sign_data/sign_data/train/029_forg/02_0104029.PNG  
      inflating: sign_data/sign_data/train/029_forg/02_0115029.PNG  
      inflating: sign_data/sign_data/train/029_forg/02_0203029.PNG  
      inflating: sign_data/sign_data/train/029_forg/03_0104029.PNG  
      inflating: sign_data/sign_data/train/029_forg/03_0115029.PNG  
      inflating: sign_data/sign_data/train/029_forg/03_0203029.PNG  
      inflating: sign_data/sign_data/train/029_forg/04_0104029.PNG  
      inflating: sign_data/sign_data/train/029_forg/04_0115029.PNG  
      inflating: sign_data/sign_data/train/029_forg/04_0203029.PNG  
      inflating: sign_data/sign_data/train/030/01_030.png  
      inflating: sign_data/sign_data/train/030/02_030.png  
      inflating: sign_data/sign_data/train/030/03_030.png  
      inflating: sign_data/sign_data/train/030/04_030.png  
      inflating: sign_data/sign_data/train/030/05_030.png  
      inflating: sign_data/sign_data/train/030/06_030.png  
      inflating: sign_data/sign_data/train/030/07_030.png  
      inflating: sign_data/sign_data/train/030/08_030.png  
      inflating: sign_data/sign_data/train/030/09_030.png  
      inflating: sign_data/sign_data/train/030/10_030.png  
      inflating: sign_data/sign_data/train/030/11_030.png  
      inflating: sign_data/sign_data/train/030/12_030.png  
      inflating: sign_data/sign_data/train/030_forg/01_0109030.PNG  
      inflating: sign_data/sign_data/train/030_forg/01_0114030.PNG  
      inflating: sign_data/sign_data/train/030_forg/01_0213030.PNG  
      inflating: sign_data/sign_data/train/030_forg/02_0109030.PNG  
      inflating: sign_data/sign_data/train/030_forg/02_0114030.PNG  
      inflating: sign_data/sign_data/train/030_forg/02_0213030.PNG  
      inflating: sign_data/sign_data/train/030_forg/03_0109030.PNG  
      inflating: sign_data/sign_data/train/030_forg/03_0114030.PNG  
      inflating: sign_data/sign_data/train/030_forg/03_0213030.PNG  
      inflating: sign_data/sign_data/train/030_forg/04_0109030.PNG  
      inflating: sign_data/sign_data/train/030_forg/04_0114030.PNG  
      inflating: sign_data/sign_data/train/030_forg/04_0213030.PNG  
      inflating: sign_data/sign_data/train/031/01_031.png  
      inflating: sign_data/sign_data/train/031/02_031.png  
      inflating: sign_data/sign_data/train/031/03_031.png  
      inflating: sign_data/sign_data/train/031/04_031.png  
      inflating: sign_data/sign_data/train/031/05_031.png  
      inflating: sign_data/sign_data/train/031/06_031.png  
      inflating: sign_data/sign_data/train/031/07_031.png  
      inflating: sign_data/sign_data/train/031/08_031.png  
      inflating: sign_data/sign_data/train/031/09_031.png  
      inflating: sign_data/sign_data/train/031/10_031.png  
      inflating: sign_data/sign_data/train/031/11_031.png  
      inflating: sign_data/sign_data/train/031/12_031.png  
      inflating: sign_data/sign_data/train/031_forg/01_0103031.PNG  
      inflating: sign_data/sign_data/train/031_forg/01_0121031.PNG  
      inflating: sign_data/sign_data/train/031_forg/02_0103031.PNG  
      inflating: sign_data/sign_data/train/031_forg/02_0121031.PNG  
      inflating: sign_data/sign_data/train/031_forg/03_0103031.PNG  
      inflating: sign_data/sign_data/train/031_forg/03_0121031.PNG  
      inflating: sign_data/sign_data/train/031_forg/04_0103031.PNG  
      inflating: sign_data/sign_data/train/031_forg/04_0121031.PNG  
      inflating: sign_data/sign_data/train/032/01_032.png  
      inflating: sign_data/sign_data/train/032/02_032.png  
      inflating: sign_data/sign_data/train/032/03_032.png  
      inflating: sign_data/sign_data/train/032/04_032.png  
      inflating: sign_data/sign_data/train/032/05_032.png  
      inflating: sign_data/sign_data/train/032/06_032.png  
      inflating: sign_data/sign_data/train/032/07_032.png  
      inflating: sign_data/sign_data/train/032/08_032.png  
      inflating: sign_data/sign_data/train/032/09_032.png  
      inflating: sign_data/sign_data/train/032/10_032.png  
      inflating: sign_data/sign_data/train/032/11_032.png  
      inflating: sign_data/sign_data/train/032/12_032.png  
      inflating: sign_data/sign_data/train/032_forg/01_0112032.PNG  
      inflating: sign_data/sign_data/train/032_forg/01_0117032.PNG  
      inflating: sign_data/sign_data/train/032_forg/01_0120032.PNG  
      inflating: sign_data/sign_data/train/032_forg/02_0112032.PNG  
      inflating: sign_data/sign_data/train/032_forg/02_0117032.PNG  
      inflating: sign_data/sign_data/train/032_forg/02_0120032.PNG  
      inflating: sign_data/sign_data/train/032_forg/03_0112032.PNG  
      inflating: sign_data/sign_data/train/032_forg/03_0117032.PNG  
      inflating: sign_data/sign_data/train/032_forg/03_0120032.PNG  
      inflating: sign_data/sign_data/train/032_forg/04_0112032.PNG  
      inflating: sign_data/sign_data/train/032_forg/04_0117032.PNG  
      inflating: sign_data/sign_data/train/032_forg/04_0120032.PNG  
      inflating: sign_data/sign_data/train/033/01_033.png  
      inflating: sign_data/sign_data/train/033/02_033.png  
      inflating: sign_data/sign_data/train/033/03_033.png  
      inflating: sign_data/sign_data/train/033/04_033.png  
      inflating: sign_data/sign_data/train/033/05_033.png  
      inflating: sign_data/sign_data/train/033/06_033.png  
      inflating: sign_data/sign_data/train/033/07_033.png  
      inflating: sign_data/sign_data/train/033/08_033.png  
      inflating: sign_data/sign_data/train/033/09_033.png  
      inflating: sign_data/sign_data/train/033/10_033.png  
      inflating: sign_data/sign_data/train/033/11_033.png  
      inflating: sign_data/sign_data/train/033/12_033.png  
      inflating: sign_data/sign_data/train/033_forg/01_0112033.PNG  
      inflating: sign_data/sign_data/train/033_forg/01_0203033.PNG  
      inflating: sign_data/sign_data/train/033_forg/01_0205033.PNG  
      inflating: sign_data/sign_data/train/033_forg/01_0213033.PNG  
      inflating: sign_data/sign_data/train/033_forg/02_0112033.PNG  
      inflating: sign_data/sign_data/train/033_forg/02_0203033.PNG  
      inflating: sign_data/sign_data/train/033_forg/02_0205033.PNG  
      inflating: sign_data/sign_data/train/033_forg/02_0213033.PNG  
      inflating: sign_data/sign_data/train/033_forg/03_0112033.PNG  
      inflating: sign_data/sign_data/train/033_forg/03_0203033.PNG  
      inflating: sign_data/sign_data/train/033_forg/03_0205033.PNG  
      inflating: sign_data/sign_data/train/033_forg/03_0213033.PNG  
      inflating: sign_data/sign_data/train/033_forg/04_0112033.PNG  
      inflating: sign_data/sign_data/train/033_forg/04_0203033.PNG  
      inflating: sign_data/sign_data/train/033_forg/04_0205033.PNG  
      inflating: sign_data/sign_data/train/033_forg/04_0213033.PNG  
      inflating: sign_data/sign_data/train/034/01_034.png  
      inflating: sign_data/sign_data/train/034/02_034.png  
      inflating: sign_data/sign_data/train/034/03_034.png  
      inflating: sign_data/sign_data/train/034/04_034.png  
      inflating: sign_data/sign_data/train/034/05_034.png  
      inflating: sign_data/sign_data/train/034/06_034.png  
      inflating: sign_data/sign_data/train/034/07_034.png  
      inflating: sign_data/sign_data/train/034/08_034.png  
      inflating: sign_data/sign_data/train/034/09_034.png  
      inflating: sign_data/sign_data/train/034/10_034.png  
      inflating: sign_data/sign_data/train/034/11_034.png  
      inflating: sign_data/sign_data/train/034/12_034.png  
      inflating: sign_data/sign_data/train/034_forg/01_0103034.PNG  
      inflating: sign_data/sign_data/train/034_forg/01_0110034.PNG  
      inflating: sign_data/sign_data/train/034_forg/01_0120034.PNG  
      inflating: sign_data/sign_data/train/034_forg/02_0103034.PNG  
      inflating: sign_data/sign_data/train/034_forg/02_0110034.PNG  
      inflating: sign_data/sign_data/train/034_forg/02_0120034.PNG  
      inflating: sign_data/sign_data/train/034_forg/03_0103034.PNG  
      inflating: sign_data/sign_data/train/034_forg/03_0110034.PNG  
      inflating: sign_data/sign_data/train/034_forg/03_0120034.PNG  
      inflating: sign_data/sign_data/train/034_forg/04_0103034.PNG  
      inflating: sign_data/sign_data/train/034_forg/04_0110034.PNG  
      inflating: sign_data/sign_data/train/034_forg/04_0120034.PNG  
      inflating: sign_data/sign_data/train/035/01_035.png  
      inflating: sign_data/sign_data/train/035/02_035.png  
      inflating: sign_data/sign_data/train/035/03_035.png  
      inflating: sign_data/sign_data/train/035/04_035.png  
      inflating: sign_data/sign_data/train/035/05_035.png  
      inflating: sign_data/sign_data/train/035/06_035.png  
      inflating: sign_data/sign_data/train/035/07_035.png  
      inflating: sign_data/sign_data/train/035/08_035.png  
      inflating: sign_data/sign_data/train/035/09_035.png  
      inflating: sign_data/sign_data/train/035/10_035.png  
      inflating: sign_data/sign_data/train/035/11_035.png  
      inflating: sign_data/sign_data/train/035/12_035.png  
      inflating: sign_data/sign_data/train/035_forg/01_0103035.PNG  
      inflating: sign_data/sign_data/train/035_forg/01_0115035.PNG  
      inflating: sign_data/sign_data/train/035_forg/01_0201035.PNG  
      inflating: sign_data/sign_data/train/035_forg/02_0103035.PNG  
      inflating: sign_data/sign_data/train/035_forg/02_0115035.PNG  
      inflating: sign_data/sign_data/train/035_forg/02_0201035.PNG  
      inflating: sign_data/sign_data/train/035_forg/03_0103035.PNG  
      inflating: sign_data/sign_data/train/035_forg/03_0115035.PNG  
      inflating: sign_data/sign_data/train/035_forg/03_0201035.PNG  
      inflating: sign_data/sign_data/train/035_forg/04_0103035.PNG  
      inflating: sign_data/sign_data/train/035_forg/04_0115035.PNG  
      inflating: sign_data/sign_data/train/035_forg/04_0201035.PNG  
      inflating: sign_data/sign_data/train/036/01_036.png  
      inflating: sign_data/sign_data/train/036/02_036.png  
      inflating: sign_data/sign_data/train/036/03_036.png  
      inflating: sign_data/sign_data/train/036/04_036.png  
      inflating: sign_data/sign_data/train/036/05_036.png  
      inflating: sign_data/sign_data/train/036/06_036.png  
      inflating: sign_data/sign_data/train/036/07_036.png  
      inflating: sign_data/sign_data/train/036/08_036.png  
      inflating: sign_data/sign_data/train/036/09_036.png  
      inflating: sign_data/sign_data/train/036/10_036.png  
      inflating: sign_data/sign_data/train/036/11_036.png  
      inflating: sign_data/sign_data/train/036/12_036.png  
      inflating: sign_data/sign_data/train/036_forg/01_0109036.PNG  
      inflating: sign_data/sign_data/train/036_forg/01_0118036.PNG  
      inflating: sign_data/sign_data/train/036_forg/01_0123036.PNG  
      inflating: sign_data/sign_data/train/036_forg/02_0109036.PNG  
      inflating: sign_data/sign_data/train/036_forg/02_0118036.PNG  
      inflating: sign_data/sign_data/train/036_forg/02_0123036.PNG  
      inflating: sign_data/sign_data/train/036_forg/03_0109036.PNG  
      inflating: sign_data/sign_data/train/036_forg/03_0118036.PNG  
      inflating: sign_data/sign_data/train/036_forg/03_0123036.PNG  
      inflating: sign_data/sign_data/train/036_forg/04_0109036.PNG  
      inflating: sign_data/sign_data/train/036_forg/04_0118036.PNG  
      inflating: sign_data/sign_data/train/036_forg/04_0123036.PNG  
      inflating: sign_data/sign_data/train/037/01_037.png  
      inflating: sign_data/sign_data/train/037/02_037.png  
      inflating: sign_data/sign_data/train/037/03_037.png  
      inflating: sign_data/sign_data/train/037/04_037.png  
      inflating: sign_data/sign_data/train/037/05_037.png  
      inflating: sign_data/sign_data/train/037/06_037.png  
      inflating: sign_data/sign_data/train/037/07_037.png  
      inflating: sign_data/sign_data/train/037/08_037.png  
      inflating: sign_data/sign_data/train/037/09_037.png  
      inflating: sign_data/sign_data/train/037/10_037.png  
      inflating: sign_data/sign_data/train/037/11_037.png  
      inflating: sign_data/sign_data/train/037/12_037.png  
      inflating: sign_data/sign_data/train/037_forg/01_0114037.PNG  
      inflating: sign_data/sign_data/train/037_forg/01_0123037.PNG  
      inflating: sign_data/sign_data/train/037_forg/01_0208037.PNG  
      inflating: sign_data/sign_data/train/037_forg/01_0214037.PNG  
      inflating: sign_data/sign_data/train/037_forg/02_0114037.PNG  
      inflating: sign_data/sign_data/train/037_forg/02_0123037.PNG  
      inflating: sign_data/sign_data/train/037_forg/02_0208037.PNG  
      inflating: sign_data/sign_data/train/037_forg/02_0214037.PNG  
      inflating: sign_data/sign_data/train/037_forg/03_0114037.PNG  
      inflating: sign_data/sign_data/train/037_forg/03_0123037.PNG  
      inflating: sign_data/sign_data/train/037_forg/03_0208037.PNG  
      inflating: sign_data/sign_data/train/037_forg/03_0214037.PNG  
      inflating: sign_data/sign_data/train/037_forg/04_0114037.PNG  
      inflating: sign_data/sign_data/train/037_forg/04_0123037.PNG  
      inflating: sign_data/sign_data/train/037_forg/04_0208037.PNG  
      inflating: sign_data/sign_data/train/037_forg/04_0214037.PNG  
      inflating: sign_data/sign_data/train/038/01_038.png  
      inflating: sign_data/sign_data/train/038/02_038.png  
      inflating: sign_data/sign_data/train/038/03_038.png  
      inflating: sign_data/sign_data/train/038/04_038.png  
      inflating: sign_data/sign_data/train/038/05_038.png  
      inflating: sign_data/sign_data/train/038/06_038.png  
      inflating: sign_data/sign_data/train/038/07_038.png  
      inflating: sign_data/sign_data/train/038/08_038.png  
      inflating: sign_data/sign_data/train/038/09_038.png  
      inflating: sign_data/sign_data/train/038/10_038.png  
      inflating: sign_data/sign_data/train/038/11_038.png  
      inflating: sign_data/sign_data/train/038/12_038.png  
      inflating: sign_data/sign_data/train/038_forg/01_0101038.PNG  
      inflating: sign_data/sign_data/train/038_forg/01_0124038.PNG  
      inflating: sign_data/sign_data/train/038_forg/01_0213038.PNG  
      inflating: sign_data/sign_data/train/038_forg/02_0101038.PNG  
      inflating: sign_data/sign_data/train/038_forg/02_0124038.PNG  
      inflating: sign_data/sign_data/train/038_forg/02_0213038.PNG  
      inflating: sign_data/sign_data/train/038_forg/03_0101038.PNG  
      inflating: sign_data/sign_data/train/038_forg/03_0124038.PNG  
      inflating: sign_data/sign_data/train/038_forg/03_0213038.PNG  
      inflating: sign_data/sign_data/train/038_forg/04_0101038.PNG  
      inflating: sign_data/sign_data/train/038_forg/04_0124038.PNG  
      inflating: sign_data/sign_data/train/038_forg/04_0213038.PNG  
      inflating: sign_data/sign_data/train/039/01_039.png  
      inflating: sign_data/sign_data/train/039/02_039.png  
      inflating: sign_data/sign_data/train/039/03_039.png  
      inflating: sign_data/sign_data/train/039/04_039.png  
      inflating: sign_data/sign_data/train/039/05_039.png  
      inflating: sign_data/sign_data/train/039/06_039.png  
      inflating: sign_data/sign_data/train/039/07_039.png  
      inflating: sign_data/sign_data/train/039/08_039.png  
      inflating: sign_data/sign_data/train/039/09_039.png  
      inflating: sign_data/sign_data/train/039/10_039.png  
      inflating: sign_data/sign_data/train/039/11_039.png  
      inflating: sign_data/sign_data/train/039/12_039.png  
      inflating: sign_data/sign_data/train/039_forg/01_0102039.PNG  
      inflating: sign_data/sign_data/train/039_forg/01_0108039.PNG  
      inflating: sign_data/sign_data/train/039_forg/01_0113039.PNG  
      inflating: sign_data/sign_data/train/039_forg/02_0102039.PNG  
      inflating: sign_data/sign_data/train/039_forg/02_0108039.PNG  
      inflating: sign_data/sign_data/train/039_forg/02_0113039.PNG  
      inflating: sign_data/sign_data/train/039_forg/03_0102039.PNG  
      inflating: sign_data/sign_data/train/039_forg/03_0108039.PNG  
      inflating: sign_data/sign_data/train/039_forg/03_0113039.PNG  
      inflating: sign_data/sign_data/train/039_forg/04_0102039.PNG  
      inflating: sign_data/sign_data/train/039_forg/04_0108039.PNG  
      inflating: sign_data/sign_data/train/039_forg/04_0113039.PNG  
      inflating: sign_data/sign_data/train/040/01_040.png  
      inflating: sign_data/sign_data/train/040/02_040.png  
      inflating: sign_data/sign_data/train/040/03_040.png  
      inflating: sign_data/sign_data/train/040/04_040.png  
      inflating: sign_data/sign_data/train/040/05_040.png  
      inflating: sign_data/sign_data/train/040/06_040.png  
      inflating: sign_data/sign_data/train/040/07_040.png  
      inflating: sign_data/sign_data/train/040/08_040.png  
      inflating: sign_data/sign_data/train/040/09_040.png  
      inflating: sign_data/sign_data/train/040/10_040.png  
      inflating: sign_data/sign_data/train/040/11_040.png  
      inflating: sign_data/sign_data/train/040/12_040.png  
      inflating: sign_data/sign_data/train/040_forg/01_0114040.PNG  
      inflating: sign_data/sign_data/train/040_forg/01_0121040.PNG  
      inflating: sign_data/sign_data/train/040_forg/02_0114040.PNG  
      inflating: sign_data/sign_data/train/040_forg/02_0121040.PNG  
      inflating: sign_data/sign_data/train/040_forg/03_0114040.PNG  
      inflating: sign_data/sign_data/train/040_forg/03_0121040.PNG  
      inflating: sign_data/sign_data/train/040_forg/04_0114040.PNG  
      inflating: sign_data/sign_data/train/040_forg/04_0121040.PNG  
      inflating: sign_data/sign_data/train/041/01_041.png  
      inflating: sign_data/sign_data/train/041/02_041.png  
      inflating: sign_data/sign_data/train/041/03_041.png  
      inflating: sign_data/sign_data/train/041/04_041.png  
      inflating: sign_data/sign_data/train/041/05_041.png  
      inflating: sign_data/sign_data/train/041/06_041.png  
      inflating: sign_data/sign_data/train/041/07_041.png  
      inflating: sign_data/sign_data/train/041/08_041.png  
      inflating: sign_data/sign_data/train/041/09_041.png  
      inflating: sign_data/sign_data/train/041/10_041.png  
      inflating: sign_data/sign_data/train/041/11_041.png  
      inflating: sign_data/sign_data/train/041/12_041.png  
      inflating: sign_data/sign_data/train/041_forg/01_0105041.PNG  
      inflating: sign_data/sign_data/train/041_forg/01_0116041.PNG  
      inflating: sign_data/sign_data/train/041_forg/01_0117041.PNG  
      inflating: sign_data/sign_data/train/041_forg/02_0105041.PNG  
      inflating: sign_data/sign_data/train/041_forg/02_0116041.PNG  
      inflating: sign_data/sign_data/train/041_forg/02_0117041.PNG  
      inflating: sign_data/sign_data/train/041_forg/03_0105041.PNG  
      inflating: sign_data/sign_data/train/041_forg/03_0116041.PNG  
      inflating: sign_data/sign_data/train/041_forg/03_0117041.PNG  
      inflating: sign_data/sign_data/train/041_forg/04_0105041.PNG  
      inflating: sign_data/sign_data/train/041_forg/04_0116041.PNG  
      inflating: sign_data/sign_data/train/041_forg/04_0117041.PNG  
      inflating: sign_data/sign_data/train/042/01_042.png  
      inflating: sign_data/sign_data/train/042/02_042.png  
      inflating: sign_data/sign_data/train/042/03_042.png  
      inflating: sign_data/sign_data/train/042/04_042.png  
      inflating: sign_data/sign_data/train/042/05_042.png  
      inflating: sign_data/sign_data/train/042/06_042.png  
      inflating: sign_data/sign_data/train/042/07_042.png  
      inflating: sign_data/sign_data/train/042/08_042.png  
      inflating: sign_data/sign_data/train/042/09_042.png  
      inflating: sign_data/sign_data/train/042/10_042.png  
      inflating: sign_data/sign_data/train/042/11_042.png  
      inflating: sign_data/sign_data/train/042/12_042.png  
      inflating: sign_data/sign_data/train/042_forg/01_0107042.PNG  
      inflating: sign_data/sign_data/train/042_forg/01_0118042.PNG  
      inflating: sign_data/sign_data/train/042_forg/01_0204042.PNG  
      inflating: sign_data/sign_data/train/042_forg/02_0107042.PNG  
      inflating: sign_data/sign_data/train/042_forg/02_0118042.PNG  
      inflating: sign_data/sign_data/train/042_forg/02_0204042.PNG  
      inflating: sign_data/sign_data/train/042_forg/03_0107042.PNG  
      inflating: sign_data/sign_data/train/042_forg/03_0118042.PNG  
      inflating: sign_data/sign_data/train/042_forg/03_0204042.PNG  
      inflating: sign_data/sign_data/train/042_forg/04_0107042.PNG  
      inflating: sign_data/sign_data/train/042_forg/04_0118042.PNG  
      inflating: sign_data/sign_data/train/042_forg/04_0204042.PNG  
      inflating: sign_data/sign_data/train/043/01_043.png  
      inflating: sign_data/sign_data/train/043/02_043.png  
      inflating: sign_data/sign_data/train/043/03_043.png  
      inflating: sign_data/sign_data/train/043/04_043.png  
      inflating: sign_data/sign_data/train/043/05_043.png  
      inflating: sign_data/sign_data/train/043/06_043.png  
      inflating: sign_data/sign_data/train/043/07_043.png  
      inflating: sign_data/sign_data/train/043/08_043.png  
      inflating: sign_data/sign_data/train/043/09_043.png  
      inflating: sign_data/sign_data/train/043/10_043.png  
      inflating: sign_data/sign_data/train/043/11_043.png  
      inflating: sign_data/sign_data/train/043/12_043.png  
      inflating: sign_data/sign_data/train/043_forg/01_0111043.PNG  
      inflating: sign_data/sign_data/train/043_forg/01_0201043.PNG  
      inflating: sign_data/sign_data/train/043_forg/01_0211043.PNG  
      inflating: sign_data/sign_data/train/043_forg/02_0111043.PNG  
      inflating: sign_data/sign_data/train/043_forg/02_0201043.PNG  
      inflating: sign_data/sign_data/train/043_forg/02_0211043.PNG  
      inflating: sign_data/sign_data/train/043_forg/03_0111043.PNG  
      inflating: sign_data/sign_data/train/043_forg/03_0201043.PNG  
      inflating: sign_data/sign_data/train/043_forg/03_0211043.PNG  
      inflating: sign_data/sign_data/train/043_forg/04_0111043.PNG  
      inflating: sign_data/sign_data/train/043_forg/04_0201043.PNG  
      inflating: sign_data/sign_data/train/043_forg/04_0211043.PNG  
      inflating: sign_data/sign_data/train/044/01_044.png  
      inflating: sign_data/sign_data/train/044/02_044.png  
      inflating: sign_data/sign_data/train/044/03_044.png  
      inflating: sign_data/sign_data/train/044/04_044.png  
      inflating: sign_data/sign_data/train/044/05_044.png  
      inflating: sign_data/sign_data/train/044/06_044.png  
      inflating: sign_data/sign_data/train/044/07_044.png  
      inflating: sign_data/sign_data/train/044/08_044.png  
      inflating: sign_data/sign_data/train/044/09_044.png  
      inflating: sign_data/sign_data/train/044/10_044.png  
      inflating: sign_data/sign_data/train/044/11_044.png  
      inflating: sign_data/sign_data/train/044/12_044.png  
      inflating: sign_data/sign_data/train/044_forg/01_0103044.PNG  
      inflating: sign_data/sign_data/train/044_forg/01_0112044.PNG  
      inflating: sign_data/sign_data/train/044_forg/01_0211044.PNG  
      inflating: sign_data/sign_data/train/044_forg/02_0103044.PNG  
      inflating: sign_data/sign_data/train/044_forg/02_0112044.PNG  
      inflating: sign_data/sign_data/train/044_forg/02_0211044.PNG  
      inflating: sign_data/sign_data/train/044_forg/03_0103044.PNG  
      inflating: sign_data/sign_data/train/044_forg/03_0112044.PNG  
      inflating: sign_data/sign_data/train/044_forg/03_0211044.PNG  
      inflating: sign_data/sign_data/train/044_forg/04_0103044.PNG  
      inflating: sign_data/sign_data/train/044_forg/04_0112044.PNG  
      inflating: sign_data/sign_data/train/044_forg/04_0211044.PNG  
      inflating: sign_data/sign_data/train/045/01_045.png  
      inflating: sign_data/sign_data/train/045/02_045.png  
      inflating: sign_data/sign_data/train/045/03_045.png  
      inflating: sign_data/sign_data/train/045/04_045.png  
      inflating: sign_data/sign_data/train/045/05_045.png  
      inflating: sign_data/sign_data/train/045/06_045.png  
      inflating: sign_data/sign_data/train/045/07_045.png  
      inflating: sign_data/sign_data/train/045/08_045.png  
      inflating: sign_data/sign_data/train/045/09_045.png  
      inflating: sign_data/sign_data/train/045/10_045.png  
      inflating: sign_data/sign_data/train/045/11_045.png  
      inflating: sign_data/sign_data/train/045/12_045.png  
      inflating: sign_data/sign_data/train/045_forg/01_0111045.PNG  
      inflating: sign_data/sign_data/train/045_forg/01_0116045.PNG  
      inflating: sign_data/sign_data/train/045_forg/01_0205045.PNG  
      inflating: sign_data/sign_data/train/045_forg/02_0111045.PNG  
      inflating: sign_data/sign_data/train/045_forg/02_0116045.PNG  
      inflating: sign_data/sign_data/train/045_forg/02_0205045.PNG  
      inflating: sign_data/sign_data/train/045_forg/03_0111045.PNG  
      inflating: sign_data/sign_data/train/045_forg/03_0116045.PNG  
      inflating: sign_data/sign_data/train/045_forg/03_0205045.PNG  
      inflating: sign_data/sign_data/train/045_forg/04_0111045.PNG  
      inflating: sign_data/sign_data/train/045_forg/04_0116045.PNG  
      inflating: sign_data/sign_data/train/045_forg/04_0205045.PNG  
      inflating: sign_data/sign_data/train/046/01_046.png  
      inflating: sign_data/sign_data/train/046/02_046.png  
      inflating: sign_data/sign_data/train/046/03_046.png  
      inflating: sign_data/sign_data/train/046/04_046.png  
      inflating: sign_data/sign_data/train/046/05_046.png  
      inflating: sign_data/sign_data/train/046/06_046.png  
      inflating: sign_data/sign_data/train/046/07_046.png  
      inflating: sign_data/sign_data/train/046/08_046.png  
      inflating: sign_data/sign_data/train/046/09_046.png  
      inflating: sign_data/sign_data/train/046/10_046.png  
      inflating: sign_data/sign_data/train/046/11_046.png  
      inflating: sign_data/sign_data/train/046/12_046.png  
      inflating: sign_data/sign_data/train/046_forg/01_0107046.PNG  
      inflating: sign_data/sign_data/train/046_forg/01_0108046.PNG  
      inflating: sign_data/sign_data/train/046_forg/01_0123046.PNG  
      inflating: sign_data/sign_data/train/046_forg/02_0107046.PNG  
      inflating: sign_data/sign_data/train/046_forg/02_0108046.PNG  
      inflating: sign_data/sign_data/train/046_forg/02_0123046.PNG  
      inflating: sign_data/sign_data/train/046_forg/03_0107046.PNG  
      inflating: sign_data/sign_data/train/046_forg/03_0108046.PNG  
      inflating: sign_data/sign_data/train/046_forg/03_0123046.PNG  
      inflating: sign_data/sign_data/train/046_forg/04_0107046.PNG  
      inflating: sign_data/sign_data/train/046_forg/04_0108046.PNG  
      inflating: sign_data/sign_data/train/046_forg/04_0123046.PNG  
      inflating: sign_data/sign_data/train/047/01_047.png  
      inflating: sign_data/sign_data/train/047/02_047.png  
      inflating: sign_data/sign_data/train/047/03_047.png  
      inflating: sign_data/sign_data/train/047/04_047.png  
      inflating: sign_data/sign_data/train/047/05_047.png  
      inflating: sign_data/sign_data/train/047/06_047.png  
      inflating: sign_data/sign_data/train/047/07_047.png  
      inflating: sign_data/sign_data/train/047/08_047.png  
      inflating: sign_data/sign_data/train/047/09_047.png  
      inflating: sign_data/sign_data/train/047/10_047.png  
      inflating: sign_data/sign_data/train/047/11_047.png  
      inflating: sign_data/sign_data/train/047/12_047.png  
      inflating: sign_data/sign_data/train/047_forg/01_0113047.PNG  
      inflating: sign_data/sign_data/train/047_forg/01_0114047.PNG  
      inflating: sign_data/sign_data/train/047_forg/01_0212047.PNG  
      inflating: sign_data/sign_data/train/047_forg/02_0113047.PNG  
      inflating: sign_data/sign_data/train/047_forg/02_0114047.PNG  
      inflating: sign_data/sign_data/train/047_forg/02_0212047.PNG  
      inflating: sign_data/sign_data/train/047_forg/03_0113047.PNG  
      inflating: sign_data/sign_data/train/047_forg/03_0114047.PNG  
      inflating: sign_data/sign_data/train/047_forg/03_0212047.PNG  
      inflating: sign_data/sign_data/train/047_forg/04_0113047.PNG  
      inflating: sign_data/sign_data/train/047_forg/04_0114047.PNG  
      inflating: sign_data/sign_data/train/047_forg/04_0212047.PNG  
      inflating: sign_data/sign_data/train/048/01_048.png  
      inflating: sign_data/sign_data/train/048/02_048.png  
      inflating: sign_data/sign_data/train/048/03_048.png  
      inflating: sign_data/sign_data/train/048/04_048.png  
      inflating: sign_data/sign_data/train/048/05_048.png  
      inflating: sign_data/sign_data/train/048/06_048.png  
      inflating: sign_data/sign_data/train/048/07_048.png  
      inflating: sign_data/sign_data/train/048/08_048.png  
      inflating: sign_data/sign_data/train/048/09_048.png  
      inflating: sign_data/sign_data/train/048/10_048.png  
      inflating: sign_data/sign_data/train/048/11_048.png  
      inflating: sign_data/sign_data/train/048/12_048.png  
      inflating: sign_data/sign_data/train/048_forg/01_0106048.PNG  
      inflating: sign_data/sign_data/train/048_forg/01_0204048.PNG  
      inflating: sign_data/sign_data/train/048_forg/02_0106048.PNG  
      inflating: sign_data/sign_data/train/048_forg/02_0204048.PNG  
      inflating: sign_data/sign_data/train/048_forg/03_0106048.PNG  
      inflating: sign_data/sign_data/train/048_forg/03_0204048.PNG  
      inflating: sign_data/sign_data/train/048_forg/04_0106048.PNG  
      inflating: sign_data/sign_data/train/048_forg/04_0204048.PNG  
      inflating: sign_data/sign_data/train/049/01_049.png  
      inflating: sign_data/sign_data/train/049/02_049.png  
      inflating: sign_data/sign_data/train/049/03_049.png  
      inflating: sign_data/sign_data/train/049/04_049.png  
      inflating: sign_data/sign_data/train/049/05_049.png  
      inflating: sign_data/sign_data/train/049/06_049.png  
      inflating: sign_data/sign_data/train/049/07_049.png  
      inflating: sign_data/sign_data/train/049/08_049.png  
      inflating: sign_data/sign_data/train/049/09_049.png  
      inflating: sign_data/sign_data/train/049/10_049.png  
      inflating: sign_data/sign_data/train/049/11_049.png  
      inflating: sign_data/sign_data/train/049/12_049.png  
      inflating: sign_data/sign_data/train/049_forg/01_0114049.PNG  
      inflating: sign_data/sign_data/train/049_forg/01_0206049.PNG  
      inflating: sign_data/sign_data/train/049_forg/01_0210049.PNG  
      inflating: sign_data/sign_data/train/049_forg/02_0114049.PNG  
      inflating: sign_data/sign_data/train/049_forg/02_0206049.PNG  
      inflating: sign_data/sign_data/train/049_forg/02_0210049.PNG  
      inflating: sign_data/sign_data/train/049_forg/03_0114049.PNG  
      inflating: sign_data/sign_data/train/049_forg/03_0206049.PNG  
      inflating: sign_data/sign_data/train/049_forg/03_0210049.PNG  
      inflating: sign_data/sign_data/train/049_forg/04_0114049.PNG  
      inflating: sign_data/sign_data/train/049_forg/04_0206049.PNG  
      inflating: sign_data/sign_data/train/049_forg/04_0210049.PNG  
      inflating: sign_data/sign_data/train/050/01_050.png  
      inflating: sign_data/sign_data/train/050/02_050.png  
      inflating: sign_data/sign_data/train/050/03_050.png  
      inflating: sign_data/sign_data/train/050/04_050.png  
      inflating: sign_data/sign_data/train/050/05_050.png  
      inflating: sign_data/sign_data/train/050/06_050.png  
      inflating: sign_data/sign_data/train/050/07_050.png  
      inflating: sign_data/sign_data/train/050/08_050.png  
      inflating: sign_data/sign_data/train/050/09_050.png  
      inflating: sign_data/sign_data/train/050/10_050.png  
      inflating: sign_data/sign_data/train/050/11_050.png  
      inflating: sign_data/sign_data/train/050/12_050.png  
      inflating: sign_data/sign_data/train/050_forg/01_0125050.PNG  
      inflating: sign_data/sign_data/train/050_forg/01_0126050.PNG  
      inflating: sign_data/sign_data/train/050_forg/01_0204050.PNG  
      inflating: sign_data/sign_data/train/050_forg/02_0125050.PNG  
      inflating: sign_data/sign_data/train/050_forg/02_0126050.PNG  
      inflating: sign_data/sign_data/train/050_forg/02_0204050.PNG  
      inflating: sign_data/sign_data/train/050_forg/03_0125050.PNG  
      inflating: sign_data/sign_data/train/050_forg/03_0126050.PNG  
      inflating: sign_data/sign_data/train/050_forg/03_0204050.PNG  
      inflating: sign_data/sign_data/train/050_forg/04_0125050.PNG  
      inflating: sign_data/sign_data/train/050_forg/04_0126050.PNG  
      inflating: sign_data/sign_data/train/050_forg/04_0204050.PNG  
      inflating: sign_data/sign_data/train/051/01_051.png  
      inflating: sign_data/sign_data/train/051/02_051.png  
      inflating: sign_data/sign_data/train/051/03_051.png  
      inflating: sign_data/sign_data/train/051/04_051.png  
      inflating: sign_data/sign_data/train/051/05_051.png  
      inflating: sign_data/sign_data/train/051/06_051.png  
      inflating: sign_data/sign_data/train/051/07_051.png  
      inflating: sign_data/sign_data/train/051/08_051.png  
      inflating: sign_data/sign_data/train/051/09_051.png  
      inflating: sign_data/sign_data/train/051/10_051.png  
      inflating: sign_data/sign_data/train/051/11_051.png  
      inflating: sign_data/sign_data/train/051/12_051.png  
      inflating: sign_data/sign_data/train/051_forg/01_0104051.PNG  
      inflating: sign_data/sign_data/train/051_forg/01_0120051.PNG  
      inflating: sign_data/sign_data/train/051_forg/02_0104051.PNG  
      inflating: sign_data/sign_data/train/051_forg/02_0120051.PNG  
      inflating: sign_data/sign_data/train/051_forg/03_0104051.PNG  
      inflating: sign_data/sign_data/train/051_forg/03_0120051.PNG  
      inflating: sign_data/sign_data/train/051_forg/04_0104051.PNG  
      inflating: sign_data/sign_data/train/051_forg/04_0120051.PNG  
      inflating: sign_data/sign_data/train/052/01_052.png  
      inflating: sign_data/sign_data/train/052/02_052.png  
      inflating: sign_data/sign_data/train/052/03_052.png  
      inflating: sign_data/sign_data/train/052/04_052.png  
      inflating: sign_data/sign_data/train/052/05_052.png  
      inflating: sign_data/sign_data/train/052/06_052.png  
      inflating: sign_data/sign_data/train/052/07_052.png  
      inflating: sign_data/sign_data/train/052/08_052.png  
      inflating: sign_data/sign_data/train/052/09_052.png  
      inflating: sign_data/sign_data/train/052/10_052.png  
      inflating: sign_data/sign_data/train/052/11_052.png  
      inflating: sign_data/sign_data/train/052/12_052.png  
      inflating: sign_data/sign_data/train/052_forg/01_0106052.PNG  
      inflating: sign_data/sign_data/train/052_forg/01_0109052.PNG  
      inflating: sign_data/sign_data/train/052_forg/01_0207052.PNG  
      inflating: sign_data/sign_data/train/052_forg/01_0210052.PNG  
      inflating: sign_data/sign_data/train/052_forg/02_0106052.PNG  
      inflating: sign_data/sign_data/train/052_forg/02_0109052.PNG  
      inflating: sign_data/sign_data/train/052_forg/02_0207052.PNG  
      inflating: sign_data/sign_data/train/052_forg/02_0210052.PNG  
      inflating: sign_data/sign_data/train/052_forg/03_0106052.PNG  
      inflating: sign_data/sign_data/train/052_forg/03_0109052.PNG  
      inflating: sign_data/sign_data/train/052_forg/03_0207052.PNG  
      inflating: sign_data/sign_data/train/052_forg/03_0210052.PNG  
      inflating: sign_data/sign_data/train/052_forg/04_0106052.PNG  
      inflating: sign_data/sign_data/train/052_forg/04_0109052.PNG  
      inflating: sign_data/sign_data/train/052_forg/04_0207052.PNG  
      inflating: sign_data/sign_data/train/052_forg/04_0210052.PNG  
      inflating: sign_data/sign_data/train/053/01_053.png  
      inflating: sign_data/sign_data/train/053/02_053.png  
      inflating: sign_data/sign_data/train/053/03_053.png  
      inflating: sign_data/sign_data/train/053/04_053.png  
      inflating: sign_data/sign_data/train/053/05_053.png  
      inflating: sign_data/sign_data/train/053/06_053.png  
      inflating: sign_data/sign_data/train/053/07_053.png  
      inflating: sign_data/sign_data/train/053/08_053.png  
      inflating: sign_data/sign_data/train/053/09_053.png  
      inflating: sign_data/sign_data/train/053/10_053.png  
      inflating: sign_data/sign_data/train/053/11_053.png  
      inflating: sign_data/sign_data/train/053/12_053.png  
      inflating: sign_data/sign_data/train/053_forg/01_0107053.PNG  
      inflating: sign_data/sign_data/train/053_forg/01_0115053.PNG  
      inflating: sign_data/sign_data/train/053_forg/01_0202053.PNG  
      inflating: sign_data/sign_data/train/053_forg/01_0207053.PNG  
      inflating: sign_data/sign_data/train/053_forg/02_0107053.PNG  
      inflating: sign_data/sign_data/train/053_forg/02_0115053.PNG  
      inflating: sign_data/sign_data/train/053_forg/02_0202053.PNG  
      inflating: sign_data/sign_data/train/053_forg/02_0207053.PNG  
      inflating: sign_data/sign_data/train/053_forg/03_0107053.PNG  
      inflating: sign_data/sign_data/train/053_forg/03_0115053.PNG  
      inflating: sign_data/sign_data/train/053_forg/03_0202053.PNG  
      inflating: sign_data/sign_data/train/053_forg/03_0207053.PNG  
      inflating: sign_data/sign_data/train/053_forg/04_0107053.PNG  
      inflating: sign_data/sign_data/train/053_forg/04_0115053.PNG  
      inflating: sign_data/sign_data/train/053_forg/04_0202053.PNG  
      inflating: sign_data/sign_data/train/053_forg/04_0207053.PNG  
      inflating: sign_data/sign_data/train/054/01_054.png  
      inflating: sign_data/sign_data/train/054/02_054.png  
      inflating: sign_data/sign_data/train/054/03_054.png  
      inflating: sign_data/sign_data/train/054/04_054.png  
      inflating: sign_data/sign_data/train/054/05_054.png  
      inflating: sign_data/sign_data/train/054/06_054.png  
      inflating: sign_data/sign_data/train/054/07_054.png  
      inflating: sign_data/sign_data/train/054/08_054.png  
      inflating: sign_data/sign_data/train/054/09_054.png  
      inflating: sign_data/sign_data/train/054/10_054.png  
      inflating: sign_data/sign_data/train/054/11_054.png  
      inflating: sign_data/sign_data/train/054/12_054.png  
      inflating: sign_data/sign_data/train/054_forg/01_0102054.PNG  
      inflating: sign_data/sign_data/train/054_forg/01_0124054.PNG  
      inflating: sign_data/sign_data/train/054_forg/01_0207054.PNG  
      inflating: sign_data/sign_data/train/054_forg/01_0208054.PNG  
      inflating: sign_data/sign_data/train/054_forg/01_0214054.PNG  
      inflating: sign_data/sign_data/train/054_forg/02_0102054.PNG  
      inflating: sign_data/sign_data/train/054_forg/02_0124054.PNG  
      inflating: sign_data/sign_data/train/054_forg/02_0207054.PNG  
      inflating: sign_data/sign_data/train/054_forg/02_0208054.PNG  
      inflating: sign_data/sign_data/train/054_forg/02_0214054.PNG  
      inflating: sign_data/sign_data/train/054_forg/03_0102054.PNG  
      inflating: sign_data/sign_data/train/054_forg/03_0124054.PNG  
      inflating: sign_data/sign_data/train/054_forg/03_0207054.PNG  
      inflating: sign_data/sign_data/train/054_forg/03_0208054.PNG  
      inflating: sign_data/sign_data/train/054_forg/03_0214054.PNG  
      inflating: sign_data/sign_data/train/054_forg/04_0102054.PNG  
      inflating: sign_data/sign_data/train/054_forg/04_0124054.PNG  
      inflating: sign_data/sign_data/train/054_forg/04_0207054.PNG  
      inflating: sign_data/sign_data/train/054_forg/04_0208054.PNG  
      inflating: sign_data/sign_data/train/054_forg/04_0214054.PNG  
      inflating: sign_data/sign_data/train/055/01_055.png  
      inflating: sign_data/sign_data/train/055/02_055.png  
      inflating: sign_data/sign_data/train/055/03_055.png  
      inflating: sign_data/sign_data/train/055/04_055.png  
      inflating: sign_data/sign_data/train/055/05_055.png  
      inflating: sign_data/sign_data/train/055/06_055.png  
      inflating: sign_data/sign_data/train/055/07_055.png  
      inflating: sign_data/sign_data/train/055/08_055.png  
      inflating: sign_data/sign_data/train/055/09_055.png  
      inflating: sign_data/sign_data/train/055/10_055.png  
      inflating: sign_data/sign_data/train/055/11_055.png  
      inflating: sign_data/sign_data/train/055/12_055.png  
      inflating: sign_data/sign_data/train/055_forg/01_0118055.PNG  
      inflating: sign_data/sign_data/train/055_forg/01_0120055.PNG  
      inflating: sign_data/sign_data/train/055_forg/01_0202055.PNG  
      inflating: sign_data/sign_data/train/055_forg/02_0118055.PNG  
      inflating: sign_data/sign_data/train/055_forg/02_0120055.PNG  
      inflating: sign_data/sign_data/train/055_forg/02_0202055.PNG  
      inflating: sign_data/sign_data/train/055_forg/03_0118055.PNG  
      inflating: sign_data/sign_data/train/055_forg/03_0120055.PNG  
      inflating: sign_data/sign_data/train/055_forg/03_0202055.PNG  
      inflating: sign_data/sign_data/train/055_forg/04_0118055.PNG  
      inflating: sign_data/sign_data/train/055_forg/04_0120055.PNG  
      inflating: sign_data/sign_data/train/055_forg/04_0202055.PNG  
      inflating: sign_data/sign_data/train/056/01_056.png  
      inflating: sign_data/sign_data/train/056/02_056.png  
      inflating: sign_data/sign_data/train/056/03_056.png  
      inflating: sign_data/sign_data/train/056/04_056.png  
      inflating: sign_data/sign_data/train/056/05_056.png  
      inflating: sign_data/sign_data/train/056/06_056.png  
      inflating: sign_data/sign_data/train/056/07_056.png  
      inflating: sign_data/sign_data/train/056/08_056.png  
      inflating: sign_data/sign_data/train/056/09_056.png  
      inflating: sign_data/sign_data/train/056/10_056.png  
      inflating: sign_data/sign_data/train/056/11_056.png  
      inflating: sign_data/sign_data/train/056/12_056.png  
      inflating: sign_data/sign_data/train/056_forg/01_0105056.PNG  
      inflating: sign_data/sign_data/train/056_forg/01_0115056.PNG  
      inflating: sign_data/sign_data/train/056_forg/02_0105056.PNG  
      inflating: sign_data/sign_data/train/056_forg/02_0115056.PNG  
      inflating: sign_data/sign_data/train/056_forg/03_0105056.PNG  
      inflating: sign_data/sign_data/train/056_forg/03_0115056.PNG  
      inflating: sign_data/sign_data/train/056_forg/04_0105056.PNG  
      inflating: sign_data/sign_data/train/056_forg/04_0115056.PNG  
      inflating: sign_data/sign_data/train/057/01_057.png  
      inflating: sign_data/sign_data/train/057/02_057.png  
      inflating: sign_data/sign_data/train/057/03_057.png  
      inflating: sign_data/sign_data/train/057/04_057.png  
      inflating: sign_data/sign_data/train/057/05_057.png  
      inflating: sign_data/sign_data/train/057/06_057.png  
      inflating: sign_data/sign_data/train/057/07_057.png  
      inflating: sign_data/sign_data/train/057/08_057.png  
      inflating: sign_data/sign_data/train/057/09_057.png  
      inflating: sign_data/sign_data/train/057/10_057.png  
      inflating: sign_data/sign_data/train/057/11_057.png  
      inflating: sign_data/sign_data/train/057/12_057.png  
      inflating: sign_data/sign_data/train/057_forg/01_0117057.PNG  
      inflating: sign_data/sign_data/train/057_forg/01_0208057.PNG  
      inflating: sign_data/sign_data/train/057_forg/01_0210057.PNG  
      inflating: sign_data/sign_data/train/057_forg/02_0117057.PNG  
      inflating: sign_data/sign_data/train/057_forg/02_0208057.PNG  
      inflating: sign_data/sign_data/train/057_forg/02_0210057.PNG  
      inflating: sign_data/sign_data/train/057_forg/03_0117057.PNG  
      inflating: sign_data/sign_data/train/057_forg/03_0208057.PNG  
      inflating: sign_data/sign_data/train/057_forg/03_0210057.PNG  
      inflating: sign_data/sign_data/train/057_forg/04_0117057.PNG  
      inflating: sign_data/sign_data/train/057_forg/04_0208057.PNG  
      inflating: sign_data/sign_data/train/057_forg/04_0210057.PNG  
      inflating: sign_data/sign_data/train/058/01_058.png  
      inflating: sign_data/sign_data/train/058/02_058.png  
      inflating: sign_data/sign_data/train/058/03_058.png  
      inflating: sign_data/sign_data/train/058/04_058.png  
      inflating: sign_data/sign_data/train/058/05_058.png  
      inflating: sign_data/sign_data/train/058/06_058.png  
      inflating: sign_data/sign_data/train/058/07_058.png  
      inflating: sign_data/sign_data/train/058/08_058.png  
      inflating: sign_data/sign_data/train/058/09_058.png  
      inflating: sign_data/sign_data/train/058/10_058.png  
      inflating: sign_data/sign_data/train/058/11_058.png  
      inflating: sign_data/sign_data/train/058/12_058.png  
      inflating: sign_data/sign_data/train/058_forg/01_0109058.PNG  
      inflating: sign_data/sign_data/train/058_forg/01_0110058.PNG  
      inflating: sign_data/sign_data/train/058_forg/01_0125058.PNG  
      inflating: sign_data/sign_data/train/058_forg/01_0127058.PNG  
      inflating: sign_data/sign_data/train/058_forg/02_0109058.PNG  
      inflating: sign_data/sign_data/train/058_forg/02_0110058.PNG  
      inflating: sign_data/sign_data/train/058_forg/02_0125058.PNG  
      inflating: sign_data/sign_data/train/058_forg/02_0127058.PNG  
      inflating: sign_data/sign_data/train/058_forg/03_0109058.PNG  
      inflating: sign_data/sign_data/train/058_forg/03_0110058.PNG  
      inflating: sign_data/sign_data/train/058_forg/03_0125058.PNG  
      inflating: sign_data/sign_data/train/058_forg/03_0127058.PNG  
      inflating: sign_data/sign_data/train/058_forg/04_0109058.PNG  
      inflating: sign_data/sign_data/train/058_forg/04_0110058.PNG  
      inflating: sign_data/sign_data/train/058_forg/04_0125058.PNG  
      inflating: sign_data/sign_data/train/058_forg/04_0127058.PNG  
      inflating: sign_data/sign_data/train/059/01_059.png  
      inflating: sign_data/sign_data/train/059/02_059.png  
      inflating: sign_data/sign_data/train/059/03_059.png  
      inflating: sign_data/sign_data/train/059/04_059.png  
      inflating: sign_data/sign_data/train/059/05_059.png  
      inflating: sign_data/sign_data/train/059/06_059.png  
      inflating: sign_data/sign_data/train/059/07_059.png  
      inflating: sign_data/sign_data/train/059/08_059.png  
      inflating: sign_data/sign_data/train/059/09_059.png  
      inflating: sign_data/sign_data/train/059/10_059.png  
      inflating: sign_data/sign_data/train/059/11_059.png  
      inflating: sign_data/sign_data/train/059/12_059.png  
      inflating: sign_data/sign_data/train/059_forg/01_0104059.PNG  
      inflating: sign_data/sign_data/train/059_forg/01_0125059.PNG  
      inflating: sign_data/sign_data/train/059_forg/02_0104059.PNG  
      inflating: sign_data/sign_data/train/059_forg/02_0125059.PNG  
      inflating: sign_data/sign_data/train/059_forg/03_0104059.PNG  
      inflating: sign_data/sign_data/train/059_forg/03_0125059.PNG  
      inflating: sign_data/sign_data/train/059_forg/04_0104059.PNG  
      inflating: sign_data/sign_data/train/059_forg/04_0125059.PNG  
      inflating: sign_data/sign_data/train/060/01_060.png  
      inflating: sign_data/sign_data/train/060/02_060.png  
      inflating: sign_data/sign_data/train/060/03_060.png  
      inflating: sign_data/sign_data/train/060/04_060.png  
      inflating: sign_data/sign_data/train/060/05_060.png  
      inflating: sign_data/sign_data/train/060/06_060.png  
      inflating: sign_data/sign_data/train/060/07_060.png  
      inflating: sign_data/sign_data/train/060/08_060.png  
      inflating: sign_data/sign_data/train/060/09_060.png  
      inflating: sign_data/sign_data/train/060/10_060.png  
      inflating: sign_data/sign_data/train/060/11_060.png  
      inflating: sign_data/sign_data/train/060/12_060.png  
      inflating: sign_data/sign_data/train/060_forg/01_0111060.PNG  
      inflating: sign_data/sign_data/train/060_forg/01_0121060.PNG  
      inflating: sign_data/sign_data/train/060_forg/01_0126060.PNG  
      inflating: sign_data/sign_data/train/060_forg/02_0111060.PNG  
      inflating: sign_data/sign_data/train/060_forg/02_0121060.PNG  
      inflating: sign_data/sign_data/train/060_forg/02_0126060.PNG  
      inflating: sign_data/sign_data/train/060_forg/03_0111060.PNG  
      inflating: sign_data/sign_data/train/060_forg/03_0121060.PNG  
      inflating: sign_data/sign_data/train/060_forg/03_0126060.PNG  
      inflating: sign_data/sign_data/train/060_forg/04_0111060.PNG  
      inflating: sign_data/sign_data/train/060_forg/04_0121060.PNG  
      inflating: sign_data/sign_data/train/060_forg/04_0126060.PNG  
      inflating: sign_data/sign_data/train/061/01_061.png  
      inflating: sign_data/sign_data/train/061/02_061.png  
      inflating: sign_data/sign_data/train/061/03_061.png  
      inflating: sign_data/sign_data/train/061/04_061.png  
      inflating: sign_data/sign_data/train/061/05_061.png  
      inflating: sign_data/sign_data/train/061/06_061.png  
      inflating: sign_data/sign_data/train/061/07_061.png  
      inflating: sign_data/sign_data/train/061/08_061.png  
      inflating: sign_data/sign_data/train/061/09_061.png  
      inflating: sign_data/sign_data/train/061/10_061.png  
      inflating: sign_data/sign_data/train/061/11_061.png  
      inflating: sign_data/sign_data/train/061/12_061.png  
      inflating: sign_data/sign_data/train/061_forg/01_0102061.PNG  
      inflating: sign_data/sign_data/train/061_forg/01_0112061.PNG  
      inflating: sign_data/sign_data/train/061_forg/01_0206061.PNG  
      inflating: sign_data/sign_data/train/061_forg/02_0102061.PNG  
      inflating: sign_data/sign_data/train/061_forg/02_0112061.PNG  
      inflating: sign_data/sign_data/train/061_forg/02_0206061.PNG  
      inflating: sign_data/sign_data/train/061_forg/03_0102061.PNG  
      inflating: sign_data/sign_data/train/061_forg/03_0112061.PNG  
      inflating: sign_data/sign_data/train/061_forg/03_0206061.PNG  
      inflating: sign_data/sign_data/train/061_forg/04_0102061.PNG  
      inflating: sign_data/sign_data/train/061_forg/04_0112061.PNG  
      inflating: sign_data/sign_data/train/061_forg/04_0206061.PNG  
      inflating: sign_data/sign_data/train/062/01_062.png  
      inflating: sign_data/sign_data/train/062/02_062.png  
      inflating: sign_data/sign_data/train/062/03_062.png  
      inflating: sign_data/sign_data/train/062/04_062.png  
      inflating: sign_data/sign_data/train/062/05_062.png  
      inflating: sign_data/sign_data/train/062/06_062.png  
      inflating: sign_data/sign_data/train/062/07_062.png  
      inflating: sign_data/sign_data/train/062/08_062.png  
      inflating: sign_data/sign_data/train/062/09_062.png  
      inflating: sign_data/sign_data/train/062/10_062.png  
      inflating: sign_data/sign_data/train/062/11_062.png  
      inflating: sign_data/sign_data/train/062/12_062.png  
      inflating: sign_data/sign_data/train/062_forg/01_0109062.PNG  
      inflating: sign_data/sign_data/train/062_forg/01_0116062.PNG  
      inflating: sign_data/sign_data/train/062_forg/01_0201062.PNG  
      inflating: sign_data/sign_data/train/062_forg/02_0109062.PNG  
      inflating: sign_data/sign_data/train/062_forg/02_0116062.PNG  
      inflating: sign_data/sign_data/train/062_forg/02_0201062.PNG  
      inflating: sign_data/sign_data/train/062_forg/03_0109062.PNG  
      inflating: sign_data/sign_data/train/062_forg/03_0116062.PNG  
      inflating: sign_data/sign_data/train/062_forg/03_0201062.PNG  
      inflating: sign_data/sign_data/train/062_forg/04_0109062.PNG  
      inflating: sign_data/sign_data/train/062_forg/04_0116062.PNG  
      inflating: sign_data/sign_data/train/062_forg/04_0201062.PNG  
      inflating: sign_data/sign_data/train/063/01_063.png  
      inflating: sign_data/sign_data/train/063/02_063.png  
      inflating: sign_data/sign_data/train/063/03_063.png  
      inflating: sign_data/sign_data/train/063/04_063.png  
      inflating: sign_data/sign_data/train/063/05_063.png  
      inflating: sign_data/sign_data/train/063/06_063.png  
      inflating: sign_data/sign_data/train/063/07_063.png  
      inflating: sign_data/sign_data/train/063/08_063.png  
      inflating: sign_data/sign_data/train/063/09_063.png  
      inflating: sign_data/sign_data/train/063/10_063.png  
      inflating: sign_data/sign_data/train/063/11_063.png  
      inflating: sign_data/sign_data/train/063/12_063.png  
      inflating: sign_data/sign_data/train/063_forg/01_0104063.PNG  
      inflating: sign_data/sign_data/train/063_forg/01_0108063.PNG  
      inflating: sign_data/sign_data/train/063_forg/01_0119063.PNG  
      inflating: sign_data/sign_data/train/063_forg/02_0104063.PNG  
      inflating: sign_data/sign_data/train/063_forg/02_0108063.PNG  
      inflating: sign_data/sign_data/train/063_forg/02_0119063.PNG  
      inflating: sign_data/sign_data/train/063_forg/03_0104063.PNG  
      inflating: sign_data/sign_data/train/063_forg/03_0108063.PNG  
      inflating: sign_data/sign_data/train/063_forg/03_0119063.PNG  
      inflating: sign_data/sign_data/train/063_forg/04_0104063.PNG  
      inflating: sign_data/sign_data/train/063_forg/04_0108063.PNG  
      inflating: sign_data/sign_data/train/063_forg/04_0119063.PNG  
      inflating: sign_data/sign_data/train/064/01_064.png  
      inflating: sign_data/sign_data/train/064/02_064.png  
      inflating: sign_data/sign_data/train/064/03_064.png  
      inflating: sign_data/sign_data/train/064/04_064.png  
      inflating: sign_data/sign_data/train/064/05_064.png  
      inflating: sign_data/sign_data/train/064/06_064.png  
      inflating: sign_data/sign_data/train/064/07_064.png  
      inflating: sign_data/sign_data/train/064/08_064.png  
      inflating: sign_data/sign_data/train/064/09_064.png  
      inflating: sign_data/sign_data/train/064/10_064.png  
      inflating: sign_data/sign_data/train/064/11_064.png  
      inflating: sign_data/sign_data/train/064/12_064.png  
      inflating: sign_data/sign_data/train/064_forg/01_0105064.PNG  
      inflating: sign_data/sign_data/train/064_forg/01_0203064.PNG  
      inflating: sign_data/sign_data/train/064_forg/02_0105064.PNG  
      inflating: sign_data/sign_data/train/064_forg/02_0203064.PNG  
      inflating: sign_data/sign_data/train/064_forg/03_0105064.PNG  
      inflating: sign_data/sign_data/train/064_forg/03_0203064.PNG  
      inflating: sign_data/sign_data/train/064_forg/04_0105064.PNG  
      inflating: sign_data/sign_data/train/064_forg/04_0203064.PNG  
      inflating: sign_data/sign_data/train/065/01_065.png  
      inflating: sign_data/sign_data/train/065/02_065.png  
      inflating: sign_data/sign_data/train/065/03_065.png  
      inflating: sign_data/sign_data/train/065/04_065.png  
      inflating: sign_data/sign_data/train/065/05_065.png  
      inflating: sign_data/sign_data/train/065/06_065.png  
      inflating: sign_data/sign_data/train/065/07_065.png  
      inflating: sign_data/sign_data/train/065/08_065.png  
      inflating: sign_data/sign_data/train/065/09_065.png  
      inflating: sign_data/sign_data/train/065/10_065.png  
      inflating: sign_data/sign_data/train/065/11_065.png  
      inflating: sign_data/sign_data/train/065/12_065.png  
      inflating: sign_data/sign_data/train/065_forg/01_0118065.PNG  
      inflating: sign_data/sign_data/train/065_forg/01_0206065.PNG  
      inflating: sign_data/sign_data/train/065_forg/02_0118065.PNG  
      inflating: sign_data/sign_data/train/065_forg/02_0206065.PNG  
      inflating: sign_data/sign_data/train/065_forg/03_0118065.PNG  
      inflating: sign_data/sign_data/train/065_forg/03_0206065.PNG  
      inflating: sign_data/sign_data/train/065_forg/04_0118065.PNG  
      inflating: sign_data/sign_data/train/065_forg/04_0206065.PNG  
      inflating: sign_data/sign_data/train/066/01_066.png  
      inflating: sign_data/sign_data/train/066/02_066.png  
      inflating: sign_data/sign_data/train/066/03_066.png  
      inflating: sign_data/sign_data/train/066/04_066.png  
      inflating: sign_data/sign_data/train/066/05_066.png  
      inflating: sign_data/sign_data/train/066/06_066.png  
      inflating: sign_data/sign_data/train/066/07_066.png  
      inflating: sign_data/sign_data/train/066/08_066.png  
      inflating: sign_data/sign_data/train/066/09_066.png  
      inflating: sign_data/sign_data/train/066/10_066.png  
      inflating: sign_data/sign_data/train/066/11_066.png  
      inflating: sign_data/sign_data/train/066/12_066.png  
      inflating: sign_data/sign_data/train/066_forg/01_0101066.PNG  
      inflating: sign_data/sign_data/train/066_forg/01_0127066.PNG  
      inflating: sign_data/sign_data/train/066_forg/01_0211066.PNG  
      inflating: sign_data/sign_data/train/066_forg/01_0212066.PNG  
      inflating: sign_data/sign_data/train/066_forg/02_0101066.PNG  
      inflating: sign_data/sign_data/train/066_forg/02_0127066.PNG  
      inflating: sign_data/sign_data/train/066_forg/02_0211066.PNG  
      inflating: sign_data/sign_data/train/066_forg/02_0212066.PNG  
      inflating: sign_data/sign_data/train/066_forg/03_0101066.PNG  
      inflating: sign_data/sign_data/train/066_forg/03_0127066.PNG  
      inflating: sign_data/sign_data/train/066_forg/03_0211066.PNG  
      inflating: sign_data/sign_data/train/066_forg/03_0212066.PNG  
      inflating: sign_data/sign_data/train/066_forg/04_0101066.PNG  
      inflating: sign_data/sign_data/train/066_forg/04_0127066.PNG  
      inflating: sign_data/sign_data/train/066_forg/04_0211066.PNG  
      inflating: sign_data/sign_data/train/066_forg/04_0212066.PNG  
      inflating: sign_data/sign_data/train/067/01_067.png  
      inflating: sign_data/sign_data/train/067/02_067.png  
      inflating: sign_data/sign_data/train/067/03_067.png  
      inflating: sign_data/sign_data/train/067/04_067.png  
      inflating: sign_data/sign_data/train/067/05_067.png  
      inflating: sign_data/sign_data/train/067/06_067.png  
      inflating: sign_data/sign_data/train/067/07_067.png  
      inflating: sign_data/sign_data/train/067/08_067.png  
      inflating: sign_data/sign_data/train/067/09_067.png  
      inflating: sign_data/sign_data/train/067/10_067.png  
      inflating: sign_data/sign_data/train/067/11_067.png  
      inflating: sign_data/sign_data/train/067/12_067.png  
      inflating: sign_data/sign_data/train/067_forg/01_0205067.PNG  
      inflating: sign_data/sign_data/train/067_forg/01_0212067.PNG  
      inflating: sign_data/sign_data/train/067_forg/02_0205067.PNG  
      inflating: sign_data/sign_data/train/067_forg/02_0212067.PNG  
      inflating: sign_data/sign_data/train/067_forg/03_0205067.PNG  
      inflating: sign_data/sign_data/train/067_forg/03_0212067.PNG  
      inflating: sign_data/sign_data/train/067_forg/04_0205067.PNG  
      inflating: sign_data/sign_data/train/067_forg/04_0212067.PNG  
      inflating: sign_data/sign_data/train/068/01_068.png  
      inflating: sign_data/sign_data/train/068/02_068.png  
      inflating: sign_data/sign_data/train/068/03_068.png  
      inflating: sign_data/sign_data/train/068/04_068.png  
      inflating: sign_data/sign_data/train/068/05_068.png  
      inflating: sign_data/sign_data/train/068/06_068.png  
      inflating: sign_data/sign_data/train/068/07_068.png  
      inflating: sign_data/sign_data/train/068/08_068.png  
      inflating: sign_data/sign_data/train/068/09_068.png  
      inflating: sign_data/sign_data/train/068/10_068.png  
      inflating: sign_data/sign_data/train/068/11_068.png  
      inflating: sign_data/sign_data/train/068/12_068.png  
      inflating: sign_data/sign_data/train/068_forg/01_0113068.PNG  
      inflating: sign_data/sign_data/train/068_forg/01_0124068.PNG  
      inflating: sign_data/sign_data/train/068_forg/02_0113068.PNG  
      inflating: sign_data/sign_data/train/068_forg/02_0124068.PNG  
      inflating: sign_data/sign_data/train/068_forg/03_0113068.PNG  
      inflating: sign_data/sign_data/train/068_forg/03_0124068.PNG  
      inflating: sign_data/sign_data/train/068_forg/04_0113068.PNG  
      inflating: sign_data/sign_data/train/068_forg/04_0124068.PNG  
      inflating: sign_data/sign_data/train/069/01_069.png  
      inflating: sign_data/sign_data/train/069/02_069.png  
      inflating: sign_data/sign_data/train/069/03_069.png  
      inflating: sign_data/sign_data/train/069/04_069.png  
      inflating: sign_data/sign_data/train/069/05_069.png  
      inflating: sign_data/sign_data/train/069/06_069.png  
      inflating: sign_data/sign_data/train/069/07_069.png  
      inflating: sign_data/sign_data/train/069/08_069.png  
      inflating: sign_data/sign_data/train/069/09_069.png  
      inflating: sign_data/sign_data/train/069/10_069.png  
      inflating: sign_data/sign_data/train/069/11_069.png  
      inflating: sign_data/sign_data/train/069/12_069.png  
      inflating: sign_data/sign_data/train/069_forg/01_0106069.PNG  
      inflating: sign_data/sign_data/train/069_forg/01_0108069.PNG  
      inflating: sign_data/sign_data/train/069_forg/01_0111069.PNG  
      inflating: sign_data/sign_data/train/069_forg/02_0106069.PNG  
      inflating: sign_data/sign_data/train/069_forg/02_0108069.PNG  
      inflating: sign_data/sign_data/train/069_forg/02_0111069.PNG  
      inflating: sign_data/sign_data/train/069_forg/03_0106069.PNG  
      inflating: sign_data/sign_data/train/069_forg/03_0108069.PNG  
      inflating: sign_data/sign_data/train/069_forg/03_0111069.PNG  
      inflating: sign_data/sign_data/train/069_forg/04_0106069.PNG  
      inflating: sign_data/sign_data/train/069_forg/04_0108069.PNG  
      inflating: sign_data/sign_data/train/069_forg/04_0111069.PNG  
      inflating: sign_data/sign_data/train_data.csv  
      inflating: sign_data/test/049/01_049.png  
      inflating: sign_data/test/049/02_049.png  
      inflating: sign_data/test/049/03_049.png  
      inflating: sign_data/test/049/04_049.png  
      inflating: sign_data/test/049/05_049.png  
      inflating: sign_data/test/049/06_049.png  
      inflating: sign_data/test/049/07_049.png  
      inflating: sign_data/test/049/08_049.png  
      inflating: sign_data/test/049/09_049.png  
      inflating: sign_data/test/049/10_049.png  
      inflating: sign_data/test/049/11_049.png  
      inflating: sign_data/test/049/12_049.png  
      inflating: sign_data/test/049_forg/01_0114049.PNG  
      inflating: sign_data/test/049_forg/01_0206049.PNG  
      inflating: sign_data/test/049_forg/01_0210049.PNG  
      inflating: sign_data/test/049_forg/02_0114049.PNG  
      inflating: sign_data/test/049_forg/02_0206049.PNG  
      inflating: sign_data/test/049_forg/02_0210049.PNG  
      inflating: sign_data/test/049_forg/03_0114049.PNG  
      inflating: sign_data/test/049_forg/03_0206049.PNG  
      inflating: sign_data/test/049_forg/03_0210049.PNG  
      inflating: sign_data/test/049_forg/04_0114049.PNG  
      inflating: sign_data/test/049_forg/04_0206049.PNG  
      inflating: sign_data/test/049_forg/04_0210049.PNG  
      inflating: sign_data/test/050/01_050.png  
      inflating: sign_data/test/050/02_050.png  
      inflating: sign_data/test/050/03_050.png  
      inflating: sign_data/test/050/04_050.png  
      inflating: sign_data/test/050/05_050.png  
      inflating: sign_data/test/050/06_050.png  
      inflating: sign_data/test/050/07_050.png  
      inflating: sign_data/test/050/08_050.png  
      inflating: sign_data/test/050/09_050.png  
      inflating: sign_data/test/050/10_050.png  
      inflating: sign_data/test/050/11_050.png  
      inflating: sign_data/test/050/12_050.png  
      inflating: sign_data/test/050_forg/01_0125050.PNG  
      inflating: sign_data/test/050_forg/01_0126050.PNG  
      inflating: sign_data/test/050_forg/01_0204050.PNG  
      inflating: sign_data/test/050_forg/02_0125050.PNG  
      inflating: sign_data/test/050_forg/02_0126050.PNG  
      inflating: sign_data/test/050_forg/02_0204050.PNG  
      inflating: sign_data/test/050_forg/03_0125050.PNG  
      inflating: sign_data/test/050_forg/03_0126050.PNG  
      inflating: sign_data/test/050_forg/03_0204050.PNG  
      inflating: sign_data/test/050_forg/04_0125050.PNG  
      inflating: sign_data/test/050_forg/04_0126050.PNG  
      inflating: sign_data/test/050_forg/04_0204050.PNG  
      inflating: sign_data/test/051/01_051.png  
      inflating: sign_data/test/051/02_051.png  
      inflating: sign_data/test/051/03_051.png  
      inflating: sign_data/test/051/04_051.png  
      inflating: sign_data/test/051/05_051.png  
      inflating: sign_data/test/051/06_051.png  
      inflating: sign_data/test/051/07_051.png  
      inflating: sign_data/test/051/08_051.png  
      inflating: sign_data/test/051/09_051.png  
      inflating: sign_data/test/051/10_051.png  
      inflating: sign_data/test/051/11_051.png  
      inflating: sign_data/test/051/12_051.png  
      inflating: sign_data/test/051_forg/01_0104051.PNG  
      inflating: sign_data/test/051_forg/01_0120051.PNG  
      inflating: sign_data/test/051_forg/02_0104051.PNG  
      inflating: sign_data/test/051_forg/02_0120051.PNG  
      inflating: sign_data/test/051_forg/03_0104051.PNG  
      inflating: sign_data/test/051_forg/03_0120051.PNG  
      inflating: sign_data/test/051_forg/04_0104051.PNG  
      inflating: sign_data/test/051_forg/04_0120051.PNG  
      inflating: sign_data/test/052/01_052.png  
      inflating: sign_data/test/052/02_052.png  
      inflating: sign_data/test/052/03_052.png  
      inflating: sign_data/test/052/04_052.png  
      inflating: sign_data/test/052/05_052.png  
      inflating: sign_data/test/052/06_052.png  
      inflating: sign_data/test/052/07_052.png  
      inflating: sign_data/test/052/08_052.png  
      inflating: sign_data/test/052/09_052.png  
      inflating: sign_data/test/052/10_052.png  
      inflating: sign_data/test/052/11_052.png  
      inflating: sign_data/test/052/12_052.png  
      inflating: sign_data/test/052_forg/01_0106052.PNG  
      inflating: sign_data/test/052_forg/01_0109052.PNG  
      inflating: sign_data/test/052_forg/01_0207052.PNG  
      inflating: sign_data/test/052_forg/01_0210052.PNG  
      inflating: sign_data/test/052_forg/02_0106052.PNG  
      inflating: sign_data/test/052_forg/02_0109052.PNG  
      inflating: sign_data/test/052_forg/02_0207052.PNG  
      inflating: sign_data/test/052_forg/02_0210052.PNG  
      inflating: sign_data/test/052_forg/03_0106052.PNG  
      inflating: sign_data/test/052_forg/03_0109052.PNG  
      inflating: sign_data/test/052_forg/03_0207052.PNG  
      inflating: sign_data/test/052_forg/03_0210052.PNG  
      inflating: sign_data/test/052_forg/04_0106052.PNG  
      inflating: sign_data/test/052_forg/04_0109052.PNG  
      inflating: sign_data/test/052_forg/04_0207052.PNG  
      inflating: sign_data/test/052_forg/04_0210052.PNG  
      inflating: sign_data/test/053/01_053.png  
      inflating: sign_data/test/053/02_053.png  
      inflating: sign_data/test/053/03_053.png  
      inflating: sign_data/test/053/04_053.png  
      inflating: sign_data/test/053/05_053.png  
      inflating: sign_data/test/053/06_053.png  
      inflating: sign_data/test/053/07_053.png  
      inflating: sign_data/test/053/08_053.png  
      inflating: sign_data/test/053/09_053.png  
      inflating: sign_data/test/053/10_053.png  
      inflating: sign_data/test/053/11_053.png  
      inflating: sign_data/test/053/12_053.png  
      inflating: sign_data/test/053_forg/01_0107053.PNG  
      inflating: sign_data/test/053_forg/01_0115053.PNG  
      inflating: sign_data/test/053_forg/01_0202053.PNG  
      inflating: sign_data/test/053_forg/01_0207053.PNG  
      inflating: sign_data/test/053_forg/02_0107053.PNG  
      inflating: sign_data/test/053_forg/02_0115053.PNG  
      inflating: sign_data/test/053_forg/02_0202053.PNG  
      inflating: sign_data/test/053_forg/02_0207053.PNG  
      inflating: sign_data/test/053_forg/03_0107053.PNG  
      inflating: sign_data/test/053_forg/03_0115053.PNG  
      inflating: sign_data/test/053_forg/03_0202053.PNG  
      inflating: sign_data/test/053_forg/03_0207053.PNG  
      inflating: sign_data/test/053_forg/04_0107053.PNG  
      inflating: sign_data/test/053_forg/04_0115053.PNG  
      inflating: sign_data/test/053_forg/04_0202053.PNG  
      inflating: sign_data/test/053_forg/04_0207053.PNG  
      inflating: sign_data/test/054/01_054.png  
      inflating: sign_data/test/054/02_054.png  
      inflating: sign_data/test/054/03_054.png  
      inflating: sign_data/test/054/04_054.png  
      inflating: sign_data/test/054/05_054.png  
      inflating: sign_data/test/054/06_054.png  
      inflating: sign_data/test/054/07_054.png  
      inflating: sign_data/test/054/08_054.png  
      inflating: sign_data/test/054/09_054.png  
      inflating: sign_data/test/054/10_054.png  
      inflating: sign_data/test/054/11_054.png  
      inflating: sign_data/test/054/12_054.png  
      inflating: sign_data/test/054_forg/01_0102054.PNG  
      inflating: sign_data/test/054_forg/01_0124054.PNG  
      inflating: sign_data/test/054_forg/01_0207054.PNG  
      inflating: sign_data/test/054_forg/01_0208054.PNG  
      inflating: sign_data/test/054_forg/01_0214054.PNG  
      inflating: sign_data/test/054_forg/02_0102054.PNG  
      inflating: sign_data/test/054_forg/02_0124054.PNG  
      inflating: sign_data/test/054_forg/02_0207054.PNG  
      inflating: sign_data/test/054_forg/02_0208054.PNG  
      inflating: sign_data/test/054_forg/02_0214054.PNG  
      inflating: sign_data/test/054_forg/03_0102054.PNG  
      inflating: sign_data/test/054_forg/03_0124054.PNG  
      inflating: sign_data/test/054_forg/03_0207054.PNG  
      inflating: sign_data/test/054_forg/03_0208054.PNG  
      inflating: sign_data/test/054_forg/03_0214054.PNG  
      inflating: sign_data/test/054_forg/04_0102054.PNG  
      inflating: sign_data/test/054_forg/04_0124054.PNG  
      inflating: sign_data/test/054_forg/04_0207054.PNG  
      inflating: sign_data/test/054_forg/04_0208054.PNG  
      inflating: sign_data/test/054_forg/04_0214054.PNG  
      inflating: sign_data/test/055/01_055.png  
      inflating: sign_data/test/055/02_055.png  
      inflating: sign_data/test/055/03_055.png  
      inflating: sign_data/test/055/04_055.png  
      inflating: sign_data/test/055/05_055.png  
      inflating: sign_data/test/055/06_055.png  
      inflating: sign_data/test/055/07_055.png  
      inflating: sign_data/test/055/08_055.png  
      inflating: sign_data/test/055/09_055.png  
      inflating: sign_data/test/055/10_055.png  
      inflating: sign_data/test/055/11_055.png  
      inflating: sign_data/test/055/12_055.png  
      inflating: sign_data/test/055_forg/01_0118055.PNG  
      inflating: sign_data/test/055_forg/01_0120055.PNG  
      inflating: sign_data/test/055_forg/01_0202055.PNG  
      inflating: sign_data/test/055_forg/02_0118055.PNG  
      inflating: sign_data/test/055_forg/02_0120055.PNG  
      inflating: sign_data/test/055_forg/02_0202055.PNG  
      inflating: sign_data/test/055_forg/03_0118055.PNG  
      inflating: sign_data/test/055_forg/03_0120055.PNG  
      inflating: sign_data/test/055_forg/03_0202055.PNG  
      inflating: sign_data/test/055_forg/04_0118055.PNG  
      inflating: sign_data/test/055_forg/04_0120055.PNG  
      inflating: sign_data/test/055_forg/04_0202055.PNG  
      inflating: sign_data/test/056/01_056.png  
      inflating: sign_data/test/056/02_056.png  
      inflating: sign_data/test/056/03_056.png  
      inflating: sign_data/test/056/04_056.png  
      inflating: sign_data/test/056/05_056.png  
      inflating: sign_data/test/056/06_056.png  
      inflating: sign_data/test/056/07_056.png  
      inflating: sign_data/test/056/08_056.png  
      inflating: sign_data/test/056/09_056.png  
      inflating: sign_data/test/056/10_056.png  
      inflating: sign_data/test/056/11_056.png  
      inflating: sign_data/test/056/12_056.png  
      inflating: sign_data/test/056_forg/01_0105056.PNG  
      inflating: sign_data/test/056_forg/01_0115056.PNG  
      inflating: sign_data/test/056_forg/02_0105056.PNG  
      inflating: sign_data/test/056_forg/02_0115056.PNG  
      inflating: sign_data/test/056_forg/03_0105056.PNG  
      inflating: sign_data/test/056_forg/03_0115056.PNG  
      inflating: sign_data/test/056_forg/04_0105056.PNG  
      inflating: sign_data/test/056_forg/04_0115056.PNG  
      inflating: sign_data/test/057/01_057.png  
      inflating: sign_data/test/057/02_057.png  
      inflating: sign_data/test/057/03_057.png  
      inflating: sign_data/test/057/04_057.png  
      inflating: sign_data/test/057/05_057.png  
      inflating: sign_data/test/057/06_057.png  
      inflating: sign_data/test/057/07_057.png  
      inflating: sign_data/test/057/08_057.png  
      inflating: sign_data/test/057/09_057.png  
      inflating: sign_data/test/057/10_057.png  
      inflating: sign_data/test/057/11_057.png  
      inflating: sign_data/test/057/12_057.png  
      inflating: sign_data/test/057_forg/01_0117057.PNG  
      inflating: sign_data/test/057_forg/01_0208057.PNG  
      inflating: sign_data/test/057_forg/01_0210057.PNG  
      inflating: sign_data/test/057_forg/02_0117057.PNG  
      inflating: sign_data/test/057_forg/02_0208057.PNG  
      inflating: sign_data/test/057_forg/02_0210057.PNG  
      inflating: sign_data/test/057_forg/03_0117057.PNG  
      inflating: sign_data/test/057_forg/03_0208057.PNG  
      inflating: sign_data/test/057_forg/03_0210057.PNG  
      inflating: sign_data/test/057_forg/04_0117057.PNG  
      inflating: sign_data/test/057_forg/04_0208057.PNG  
      inflating: sign_data/test/057_forg/04_0210057.PNG  
      inflating: sign_data/test/058/01_058.png  
      inflating: sign_data/test/058/02_058.png  
      inflating: sign_data/test/058/03_058.png  
      inflating: sign_data/test/058/04_058.png  
      inflating: sign_data/test/058/05_058.png  
      inflating: sign_data/test/058/06_058.png  
      inflating: sign_data/test/058/07_058.png  
      inflating: sign_data/test/058/08_058.png  
      inflating: sign_data/test/058/09_058.png  
      inflating: sign_data/test/058/10_058.png  
      inflating: sign_data/test/058/11_058.png  
      inflating: sign_data/test/058/12_058.png  
      inflating: sign_data/test/058_forg/01_0109058.PNG  
      inflating: sign_data/test/058_forg/01_0110058.PNG  
      inflating: sign_data/test/058_forg/01_0125058.PNG  
      inflating: sign_data/test/058_forg/01_0127058.PNG  
      inflating: sign_data/test/058_forg/02_0109058.PNG  
      inflating: sign_data/test/058_forg/02_0110058.PNG  
      inflating: sign_data/test/058_forg/02_0125058.PNG  
      inflating: sign_data/test/058_forg/02_0127058.PNG  
      inflating: sign_data/test/058_forg/03_0109058.PNG  
      inflating: sign_data/test/058_forg/03_0110058.PNG  
      inflating: sign_data/test/058_forg/03_0125058.PNG  
      inflating: sign_data/test/058_forg/03_0127058.PNG  
      inflating: sign_data/test/058_forg/04_0109058.PNG  
      inflating: sign_data/test/058_forg/04_0110058.PNG  
      inflating: sign_data/test/058_forg/04_0125058.PNG  
      inflating: sign_data/test/058_forg/04_0127058.PNG  
      inflating: sign_data/test/059/01_059.png  
      inflating: sign_data/test/059/02_059.png  
      inflating: sign_data/test/059/03_059.png  
      inflating: sign_data/test/059/04_059.png  
      inflating: sign_data/test/059/05_059.png  
      inflating: sign_data/test/059/06_059.png  
      inflating: sign_data/test/059/07_059.png  
      inflating: sign_data/test/059/08_059.png  
      inflating: sign_data/test/059/09_059.png  
      inflating: sign_data/test/059/10_059.png  
      inflating: sign_data/test/059/11_059.png  
      inflating: sign_data/test/059/12_059.png  
      inflating: sign_data/test/059_forg/01_0104059.PNG  
      inflating: sign_data/test/059_forg/01_0125059.PNG  
      inflating: sign_data/test/059_forg/02_0104059.PNG  
      inflating: sign_data/test/059_forg/02_0125059.PNG  
      inflating: sign_data/test/059_forg/03_0104059.PNG  
      inflating: sign_data/test/059_forg/03_0125059.PNG  
      inflating: sign_data/test/059_forg/04_0104059.PNG  
      inflating: sign_data/test/059_forg/04_0125059.PNG  
      inflating: sign_data/test/060/01_060.png  
      inflating: sign_data/test/060/02_060.png  
      inflating: sign_data/test/060/03_060.png  
      inflating: sign_data/test/060/04_060.png  
      inflating: sign_data/test/060/05_060.png  
      inflating: sign_data/test/060/06_060.png  
      inflating: sign_data/test/060/07_060.png  
      inflating: sign_data/test/060/08_060.png  
      inflating: sign_data/test/060/09_060.png  
      inflating: sign_data/test/060/10_060.png  
      inflating: sign_data/test/060/11_060.png  
      inflating: sign_data/test/060/12_060.png  
      inflating: sign_data/test/060_forg/01_0111060.PNG  
      inflating: sign_data/test/060_forg/01_0121060.PNG  
      inflating: sign_data/test/060_forg/01_0126060.PNG  
      inflating: sign_data/test/060_forg/02_0111060.PNG  
      inflating: sign_data/test/060_forg/02_0121060.PNG  
      inflating: sign_data/test/060_forg/02_0126060.PNG  
      inflating: sign_data/test/060_forg/03_0111060.PNG  
      inflating: sign_data/test/060_forg/03_0121060.PNG  
      inflating: sign_data/test/060_forg/03_0126060.PNG  
      inflating: sign_data/test/060_forg/04_0111060.PNG  
      inflating: sign_data/test/060_forg/04_0121060.PNG  
      inflating: sign_data/test/060_forg/04_0126060.PNG  
      inflating: sign_data/test/061/01_061.png  
      inflating: sign_data/test/061/02_061.png  
      inflating: sign_data/test/061/03_061.png  
      inflating: sign_data/test/061/04_061.png  
      inflating: sign_data/test/061/05_061.png  
      inflating: sign_data/test/061/06_061.png  
      inflating: sign_data/test/061/07_061.png  
      inflating: sign_data/test/061/08_061.png  
      inflating: sign_data/test/061/09_061.png  
      inflating: sign_data/test/061/10_061.png  
      inflating: sign_data/test/061/11_061.png  
      inflating: sign_data/test/061/12_061.png  
      inflating: sign_data/test/061_forg/01_0102061.PNG  
      inflating: sign_data/test/061_forg/01_0112061.PNG  
      inflating: sign_data/test/061_forg/01_0206061.PNG  
      inflating: sign_data/test/061_forg/02_0102061.PNG  
      inflating: sign_data/test/061_forg/02_0112061.PNG  
      inflating: sign_data/test/061_forg/02_0206061.PNG  
      inflating: sign_data/test/061_forg/03_0102061.PNG  
      inflating: sign_data/test/061_forg/03_0112061.PNG  
      inflating: sign_data/test/061_forg/03_0206061.PNG  
      inflating: sign_data/test/061_forg/04_0102061.PNG  
      inflating: sign_data/test/061_forg/04_0112061.PNG  
      inflating: sign_data/test/061_forg/04_0206061.PNG  
      inflating: sign_data/test/062/01_062.png  
      inflating: sign_data/test/062/02_062.png  
      inflating: sign_data/test/062/03_062.png  
      inflating: sign_data/test/062/04_062.png  
      inflating: sign_data/test/062/05_062.png  
      inflating: sign_data/test/062/06_062.png  
      inflating: sign_data/test/062/07_062.png  
      inflating: sign_data/test/062/08_062.png  
      inflating: sign_data/test/062/09_062.png  
      inflating: sign_data/test/062/10_062.png  
      inflating: sign_data/test/062/11_062.png  
      inflating: sign_data/test/062/12_062.png  
      inflating: sign_data/test/062_forg/01_0109062.PNG  
      inflating: sign_data/test/062_forg/01_0116062.PNG  
      inflating: sign_data/test/062_forg/01_0201062.PNG  
      inflating: sign_data/test/062_forg/02_0109062.PNG  
      inflating: sign_data/test/062_forg/02_0116062.PNG  
      inflating: sign_data/test/062_forg/02_0201062.PNG  
      inflating: sign_data/test/062_forg/03_0109062.PNG  
      inflating: sign_data/test/062_forg/03_0116062.PNG  
      inflating: sign_data/test/062_forg/03_0201062.PNG  
      inflating: sign_data/test/062_forg/04_0109062.PNG  
      inflating: sign_data/test/062_forg/04_0116062.PNG  
      inflating: sign_data/test/062_forg/04_0201062.PNG  
      inflating: sign_data/test/063/01_063.png  
      inflating: sign_data/test/063/02_063.png  
      inflating: sign_data/test/063/03_063.png  
      inflating: sign_data/test/063/04_063.png  
      inflating: sign_data/test/063/05_063.png  
      inflating: sign_data/test/063/06_063.png  
      inflating: sign_data/test/063/07_063.png  
      inflating: sign_data/test/063/08_063.png  
      inflating: sign_data/test/063/09_063.png  
      inflating: sign_data/test/063/10_063.png  
      inflating: sign_data/test/063/11_063.png  
      inflating: sign_data/test/063/12_063.png  
      inflating: sign_data/test/063_forg/01_0104063.PNG  
      inflating: sign_data/test/063_forg/01_0108063.PNG  
      inflating: sign_data/test/063_forg/01_0119063.PNG  
      inflating: sign_data/test/063_forg/02_0104063.PNG  
      inflating: sign_data/test/063_forg/02_0108063.PNG  
      inflating: sign_data/test/063_forg/02_0119063.PNG  
      inflating: sign_data/test/063_forg/03_0104063.PNG  
      inflating: sign_data/test/063_forg/03_0108063.PNG  
      inflating: sign_data/test/063_forg/03_0119063.PNG  
      inflating: sign_data/test/063_forg/04_0104063.PNG  
      inflating: sign_data/test/063_forg/04_0108063.PNG  
      inflating: sign_data/test/063_forg/04_0119063.PNG  
      inflating: sign_data/test/064/01_064.png  
      inflating: sign_data/test/064/02_064.png  
      inflating: sign_data/test/064/03_064.png  
      inflating: sign_data/test/064/04_064.png  
      inflating: sign_data/test/064/05_064.png  
      inflating: sign_data/test/064/06_064.png  
      inflating: sign_data/test/064/07_064.png  
      inflating: sign_data/test/064/08_064.png  
      inflating: sign_data/test/064/09_064.png  
      inflating: sign_data/test/064/10_064.png  
      inflating: sign_data/test/064/11_064.png  
      inflating: sign_data/test/064/12_064.png  
      inflating: sign_data/test/064_forg/01_0105064.PNG  
      inflating: sign_data/test/064_forg/01_0203064.PNG  
      inflating: sign_data/test/064_forg/02_0105064.PNG  
      inflating: sign_data/test/064_forg/02_0203064.PNG  
      inflating: sign_data/test/064_forg/03_0105064.PNG  
      inflating: sign_data/test/064_forg/03_0203064.PNG  
      inflating: sign_data/test/064_forg/04_0105064.PNG  
      inflating: sign_data/test/064_forg/04_0203064.PNG  
      inflating: sign_data/test/065/01_065.png  
      inflating: sign_data/test/065/02_065.png  
      inflating: sign_data/test/065/03_065.png  
      inflating: sign_data/test/065/04_065.png  
      inflating: sign_data/test/065/05_065.png  
      inflating: sign_data/test/065/06_065.png  
      inflating: sign_data/test/065/07_065.png  
      inflating: sign_data/test/065/08_065.png  
      inflating: sign_data/test/065/09_065.png  
      inflating: sign_data/test/065/10_065.png  
      inflating: sign_data/test/065/11_065.png  
      inflating: sign_data/test/065/12_065.png  
      inflating: sign_data/test/065_forg/01_0118065.PNG  
      inflating: sign_data/test/065_forg/01_0206065.PNG  
      inflating: sign_data/test/065_forg/02_0118065.PNG  
      inflating: sign_data/test/065_forg/02_0206065.PNG  
      inflating: sign_data/test/065_forg/03_0118065.PNG  
      inflating: sign_data/test/065_forg/03_0206065.PNG  
      inflating: sign_data/test/065_forg/04_0118065.PNG  
      inflating: sign_data/test/065_forg/04_0206065.PNG  
      inflating: sign_data/test/066/01_066.png  
      inflating: sign_data/test/066/02_066.png  
      inflating: sign_data/test/066/03_066.png  
      inflating: sign_data/test/066/04_066.png  
      inflating: sign_data/test/066/05_066.png  
      inflating: sign_data/test/066/06_066.png  
      inflating: sign_data/test/066/07_066.png  
      inflating: sign_data/test/066/08_066.png  
      inflating: sign_data/test/066/09_066.png  
      inflating: sign_data/test/066/10_066.png  
      inflating: sign_data/test/066/11_066.png  
      inflating: sign_data/test/066/12_066.png  
      inflating: sign_data/test/066_forg/01_0101066.PNG  
      inflating: sign_data/test/066_forg/01_0127066.PNG  
      inflating: sign_data/test/066_forg/01_0211066.PNG  
      inflating: sign_data/test/066_forg/01_0212066.PNG  
      inflating: sign_data/test/066_forg/02_0101066.PNG  
      inflating: sign_data/test/066_forg/02_0127066.PNG  
      inflating: sign_data/test/066_forg/02_0211066.PNG  
      inflating: sign_data/test/066_forg/02_0212066.PNG  
      inflating: sign_data/test/066_forg/03_0101066.PNG  
      inflating: sign_data/test/066_forg/03_0127066.PNG  
      inflating: sign_data/test/066_forg/03_0211066.PNG  
      inflating: sign_data/test/066_forg/03_0212066.PNG  
      inflating: sign_data/test/066_forg/04_0101066.PNG  
      inflating: sign_data/test/066_forg/04_0127066.PNG  
      inflating: sign_data/test/066_forg/04_0211066.PNG  
      inflating: sign_data/test/066_forg/04_0212066.PNG  
      inflating: sign_data/test/067/01_067.png  
      inflating: sign_data/test/067/02_067.png  
      inflating: sign_data/test/067/03_067.png  
      inflating: sign_data/test/067/04_067.png  
      inflating: sign_data/test/067/05_067.png  
      inflating: sign_data/test/067/06_067.png  
      inflating: sign_data/test/067/07_067.png  
      inflating: sign_data/test/067/08_067.png  
      inflating: sign_data/test/067/09_067.png  
      inflating: sign_data/test/067/10_067.png  
      inflating: sign_data/test/067/11_067.png  
      inflating: sign_data/test/067/12_067.png  
      inflating: sign_data/test/067_forg/01_0205067.PNG  
      inflating: sign_data/test/067_forg/01_0212067.PNG  
      inflating: sign_data/test/067_forg/02_0205067.PNG  
      inflating: sign_data/test/067_forg/02_0212067.PNG  
      inflating: sign_data/test/067_forg/03_0205067.PNG  
      inflating: sign_data/test/067_forg/03_0212067.PNG  
      inflating: sign_data/test/067_forg/04_0205067.PNG  
      inflating: sign_data/test/067_forg/04_0212067.PNG  
      inflating: sign_data/test/068/01_068.png  
      inflating: sign_data/test/068/02_068.png  
      inflating: sign_data/test/068/03_068.png  
      inflating: sign_data/test/068/04_068.png  
      inflating: sign_data/test/068/05_068.png  
      inflating: sign_data/test/068/06_068.png  
      inflating: sign_data/test/068/07_068.png  
      inflating: sign_data/test/068/08_068.png  
      inflating: sign_data/test/068/09_068.png  
      inflating: sign_data/test/068/10_068.png  
      inflating: sign_data/test/068/11_068.png  
      inflating: sign_data/test/068/12_068.png  
      inflating: sign_data/test/068_forg/01_0113068.PNG  
      inflating: sign_data/test/068_forg/01_0124068.PNG  
      inflating: sign_data/test/068_forg/02_0113068.PNG  
      inflating: sign_data/test/068_forg/02_0124068.PNG  
      inflating: sign_data/test/068_forg/03_0113068.PNG  
      inflating: sign_data/test/068_forg/03_0124068.PNG  
      inflating: sign_data/test/068_forg/04_0113068.PNG  
      inflating: sign_data/test/068_forg/04_0124068.PNG  
      inflating: sign_data/test/069/01_069.png  
      inflating: sign_data/test/069/02_069.png  
      inflating: sign_data/test/069/03_069.png  
      inflating: sign_data/test/069/04_069.png  
      inflating: sign_data/test/069/05_069.png  
      inflating: sign_data/test/069/06_069.png  
      inflating: sign_data/test/069/07_069.png  
      inflating: sign_data/test/069/08_069.png  
      inflating: sign_data/test/069/09_069.png  
      inflating: sign_data/test/069/10_069.png  
      inflating: sign_data/test/069/11_069.png  
      inflating: sign_data/test/069/12_069.png  
      inflating: sign_data/test/069_forg/01_0106069.PNG  
      inflating: sign_data/test/069_forg/01_0108069.PNG  
      inflating: sign_data/test/069_forg/01_0111069.PNG  
      inflating: sign_data/test/069_forg/02_0106069.PNG  
      inflating: sign_data/test/069_forg/02_0108069.PNG  
      inflating: sign_data/test/069_forg/02_0111069.PNG  
      inflating: sign_data/test/069_forg/03_0106069.PNG  
      inflating: sign_data/test/069_forg/03_0108069.PNG  
      inflating: sign_data/test/069_forg/03_0111069.PNG  
      inflating: sign_data/test/069_forg/04_0106069.PNG  
      inflating: sign_data/test/069_forg/04_0108069.PNG  
      inflating: sign_data/test/069_forg/04_0111069.PNG  
      inflating: sign_data/test_data.csv  
      inflating: sign_data/train/001/001_01.PNG  
      inflating: sign_data/train/001/001_02.PNG  
      inflating: sign_data/train/001/001_03.PNG  
      inflating: sign_data/train/001/001_04.PNG  
      inflating: sign_data/train/001/001_05.PNG  
      inflating: sign_data/train/001/001_06.PNG  
      inflating: sign_data/train/001/001_07.PNG  
      inflating: sign_data/train/001/001_08.PNG  
      inflating: sign_data/train/001/001_09.PNG  
      inflating: sign_data/train/001/001_10.PNG  
      inflating: sign_data/train/001/001_11.PNG  
      inflating: sign_data/train/001/001_12.PNG  
      inflating: sign_data/train/001/001_13.PNG  
      inflating: sign_data/train/001/001_14.PNG  
      inflating: sign_data/train/001/001_15.PNG  
      inflating: sign_data/train/001/001_16.PNG  
      inflating: sign_data/train/001/001_17.PNG  
      inflating: sign_data/train/001/001_18.PNG  
      inflating: sign_data/train/001/001_19.PNG  
      inflating: sign_data/train/001/001_20.PNG  
      inflating: sign_data/train/001/001_21.PNG  
      inflating: sign_data/train/001/001_22.PNG  
      inflating: sign_data/train/001/001_23.PNG  
      inflating: sign_data/train/001/001_24.PNG  
      inflating: sign_data/train/001_forg/0119001_01.png  
      inflating: sign_data/train/001_forg/0119001_02.png  
      inflating: sign_data/train/001_forg/0119001_03.png  
      inflating: sign_data/train/001_forg/0119001_04.png  
      inflating: sign_data/train/001_forg/0201001_01.png  
      inflating: sign_data/train/001_forg/0201001_02.png  
      inflating: sign_data/train/001_forg/0201001_03.png  
      inflating: sign_data/train/001_forg/0201001_04.png  
      inflating: sign_data/train/002/002_01.PNG  
      inflating: sign_data/train/002/002_02.PNG  
      inflating: sign_data/train/002/002_03.PNG  
      inflating: sign_data/train/002/002_04.PNG  
      inflating: sign_data/train/002/002_05.PNG  
      inflating: sign_data/train/002/002_06.PNG  
      inflating: sign_data/train/002/002_07.PNG  
      inflating: sign_data/train/002/002_08.PNG  
      inflating: sign_data/train/002/002_09.PNG  
      inflating: sign_data/train/002/002_10.PNG  
      inflating: sign_data/train/002/002_11.PNG  
      inflating: sign_data/train/002/002_12.PNG  
      inflating: sign_data/train/002/002_13.PNG  
      inflating: sign_data/train/002/002_14.PNG  
      inflating: sign_data/train/002/002_15.PNG  
      inflating: sign_data/train/002/002_16.PNG  
      inflating: sign_data/train/002/002_17.PNG  
      inflating: sign_data/train/002/002_18.PNG  
      inflating: sign_data/train/002/002_19.PNG  
      inflating: sign_data/train/002/002_20.PNG  
      inflating: sign_data/train/002/002_21.PNG  
      inflating: sign_data/train/002/002_22.PNG  
      inflating: sign_data/train/002/002_23.PNG  
      inflating: sign_data/train/002/002_24.PNG  
      inflating: sign_data/train/002_forg/0108002_01.png  
      inflating: sign_data/train/002_forg/0108002_02.png  
      inflating: sign_data/train/002_forg/0108002_03.png  
      inflating: sign_data/train/002_forg/0108002_04.png  
      inflating: sign_data/train/002_forg/0110002_01.png  
      inflating: sign_data/train/002_forg/0110002_02.png  
      inflating: sign_data/train/002_forg/0110002_03.png  
      inflating: sign_data/train/002_forg/0110002_04.png  
      inflating: sign_data/train/002_forg/0118002_01.png  
      inflating: sign_data/train/002_forg/0118002_02.png  
      inflating: sign_data/train/002_forg/0118002_03.png  
      inflating: sign_data/train/002_forg/0118002_04.png  
      inflating: sign_data/train/003/003_01.PNG  
      inflating: sign_data/train/003/003_02.PNG  
      inflating: sign_data/train/003/003_03.PNG  
      inflating: sign_data/train/003/003_04.PNG  
      inflating: sign_data/train/003/003_05.PNG  
      inflating: sign_data/train/003/003_06.PNG  
      inflating: sign_data/train/003/003_07.PNG  
      inflating: sign_data/train/003/003_08.PNG  
      inflating: sign_data/train/003/003_09.PNG  
      inflating: sign_data/train/003/003_10.PNG  
      inflating: sign_data/train/003/003_11.PNG  
      inflating: sign_data/train/003/003_12.PNG  
      inflating: sign_data/train/003/003_13.PNG  
      inflating: sign_data/train/003/003_14.PNG  
      inflating: sign_data/train/003/003_15.PNG  
      inflating: sign_data/train/003/003_16.PNG  
      inflating: sign_data/train/003/003_17.PNG  
      inflating: sign_data/train/003/003_18.PNG  
      inflating: sign_data/train/003/003_19.PNG  
      inflating: sign_data/train/003/003_20.PNG  
      inflating: sign_data/train/003/003_21.PNG  
      inflating: sign_data/train/003/003_22.PNG  
      inflating: sign_data/train/003/003_23.PNG  
      inflating: sign_data/train/003/003_24.PNG  
      inflating: sign_data/train/003_forg/0121003_01.png  
      inflating: sign_data/train/003_forg/0121003_02.png  
      inflating: sign_data/train/003_forg/0121003_03.png  
      inflating: sign_data/train/003_forg/0121003_04.png  
      inflating: sign_data/train/003_forg/0126003_01.png  
      inflating: sign_data/train/003_forg/0126003_02.png  
      inflating: sign_data/train/003_forg/0126003_03.png  
      inflating: sign_data/train/003_forg/0126003_04.png  
      inflating: sign_data/train/003_forg/0206003_01.png  
      inflating: sign_data/train/003_forg/0206003_02.png  
      inflating: sign_data/train/003_forg/0206003_03.png  
      inflating: sign_data/train/003_forg/0206003_04.png  
      inflating: sign_data/train/004/004_01.PNG  
      inflating: sign_data/train/004/004_02.PNG  
      inflating: sign_data/train/004/004_03.PNG  
      inflating: sign_data/train/004/004_04.PNG  
      inflating: sign_data/train/004/004_05.PNG  
      inflating: sign_data/train/004/004_06.PNG  
      inflating: sign_data/train/004/004_07.PNG  
      inflating: sign_data/train/004/004_08.PNG  
      inflating: sign_data/train/004/004_09.PNG  
      inflating: sign_data/train/004/004_10.PNG  
      inflating: sign_data/train/004/004_11.PNG  
      inflating: sign_data/train/004/004_12.PNG  
      inflating: sign_data/train/004/004_13.PNG  
      inflating: sign_data/train/004/004_14.PNG  
      inflating: sign_data/train/004/004_15.PNG  
      inflating: sign_data/train/004/004_16.PNG  
      inflating: sign_data/train/004/004_17.PNG  
      inflating: sign_data/train/004/004_18.PNG  
      inflating: sign_data/train/004/004_19.PNG  
      inflating: sign_data/train/004/004_20.PNG  
      inflating: sign_data/train/004/004_21.PNG  
      inflating: sign_data/train/004/004_22.PNG  
      inflating: sign_data/train/004/004_23.PNG  
      inflating: sign_data/train/004/004_24.PNG  
      inflating: sign_data/train/004_forg/0103004_02.png  
      inflating: sign_data/train/004_forg/0103004_03.png  
      inflating: sign_data/train/004_forg/0103004_04.png  
      inflating: sign_data/train/004_forg/0105004_01.png  
      inflating: sign_data/train/004_forg/0105004_02.png  
      inflating: sign_data/train/004_forg/0105004_03.png  
      inflating: sign_data/train/004_forg/0105004_04.png  
      inflating: sign_data/train/004_forg/0124004_01.png  
      inflating: sign_data/train/004_forg/0124004_02.png  
      inflating: sign_data/train/004_forg/0124004_03.png  
      inflating: sign_data/train/004_forg/0124004_04.png  
      inflating: sign_data/train/006/006_01.PNG  
      inflating: sign_data/train/006/006_02.PNG  
      inflating: sign_data/train/006/006_03.PNG  
      inflating: sign_data/train/006/006_04.PNG  
      inflating: sign_data/train/006/006_05.PNG  
      inflating: sign_data/train/006/006_06.PNG  
      inflating: sign_data/train/006/006_07.PNG  
      inflating: sign_data/train/006/006_08.PNG  
      inflating: sign_data/train/006/006_09.PNG  
      inflating: sign_data/train/006/006_10.PNG  
      inflating: sign_data/train/006/006_11.PNG  
      inflating: sign_data/train/006/006_12.PNG  
      inflating: sign_data/train/006/006_13.PNG  
      inflating: sign_data/train/006/006_14.PNG  
      inflating: sign_data/train/006/006_15.PNG  
      inflating: sign_data/train/006/006_16.PNG  
      inflating: sign_data/train/006/006_17.PNG  
      inflating: sign_data/train/006/006_18.PNG  
      inflating: sign_data/train/006/006_19.PNG  
      inflating: sign_data/train/006/006_20.PNG  
      inflating: sign_data/train/006/006_21.PNG  
      inflating: sign_data/train/006/006_22.PNG  
      inflating: sign_data/train/006/006_23.PNG  
      inflating: sign_data/train/006/006_24.PNG  
      inflating: sign_data/train/006_forg/0111006_01.png  
      inflating: sign_data/train/006_forg/0111006_02.png  
      inflating: sign_data/train/006_forg/0111006_03.png  
      inflating: sign_data/train/006_forg/0111006_04.png  
      inflating: sign_data/train/006_forg/0202006_01.png  
      inflating: sign_data/train/006_forg/0202006_02.png  
      inflating: sign_data/train/006_forg/0202006_03.png  
      inflating: sign_data/train/006_forg/0202006_04.png  
      inflating: sign_data/train/006_forg/0205006_01.png  
      inflating: sign_data/train/006_forg/0205006_02.png  
      inflating: sign_data/train/006_forg/0205006_03.png  
      inflating: sign_data/train/006_forg/0205006_04.png  
      inflating: sign_data/train/009/009_01.PNG  
      inflating: sign_data/train/009/009_02.PNG  
      inflating: sign_data/train/009/009_03.PNG  
      inflating: sign_data/train/009/009_04.PNG  
      inflating: sign_data/train/009/009_05.PNG  
      inflating: sign_data/train/009/009_06.PNG  
      inflating: sign_data/train/009/009_07.PNG  
      inflating: sign_data/train/009/009_08.PNG  
      inflating: sign_data/train/009/009_09.PNG  
      inflating: sign_data/train/009/009_10.PNG  
      inflating: sign_data/train/009/009_11.PNG  
      inflating: sign_data/train/009/009_12.PNG  
      inflating: sign_data/train/009/009_13.PNG  
      inflating: sign_data/train/009/009_14.PNG  
      inflating: sign_data/train/009/009_15.PNG  
      inflating: sign_data/train/009/009_16.PNG  
      inflating: sign_data/train/009/009_17.PNG  
      inflating: sign_data/train/009/009_18.PNG  
      inflating: sign_data/train/009/009_19.PNG  
      inflating: sign_data/train/009/009_20.PNG  
      inflating: sign_data/train/009/009_21.PNG  
      inflating: sign_data/train/009/009_22.PNG  
      inflating: sign_data/train/009/009_23.PNG  
      inflating: sign_data/train/009/009_24.PNG  
      inflating: sign_data/train/009_forg/0117009_01.png  
      inflating: sign_data/train/009_forg/0117009_02.png  
      inflating: sign_data/train/009_forg/0117009_03.png  
      inflating: sign_data/train/009_forg/0117009_04.png  
      inflating: sign_data/train/009_forg/0123009_01.png  
      inflating: sign_data/train/009_forg/0123009_02.png  
      inflating: sign_data/train/009_forg/0123009_03.png  
      inflating: sign_data/train/009_forg/0123009_04.png  
      inflating: sign_data/train/009_forg/0201009_01.png  
      inflating: sign_data/train/009_forg/0201009_02.png  
      inflating: sign_data/train/009_forg/0201009_03.png  
      inflating: sign_data/train/009_forg/0201009_04.png  
      inflating: sign_data/train/012/012_01.PNG  
      inflating: sign_data/train/012/012_02.PNG  
      inflating: sign_data/train/012/012_03.PNG  
      inflating: sign_data/train/012/012_04.PNG  
      inflating: sign_data/train/012/012_05.PNG  
      inflating: sign_data/train/012/012_06.PNG  
      inflating: sign_data/train/012/012_07.PNG  
      inflating: sign_data/train/012/012_08.PNG  
      inflating: sign_data/train/012/012_09.PNG  
      inflating: sign_data/train/012/012_10.PNG  
      inflating: sign_data/train/012/012_11.PNG  
      inflating: sign_data/train/012/012_12.PNG  
      inflating: sign_data/train/012/012_13.PNG  
      inflating: sign_data/train/012/012_14.PNG  
      inflating: sign_data/train/012/012_15.PNG  
      inflating: sign_data/train/012/012_16.PNG  
      inflating: sign_data/train/012/012_17.PNG  
      inflating: sign_data/train/012/012_18.PNG  
      inflating: sign_data/train/012/012_19.PNG  
      inflating: sign_data/train/012/012_20.PNG  
      inflating: sign_data/train/012/012_21.PNG  
      inflating: sign_data/train/012/012_22.PNG  
      inflating: sign_data/train/012/012_23.PNG  
      inflating: sign_data/train/012/012_24.PNG  
      inflating: sign_data/train/012_forg/0113012_01.png  
      inflating: sign_data/train/012_forg/0113012_02.png  
      inflating: sign_data/train/012_forg/0113012_03.png  
      inflating: sign_data/train/012_forg/0113012_04.png  
      inflating: sign_data/train/012_forg/0206012_01.png  
      inflating: sign_data/train/012_forg/0206012_02.png  
      inflating: sign_data/train/012_forg/0206012_03.png  
      inflating: sign_data/train/012_forg/0206012_04.png  
      inflating: sign_data/train/012_forg/0210012_01.png  
      inflating: sign_data/train/012_forg/0210012_02.png  
      inflating: sign_data/train/012_forg/0210012_03.png  
      inflating: sign_data/train/012_forg/0210012_04.png  
      inflating: sign_data/train/013/01_013.png  
      inflating: sign_data/train/013/02_013.png  
      inflating: sign_data/train/013/03_013.png  
      inflating: sign_data/train/013/04_013.png  
      inflating: sign_data/train/013/05_013.png  
      inflating: sign_data/train/013/06_013.png  
      inflating: sign_data/train/013/07_013.png  
      inflating: sign_data/train/013/08_013.png  
      inflating: sign_data/train/013/09_013.png  
      inflating: sign_data/train/013/10_013.png  
      inflating: sign_data/train/013/11_013.png  
      inflating: sign_data/train/013/12_013.png  
      inflating: sign_data/train/013_forg/01_0113013.PNG  
      inflating: sign_data/train/013_forg/01_0203013.PNG  
      inflating: sign_data/train/013_forg/01_0204013.PNG  
      inflating: sign_data/train/013_forg/02_0113013.PNG  
      inflating: sign_data/train/013_forg/02_0203013.PNG  
      inflating: sign_data/train/013_forg/02_0204013.PNG  
      inflating: sign_data/train/013_forg/03_0113013.PNG  
      inflating: sign_data/train/013_forg/03_0203013.PNG  
      inflating: sign_data/train/013_forg/03_0204013.PNG  
      inflating: sign_data/train/013_forg/04_0113013.PNG  
      inflating: sign_data/train/013_forg/04_0203013.PNG  
      inflating: sign_data/train/013_forg/04_0204013.PNG  
      inflating: sign_data/train/014/014_01.PNG  
      inflating: sign_data/train/014/014_02.PNG  
      inflating: sign_data/train/014/014_03.PNG  
      inflating: sign_data/train/014/014_04.PNG  
      inflating: sign_data/train/014/014_05.PNG  
      inflating: sign_data/train/014/014_06.PNG  
      inflating: sign_data/train/014/014_07.PNG  
      inflating: sign_data/train/014/014_08.PNG  
      inflating: sign_data/train/014/014_09.PNG  
      inflating: sign_data/train/014/014_10.PNG  
      inflating: sign_data/train/014/014_11.PNG  
      inflating: sign_data/train/014/014_12.PNG  
      inflating: sign_data/train/014/014_13.PNG  
      inflating: sign_data/train/014/014_14.PNG  
      inflating: sign_data/train/014/014_15.PNG  
      inflating: sign_data/train/014/014_16.PNG  
      inflating: sign_data/train/014/014_17.PNG  
      inflating: sign_data/train/014/014_18.PNG  
      inflating: sign_data/train/014/014_19.PNG  
      inflating: sign_data/train/014/014_20.PNG  
      inflating: sign_data/train/014/014_21.PNG  
      inflating: sign_data/train/014/014_22.PNG  
      inflating: sign_data/train/014/014_23.PNG  
      inflating: sign_data/train/014/014_24.PNG  
      inflating: sign_data/train/014_forg/0102014_01.png  
      inflating: sign_data/train/014_forg/0102014_02.png  
      inflating: sign_data/train/014_forg/0102014_03.png  
      inflating: sign_data/train/014_forg/0102014_04.png  
      inflating: sign_data/train/014_forg/0104014_01.png  
      inflating: sign_data/train/014_forg/0104014_02.png  
      inflating: sign_data/train/014_forg/0104014_03.png  
      inflating: sign_data/train/014_forg/0104014_04.png  
      inflating: sign_data/train/014_forg/0208014_01.png  
      inflating: sign_data/train/014_forg/0208014_02.png  
      inflating: sign_data/train/014_forg/0208014_03.png  
      inflating: sign_data/train/014_forg/0208014_04.png  
      inflating: sign_data/train/014_forg/0214014_01.png  
      inflating: sign_data/train/014_forg/0214014_02.png  
      inflating: sign_data/train/014_forg/0214014_03.png  
      inflating: sign_data/train/014_forg/0214014_04.png  
      inflating: sign_data/train/015/015_01.PNG  
      inflating: sign_data/train/015/015_02.PNG  
      inflating: sign_data/train/015/015_03.PNG  
      inflating: sign_data/train/015/015_04.PNG  
      inflating: sign_data/train/015/015_05.PNG  
      inflating: sign_data/train/015/015_06.PNG  
      inflating: sign_data/train/015/015_07.PNG  
      inflating: sign_data/train/015/015_08.PNG  
      inflating: sign_data/train/015/015_09.PNG  
      inflating: sign_data/train/015/015_10.PNG  
      inflating: sign_data/train/015/015_11.PNG  
      inflating: sign_data/train/015/015_12.PNG  
      inflating: sign_data/train/015/015_13.PNG  
      inflating: sign_data/train/015/015_14.PNG  
      inflating: sign_data/train/015/015_15.PNG  
      inflating: sign_data/train/015/015_16.PNG  
      inflating: sign_data/train/015/015_17.PNG  
      inflating: sign_data/train/015/015_18.PNG  
      inflating: sign_data/train/015/015_19.PNG  
      inflating: sign_data/train/015/015_20.PNG  
      inflating: sign_data/train/015/015_21.PNG  
      inflating: sign_data/train/015/015_22.PNG  
      inflating: sign_data/train/015/015_23.PNG  
      inflating: sign_data/train/015/015_24.PNG  
      inflating: sign_data/train/015_forg/0106015_01.png  
      inflating: sign_data/train/015_forg/0106015_02.png  
      inflating: sign_data/train/015_forg/0106015_03.png  
      inflating: sign_data/train/015_forg/0106015_04.png  
      inflating: sign_data/train/015_forg/0210015_01.png  
      inflating: sign_data/train/015_forg/0210015_02.png  
      inflating: sign_data/train/015_forg/0210015_03.png  
      inflating: sign_data/train/015_forg/0210015_04.png  
      inflating: sign_data/train/015_forg/0213015_01.png  
      inflating: sign_data/train/015_forg/0213015_02.png  
      inflating: sign_data/train/015_forg/0213015_03.png  
      inflating: sign_data/train/015_forg/0213015_04.png  
      inflating: sign_data/train/016/016_01.PNG  
      inflating: sign_data/train/016/016_02.PNG  
      inflating: sign_data/train/016/016_03.PNG  
      inflating: sign_data/train/016/016_04.PNG  
      inflating: sign_data/train/016/016_05.PNG  
      inflating: sign_data/train/016/016_06.PNG  
      inflating: sign_data/train/016/016_07.PNG  
      inflating: sign_data/train/016/016_08.PNG  
      inflating: sign_data/train/016/016_09.PNG  
      inflating: sign_data/train/016/016_10.PNG  
      inflating: sign_data/train/016/016_11.PNG  
      inflating: sign_data/train/016/016_12.PNG  
      inflating: sign_data/train/016/016_13.PNG  
      inflating: sign_data/train/016/016_14.PNG  
      inflating: sign_data/train/016/016_15.PNG  
      inflating: sign_data/train/016/016_16.PNG  
      inflating: sign_data/train/016/016_17.PNG  
      inflating: sign_data/train/016/016_18.PNG  
      inflating: sign_data/train/016/016_20.PNG  
      inflating: sign_data/train/016/016_21.PNG  
      inflating: sign_data/train/016/016_22.PNG  
      inflating: sign_data/train/016/016_23.PNG  
      inflating: sign_data/train/016/016_24.PNG  
      inflating: sign_data/train/016_forg/0107016_01.png  
      inflating: sign_data/train/016_forg/0107016_02.png  
      inflating: sign_data/train/016_forg/0107016_03.png  
      inflating: sign_data/train/016_forg/0107016_04.png  
      inflating: sign_data/train/016_forg/0110016_01.png  
      inflating: sign_data/train/016_forg/0110016_02.png  
      inflating: sign_data/train/016_forg/0110016_03.png  
      inflating: sign_data/train/016_forg/0110016_04.png  
      inflating: sign_data/train/016_forg/0127016_01.png  
      inflating: sign_data/train/016_forg/0127016_02.png  
      inflating: sign_data/train/016_forg/0127016_03.png  
      inflating: sign_data/train/016_forg/0127016_04.png  
      inflating: sign_data/train/016_forg/0202016_01.png  
      inflating: sign_data/train/016_forg/0202016_02.png  
      inflating: sign_data/train/016_forg/0202016_03.png  
      inflating: sign_data/train/016_forg/0202016_04.png  
      inflating: sign_data/train/017/01_017.png  
      inflating: sign_data/train/017/02_017.png  
      inflating: sign_data/train/017/03_017.png  
      inflating: sign_data/train/017/04_017.png  
      inflating: sign_data/train/017/05_017.png  
      inflating: sign_data/train/017/06_017.png  
      inflating: sign_data/train/017/07_017.png  
      inflating: sign_data/train/017/08_017.png  
      inflating: sign_data/train/017/09_017.png  
      inflating: sign_data/train/017/10_017.png  
      inflating: sign_data/train/017/11_017.png  
      inflating: sign_data/train/017/12_017.png  
      inflating: sign_data/train/017_forg/01_0107017.PNG  
      inflating: sign_data/train/017_forg/01_0124017.PNG  
      inflating: sign_data/train/017_forg/01_0211017.PNG  
      inflating: sign_data/train/017_forg/02_0107017.PNG  
      inflating: sign_data/train/017_forg/02_0124017.PNG  
      inflating: sign_data/train/017_forg/02_0211017.PNG  
      inflating: sign_data/train/017_forg/03_0107017.PNG  
      inflating: sign_data/train/017_forg/03_0124017.PNG  
      inflating: sign_data/train/017_forg/03_0211017.PNG  
      inflating: sign_data/train/017_forg/04_0107017.PNG  
      inflating: sign_data/train/017_forg/04_0124017.PNG  
      inflating: sign_data/train/017_forg/04_0211017.PNG  
      inflating: sign_data/train/018/01_018.png  
      inflating: sign_data/train/018/02_018.png  
      inflating: sign_data/train/018/03_018.png  
      inflating: sign_data/train/018/04_018.png  
      inflating: sign_data/train/018/05_018.png  
      inflating: sign_data/train/018/06_018.png  
      inflating: sign_data/train/018/07_018.png  
      inflating: sign_data/train/018/08_018.png  
      inflating: sign_data/train/018/09_018.png  
      inflating: sign_data/train/018/10_018.png  
      inflating: sign_data/train/018/11_018.png  
      inflating: sign_data/train/018/12_018.png  
      inflating: sign_data/train/018_forg/01_0106018.PNG  
      inflating: sign_data/train/018_forg/01_0112018.PNG  
      inflating: sign_data/train/018_forg/01_0202018.PNG  
      inflating: sign_data/train/018_forg/02_0106018.PNG  
      inflating: sign_data/train/018_forg/02_0112018.PNG  
      inflating: sign_data/train/018_forg/02_0202018.PNG  
      inflating: sign_data/train/018_forg/03_0106018.PNG  
      inflating: sign_data/train/018_forg/03_0112018.PNG  
      inflating: sign_data/train/018_forg/03_0202018.PNG  
      inflating: sign_data/train/018_forg/04_0106018.PNG  
      inflating: sign_data/train/018_forg/04_0112018.PNG  
      inflating: sign_data/train/018_forg/04_0202018.PNG  
      inflating: sign_data/train/019/01_019.png  
      inflating: sign_data/train/019/02_019.png  
      inflating: sign_data/train/019/03_019.png  
      inflating: sign_data/train/019/04_019.png  
      inflating: sign_data/train/019/05_019.png  
      inflating: sign_data/train/019/06_019.png  
      inflating: sign_data/train/019/07_019.png  
      inflating: sign_data/train/019/08_019.png  
      inflating: sign_data/train/019/09_019.png  
      inflating: sign_data/train/019/10_019.png  
      inflating: sign_data/train/019/11_019.png  
      inflating: sign_data/train/019/12_019.png  
      inflating: sign_data/train/019_forg/01_0115019.PNG  
      inflating: sign_data/train/019_forg/01_0116019.PNG  
      inflating: sign_data/train/019_forg/01_0119019.PNG  
      inflating: sign_data/train/019_forg/02_0115019.PNG  
      inflating: sign_data/train/019_forg/02_0116019.PNG  
      inflating: sign_data/train/019_forg/02_0119019.PNG  
      inflating: sign_data/train/019_forg/03_0115019.PNG  
      inflating: sign_data/train/019_forg/03_0116019.PNG  
      inflating: sign_data/train/019_forg/03_0119019.PNG  
      inflating: sign_data/train/019_forg/04_0115019.PNG  
      inflating: sign_data/train/019_forg/04_0116019.PNG  
      inflating: sign_data/train/019_forg/04_0119019.PNG  
      inflating: sign_data/train/020/01_020.png  
      inflating: sign_data/train/020/02_020.png  
      inflating: sign_data/train/020/03_020.png  
      inflating: sign_data/train/020/04_020.png  
      inflating: sign_data/train/020/05_020.png  
      inflating: sign_data/train/020/06_020.png  
      inflating: sign_data/train/020/07_020.png  
      inflating: sign_data/train/020/08_020.png  
      inflating: sign_data/train/020/09_020.png  
      inflating: sign_data/train/020/10_020.png  
      inflating: sign_data/train/020/11_020.png  
      inflating: sign_data/train/020/12_020.png  
      inflating: sign_data/train/020_forg/01_0105020.PNG  
      inflating: sign_data/train/020_forg/01_0117020.PNG  
      inflating: sign_data/train/020_forg/01_0127020.PNG  
      inflating: sign_data/train/020_forg/01_0213020.PNG  
      inflating: sign_data/train/020_forg/02_0101020.PNG  
      inflating: sign_data/train/020_forg/02_0105020.PNG  
      inflating: sign_data/train/020_forg/02_0117020.PNG  
      inflating: sign_data/train/020_forg/02_0127020.PNG  
      inflating: sign_data/train/020_forg/02_0213020.PNG  
      inflating: sign_data/train/020_forg/03_0101020.PNG  
      inflating: sign_data/train/020_forg/03_0105020.PNG  
      inflating: sign_data/train/020_forg/03_0117020.PNG  
      inflating: sign_data/train/020_forg/03_0127020.PNG  
      inflating: sign_data/train/020_forg/03_0213020.PNG  
      inflating: sign_data/train/020_forg/04_0101020.PNG  
      inflating: sign_data/train/020_forg/04_0105020.PNG  
      inflating: sign_data/train/020_forg/04_0117020.PNG  
      inflating: sign_data/train/020_forg/04_0127020.PNG  
      inflating: sign_data/train/020_forg/04_0213020.PNG  
      inflating: sign_data/train/021/01_021.png  
      inflating: sign_data/train/021/02_021.png  
      inflating: sign_data/train/021/03_021.png  
      inflating: sign_data/train/021/04_021.png  
      inflating: sign_data/train/021/05_021.png  
      inflating: sign_data/train/021/06_021.png  
      inflating: sign_data/train/021/07_021.png  
      inflating: sign_data/train/021/08_021.png  
      inflating: sign_data/train/021/09_021.png  
      inflating: sign_data/train/021/10_021.png  
      inflating: sign_data/train/021/11_021.png  
      inflating: sign_data/train/021/12_021.png  
      inflating: sign_data/train/021_forg/01_0110021.PNG  
      inflating: sign_data/train/021_forg/01_0204021.PNG  
      inflating: sign_data/train/021_forg/01_0211021.PNG  
      inflating: sign_data/train/021_forg/02_0110021.PNG  
      inflating: sign_data/train/021_forg/02_0204021.PNG  
      inflating: sign_data/train/021_forg/02_0211021.PNG  
      inflating: sign_data/train/021_forg/03_0110021.PNG  
      inflating: sign_data/train/021_forg/03_0204021.PNG  
      inflating: sign_data/train/021_forg/03_0211021.PNG  
      inflating: sign_data/train/021_forg/04_0110021.PNG  
      inflating: sign_data/train/021_forg/04_0204021.PNG  
      inflating: sign_data/train/021_forg/04_0211021.PNG  
      inflating: sign_data/train/022/01_022.png  
      inflating: sign_data/train/022/02_022.png  
      inflating: sign_data/train/022/03_022.png  
      inflating: sign_data/train/022/04_022.png  
      inflating: sign_data/train/022/05_022.png  
      inflating: sign_data/train/022/06_022.png  
      inflating: sign_data/train/022/07_022.png  
      inflating: sign_data/train/022/08_022.png  
      inflating: sign_data/train/022/09_022.png  
      inflating: sign_data/train/022/10_022.png  
      inflating: sign_data/train/022/11_022.png  
      inflating: sign_data/train/022/12_022.png  
      inflating: sign_data/train/022_forg/01_0125022.PNG  
      inflating: sign_data/train/022_forg/01_0127022.PNG  
      inflating: sign_data/train/022_forg/01_0208022.PNG  
      inflating: sign_data/train/022_forg/01_0214022.PNG  
      inflating: sign_data/train/022_forg/02_0125022.PNG  
      inflating: sign_data/train/022_forg/02_0127022.PNG  
      inflating: sign_data/train/022_forg/02_0208022.PNG  
      inflating: sign_data/train/022_forg/02_0214022.PNG  
      inflating: sign_data/train/022_forg/03_0125022.PNG  
      inflating: sign_data/train/022_forg/03_0127022.PNG  
      inflating: sign_data/train/022_forg/03_0208022.PNG  
      inflating: sign_data/train/022_forg/03_0214022.PNG  
      inflating: sign_data/train/022_forg/04_0125022.PNG  
      inflating: sign_data/train/022_forg/04_0127022.PNG  
      inflating: sign_data/train/022_forg/04_0208022.PNG  
      inflating: sign_data/train/022_forg/04_0214022.PNG  
      inflating: sign_data/train/023/01_023.png  
      inflating: sign_data/train/023/02_023.png  
      inflating: sign_data/train/023/03_023.png  
      inflating: sign_data/train/023/04_023.png  
      inflating: sign_data/train/023/05_023.png  
      inflating: sign_data/train/023/06_023.png  
      inflating: sign_data/train/023/07_023.png  
      inflating: sign_data/train/023/08_023.png  
      inflating: sign_data/train/023/09_023.png  
      inflating: sign_data/train/023/10_023.png  
      inflating: sign_data/train/023/11_023.png  
      inflating: sign_data/train/023/12_023.png  
      inflating: sign_data/train/023_forg/01_0126023.PNG  
      inflating: sign_data/train/023_forg/01_0203023.PNG  
      inflating: sign_data/train/023_forg/02_0126023.PNG  
      inflating: sign_data/train/023_forg/02_0203023.PNG  
      inflating: sign_data/train/023_forg/03_0126023.PNG  
      inflating: sign_data/train/023_forg/03_0203023.PNG  
      inflating: sign_data/train/023_forg/04_0126023.PNG  
      inflating: sign_data/train/023_forg/04_0203023.PNG  
      inflating: sign_data/train/024/01_024.png  
      inflating: sign_data/train/024/02_024.png  
      inflating: sign_data/train/024/03_024.png  
      inflating: sign_data/train/024/04_024.png  
      inflating: sign_data/train/024/05_024.png  
      inflating: sign_data/train/024/06_024.png  
      inflating: sign_data/train/024/07_024.png  
      inflating: sign_data/train/024/08_024.png  
      inflating: sign_data/train/024/09_024.png  
      inflating: sign_data/train/024/10_024.png  
      inflating: sign_data/train/024/11_024.png  
      inflating: sign_data/train/024/12_024.png  
      inflating: sign_data/train/024_forg/01_0102024.PNG  
      inflating: sign_data/train/024_forg/01_0119024.PNG  
      inflating: sign_data/train/024_forg/01_0120024.PNG  
      inflating: sign_data/train/024_forg/02_0102024.PNG  
      inflating: sign_data/train/024_forg/02_0119024.PNG  
      inflating: sign_data/train/024_forg/02_0120024.PNG  
      inflating: sign_data/train/024_forg/03_0102024.PNG  
      inflating: sign_data/train/024_forg/03_0119024.PNG  
      inflating: sign_data/train/024_forg/03_0120024.PNG  
      inflating: sign_data/train/024_forg/04_0102024.PNG  
      inflating: sign_data/train/024_forg/04_0119024.PNG  
      inflating: sign_data/train/024_forg/04_0120024.PNG  
      inflating: sign_data/train/025/01_025.png  
      inflating: sign_data/train/025/02_025.png  
      inflating: sign_data/train/025/03_025.png  
      inflating: sign_data/train/025/04_025.png  
      inflating: sign_data/train/025/05_025.png  
      inflating: sign_data/train/025/06_025.png  
      inflating: sign_data/train/025/07_025.png  
      inflating: sign_data/train/025/08_025.png  
      inflating: sign_data/train/025/09_025.png  
      inflating: sign_data/train/025/10_025.png  
      inflating: sign_data/train/025/11_025.png  
      inflating: sign_data/train/025/12_025.png  
      inflating: sign_data/train/025_forg/01_0116025.PNG  
      inflating: sign_data/train/025_forg/01_0121025.PNG  
      inflating: sign_data/train/025_forg/02_0116025.PNG  
      inflating: sign_data/train/025_forg/02_0121025.PNG  
      inflating: sign_data/train/025_forg/03_0116025.PNG  
      inflating: sign_data/train/025_forg/03_0121025.PNG  
      inflating: sign_data/train/025_forg/04_0116025.PNG  
      inflating: sign_data/train/025_forg/04_0121025.PNG  
      inflating: sign_data/train/026/01_026.png  
      inflating: sign_data/train/026/02_026.png  
      inflating: sign_data/train/026/03_026.png  
      inflating: sign_data/train/026/04_026.png  
      inflating: sign_data/train/026/05_026.png  
      inflating: sign_data/train/026/06_026.png  
      inflating: sign_data/train/026/07_026.png  
      inflating: sign_data/train/026/08_026.png  
      inflating: sign_data/train/026/09_026.png  
      inflating: sign_data/train/026/10_026.png  
      inflating: sign_data/train/026/11_026.png  
      inflating: sign_data/train/026/12_026.png  
      inflating: sign_data/train/026_forg/01_0119026.PNG  
      inflating: sign_data/train/026_forg/01_0123026.PNG  
      inflating: sign_data/train/026_forg/01_0125026.PNG  
      inflating: sign_data/train/026_forg/02_0119026.PNG  
      inflating: sign_data/train/026_forg/02_0123026.PNG  
      inflating: sign_data/train/026_forg/02_0125026.PNG  
      inflating: sign_data/train/026_forg/03_0119026.PNG  
      inflating: sign_data/train/026_forg/03_0123026.PNG  
      inflating: sign_data/train/026_forg/03_0125026.PNG  
      inflating: sign_data/train/026_forg/04_0119026.PNG  
      inflating: sign_data/train/026_forg/04_0123026.PNG  
      inflating: sign_data/train/026_forg/04_0125026.PNG  
      inflating: sign_data/train/027/01_027.png  
      inflating: sign_data/train/027/02_027.png  
      inflating: sign_data/train/027/03_027.png  
      inflating: sign_data/train/027/04_027.png  
      inflating: sign_data/train/027/05_027.png  
      inflating: sign_data/train/027/06_027.png  
      inflating: sign_data/train/027/07_027.png  
      inflating: sign_data/train/027/08_027.png  
      inflating: sign_data/train/027/09_027.png  
      inflating: sign_data/train/027/10_027.png  
      inflating: sign_data/train/027/11_027.png  
      inflating: sign_data/train/027/12_027.png  
      inflating: sign_data/train/027_forg/01_0101027.PNG  
      inflating: sign_data/train/027_forg/01_0212027.PNG  
      inflating: sign_data/train/027_forg/02_0101027.PNG  
      inflating: sign_data/train/027_forg/02_0212027.PNG  
      inflating: sign_data/train/027_forg/03_0101027.PNG  
      inflating: sign_data/train/027_forg/03_0212027.PNG  
      inflating: sign_data/train/027_forg/04_0101027.PNG  
      inflating: sign_data/train/027_forg/04_0212027.PNG  
      inflating: sign_data/train/028/01_028.png  
      inflating: sign_data/train/028/02_028.png  
      inflating: sign_data/train/028/03_028.png  
      inflating: sign_data/train/028/04_028.png  
      inflating: sign_data/train/028/05_028.png  
      inflating: sign_data/train/028/06_028.png  
      inflating: sign_data/train/028/07_028.png  
      inflating: sign_data/train/028/08_028.png  
      inflating: sign_data/train/028/09_028.png  
      inflating: sign_data/train/028/10_028.png  
      inflating: sign_data/train/028/11_028.png  
      inflating: sign_data/train/028/12_028.png  
      inflating: sign_data/train/028_forg/01_0126028.PNG  
      inflating: sign_data/train/028_forg/01_0205028.PNG  
      inflating: sign_data/train/028_forg/01_0212028.PNG  
      inflating: sign_data/train/028_forg/02_0126028.PNG  
      inflating: sign_data/train/028_forg/02_0205028.PNG  
      inflating: sign_data/train/028_forg/02_0212028.PNG  
      inflating: sign_data/train/028_forg/03_0126028.PNG  
      inflating: sign_data/train/028_forg/03_0205028.PNG  
      inflating: sign_data/train/028_forg/03_0212028.PNG  
      inflating: sign_data/train/028_forg/04_0126028.PNG  
      inflating: sign_data/train/028_forg/04_0205028.PNG  
      inflating: sign_data/train/028_forg/04_0212028.PNG  
      inflating: sign_data/train/029/01_029.png  
      inflating: sign_data/train/029/02_029.png  
      inflating: sign_data/train/029/03_029.png  
      inflating: sign_data/train/029/04_029.png  
      inflating: sign_data/train/029/05_029.png  
      inflating: sign_data/train/029/06_029.png  
      inflating: sign_data/train/029/07_029.png  
      inflating: sign_data/train/029/08_029.png  
      inflating: sign_data/train/029/09_029.png  
      inflating: sign_data/train/029/10_029.png  
      inflating: sign_data/train/029/11_029.png  
      inflating: sign_data/train/029/12_029.png  
      inflating: sign_data/train/029_forg/01_0104029.PNG  
      inflating: sign_data/train/029_forg/01_0115029.PNG  
      inflating: sign_data/train/029_forg/01_0203029.PNG  
      inflating: sign_data/train/029_forg/02_0104029.PNG  
      inflating: sign_data/train/029_forg/02_0115029.PNG  
      inflating: sign_data/train/029_forg/02_0203029.PNG  
      inflating: sign_data/train/029_forg/03_0104029.PNG  
      inflating: sign_data/train/029_forg/03_0115029.PNG  
      inflating: sign_data/train/029_forg/03_0203029.PNG  
      inflating: sign_data/train/029_forg/04_0104029.PNG  
      inflating: sign_data/train/029_forg/04_0115029.PNG  
      inflating: sign_data/train/029_forg/04_0203029.PNG  
      inflating: sign_data/train/030/01_030.png  
      inflating: sign_data/train/030/02_030.png  
      inflating: sign_data/train/030/03_030.png  
      inflating: sign_data/train/030/04_030.png  
      inflating: sign_data/train/030/05_030.png  
      inflating: sign_data/train/030/06_030.png  
      inflating: sign_data/train/030/07_030.png  
      inflating: sign_data/train/030/08_030.png  
      inflating: sign_data/train/030/09_030.png  
      inflating: sign_data/train/030/10_030.png  
      inflating: sign_data/train/030/11_030.png  
      inflating: sign_data/train/030/12_030.png  
      inflating: sign_data/train/030_forg/01_0109030.PNG  
      inflating: sign_data/train/030_forg/01_0114030.PNG  
      inflating: sign_data/train/030_forg/01_0213030.PNG  
      inflating: sign_data/train/030_forg/02_0109030.PNG  
      inflating: sign_data/train/030_forg/02_0114030.PNG  
      inflating: sign_data/train/030_forg/02_0213030.PNG  
      inflating: sign_data/train/030_forg/03_0109030.PNG  
      inflating: sign_data/train/030_forg/03_0114030.PNG  
      inflating: sign_data/train/030_forg/03_0213030.PNG  
      inflating: sign_data/train/030_forg/04_0109030.PNG  
      inflating: sign_data/train/030_forg/04_0114030.PNG  
      inflating: sign_data/train/030_forg/04_0213030.PNG  
      inflating: sign_data/train/031/01_031.png  
      inflating: sign_data/train/031/02_031.png  
      inflating: sign_data/train/031/03_031.png  
      inflating: sign_data/train/031/04_031.png  
      inflating: sign_data/train/031/05_031.png  
      inflating: sign_data/train/031/06_031.png  
      inflating: sign_data/train/031/07_031.png  
      inflating: sign_data/train/031/08_031.png  
      inflating: sign_data/train/031/09_031.png  
      inflating: sign_data/train/031/10_031.png  
      inflating: sign_data/train/031/11_031.png  
      inflating: sign_data/train/031/12_031.png  
      inflating: sign_data/train/031_forg/01_0103031.PNG  
      inflating: sign_data/train/031_forg/01_0121031.PNG  
      inflating: sign_data/train/031_forg/02_0103031.PNG  
      inflating: sign_data/train/031_forg/02_0121031.PNG  
      inflating: sign_data/train/031_forg/03_0103031.PNG  
      inflating: sign_data/train/031_forg/03_0121031.PNG  
      inflating: sign_data/train/031_forg/04_0103031.PNG  
      inflating: sign_data/train/031_forg/04_0121031.PNG  
      inflating: sign_data/train/032/01_032.png  
      inflating: sign_data/train/032/02_032.png  
      inflating: sign_data/train/032/03_032.png  
      inflating: sign_data/train/032/04_032.png  
      inflating: sign_data/train/032/05_032.png  
      inflating: sign_data/train/032/06_032.png  
      inflating: sign_data/train/032/07_032.png  
      inflating: sign_data/train/032/08_032.png  
      inflating: sign_data/train/032/09_032.png  
      inflating: sign_data/train/032/10_032.png  
      inflating: sign_data/train/032/11_032.png  
      inflating: sign_data/train/032/12_032.png  
      inflating: sign_data/train/032_forg/01_0112032.PNG  
      inflating: sign_data/train/032_forg/01_0117032.PNG  
      inflating: sign_data/train/032_forg/01_0120032.PNG  
      inflating: sign_data/train/032_forg/02_0112032.PNG  
      inflating: sign_data/train/032_forg/02_0117032.PNG  
      inflating: sign_data/train/032_forg/02_0120032.PNG  
      inflating: sign_data/train/032_forg/03_0112032.PNG  
      inflating: sign_data/train/032_forg/03_0117032.PNG  
      inflating: sign_data/train/032_forg/03_0120032.PNG  
      inflating: sign_data/train/032_forg/04_0112032.PNG  
      inflating: sign_data/train/032_forg/04_0117032.PNG  
      inflating: sign_data/train/032_forg/04_0120032.PNG  
      inflating: sign_data/train/033/01_033.png  
      inflating: sign_data/train/033/02_033.png  
      inflating: sign_data/train/033/03_033.png  
      inflating: sign_data/train/033/04_033.png  
      inflating: sign_data/train/033/05_033.png  
      inflating: sign_data/train/033/06_033.png  
      inflating: sign_data/train/033/07_033.png  
      inflating: sign_data/train/033/08_033.png  
      inflating: sign_data/train/033/09_033.png  
      inflating: sign_data/train/033/10_033.png  
      inflating: sign_data/train/033/11_033.png  
      inflating: sign_data/train/033/12_033.png  
      inflating: sign_data/train/033_forg/01_0112033.PNG  
      inflating: sign_data/train/033_forg/01_0203033.PNG  
      inflating: sign_data/train/033_forg/01_0205033.PNG  
      inflating: sign_data/train/033_forg/01_0213033.PNG  
      inflating: sign_data/train/033_forg/02_0112033.PNG  
      inflating: sign_data/train/033_forg/02_0203033.PNG  
      inflating: sign_data/train/033_forg/02_0205033.PNG  
      inflating: sign_data/train/033_forg/02_0213033.PNG  
      inflating: sign_data/train/033_forg/03_0112033.PNG  
      inflating: sign_data/train/033_forg/03_0203033.PNG  
      inflating: sign_data/train/033_forg/03_0205033.PNG  
      inflating: sign_data/train/033_forg/03_0213033.PNG  
      inflating: sign_data/train/033_forg/04_0112033.PNG  
      inflating: sign_data/train/033_forg/04_0203033.PNG  
      inflating: sign_data/train/033_forg/04_0205033.PNG  
      inflating: sign_data/train/033_forg/04_0213033.PNG  
      inflating: sign_data/train/034/01_034.png  
      inflating: sign_data/train/034/02_034.png  
      inflating: sign_data/train/034/03_034.png  
      inflating: sign_data/train/034/04_034.png  
      inflating: sign_data/train/034/05_034.png  
      inflating: sign_data/train/034/06_034.png  
      inflating: sign_data/train/034/07_034.png  
      inflating: sign_data/train/034/08_034.png  
      inflating: sign_data/train/034/09_034.png  
      inflating: sign_data/train/034/10_034.png  
      inflating: sign_data/train/034/11_034.png  
      inflating: sign_data/train/034/12_034.png  
      inflating: sign_data/train/034_forg/01_0103034.PNG  
      inflating: sign_data/train/034_forg/01_0110034.PNG  
      inflating: sign_data/train/034_forg/01_0120034.PNG  
      inflating: sign_data/train/034_forg/02_0103034.PNG  
      inflating: sign_data/train/034_forg/02_0110034.PNG  
      inflating: sign_data/train/034_forg/02_0120034.PNG  
      inflating: sign_data/train/034_forg/03_0103034.PNG  
      inflating: sign_data/train/034_forg/03_0110034.PNG  
      inflating: sign_data/train/034_forg/03_0120034.PNG  
      inflating: sign_data/train/034_forg/04_0103034.PNG  
      inflating: sign_data/train/034_forg/04_0110034.PNG  
      inflating: sign_data/train/034_forg/04_0120034.PNG  
      inflating: sign_data/train/035/01_035.png  
      inflating: sign_data/train/035/02_035.png  
      inflating: sign_data/train/035/03_035.png  
      inflating: sign_data/train/035/04_035.png  
      inflating: sign_data/train/035/05_035.png  
      inflating: sign_data/train/035/06_035.png  
      inflating: sign_data/train/035/07_035.png  
      inflating: sign_data/train/035/08_035.png  
      inflating: sign_data/train/035/09_035.png  
      inflating: sign_data/train/035/10_035.png  
      inflating: sign_data/train/035/11_035.png  
      inflating: sign_data/train/035/12_035.png  
      inflating: sign_data/train/035_forg/01_0103035.PNG  
      inflating: sign_data/train/035_forg/01_0115035.PNG  
      inflating: sign_data/train/035_forg/01_0201035.PNG  
      inflating: sign_data/train/035_forg/02_0103035.PNG  
      inflating: sign_data/train/035_forg/02_0115035.PNG  
      inflating: sign_data/train/035_forg/02_0201035.PNG  
      inflating: sign_data/train/035_forg/03_0103035.PNG  
      inflating: sign_data/train/035_forg/03_0115035.PNG  
      inflating: sign_data/train/035_forg/03_0201035.PNG  
      inflating: sign_data/train/035_forg/04_0103035.PNG  
      inflating: sign_data/train/035_forg/04_0115035.PNG  
      inflating: sign_data/train/035_forg/04_0201035.PNG  
      inflating: sign_data/train/036/01_036.png  
      inflating: sign_data/train/036/02_036.png  
      inflating: sign_data/train/036/03_036.png  
      inflating: sign_data/train/036/04_036.png  
      inflating: sign_data/train/036/05_036.png  
      inflating: sign_data/train/036/06_036.png  
      inflating: sign_data/train/036/07_036.png  
      inflating: sign_data/train/036/08_036.png  
      inflating: sign_data/train/036/09_036.png  
      inflating: sign_data/train/036/10_036.png  
      inflating: sign_data/train/036/11_036.png  
      inflating: sign_data/train/036/12_036.png  
      inflating: sign_data/train/036_forg/01_0109036.PNG  
      inflating: sign_data/train/036_forg/01_0118036.PNG  
      inflating: sign_data/train/036_forg/01_0123036.PNG  
      inflating: sign_data/train/036_forg/02_0109036.PNG  
      inflating: sign_data/train/036_forg/02_0118036.PNG  
      inflating: sign_data/train/036_forg/02_0123036.PNG  
      inflating: sign_data/train/036_forg/03_0109036.PNG  
      inflating: sign_data/train/036_forg/03_0118036.PNG  
      inflating: sign_data/train/036_forg/03_0123036.PNG  
      inflating: sign_data/train/036_forg/04_0109036.PNG  
      inflating: sign_data/train/036_forg/04_0118036.PNG  
      inflating: sign_data/train/036_forg/04_0123036.PNG  
      inflating: sign_data/train/037/01_037.png  
      inflating: sign_data/train/037/02_037.png  
      inflating: sign_data/train/037/03_037.png  
      inflating: sign_data/train/037/04_037.png  
      inflating: sign_data/train/037/05_037.png  
      inflating: sign_data/train/037/06_037.png  
      inflating: sign_data/train/037/07_037.png  
      inflating: sign_data/train/037/08_037.png  
      inflating: sign_data/train/037/09_037.png  
      inflating: sign_data/train/037/10_037.png  
      inflating: sign_data/train/037/11_037.png  
      inflating: sign_data/train/037/12_037.png  
      inflating: sign_data/train/037_forg/01_0114037.PNG  
      inflating: sign_data/train/037_forg/01_0123037.PNG  
      inflating: sign_data/train/037_forg/01_0208037.PNG  
      inflating: sign_data/train/037_forg/01_0214037.PNG  
      inflating: sign_data/train/037_forg/02_0114037.PNG  
      inflating: sign_data/train/037_forg/02_0123037.PNG  
      inflating: sign_data/train/037_forg/02_0208037.PNG  
      inflating: sign_data/train/037_forg/02_0214037.PNG  
      inflating: sign_data/train/037_forg/03_0114037.PNG  
      inflating: sign_data/train/037_forg/03_0123037.PNG  
      inflating: sign_data/train/037_forg/03_0208037.PNG  
      inflating: sign_data/train/037_forg/03_0214037.PNG  
      inflating: sign_data/train/037_forg/04_0114037.PNG  
      inflating: sign_data/train/037_forg/04_0123037.PNG  
      inflating: sign_data/train/037_forg/04_0208037.PNG  
      inflating: sign_data/train/037_forg/04_0214037.PNG  
      inflating: sign_data/train/038/01_038.png  
      inflating: sign_data/train/038/02_038.png  
      inflating: sign_data/train/038/03_038.png  
      inflating: sign_data/train/038/04_038.png  
      inflating: sign_data/train/038/05_038.png  
      inflating: sign_data/train/038/06_038.png  
      inflating: sign_data/train/038/07_038.png  
      inflating: sign_data/train/038/08_038.png  
      inflating: sign_data/train/038/09_038.png  
      inflating: sign_data/train/038/10_038.png  
      inflating: sign_data/train/038/11_038.png  
      inflating: sign_data/train/038/12_038.png  
      inflating: sign_data/train/038_forg/01_0101038.PNG  
      inflating: sign_data/train/038_forg/01_0124038.PNG  
      inflating: sign_data/train/038_forg/01_0213038.PNG  
      inflating: sign_data/train/038_forg/02_0101038.PNG  
      inflating: sign_data/train/038_forg/02_0124038.PNG  
      inflating: sign_data/train/038_forg/02_0213038.PNG  
      inflating: sign_data/train/038_forg/03_0101038.PNG  
      inflating: sign_data/train/038_forg/03_0124038.PNG  
      inflating: sign_data/train/038_forg/03_0213038.PNG  
      inflating: sign_data/train/038_forg/04_0101038.PNG  
      inflating: sign_data/train/038_forg/04_0124038.PNG  
      inflating: sign_data/train/038_forg/04_0213038.PNG  
      inflating: sign_data/train/039/01_039.png  
      inflating: sign_data/train/039/02_039.png  
      inflating: sign_data/train/039/03_039.png  
      inflating: sign_data/train/039/04_039.png  
      inflating: sign_data/train/039/05_039.png  
      inflating: sign_data/train/039/06_039.png  
      inflating: sign_data/train/039/07_039.png  
      inflating: sign_data/train/039/08_039.png  
      inflating: sign_data/train/039/09_039.png  
      inflating: sign_data/train/039/10_039.png  
      inflating: sign_data/train/039/11_039.png  
      inflating: sign_data/train/039/12_039.png  
      inflating: sign_data/train/039_forg/01_0102039.PNG  
      inflating: sign_data/train/039_forg/01_0108039.PNG  
      inflating: sign_data/train/039_forg/01_0113039.PNG  
      inflating: sign_data/train/039_forg/02_0102039.PNG  
      inflating: sign_data/train/039_forg/02_0108039.PNG  
      inflating: sign_data/train/039_forg/02_0113039.PNG  
      inflating: sign_data/train/039_forg/03_0102039.PNG  
      inflating: sign_data/train/039_forg/03_0108039.PNG  
      inflating: sign_data/train/039_forg/03_0113039.PNG  
      inflating: sign_data/train/039_forg/04_0102039.PNG  
      inflating: sign_data/train/039_forg/04_0108039.PNG  
      inflating: sign_data/train/039_forg/04_0113039.PNG  
      inflating: sign_data/train/040/01_040.png  
      inflating: sign_data/train/040/02_040.png  
      inflating: sign_data/train/040/03_040.png  
      inflating: sign_data/train/040/04_040.png  
      inflating: sign_data/train/040/05_040.png  
      inflating: sign_data/train/040/06_040.png  
      inflating: sign_data/train/040/07_040.png  
      inflating: sign_data/train/040/08_040.png  
      inflating: sign_data/train/040/09_040.png  
      inflating: sign_data/train/040/10_040.png  
      inflating: sign_data/train/040/11_040.png  
      inflating: sign_data/train/040/12_040.png  
      inflating: sign_data/train/040_forg/01_0114040.PNG  
      inflating: sign_data/train/040_forg/01_0121040.PNG  
      inflating: sign_data/train/040_forg/02_0114040.PNG  
      inflating: sign_data/train/040_forg/02_0121040.PNG  
      inflating: sign_data/train/040_forg/03_0114040.PNG  
      inflating: sign_data/train/040_forg/03_0121040.PNG  
      inflating: sign_data/train/040_forg/04_0114040.PNG  
      inflating: sign_data/train/040_forg/04_0121040.PNG  
      inflating: sign_data/train/041/01_041.png  
      inflating: sign_data/train/041/02_041.png  
      inflating: sign_data/train/041/03_041.png  
      inflating: sign_data/train/041/04_041.png  
      inflating: sign_data/train/041/05_041.png  
      inflating: sign_data/train/041/06_041.png  
      inflating: sign_data/train/041/07_041.png  
      inflating: sign_data/train/041/08_041.png  
      inflating: sign_data/train/041/09_041.png  
      inflating: sign_data/train/041/10_041.png  
      inflating: sign_data/train/041/11_041.png  
      inflating: sign_data/train/041/12_041.png  
      inflating: sign_data/train/041_forg/01_0105041.PNG  
      inflating: sign_data/train/041_forg/01_0116041.PNG  
      inflating: sign_data/train/041_forg/01_0117041.PNG  
      inflating: sign_data/train/041_forg/02_0105041.PNG  
      inflating: sign_data/train/041_forg/02_0116041.PNG  
      inflating: sign_data/train/041_forg/02_0117041.PNG  
      inflating: sign_data/train/041_forg/03_0105041.PNG  
      inflating: sign_data/train/041_forg/03_0116041.PNG  
      inflating: sign_data/train/041_forg/03_0117041.PNG  
      inflating: sign_data/train/041_forg/04_0105041.PNG  
      inflating: sign_data/train/041_forg/04_0116041.PNG  
      inflating: sign_data/train/041_forg/04_0117041.PNG  
      inflating: sign_data/train/042/01_042.png  
      inflating: sign_data/train/042/02_042.png  
      inflating: sign_data/train/042/03_042.png  
      inflating: sign_data/train/042/04_042.png  
      inflating: sign_data/train/042/05_042.png  
      inflating: sign_data/train/042/06_042.png  
      inflating: sign_data/train/042/07_042.png  
      inflating: sign_data/train/042/08_042.png  
      inflating: sign_data/train/042/09_042.png  
      inflating: sign_data/train/042/10_042.png  
      inflating: sign_data/train/042/11_042.png  
      inflating: sign_data/train/042/12_042.png  
      inflating: sign_data/train/042_forg/01_0107042.PNG  
      inflating: sign_data/train/042_forg/01_0118042.PNG  
      inflating: sign_data/train/042_forg/01_0204042.PNG  
      inflating: sign_data/train/042_forg/02_0107042.PNG  
      inflating: sign_data/train/042_forg/02_0118042.PNG  
      inflating: sign_data/train/042_forg/02_0204042.PNG  
      inflating: sign_data/train/042_forg/03_0107042.PNG  
      inflating: sign_data/train/042_forg/03_0118042.PNG  
      inflating: sign_data/train/042_forg/03_0204042.PNG  
      inflating: sign_data/train/042_forg/04_0107042.PNG  
      inflating: sign_data/train/042_forg/04_0118042.PNG  
      inflating: sign_data/train/042_forg/04_0204042.PNG  
      inflating: sign_data/train/043/01_043.png  
      inflating: sign_data/train/043/02_043.png  
      inflating: sign_data/train/043/03_043.png  
      inflating: sign_data/train/043/04_043.png  
      inflating: sign_data/train/043/05_043.png  
      inflating: sign_data/train/043/06_043.png  
      inflating: sign_data/train/043/07_043.png  
      inflating: sign_data/train/043/08_043.png  
      inflating: sign_data/train/043/09_043.png  
      inflating: sign_data/train/043/10_043.png  
      inflating: sign_data/train/043/11_043.png  
      inflating: sign_data/train/043/12_043.png  
      inflating: sign_data/train/043_forg/01_0111043.PNG  
      inflating: sign_data/train/043_forg/01_0201043.PNG  
      inflating: sign_data/train/043_forg/01_0211043.PNG  
      inflating: sign_data/train/043_forg/02_0111043.PNG  
      inflating: sign_data/train/043_forg/02_0201043.PNG  
      inflating: sign_data/train/043_forg/02_0211043.PNG  
      inflating: sign_data/train/043_forg/03_0111043.PNG  
      inflating: sign_data/train/043_forg/03_0201043.PNG  
      inflating: sign_data/train/043_forg/03_0211043.PNG  
      inflating: sign_data/train/043_forg/04_0111043.PNG  
      inflating: sign_data/train/043_forg/04_0201043.PNG  
      inflating: sign_data/train/043_forg/04_0211043.PNG  
      inflating: sign_data/train/044/01_044.png  
      inflating: sign_data/train/044/02_044.png  
      inflating: sign_data/train/044/03_044.png  
      inflating: sign_data/train/044/04_044.png  
      inflating: sign_data/train/044/05_044.png  
      inflating: sign_data/train/044/06_044.png  
      inflating: sign_data/train/044/07_044.png  
      inflating: sign_data/train/044/08_044.png  
      inflating: sign_data/train/044/09_044.png  
      inflating: sign_data/train/044/10_044.png  
      inflating: sign_data/train/044/11_044.png  
      inflating: sign_data/train/044/12_044.png  
      inflating: sign_data/train/044_forg/01_0103044.PNG  
      inflating: sign_data/train/044_forg/01_0112044.PNG  
      inflating: sign_data/train/044_forg/01_0211044.PNG  
      inflating: sign_data/train/044_forg/02_0103044.PNG  
      inflating: sign_data/train/044_forg/02_0112044.PNG  
      inflating: sign_data/train/044_forg/02_0211044.PNG  
      inflating: sign_data/train/044_forg/03_0103044.PNG  
      inflating: sign_data/train/044_forg/03_0112044.PNG  
      inflating: sign_data/train/044_forg/03_0211044.PNG  
      inflating: sign_data/train/044_forg/04_0103044.PNG  
      inflating: sign_data/train/044_forg/04_0112044.PNG  
      inflating: sign_data/train/044_forg/04_0211044.PNG  
      inflating: sign_data/train/045/01_045.png  
      inflating: sign_data/train/045/02_045.png  
      inflating: sign_data/train/045/03_045.png  
      inflating: sign_data/train/045/04_045.png  
      inflating: sign_data/train/045/05_045.png  
      inflating: sign_data/train/045/06_045.png  
      inflating: sign_data/train/045/07_045.png  
      inflating: sign_data/train/045/08_045.png  
      inflating: sign_data/train/045/09_045.png  
      inflating: sign_data/train/045/10_045.png  
      inflating: sign_data/train/045/11_045.png  
      inflating: sign_data/train/045/12_045.png  
      inflating: sign_data/train/045_forg/01_0111045.PNG  
      inflating: sign_data/train/045_forg/01_0116045.PNG  
      inflating: sign_data/train/045_forg/01_0205045.PNG  
      inflating: sign_data/train/045_forg/02_0111045.PNG  
      inflating: sign_data/train/045_forg/02_0116045.PNG  
      inflating: sign_data/train/045_forg/02_0205045.PNG  
      inflating: sign_data/train/045_forg/03_0111045.PNG  
      inflating: sign_data/train/045_forg/03_0116045.PNG  
      inflating: sign_data/train/045_forg/03_0205045.PNG  
      inflating: sign_data/train/045_forg/04_0111045.PNG  
      inflating: sign_data/train/045_forg/04_0116045.PNG  
      inflating: sign_data/train/045_forg/04_0205045.PNG  
      inflating: sign_data/train/046/01_046.png  
      inflating: sign_data/train/046/02_046.png  
      inflating: sign_data/train/046/03_046.png  
      inflating: sign_data/train/046/04_046.png  
      inflating: sign_data/train/046/05_046.png  
      inflating: sign_data/train/046/06_046.png  
      inflating: sign_data/train/046/07_046.png  
      inflating: sign_data/train/046/08_046.png  
      inflating: sign_data/train/046/09_046.png  
      inflating: sign_data/train/046/10_046.png  
      inflating: sign_data/train/046/11_046.png  
      inflating: sign_data/train/046/12_046.png  
      inflating: sign_data/train/046_forg/01_0107046.PNG  
      inflating: sign_data/train/046_forg/01_0108046.PNG  
      inflating: sign_data/train/046_forg/01_0123046.PNG  
      inflating: sign_data/train/046_forg/02_0107046.PNG  
      inflating: sign_data/train/046_forg/02_0108046.PNG  
      inflating: sign_data/train/046_forg/02_0123046.PNG  
      inflating: sign_data/train/046_forg/03_0107046.PNG  
      inflating: sign_data/train/046_forg/03_0108046.PNG  
      inflating: sign_data/train/046_forg/03_0123046.PNG  
      inflating: sign_data/train/046_forg/04_0107046.PNG  
      inflating: sign_data/train/046_forg/04_0108046.PNG  
      inflating: sign_data/train/046_forg/04_0123046.PNG  
      inflating: sign_data/train/047/01_047.png  
      inflating: sign_data/train/047/02_047.png  
      inflating: sign_data/train/047/03_047.png  
      inflating: sign_data/train/047/04_047.png  
      inflating: sign_data/train/047/05_047.png  
      inflating: sign_data/train/047/06_047.png  
      inflating: sign_data/train/047/07_047.png  
      inflating: sign_data/train/047/08_047.png  
      inflating: sign_data/train/047/09_047.png  
      inflating: sign_data/train/047/10_047.png  
      inflating: sign_data/train/047/11_047.png  
      inflating: sign_data/train/047/12_047.png  
      inflating: sign_data/train/047_forg/01_0113047.PNG  
      inflating: sign_data/train/047_forg/01_0114047.PNG  
      inflating: sign_data/train/047_forg/01_0212047.PNG  
      inflating: sign_data/train/047_forg/02_0113047.PNG  
      inflating: sign_data/train/047_forg/02_0114047.PNG  
      inflating: sign_data/train/047_forg/02_0212047.PNG  
      inflating: sign_data/train/047_forg/03_0113047.PNG  
      inflating: sign_data/train/047_forg/03_0114047.PNG  
      inflating: sign_data/train/047_forg/03_0212047.PNG  
      inflating: sign_data/train/047_forg/04_0113047.PNG  
      inflating: sign_data/train/047_forg/04_0114047.PNG  
      inflating: sign_data/train/047_forg/04_0212047.PNG  
      inflating: sign_data/train/048/01_048.png  
      inflating: sign_data/train/048/02_048.png  
      inflating: sign_data/train/048/03_048.png  
      inflating: sign_data/train/048/04_048.png  
      inflating: sign_data/train/048/05_048.png  
      inflating: sign_data/train/048/06_048.png  
      inflating: sign_data/train/048/07_048.png  
      inflating: sign_data/train/048/08_048.png  
      inflating: sign_data/train/048/09_048.png  
      inflating: sign_data/train/048/10_048.png  
      inflating: sign_data/train/048/11_048.png  
      inflating: sign_data/train/048/12_048.png  
      inflating: sign_data/train/048_forg/01_0106048.PNG  
      inflating: sign_data/train/048_forg/01_0204048.PNG  
      inflating: sign_data/train/048_forg/02_0106048.PNG  
      inflating: sign_data/train/048_forg/02_0204048.PNG  
      inflating: sign_data/train/048_forg/03_0106048.PNG  
      inflating: sign_data/train/048_forg/03_0204048.PNG  
      inflating: sign_data/train/048_forg/04_0106048.PNG  
      inflating: sign_data/train/048_forg/04_0204048.PNG  
      inflating: sign_data/train/049/01_049.png  
      inflating: sign_data/train/049/02_049.png  
      inflating: sign_data/train/049/03_049.png  
      inflating: sign_data/train/049/04_049.png  
      inflating: sign_data/train/049/05_049.png  
      inflating: sign_data/train/049/06_049.png  
      inflating: sign_data/train/049/07_049.png  
      inflating: sign_data/train/049/08_049.png  
      inflating: sign_data/train/049/09_049.png  
      inflating: sign_data/train/049/10_049.png  
      inflating: sign_data/train/049/11_049.png  
      inflating: sign_data/train/049/12_049.png  
      inflating: sign_data/train/049_forg/01_0114049.PNG  
      inflating: sign_data/train/049_forg/01_0206049.PNG  
      inflating: sign_data/train/049_forg/01_0210049.PNG  
      inflating: sign_data/train/049_forg/02_0114049.PNG  
      inflating: sign_data/train/049_forg/02_0206049.PNG  
      inflating: sign_data/train/049_forg/02_0210049.PNG  
      inflating: sign_data/train/049_forg/03_0114049.PNG  
      inflating: sign_data/train/049_forg/03_0206049.PNG  
      inflating: sign_data/train/049_forg/03_0210049.PNG  
      inflating: sign_data/train/049_forg/04_0114049.PNG  
      inflating: sign_data/train/049_forg/04_0206049.PNG  
      inflating: sign_data/train/049_forg/04_0210049.PNG  
      inflating: sign_data/train/050/01_050.png  
      inflating: sign_data/train/050/02_050.png  
      inflating: sign_data/train/050/03_050.png  
      inflating: sign_data/train/050/04_050.png  
      inflating: sign_data/train/050/05_050.png  
      inflating: sign_data/train/050/06_050.png  
      inflating: sign_data/train/050/07_050.png  
      inflating: sign_data/train/050/08_050.png  
      inflating: sign_data/train/050/09_050.png  
      inflating: sign_data/train/050/10_050.png  
      inflating: sign_data/train/050/11_050.png  
      inflating: sign_data/train/050/12_050.png  
      inflating: sign_data/train/050_forg/01_0125050.PNG  
      inflating: sign_data/train/050_forg/01_0126050.PNG  
      inflating: sign_data/train/050_forg/01_0204050.PNG  
      inflating: sign_data/train/050_forg/02_0125050.PNG  
      inflating: sign_data/train/050_forg/02_0126050.PNG  
      inflating: sign_data/train/050_forg/02_0204050.PNG  
      inflating: sign_data/train/050_forg/03_0125050.PNG  
      inflating: sign_data/train/050_forg/03_0126050.PNG  
      inflating: sign_data/train/050_forg/03_0204050.PNG  
      inflating: sign_data/train/050_forg/04_0125050.PNG  
      inflating: sign_data/train/050_forg/04_0126050.PNG  
      inflating: sign_data/train/050_forg/04_0204050.PNG  
      inflating: sign_data/train/051/01_051.png  
      inflating: sign_data/train/051/02_051.png  
      inflating: sign_data/train/051/03_051.png  
      inflating: sign_data/train/051/04_051.png  
      inflating: sign_data/train/051/05_051.png  
      inflating: sign_data/train/051/06_051.png  
      inflating: sign_data/train/051/07_051.png  
      inflating: sign_data/train/051/08_051.png  
      inflating: sign_data/train/051/09_051.png  
      inflating: sign_data/train/051/10_051.png  
      inflating: sign_data/train/051/11_051.png  
      inflating: sign_data/train/051/12_051.png  
      inflating: sign_data/train/051_forg/01_0104051.PNG  
      inflating: sign_data/train/051_forg/01_0120051.PNG  
      inflating: sign_data/train/051_forg/02_0104051.PNG  
      inflating: sign_data/train/051_forg/02_0120051.PNG  
      inflating: sign_data/train/051_forg/03_0104051.PNG  
      inflating: sign_data/train/051_forg/03_0120051.PNG  
      inflating: sign_data/train/051_forg/04_0104051.PNG  
      inflating: sign_data/train/051_forg/04_0120051.PNG  
      inflating: sign_data/train/052/01_052.png  
      inflating: sign_data/train/052/02_052.png  
      inflating: sign_data/train/052/03_052.png  
      inflating: sign_data/train/052/04_052.png  
      inflating: sign_data/train/052/05_052.png  
      inflating: sign_data/train/052/06_052.png  
      inflating: sign_data/train/052/07_052.png  
      inflating: sign_data/train/052/08_052.png  
      inflating: sign_data/train/052/09_052.png  
      inflating: sign_data/train/052/10_052.png  
      inflating: sign_data/train/052/11_052.png  
      inflating: sign_data/train/052/12_052.png  
      inflating: sign_data/train/052_forg/01_0106052.PNG  
      inflating: sign_data/train/052_forg/01_0109052.PNG  
      inflating: sign_data/train/052_forg/01_0207052.PNG  
      inflating: sign_data/train/052_forg/01_0210052.PNG  
      inflating: sign_data/train/052_forg/02_0106052.PNG  
      inflating: sign_data/train/052_forg/02_0109052.PNG  
      inflating: sign_data/train/052_forg/02_0207052.PNG  
      inflating: sign_data/train/052_forg/02_0210052.PNG  
      inflating: sign_data/train/052_forg/03_0106052.PNG  
      inflating: sign_data/train/052_forg/03_0109052.PNG  
      inflating: sign_data/train/052_forg/03_0207052.PNG  
      inflating: sign_data/train/052_forg/03_0210052.PNG  
      inflating: sign_data/train/052_forg/04_0106052.PNG  
      inflating: sign_data/train/052_forg/04_0109052.PNG  
      inflating: sign_data/train/052_forg/04_0207052.PNG  
      inflating: sign_data/train/052_forg/04_0210052.PNG  
      inflating: sign_data/train/053/01_053.png  
      inflating: sign_data/train/053/02_053.png  
      inflating: sign_data/train/053/03_053.png  
      inflating: sign_data/train/053/04_053.png  
      inflating: sign_data/train/053/05_053.png  
      inflating: sign_data/train/053/06_053.png  
      inflating: sign_data/train/053/07_053.png  
      inflating: sign_data/train/053/08_053.png  
      inflating: sign_data/train/053/09_053.png  
      inflating: sign_data/train/053/10_053.png  
      inflating: sign_data/train/053/11_053.png  
      inflating: sign_data/train/053/12_053.png  
      inflating: sign_data/train/053_forg/01_0107053.PNG  
      inflating: sign_data/train/053_forg/01_0115053.PNG  
      inflating: sign_data/train/053_forg/01_0202053.PNG  
      inflating: sign_data/train/053_forg/01_0207053.PNG  
      inflating: sign_data/train/053_forg/02_0107053.PNG  
      inflating: sign_data/train/053_forg/02_0115053.PNG  
      inflating: sign_data/train/053_forg/02_0202053.PNG  
      inflating: sign_data/train/053_forg/02_0207053.PNG  
      inflating: sign_data/train/053_forg/03_0107053.PNG  
      inflating: sign_data/train/053_forg/03_0115053.PNG  
      inflating: sign_data/train/053_forg/03_0202053.PNG  
      inflating: sign_data/train/053_forg/03_0207053.PNG  
      inflating: sign_data/train/053_forg/04_0107053.PNG  
      inflating: sign_data/train/053_forg/04_0115053.PNG  
      inflating: sign_data/train/053_forg/04_0202053.PNG  
      inflating: sign_data/train/053_forg/04_0207053.PNG  
      inflating: sign_data/train/054/01_054.png  
      inflating: sign_data/train/054/02_054.png  
      inflating: sign_data/train/054/03_054.png  
      inflating: sign_data/train/054/04_054.png  
      inflating: sign_data/train/054/05_054.png  
      inflating: sign_data/train/054/06_054.png  
      inflating: sign_data/train/054/07_054.png  
      inflating: sign_data/train/054/08_054.png  
      inflating: sign_data/train/054/09_054.png  
      inflating: sign_data/train/054/10_054.png  
      inflating: sign_data/train/054/11_054.png  
      inflating: sign_data/train/054/12_054.png  
      inflating: sign_data/train/054_forg/01_0102054.PNG  
      inflating: sign_data/train/054_forg/01_0124054.PNG  
      inflating: sign_data/train/054_forg/01_0207054.PNG  
      inflating: sign_data/train/054_forg/01_0208054.PNG  
      inflating: sign_data/train/054_forg/01_0214054.PNG  
      inflating: sign_data/train/054_forg/02_0102054.PNG  
      inflating: sign_data/train/054_forg/02_0124054.PNG  
      inflating: sign_data/train/054_forg/02_0207054.PNG  
      inflating: sign_data/train/054_forg/02_0208054.PNG  
      inflating: sign_data/train/054_forg/02_0214054.PNG  
      inflating: sign_data/train/054_forg/03_0102054.PNG  
      inflating: sign_data/train/054_forg/03_0124054.PNG  
      inflating: sign_data/train/054_forg/03_0207054.PNG  
      inflating: sign_data/train/054_forg/03_0208054.PNG  
      inflating: sign_data/train/054_forg/03_0214054.PNG  
      inflating: sign_data/train/054_forg/04_0102054.PNG  
      inflating: sign_data/train/054_forg/04_0124054.PNG  
      inflating: sign_data/train/054_forg/04_0207054.PNG  
      inflating: sign_data/train/054_forg/04_0208054.PNG  
      inflating: sign_data/train/054_forg/04_0214054.PNG  
      inflating: sign_data/train/055/01_055.png  
      inflating: sign_data/train/055/02_055.png  
      inflating: sign_data/train/055/03_055.png  
      inflating: sign_data/train/055/04_055.png  
      inflating: sign_data/train/055/05_055.png  
      inflating: sign_data/train/055/06_055.png  
      inflating: sign_data/train/055/07_055.png  
      inflating: sign_data/train/055/08_055.png  
      inflating: sign_data/train/055/09_055.png  
      inflating: sign_data/train/055/10_055.png  
      inflating: sign_data/train/055/11_055.png  
      inflating: sign_data/train/055/12_055.png  
      inflating: sign_data/train/055_forg/01_0118055.PNG  
      inflating: sign_data/train/055_forg/01_0120055.PNG  
      inflating: sign_data/train/055_forg/01_0202055.PNG  
      inflating: sign_data/train/055_forg/02_0118055.PNG  
      inflating: sign_data/train/055_forg/02_0120055.PNG  
      inflating: sign_data/train/055_forg/02_0202055.PNG  
      inflating: sign_data/train/055_forg/03_0118055.PNG  
      inflating: sign_data/train/055_forg/03_0120055.PNG  
      inflating: sign_data/train/055_forg/03_0202055.PNG  
      inflating: sign_data/train/055_forg/04_0118055.PNG  
      inflating: sign_data/train/055_forg/04_0120055.PNG  
      inflating: sign_data/train/055_forg/04_0202055.PNG  
      inflating: sign_data/train/056/01_056.png  
      inflating: sign_data/train/056/02_056.png  
      inflating: sign_data/train/056/03_056.png  
      inflating: sign_data/train/056/04_056.png  
      inflating: sign_data/train/056/05_056.png  
      inflating: sign_data/train/056/06_056.png  
      inflating: sign_data/train/056/07_056.png  
      inflating: sign_data/train/056/08_056.png  
      inflating: sign_data/train/056/09_056.png  
      inflating: sign_data/train/056/10_056.png  
      inflating: sign_data/train/056/11_056.png  
      inflating: sign_data/train/056/12_056.png  
      inflating: sign_data/train/056_forg/01_0105056.PNG  
      inflating: sign_data/train/056_forg/01_0115056.PNG  
      inflating: sign_data/train/056_forg/02_0105056.PNG  
      inflating: sign_data/train/056_forg/02_0115056.PNG  
      inflating: sign_data/train/056_forg/03_0105056.PNG  
      inflating: sign_data/train/056_forg/03_0115056.PNG  
      inflating: sign_data/train/056_forg/04_0105056.PNG  
      inflating: sign_data/train/056_forg/04_0115056.PNG  
      inflating: sign_data/train/057/01_057.png  
      inflating: sign_data/train/057/02_057.png  
      inflating: sign_data/train/057/03_057.png  
      inflating: sign_data/train/057/04_057.png  
      inflating: sign_data/train/057/05_057.png  
      inflating: sign_data/train/057/06_057.png  
      inflating: sign_data/train/057/07_057.png  
      inflating: sign_data/train/057/08_057.png  
      inflating: sign_data/train/057/09_057.png  
      inflating: sign_data/train/057/10_057.png  
      inflating: sign_data/train/057/11_057.png  
      inflating: sign_data/train/057/12_057.png  
      inflating: sign_data/train/057_forg/01_0117057.PNG  
      inflating: sign_data/train/057_forg/01_0208057.PNG  
      inflating: sign_data/train/057_forg/01_0210057.PNG  
      inflating: sign_data/train/057_forg/02_0117057.PNG  
      inflating: sign_data/train/057_forg/02_0208057.PNG  
      inflating: sign_data/train/057_forg/02_0210057.PNG  
      inflating: sign_data/train/057_forg/03_0117057.PNG  
      inflating: sign_data/train/057_forg/03_0208057.PNG  
      inflating: sign_data/train/057_forg/03_0210057.PNG  
      inflating: sign_data/train/057_forg/04_0117057.PNG  
      inflating: sign_data/train/057_forg/04_0208057.PNG  
      inflating: sign_data/train/057_forg/04_0210057.PNG  
      inflating: sign_data/train/058/01_058.png  
      inflating: sign_data/train/058/02_058.png  
      inflating: sign_data/train/058/03_058.png  
      inflating: sign_data/train/058/04_058.png  
      inflating: sign_data/train/058/05_058.png  
      inflating: sign_data/train/058/06_058.png  
      inflating: sign_data/train/058/07_058.png  
      inflating: sign_data/train/058/08_058.png  
      inflating: sign_data/train/058/09_058.png  
      inflating: sign_data/train/058/10_058.png  
      inflating: sign_data/train/058/11_058.png  
      inflating: sign_data/train/058/12_058.png  
      inflating: sign_data/train/058_forg/01_0109058.PNG  
      inflating: sign_data/train/058_forg/01_0110058.PNG  
      inflating: sign_data/train/058_forg/01_0125058.PNG  
      inflating: sign_data/train/058_forg/01_0127058.PNG  
      inflating: sign_data/train/058_forg/02_0109058.PNG  
      inflating: sign_data/train/058_forg/02_0110058.PNG  
      inflating: sign_data/train/058_forg/02_0125058.PNG  
      inflating: sign_data/train/058_forg/02_0127058.PNG  
      inflating: sign_data/train/058_forg/03_0109058.PNG  
      inflating: sign_data/train/058_forg/03_0110058.PNG  
      inflating: sign_data/train/058_forg/03_0125058.PNG  
      inflating: sign_data/train/058_forg/03_0127058.PNG  
      inflating: sign_data/train/058_forg/04_0109058.PNG  
      inflating: sign_data/train/058_forg/04_0110058.PNG  
      inflating: sign_data/train/058_forg/04_0125058.PNG  
      inflating: sign_data/train/058_forg/04_0127058.PNG  
      inflating: sign_data/train/059/01_059.png  
      inflating: sign_data/train/059/02_059.png  
      inflating: sign_data/train/059/03_059.png  
      inflating: sign_data/train/059/04_059.png  
      inflating: sign_data/train/059/05_059.png  
      inflating: sign_data/train/059/06_059.png  
      inflating: sign_data/train/059/07_059.png  
      inflating: sign_data/train/059/08_059.png  
      inflating: sign_data/train/059/09_059.png  
      inflating: sign_data/train/059/10_059.png  
      inflating: sign_data/train/059/11_059.png  
      inflating: sign_data/train/059/12_059.png  
      inflating: sign_data/train/059_forg/01_0104059.PNG  
      inflating: sign_data/train/059_forg/01_0125059.PNG  
      inflating: sign_data/train/059_forg/02_0104059.PNG  
      inflating: sign_data/train/059_forg/02_0125059.PNG  
      inflating: sign_data/train/059_forg/03_0104059.PNG  
      inflating: sign_data/train/059_forg/03_0125059.PNG  
      inflating: sign_data/train/059_forg/04_0104059.PNG  
      inflating: sign_data/train/059_forg/04_0125059.PNG  
      inflating: sign_data/train/060/01_060.png  
      inflating: sign_data/train/060/02_060.png  
      inflating: sign_data/train/060/03_060.png  
      inflating: sign_data/train/060/04_060.png  
      inflating: sign_data/train/060/05_060.png  
      inflating: sign_data/train/060/06_060.png  
      inflating: sign_data/train/060/07_060.png  
      inflating: sign_data/train/060/08_060.png  
      inflating: sign_data/train/060/09_060.png  
      inflating: sign_data/train/060/10_060.png  
      inflating: sign_data/train/060/11_060.png  
      inflating: sign_data/train/060/12_060.png  
      inflating: sign_data/train/060_forg/01_0111060.PNG  
      inflating: sign_data/train/060_forg/01_0121060.PNG  
      inflating: sign_data/train/060_forg/01_0126060.PNG  
      inflating: sign_data/train/060_forg/02_0111060.PNG  
      inflating: sign_data/train/060_forg/02_0121060.PNG  
      inflating: sign_data/train/060_forg/02_0126060.PNG  
      inflating: sign_data/train/060_forg/03_0111060.PNG  
      inflating: sign_data/train/060_forg/03_0121060.PNG  
      inflating: sign_data/train/060_forg/03_0126060.PNG  
      inflating: sign_data/train/060_forg/04_0111060.PNG  
      inflating: sign_data/train/060_forg/04_0121060.PNG  
      inflating: sign_data/train/060_forg/04_0126060.PNG  
      inflating: sign_data/train/061/01_061.png  
      inflating: sign_data/train/061/02_061.png  
      inflating: sign_data/train/061/03_061.png  
      inflating: sign_data/train/061/04_061.png  
      inflating: sign_data/train/061/05_061.png  
      inflating: sign_data/train/061/06_061.png  
      inflating: sign_data/train/061/07_061.png  
      inflating: sign_data/train/061/08_061.png  
      inflating: sign_data/train/061/09_061.png  
      inflating: sign_data/train/061/10_061.png  
      inflating: sign_data/train/061/11_061.png  
      inflating: sign_data/train/061/12_061.png  
      inflating: sign_data/train/061_forg/01_0102061.PNG  
      inflating: sign_data/train/061_forg/01_0112061.PNG  
      inflating: sign_data/train/061_forg/01_0206061.PNG  
      inflating: sign_data/train/061_forg/02_0102061.PNG  
      inflating: sign_data/train/061_forg/02_0112061.PNG  
      inflating: sign_data/train/061_forg/02_0206061.PNG  
      inflating: sign_data/train/061_forg/03_0102061.PNG  
      inflating: sign_data/train/061_forg/03_0112061.PNG  
      inflating: sign_data/train/061_forg/03_0206061.PNG  
      inflating: sign_data/train/061_forg/04_0102061.PNG  
      inflating: sign_data/train/061_forg/04_0112061.PNG  
      inflating: sign_data/train/061_forg/04_0206061.PNG  
      inflating: sign_data/train/062/01_062.png  
      inflating: sign_data/train/062/02_062.png  
      inflating: sign_data/train/062/03_062.png  
      inflating: sign_data/train/062/04_062.png  
      inflating: sign_data/train/062/05_062.png  
      inflating: sign_data/train/062/06_062.png  
      inflating: sign_data/train/062/07_062.png  
      inflating: sign_data/train/062/08_062.png  
      inflating: sign_data/train/062/09_062.png  
      inflating: sign_data/train/062/10_062.png  
      inflating: sign_data/train/062/11_062.png  
      inflating: sign_data/train/062/12_062.png  
      inflating: sign_data/train/062_forg/01_0109062.PNG  
      inflating: sign_data/train/062_forg/01_0116062.PNG  
      inflating: sign_data/train/062_forg/01_0201062.PNG  
      inflating: sign_data/train/062_forg/02_0109062.PNG  
      inflating: sign_data/train/062_forg/02_0116062.PNG  
      inflating: sign_data/train/062_forg/02_0201062.PNG  
      inflating: sign_data/train/062_forg/03_0109062.PNG  
      inflating: sign_data/train/062_forg/03_0116062.PNG  
      inflating: sign_data/train/062_forg/03_0201062.PNG  
      inflating: sign_data/train/062_forg/04_0109062.PNG  
      inflating: sign_data/train/062_forg/04_0116062.PNG  
      inflating: sign_data/train/062_forg/04_0201062.PNG  
      inflating: sign_data/train/063/01_063.png  
      inflating: sign_data/train/063/02_063.png  
      inflating: sign_data/train/063/03_063.png  
      inflating: sign_data/train/063/04_063.png  
      inflating: sign_data/train/063/05_063.png  
      inflating: sign_data/train/063/06_063.png  
      inflating: sign_data/train/063/07_063.png  
      inflating: sign_data/train/063/08_063.png  
      inflating: sign_data/train/063/09_063.png  
      inflating: sign_data/train/063/10_063.png  
      inflating: sign_data/train/063/11_063.png  
      inflating: sign_data/train/063/12_063.png  
      inflating: sign_data/train/063_forg/01_0104063.PNG  
      inflating: sign_data/train/063_forg/01_0108063.PNG  
      inflating: sign_data/train/063_forg/01_0119063.PNG  
      inflating: sign_data/train/063_forg/02_0104063.PNG  
      inflating: sign_data/train/063_forg/02_0108063.PNG  
      inflating: sign_data/train/063_forg/02_0119063.PNG  
      inflating: sign_data/train/063_forg/03_0104063.PNG  
      inflating: sign_data/train/063_forg/03_0108063.PNG  
      inflating: sign_data/train/063_forg/03_0119063.PNG  
      inflating: sign_data/train/063_forg/04_0104063.PNG  
      inflating: sign_data/train/063_forg/04_0108063.PNG  
      inflating: sign_data/train/063_forg/04_0119063.PNG  
      inflating: sign_data/train/064/01_064.png  
      inflating: sign_data/train/064/02_064.png  
      inflating: sign_data/train/064/03_064.png  
      inflating: sign_data/train/064/04_064.png  
      inflating: sign_data/train/064/05_064.png  
      inflating: sign_data/train/064/06_064.png  
      inflating: sign_data/train/064/07_064.png  
      inflating: sign_data/train/064/08_064.png  
      inflating: sign_data/train/064/09_064.png  
      inflating: sign_data/train/064/10_064.png  
      inflating: sign_data/train/064/11_064.png  
      inflating: sign_data/train/064/12_064.png  
      inflating: sign_data/train/064_forg/01_0105064.PNG  
      inflating: sign_data/train/064_forg/01_0203064.PNG  
      inflating: sign_data/train/064_forg/02_0105064.PNG  
      inflating: sign_data/train/064_forg/02_0203064.PNG  
      inflating: sign_data/train/064_forg/03_0105064.PNG  
      inflating: sign_data/train/064_forg/03_0203064.PNG  
      inflating: sign_data/train/064_forg/04_0105064.PNG  
      inflating: sign_data/train/064_forg/04_0203064.PNG  
      inflating: sign_data/train/065/01_065.png  
      inflating: sign_data/train/065/02_065.png  
      inflating: sign_data/train/065/03_065.png  
      inflating: sign_data/train/065/04_065.png  
      inflating: sign_data/train/065/05_065.png  
      inflating: sign_data/train/065/06_065.png  
      inflating: sign_data/train/065/07_065.png  
      inflating: sign_data/train/065/08_065.png  
      inflating: sign_data/train/065/09_065.png  
      inflating: sign_data/train/065/10_065.png  
      inflating: sign_data/train/065/11_065.png  
      inflating: sign_data/train/065/12_065.png  
      inflating: sign_data/train/065_forg/01_0118065.PNG  
      inflating: sign_data/train/065_forg/01_0206065.PNG  
      inflating: sign_data/train/065_forg/02_0118065.PNG  
      inflating: sign_data/train/065_forg/02_0206065.PNG  
      inflating: sign_data/train/065_forg/03_0118065.PNG  
      inflating: sign_data/train/065_forg/03_0206065.PNG  
      inflating: sign_data/train/065_forg/04_0118065.PNG  
      inflating: sign_data/train/065_forg/04_0206065.PNG  
      inflating: sign_data/train/066/01_066.png  
      inflating: sign_data/train/066/02_066.png  
      inflating: sign_data/train/066/03_066.png  
      inflating: sign_data/train/066/04_066.png  
      inflating: sign_data/train/066/05_066.png  
      inflating: sign_data/train/066/06_066.png  
      inflating: sign_data/train/066/07_066.png  
      inflating: sign_data/train/066/08_066.png  
      inflating: sign_data/train/066/09_066.png  
      inflating: sign_data/train/066/10_066.png  
      inflating: sign_data/train/066/11_066.png  
      inflating: sign_data/train/066/12_066.png  
      inflating: sign_data/train/066_forg/01_0101066.PNG  
      inflating: sign_data/train/066_forg/01_0127066.PNG  
      inflating: sign_data/train/066_forg/01_0211066.PNG  
      inflating: sign_data/train/066_forg/01_0212066.PNG  
      inflating: sign_data/train/066_forg/02_0101066.PNG  
      inflating: sign_data/train/066_forg/02_0127066.PNG  
      inflating: sign_data/train/066_forg/02_0211066.PNG  
      inflating: sign_data/train/066_forg/02_0212066.PNG  
      inflating: sign_data/train/066_forg/03_0101066.PNG  
      inflating: sign_data/train/066_forg/03_0127066.PNG  
      inflating: sign_data/train/066_forg/03_0211066.PNG  
      inflating: sign_data/train/066_forg/03_0212066.PNG  
      inflating: sign_data/train/066_forg/04_0101066.PNG  
      inflating: sign_data/train/066_forg/04_0127066.PNG  
      inflating: sign_data/train/066_forg/04_0211066.PNG  
      inflating: sign_data/train/066_forg/04_0212066.PNG  
      inflating: sign_data/train/067/01_067.png  
      inflating: sign_data/train/067/02_067.png  
      inflating: sign_data/train/067/03_067.png  
      inflating: sign_data/train/067/04_067.png  
      inflating: sign_data/train/067/05_067.png  
      inflating: sign_data/train/067/06_067.png  
      inflating: sign_data/train/067/07_067.png  
      inflating: sign_data/train/067/08_067.png  
      inflating: sign_data/train/067/09_067.png  
      inflating: sign_data/train/067/10_067.png  
      inflating: sign_data/train/067/11_067.png  
      inflating: sign_data/train/067/12_067.png  
      inflating: sign_data/train/067_forg/01_0205067.PNG  
      inflating: sign_data/train/067_forg/01_0212067.PNG  
      inflating: sign_data/train/067_forg/02_0205067.PNG  
      inflating: sign_data/train/067_forg/02_0212067.PNG  
      inflating: sign_data/train/067_forg/03_0205067.PNG  
      inflating: sign_data/train/067_forg/03_0212067.PNG  
      inflating: sign_data/train/067_forg/04_0205067.PNG  
      inflating: sign_data/train/067_forg/04_0212067.PNG  
      inflating: sign_data/train/068/01_068.png  
      inflating: sign_data/train/068/02_068.png  
      inflating: sign_data/train/068/03_068.png  
      inflating: sign_data/train/068/04_068.png  
      inflating: sign_data/train/068/05_068.png  
      inflating: sign_data/train/068/06_068.png  
      inflating: sign_data/train/068/07_068.png  
      inflating: sign_data/train/068/08_068.png  
      inflating: sign_data/train/068/09_068.png  
      inflating: sign_data/train/068/10_068.png  
      inflating: sign_data/train/068/11_068.png  
      inflating: sign_data/train/068/12_068.png  
      inflating: sign_data/train/068_forg/01_0113068.PNG  
      inflating: sign_data/train/068_forg/01_0124068.PNG  
      inflating: sign_data/train/068_forg/02_0113068.PNG  
      inflating: sign_data/train/068_forg/02_0124068.PNG  
      inflating: sign_data/train/068_forg/03_0113068.PNG  
      inflating: sign_data/train/068_forg/03_0124068.PNG  
      inflating: sign_data/train/068_forg/04_0113068.PNG  
      inflating: sign_data/train/068_forg/04_0124068.PNG  
      inflating: sign_data/train/069/01_069.png  
      inflating: sign_data/train/069/02_069.png  
      inflating: sign_data/train/069/03_069.png  
      inflating: sign_data/train/069/04_069.png  
      inflating: sign_data/train/069/05_069.png  
      inflating: sign_data/train/069/06_069.png  
      inflating: sign_data/train/069/07_069.png  
      inflating: sign_data/train/069/08_069.png  
      inflating: sign_data/train/069/09_069.png  
      inflating: sign_data/train/069/10_069.png  
      inflating: sign_data/train/069/11_069.png  
      inflating: sign_data/train/069/12_069.png  
      inflating: sign_data/train/069_forg/01_0106069.PNG  
      inflating: sign_data/train/069_forg/01_0108069.PNG  
      inflating: sign_data/train/069_forg/01_0111069.PNG  
      inflating: sign_data/train/069_forg/02_0106069.PNG  
      inflating: sign_data/train/069_forg/02_0108069.PNG  
      inflating: sign_data/train/069_forg/02_0111069.PNG  
      inflating: sign_data/train/069_forg/03_0106069.PNG  
      inflating: sign_data/train/069_forg/03_0108069.PNG  
      inflating: sign_data/train/069_forg/03_0111069.PNG  
      inflating: sign_data/train/069_forg/04_0106069.PNG  
      inflating: sign_data/train/069_forg/04_0108069.PNG  
      inflating: sign_data/train/069_forg/04_0111069.PNG  
      inflating: sign_data/train_data.csv  


Image **preprocessing function** and **data loader** class:


```python
def img_norm(x):
  # a simple image preprocessing function
  return (x - x.mean(axis=(0,1,2), keepdims=True)) / x.std(axis=(0,1,2), keepdims=True)

class DataLoader:
  #constructor
  def __init__(self, dataset, batch_size=32, img_size=112, dir='./'):
    self.dataset = dataset
    self.batch_size = batch_size
    self.dir = dir
    self.img_size = img_size
  #shuffler
  def shuffle(self):
    return self.dataset.sample(frac=1)
  #generator
  def datagen(self, repeat_flag=True):
    num_samples = len(self.dataset)
    while True:
        # shuffling the samples
        self.dataset = self.shuffle()
        for batch in range(1, num_samples, self.batch_size):
            image1_batch_samples = self.dir + "/" + self.dataset.iloc[:, 0][batch:batch + self.batch_size]
            image2_batch_samples = self.dir + "/" + self.dataset.iloc[:, 1][batch:batch + self.batch_size]
            label_batch_samples = self.dataset.iloc[:, 2][batch:batch + self.batch_size]
            Image1, Image2, Label = [], [], []
            for image1, image2, label in zip(image1_batch_samples, image2_batch_samples, label_batch_samples):
                # append them to Images directly
                image1_data = Image.open(image1)
                image2_data = Image.open(image2)
                # resizing the images
                image1_data = image1_data.resize((self.img_size, self.img_size))
                image2_data = image2_data.resize((self.img_size, self.img_size))
                # converting to array
                image1_data = img_to_array(image1_data)
                image2_data = img_to_array(image2_data)

                # image1_data = preprocess_input(image1_data)
                # image2_data = preprocess_input(image2_data)
                image1_data = img_norm(image1_data)
                image2_data = img_norm(image2_data)

                Image1.append(image1_data)
                Image2.append(image2_data)
                Label.append(label)
            # convert each list to numpy arrays to ensure that they get processed by fit function
            Image1 = np.asarray(Image1).astype(np.float32)
            Image2 = np.asarray(Image2).astype(np.float32)

            Label = np.asarray(Label).astype(np.float32)
            yield [Image1, Image2], Label
        if not repeat_flag:
          break
```

generators:


```python
train_set_file = "./sign_data/train_data.csv"
test_set_file = "./sign_data/test_data.csv"

train_val_set = pd.read_csv(train_set_file)
train_set, val_set = train_test_split(train_val_set, test_size=0.2)
test_set = pd.read_csv(test_set_file)

train_gen= DataLoader(train_set, batch_size, img_size, "./sign_data/train/")
test_gen = DataLoader(test_set, batch_size, img_size, "./sign_data/test/")
val_gen= DataLoader(val_set, batch_size, img_size, "./sign_data/train/")
```

Test the train generator:


```python
train_batch = next(train_gen.datagen())
print("Train batch images shape:", train_batch[0][0].shape, train_batch[0][1].shape)
print("Train batch labels shape:", train_batch[1].shape)
```

    Train batch images shape: (64, 224, 224, 3) (64, 224, 224, 3)
    Train batch labels shape: (64,)


# **Model**

Base Model:


```python
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


def custom_cnn():
  model = Sequential()
  model.add(Conv2D(4, (3,3), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(2,2))
  model.add(Dropout(0.25))

  model.add(Conv2D(16, (3,3), activation='relu'))
  model.add(MaxPooling2D(5,5))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256, activation='relu'))

  return model


def def_base_model(backbone='xception', freeze_conv_layers=True):
  print('backbone model: ' + backbone)
  if backbone == 'Xception':
    base_model = Xception(weights='imagenet', include_top=False)
  elif backbone == 'InceptionV3':
    base_model = InceptionV3(weights='imagenet', include_top=False)
  elif backbone == 'ResNet50':
    base_model = ResNet50(weights='imagenet', include_top=False)
  elif backbone == 'MobileNetV2':
    base_model = MobileNetV2(weights='imagenet', include_top=False)
  else:
    raise("unexpected backbone model. Backbone model can be choosen from: "
    "'Xception', 'InceptionV3', 'MobileNetV2', and 'ResNet50'")

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = Dense(128, activation='relu')(x)

  # first: train only the top layers (which were randomly initialized)
  # i.e. freeze all convolutional layers
  if freeze_conv_layers:
    print('freeze convolutional layers ...')
    for layer in base_model.layers:
        layer.trainable = False
  model = Model(inputs=base_model.input, outputs=x)
  return model
```

Siamese Model:


```python
def siamese_model(input_shape, backbone_model='custom_cnn',
                  freeze_conv_layers=True):
    input1 = Input(input_shape)
    input2 = Input(input_shape)

    if backbone_model=='custom_cnn':
      base_model = custom_cnn()
    else:
      base_model = def_base_model(backbone_model, freeze_conv_layers)

    # Call the model with the inputs:
    embedding1 = base_model(input1)
    embedding2 = base_model(input2)

    # custom loss layer:
    loss_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    manhattan_distance = loss_layer([embedding1, embedding2])

    # add a dense layer for 2-class classification (genuine and fraud):
    output = Dense(1, activation='sigmoid')(manhattan_distance)

    network = Model(inputs=[input1, input2], outputs=output)
    return network
```


```python
model = siamese_model((img_size, img_size, 3), backbone_model, freeze_conv_layers)
model.summary()
```

    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.


    backbone model: MobileNetV2
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
    9406464/9406464 [==============================] - 1s 0us/step
    freeze convolutional layers ...
    Model: "model_1"
    __________________________________________________________________________________________________
     Layer (type)                Output Shape                 Param #   Connected to                  
    ==================================================================================================
     input_1 (InputLayer)        [(None, 224, 224, 3)]        0         []                            
                                                                                                      
     input_2 (InputLayer)        [(None, 224, 224, 3)]        0         []                            
                                                                                                      
     model (Functional)          (None, 128)                  2421952   ['input_1[0][0]',             
                                                                         'input_2[0][0]']             
                                                                                                      
     lambda (Lambda)             (None, 128)                  0         ['model[0][0]',               
                                                                         'model[1][0]']               
                                                                                                      
     dense_1 (Dense)             (None, 1)                    129       ['lambda[0][0]']              
                                                                                                      
    ==================================================================================================
    Total params: 2422081 (9.24 MB)
    Trainable params: 164097 (641.00 KB)
    Non-trainable params: 2257984 (8.61 MB)
    __________________________________________________________________________________________________


# **Train & Test**

Define f1_score function to use during training:


```python
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
```

Compile the model:


```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=5*steps_per_epoch,
    decay_rate=0.5)

optimizer = Adam(learning_rate=lr_schedule, weight_decay=0.2)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy', f1_score])
```


```python
early_stopper =  EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)
custom_callback = [early_stopper]
```

Train:


```python
print("Training!")
checkpoint_filepath = data_path + '/best_model.hdf5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(
    train_gen.datagen(),
    verbose=1,
    steps_per_epoch=steps_per_epoch,  # set appropriate steps_per_epoch
    epochs=num_epoches,
    validation_data=val_gen.datagen(),
    validation_steps=1,  # set appropriate validation_steps
    callbacks=[model_checkpoint_callback]
)
```

    Training!
    Epoch 1/5
    100/100 [==============================] - ETA: 0s - loss: 0.3199 - accuracy: 0.8631 - f1_score: 0.8480

    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(


    100/100 [==============================] - 111s 892ms/step - loss: 0.3199 - accuracy: 0.8631 - f1_score: 0.8480 - val_loss: 0.1596 - val_accuracy: 0.9375 - val_f1_score: 0.9375
    Epoch 2/5
    100/100 [==============================] - 86s 867ms/step - loss: 0.0867 - accuracy: 0.9709 - f1_score: 0.9671 - val_loss: 0.0763 - val_accuracy: 0.9688 - val_f1_score: 0.9643
    Epoch 3/5
    100/100 [==============================] - 86s 870ms/step - loss: 0.0465 - accuracy: 0.9883 - f1_score: 0.9870 - val_loss: 0.0413 - val_accuracy: 0.9844 - val_f1_score: 0.9818
    Epoch 4/5
    100/100 [==============================] - 86s 871ms/step - loss: 0.0193 - accuracy: 0.9978 - f1_score: 0.9974 - val_loss: 0.0414 - val_accuracy: 0.9844 - val_f1_score: 0.9831
    Epoch 5/5
    100/100 [==============================] - 87s 879ms/step - loss: 0.0115 - accuracy: 0.9991 - f1_score: 0.9990 - val_loss: 0.0152 - val_accuracy: 1.0000 - val_f1_score: 1.0000


Plot traning curves:


```python
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for f1_score
plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('model f1_score')
plt.ylabel('f1_score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
```

    dict_keys(['loss', 'accuracy', 'f1_score', 'val_loss', 'val_accuracy', 'val_f1_score'])



    
![png](README_files/README_31_1.png)
    



    
![png](README_files/README_31_2.png)
    



    
![png](README_files/README_31_3.png)
    


**Save the trained model**


```python
keras.saving.save_model(model, backbone_model + '.h5', overwrite=True)
```

    <ipython-input-16-910306078cc1>:1: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      keras.saving.save_model(model, backbone_model + '.h5', overwrite=True)


**Test**


```python
# loaded_model = keras.saving.load_model(backbone_model + '.h5')
loaded_model = keras.saving.load_model(checkpoint_filepath,custom_objects={"f1_score": f1_score})
result = loaded_model.evaluate(test_gen.datagen(repeat_flag=False), batch_size=None,
                               verbose=1, sample_weight=None, steps=None,
                               callbacks=None, max_queue_size=10, workers=1,
                               use_multiprocessing=False, return_dict=False)
```

    90/90 [==============================] - 87s 953ms/step - loss: 0.0122 - accuracy: 0.9981 - f1_score: 0.9981


confusion matrix for 2 classes:


```python
y_gt = []
y_pr = []
for data in test_gen.datagen(repeat_flag=False):
  labels = data[1]
  predictions = loaded_model.predict(data[0], verbose=0)
  for i, label in enumerate(labels):
    y_gt.append(label)
    y_pr.append(predictions[i])
```


```python
y_pr = np.round(y_pr)
cm = confusion_matrix(y_gt, y_pr, normalize='true')
print(cm)
```

    [[9.99278499e-01 7.21500722e-04]
     [3.02622730e-03 9.96973773e-01]]


Calculate classification metrics on test data:


```python
from sklearn.metrics import classification_report
print(classification_report(y_gt, y_pr))
```

                  precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00      2772
             1.0       1.00      1.00      1.00      2974
    
        accuracy                           1.00      5746
       macro avg       1.00      1.00      1.00      5746
    weighted avg       1.00      1.00      1.00      5746
    

