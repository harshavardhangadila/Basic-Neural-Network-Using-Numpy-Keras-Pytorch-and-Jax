# Basic-Neural-Network-Using-Numpy-Keras-Pytorch-and-Jax

### 📘 1) Neural_Network_using_Numpy.ipynb
Implemented a 3-layer feedforward neural network using only NumPy.  
Performed manual forward and backward propagation with chain rule.  
Used a custom 3-variable nonlinear function to generate regression data.  
Visualized 4D input-output space and plotted loss over 1000 epochs.  
Predicted on new data and confirmed generalization with low final MSE.

---

### 📘 2) Neural_Network_TensorFlow_Using_Einsum.ipynb
Built a low-level TensorFlow model using `tf.Variable` and `tf.einsum` instead of `matmul`.  
Used custom activation and training loop with `tf.GradientTape`.  
Handled manual weight updates for a 3-layer network.  
Generated 4D plots and tracked training loss with visual feedback.  
Included predictions and custom test case evaluation.

---

### 📘 3) PyTorch_From_Scratch_(Manual_Layers,_No_nn_Module).ipynb
Built a neural net from scratch in PyTorch without using `nn.Module` or `torch.nn`.  
Defined weights manually, applied autograd for gradient computation.  
Trained the 3-layer network using MSE loss and custom update logic.  
Visualized predictions in 3D space and validated on test points.  
Plotted training loss and evaluated model on unseen data.

---

### 📘 4) PyTorch_with_nn_Module.ipynb
Used `torch.nn.Module` to define a class-based 3-layer network.  
Trained with Adam optimizer and MSE loss via `optimizer.step()`.  
Included 4D visualization, actual vs predicted scatter, and test predictions.  
Tracked loss across epochs and evaluated performance visually.  
Used ReLU activations and dense layers throughout.

---

### 📘 5) PyTorch_Lightning_Version.ipynb
Refactored the previous model into a `LightningModule` using PyTorch Lightning.  
Simplified training and validation steps with built-in logging.  
Used Lightning’s trainer for epoch management and device flexibility.  
Maintained identical architecture and evaluation pipeline.  
Validated performance with visual diagnostics and predictions.

---

### 📘 6) TensorFlow_Sequential_API.ipynb
Used `tf.keras.Sequential` to define a 3-layer neural network with built-in `Dense` layers.  
Trained using `compile()` + `fit()` on synthetic regression data.  
Included validation split, loss curve, 4D prediction plot, and residual histogram.  
Made predictions on custom test inputs.  
Saved model and exported results to CSV.

---

### 📘 7) TensorFlow_Functional_API.ipynb
Built the same 3-layer model using the Functional API with `Input` and `Model`.  
Utilized ReLU activations and Dense layers explicitly.  
Trained using MSE loss and validated via loss curve and predictions.  
Included 4D visualizations and actual vs predicted plots.  
Supported flexible architecture and clear code structure.

---

### 📘 8) TensorFlow_Subclassed_API.ipynb
Created a custom model class by subclassing `tf.keras.Model`.  
Defined layers in `__init__` and forward logic in `call()`.  
Trained using `compile()` + `fit()` and visualized residuals and outputs.  
Generated 4D plots of prediction space and supported custom inference.  
Great demonstration of OOP in TensorFlow.

---

### 📘 9) TensorFlow_Low_Level_Neural_Network.ipynb
Implemented another manual TensorFlow model with a different nonlinear equation.  
Used `tf.Variable`, `tf.GradientTape`, and `tf.einsum` for all operations.  
Tracked training loss and evaluated predictions in detail.  
Visualized 4D output space and residuals.  
Included sorted prediction plots and final test predictions.

---

### 📘 10) TensorFlow_Built_in_Layers_(Sequential_+_Dense).ipynb
Used Sequential API with built-in layers and added advanced diagnostics.  
Included training, residual analysis, sorted comparisons, and feature sensitivity.  
Visualized per-feature error trends and saved predictions to CSV.  
This notebook validates the built-in training pipeline.  
Final results include RMSE and sample evaluations.

---

### 📘 11) Neural_Network_using_JAX.ipynb
Built a 3-layer MLP in JAX using `jax.numpy`, `grad`, `jit`, and `vmap`.  
Implemented He initialization, manual forward pass, and full training loop.  
Added advanced features like feature sensitivity, tree flattening, and per-sample loss.  
Visualized predictions, residuals.  
---
Youtube: [Basic-Neural-Network-Using-Numpy-Keras-Pytorch-and-Jax](https://www.youtube.com/playlist?list=PLCGwaUpxPWO3ySYESDUf6mK1Vz4j9Mvus)
